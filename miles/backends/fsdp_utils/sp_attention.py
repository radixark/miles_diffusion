"""Sequence parallelism for diffusers DiTs: self-attention runs USP (sp_ops).

Each sp rank holds S/sp latent tokens; attention internally gathers to the full
sequence via Ulysses all-to-all (+Ring) and scatters back. Shard/gather points are
taken from the model's diffusers ``_cp_plan``, so model outputs stay full-sequence
and every parameter sees a partial gradient; the actor's cross-sp grad all-reduce
applies uniformly.
"""

import inspect

import torch
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

from .sp_ops import gather_sequence, shard_sequence, usp_attention


class WanUSPAttnProcessor:
    """Wan attention processor: self-attn via USP, cross-attn via local SDPA.

    Reuses Wan's QKV/RMSNorm/RoPE; rotary_emb arrives pre-sharded to S_local.
    With parallel_state=None the self-attn falls back to plain SDPA (reference mode).
    """

    def __init__(self, parallel_state=None):
        self.ulysses_group = parallel_state.ulysses_group if parallel_state is not None else None
        self.ring_group = parallel_state.ring_group if parallel_state is not None else None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        is_self_attn = encoder_hidden_states is None

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        if is_self_attn:
            hidden_states = usp_attention(query, key, value, self.ulysses_group, self.ring_group)
        else:
            hidden_states = torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if encoder_hidden_states_img is not None:
            key_img = attn.norm_added_k(attn.add_k_proj(encoder_hidden_states_img)).unflatten(2, (attn.heads, -1))
            value_img = attn.add_v_proj(encoder_hidden_states_img).unflatten(2, (attn.heads, -1))
            hidden_states_img = (
                torch.nn.functional.scaled_dot_product_attention(
                    query.transpose(1, 2),
                    key_img.transpose(1, 2),
                    value_img.transpose(1, 2),
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )
                .transpose(1, 2)
                .flatten(2, 3)
                .type_as(query)
            )
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def _apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def _split_if_expected(x, spec, parallel_state):
    if not isinstance(x, torch.Tensor):
        return x
    if spec.expected_dims is not None and x.ndim != spec.expected_dims:
        return x
    return shard_sequence(x, parallel_state.sp_rank, parallel_state.sp_size, dim=spec.split_dim)


def _resolve_submodule(root, path):
    # getattr-based walk: unlike get_submodule, it also traverses attribute
    # forwarding of wrappers such as peft's PeftModel.
    module = root
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _install_cp_plan_hooks(transformer, parallel_state):
    """Install shard/gather hooks from the model's diffusers ``_cp_plan``.

    ContextParallelInput entries split module inputs (or outputs when
    split_output=True); ContextParallelOutput entries gather module outputs.
    The gather is a differentiable all-gather, so backward stays partial-grad.
    """
    for path, spec in transformer._cp_plan.items():
        module = _resolve_submodule(transformer, path) if path else transformer

        if isinstance(spec, ContextParallelOutput):

            def gather_output(mod, args, output, _spec=spec):
                assert isinstance(output, torch.Tensor)
                return gather_sequence(
                    output, parallel_state.sp_group, parallel_state.sp_rank, parallel_state.sp_size, dim=_spec.gather_dim
                )

            module.register_forward_hook(gather_output)
            continue

        input_specs = {k: v for k, v in spec.items() if not v.split_output}
        output_specs = {k: v for k, v in spec.items() if v.split_output}

        if input_specs:
            param_names = list(inspect.signature(module.forward).parameters)

            def split_inputs(mod, args, kwargs, _specs=input_specs, _names=param_names):
                args = list(args)
                for key, s in _specs.items():
                    if isinstance(key, str) and key in kwargs:
                        kwargs[key] = _split_if_expected(kwargs[key], s, parallel_state)
                        continue
                    index = key if isinstance(key, int) else (_names.index(key) if key in _names else None)
                    if index is not None and index < len(args):
                        args[index] = _split_if_expected(args[index], s, parallel_state)
                return tuple(args), kwargs

            module.register_forward_pre_hook(split_inputs, with_kwargs=True)

        if output_specs:

            def split_outputs(mod, args, kwargs, output, _specs=output_specs):
                single = not isinstance(output, tuple)
                out = [output] if single else list(output)
                for index, s in _specs.items():
                    out[index] = _split_if_expected(out[index], s, parallel_state)
                return out[0] if single else tuple(out)

            module.register_forward_hook(split_outputs, with_kwargs=True)


def apply_sequence_parallel(transformer, parallel_state):
    """Wire SP into one transformer: replace self-attn processors and install
    the shard/gather hooks declared by its _cp_plan. Call once per transformer
    after FSDP wrapping."""
    from diffusers import WanTransformer3DModel

    base = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
    if not isinstance(base, WanTransformer3DModel):
        raise ValueError(
            f"SP attention processor currently supports WanTransformer3DModel only, got {base.__class__.__name__}"
        )
    heads = transformer.config.num_attention_heads
    # args carry no head count, so the startup divisibility check must happen
    # here where the real model config is available.
    if heads % parallel_state.ulysses_degree != 0:
        raise ValueError(
            f"num_attention_heads({heads}) is not divisible by ulysses_degree({parallel_state.ulysses_degree})"
        )
    transformer.set_attn_processor(WanUSPAttnProcessor(parallel_state))
    _install_cp_plan_hooks(transformer, parallel_state)
