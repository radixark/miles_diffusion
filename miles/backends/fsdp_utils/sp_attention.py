"""Sequence parallelism for diffusion DiTs: self-attention runs USP (sp_ops).

Each sp rank holds S/sp latent tokens; attention internally gathers to the full
sequence via Ulysses all-to-all (+Ring) and scatters back. A model family opts in
through a ``SequenceParallelPlan``; model outputs stay full-sequence, so every
parameter sees a partial gradient and loss/log_prob code is untouched.
"""

import functools
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass

import torch
from diffusers.models._modeling_parallel import ContextParallelOutput

from .sp_ops import gather_sequence, shard_sequence, usp_attention


@dataclass(frozen=True)
class SequenceParallelPlan:
    """What one transformer family declares to run under SP.

    ``boundaries``: fqn -> ``ContextParallelInput``/``ContextParallelOutput``
    (the diffusers ``_cp_plan`` vocabulary) — where the sequence dim is sharded
    to S/sp and where full-sequence outputs are gathered back.
    ``attention``: called with (transformer, parallel_state); routes the model's
    self-attention through ``sp_ops.usp_attention``.
    """

    boundaries: dict
    attention: Callable[[torch.nn.Module, object], None]
    num_attention_heads: int


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


def _install_boundary_hooks(transformer, boundaries, parallel_state, sum_grad=True):
    """Install shard/gather hooks from the plan's boundary specs.

    ContextParallelInput entries split module inputs (or outputs when
    split_output=True); ContextParallelOutput entries gather module outputs.
    The gather is a differentiable all-gather; its sum_grad backward pairs
    with FSDP's 1/(dp*sp) mean (sum_grad=False is the sp-replicated-parameter
    variant, used by tests as a validation anchor).
    """
    for path, spec in boundaries.items():
        module = _resolve_submodule(transformer, path) if path else transformer

        if isinstance(spec, ContextParallelOutput):

            def gather_output(mod, args, output, _spec=spec):
                assert isinstance(output, torch.Tensor)
                return gather_sequence(
                    output,
                    parallel_state.sp_group,
                    parallel_state.sp_rank,
                    parallel_state.sp_size,
                    dim=_spec.gather_dim,
                    sum_grad=sum_grad,
                )

            module.register_forward_hook(gather_output)
            continue

        input_specs = {k: v for k, v in spec.items() if not v.split_output}
        output_specs = {k: v for k, v in spec.items() if v.split_output}

        if input_specs:
            # Wrappers like PeftModel expose forward(*args, **kwargs); the real
            # parameter names live on the base module.
            sig_module = module.get_base_model() if hasattr(module, "get_base_model") else module
            param_names = list(inspect.signature(sig_module.forward).parameters)
            missing = [k for k in input_specs if isinstance(k, str) and k not in param_names]
            if missing:
                raise ValueError(
                    f"boundary keys {missing} at '{path}' are not parameters of "
                    f"{type(sig_module).__name__}.forward"
                )

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


class _USPDispatchConfig:
    """Marker consumed by the wrapped dispatch_attention_fn; models pass it
    through ``_parallel_config`` for self-attention call sites only."""

    def __init__(self, parallel_state):
        self.ulysses_group = parallel_state.ulysses_group
        self.ring_group = parallel_state.ring_group


def _wrap_dispatch(module):
    original = module.dispatch_attention_fn
    if getattr(original, "_miles_usp_wrapped", False):
        return

    @functools.wraps(original)
    def dispatch(query, key, value, *args, parallel_config=None, **kwargs):
        if not isinstance(parallel_config, _USPDispatchConfig):
            return original(query, key, value, *args, parallel_config=parallel_config, **kwargs)
        attn_mask = args[0] if args else kwargs.get("attn_mask")
        if attn_mask is not None:
            raise ValueError("USP self-attention does not support attention masks")
        if (len(args) > 1 and args[1]) or kwargs.get("dropout_p"):
            raise ValueError("USP self-attention requires dropout_p=0 (per-rank RNG streams diverge)")
        if (len(args) > 2 and args[2]) or kwargs.get("is_causal"):
            raise ValueError("USP self-attention does not support is_causal yet")
        if (len(args) > 3 and args[3] is not None) or kwargs.get("scale") is not None:
            raise ValueError("USP self-attention uses the default 1/sqrt(d) scale")
        return usp_attention(query, key, value, parallel_config.ulysses_group, parallel_config.ring_group)

    dispatch._miles_usp_wrapped = True
    module.dispatch_attention_fn = dispatch


def apply_dispatch_sp_attention(transformer, parallel_state):
    """Default SP attention: intercept the model module's dispatch_attention_fn
    so self-attention call sites (which pass ``_parallel_config`` per upstream
    convention) route through usp_attention; the model's own processors and
    cross-attention stay untouched."""
    base = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
    # fully_shard swizzles the class (FSDP<Name>, defined in torch's fsdp
    # module); the modeling module that imported dispatch_attention_fn is
    # found through the MRO.
    module = next(
        (
            mod
            for cls in type(base).__mro__
            if (mod := sys.modules.get(cls.__module__)) is not None and hasattr(mod, "dispatch_attention_fn")
        ),
        None,
    )
    if module is None:
        raise ValueError(
            f"{type(base).__name__} does not route attention through diffusers' "
            "dispatch_attention_fn; the family must override apply_sp_attention"
        )
    _wrap_dispatch(module)
    config = _USPDispatchConfig(parallel_state)
    for processor in base.attn_processors.values():
        processor._parallel_config = config


def apply_sequence_parallel(transformer, parallel_state, plan, sum_grad=True):
    """Wire SP into one transformer per its plan: install the family's SP
    self-attention and the shard/gather boundary hooks. Call once per
    transformer after FSDP wrapping."""
    if plan.num_attention_heads % parallel_state.ulysses_degree != 0:
        raise ValueError(
            f"num_attention_heads({plan.num_attention_heads}) is not divisible by "
            f"ulysses_degree({parallel_state.ulysses_degree})"
        )
    plan.attention(transformer, parallel_state)
    _install_boundary_hooks(transformer, plan.boundaries, parallel_state, sum_grad=sum_grad)
