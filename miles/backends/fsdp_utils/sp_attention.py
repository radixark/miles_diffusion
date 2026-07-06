"""Sequence parallelism for diffusers Wan DiT: self-attention runs sglang's USPAttention.

Each sp rank holds S/sp latent tokens; attention internally gathers to the full
sequence via Ulysses all-to-all (+Ring) and scatters back. The sequence is sharded
before the first block and gathered after proj_out, so every parameter sees a
partial gradient and the actor's cross-sp grad all-reduce applies uniformly.
"""

import functools

import torch
import torch.distributed as dist


class _GatherSequence(torch.autograd.Function):
    """All-gather S_local shards along dim=1; backward returns each rank's slice."""

    @staticmethod
    def forward(ctx, x, group, sp_rank, sp_size):
        ctx.sp_rank = sp_rank
        ctx.s_local = x.shape[1]
        parts = [torch.empty_like(x) for _ in range(sp_size)]
        dist.all_gather(parts, x.contiguous(), group=group)
        return torch.cat(parts, dim=1)

    @staticmethod
    def backward(ctx, grad):
        start = ctx.sp_rank * ctx.s_local
        return grad[:, start : start + ctx.s_local], None, None, None


def shard_sequence(x, parallel_state):
    sp = parallel_state.sp_size
    s = x.shape[1]
    if s % sp:
        raise ValueError(f"sequence length {s} is not divisible by sp_size {sp}")
    s_local = s // sp
    start = parallel_state.sp_rank * s_local
    return x[:, start : start + s_local]


def gather_sequence(x, parallel_state):
    return _GatherSequence.apply(x, parallel_state.sp_group, parallel_state.sp_rank, parallel_state.sp_size)


class WanUSPAttnProcessor:
    """Wan attention processor: self-attn via USPAttention, cross-attn via local SDPA.

    Reuses Wan's QKV/RMSNorm/RoPE; rotary_emb arrives pre-sharded to S_local.
    """

    def __init__(self, num_heads, head_dim, compute_dtype=torch.bfloat16):
        from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention

        init_sp_backend(compute_dtype)
        self.usp_attn = USPAttention(num_heads=num_heads, head_size=head_dim, causal=False)

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
            hidden_states = self.usp_attn(query, key, value)
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


def init_sp_backend(compute_dtype):
    """Set up the sglang runtime state USPAttention depends on.

    The training process never goes through the sglang server launch, so the
    attention backend, mixed-precision policy, and forward context are unset.
    FA requires fp16/bf16; the forward context is persisted module-globally
    because checkpoint recompute runs after any with-scope has exited.
    """
    from sglang.multimodal_gen.runtime.layers.attention.selector import global_force_attn_backend
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang.multimodal_gen.utils import set_mixed_precision_policy

    half = compute_dtype in (torch.float16, torch.bfloat16)
    global_force_attn_backend(AttentionBackendEnum.FA if half else AttentionBackendEnum.TORCH_SDPA)
    set_mixed_precision_policy(param_dtype=compute_dtype, reduce_dtype=torch.float32)

    from sglang.multimodal_gen.runtime.managers import forward_context as fc

    if fc._forward_context is None:
        fc._forward_context = fc.ForwardContext(current_timestep=0, attn_metadata=None)


def _apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def apply_sequence_parallel(transformer, parallel_state, compute_dtype=None):
    """Wire SP into one Wan transformer: replace self-attn processors and install
    the shard/gather hooks. Call once per transformer after FSDP wrapping."""
    heads = transformer.config.num_attention_heads
    head_dim = transformer.config.attention_head_dim
    # args carry no head count, so the startup divisibility check must happen
    # here where the real model config is available.
    if heads % parallel_state.ulysses_degree != 0:
        raise ValueError(
            f"num_attention_heads({heads}) is not divisible by ulysses_degree({parallel_state.ulysses_degree})"
        )
    if compute_dtype is None:
        compute_dtype = next(transformer.parameters()).dtype
    transformer.set_attn_processor(WanUSPAttnProcessor(heads, head_dim, compute_dtype))

    # RoPE runs once per forward and its output is reused by every block; shard at the source.
    rope = transformer.rope
    orig_rope_forward = rope.forward

    @functools.wraps(orig_rope_forward)
    def sp_rope_forward(hidden_states):
        cos, sin = orig_rope_forward(hidden_states)
        return shard_sequence(cos, parallel_state), shard_sequence(sin, parallel_state)

    rope.forward = sp_rope_forward

    def slice_block_input(module, args):
        hs, *rest = args
        return (shard_sequence(hs, parallel_state), *rest)

    def gather_proj_output(module, args, output):
        return gather_sequence(output, parallel_state)

    transformer.blocks[0].register_forward_pre_hook(slice_block_input)
    transformer.proj_out.register_forward_hook(gather_proj_output)
