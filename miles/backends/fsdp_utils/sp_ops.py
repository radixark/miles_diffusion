"""Differentiable USP (Ulysses x Ring) attention operators, owned by the trainer.

Layout convention matches sglang-diffusion's USP (heads sharded across the ulysses
group inside attention, sequence sharded outside), so training numerics stay
aligned with rollout; the collectives only move data. Local attention is torch
SDPA; ring attention uses torch's ring templates with the aten flash op.
"""

import torch
import torch.distributed as dist


class _AllToAllSingle(torch.autograd.Function):
    """Even-split all-to-all; an involution, so the adjoint is the same collective."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        out = torch.empty_like(grad)
        dist.all_to_all_single(out, grad.contiguous(), group=ctx.group)
        return out, None


def _all_to_all_4d(x, group):
    shape = x.shape
    return _AllToAllSingle.apply(x.flatten(), group).reshape(shape)


def ulysses_input_all_to_all(x, group):
    """[b, s_local, h, d] -> [b, s_local * world, h / world, d]: shard heads, gather sequence."""
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return x
    b, s_local, h_global, d = x.shape
    if h_global % world_size:
        raise ValueError(f"num_heads({h_global}) is not divisible by ulysses world size({world_size})")
    h_local, s_global = h_global // world_size, s_local * world_size

    x = x.permute(2, 0, 1, 3).contiguous()  # [h_global, b, s_local, d]
    x = _all_to_all_4d(x, group)
    x = x.reshape(world_size, h_local, b, s_local, d)
    return x.permute(2, 0, 3, 1, 4).contiguous().reshape(b, s_global, h_local, d)


def ulysses_output_all_to_all(x, group):
    """[b, s_global, h_local, d] -> [b, s_global / world, h_local * world, d]: inverse of input."""
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return x
    b, s_global, h_local, d = x.shape
    if s_global % world_size:
        raise ValueError(f"sequence({s_global}) is not divisible by ulysses world size({world_size})")
    s_local, h_global = s_global // world_size, h_local * world_size

    x = x.permute(1, 0, 2, 3).contiguous()  # [s_global, b, h_local, d]
    x = _all_to_all_4d(x, group)
    x = x.reshape(world_size, s_local, b, h_local, d)
    return x.permute(2, 1, 0, 3, 4).contiguous().reshape(b, s_local, h_global, d)


class _RingFlashAttention(torch.autograd.Function):
    """Ring attention via torch's ring templates (fwd + reverse-ring bwd), aten flash op.

    q/k/v: [B, H, S, D].
    """

    @staticmethod
    def forward(ctx, query, key, value, group, scale, is_causal):
        from torch.distributed.tensor.experimental._attention import _templated_ring_attention

        out, lse, cum_q, cum_k, max_q, max_k, philox_seed, philox_offset, _dbg = _templated_ring_attention(
            group,
            2,
            torch.ops.aten._scaled_dot_product_flash_attention,
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            dropout_p=0.0,
            scale=scale,
        )
        out = out.to(query.dtype)
        ctx.save_for_backward(query, key, value, out, lse, cum_q, cum_k, philox_seed, philox_offset)
        ctx.group, ctx.scale, ctx.is_causal, ctx.max_q, ctx.max_k = group, scale, is_causal, max_q, max_k
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from torch.distributed.tensor.experimental._attention import (
            _templated_ring_attention_backward,
        )

        query, key, value, out, lse, cum_q, cum_k, philox_seed, philox_offset = ctx.saved_tensors
        grad_q, grad_k, grad_v, *_ = _templated_ring_attention_backward(
            ctx.group,
            2,
            torch.ops.aten._scaled_dot_product_flash_attention_backward.default,
            grad_out=grad_out.contiguous(),
            grad_out_name="grad_out",
            query=query,
            key=key,
            value=value,
            out=out,
            logsumexp=lse,
            is_causal=ctx.is_causal,
            cum_seq_q=cum_q,
            cum_seq_k=cum_k,
            max_q=ctx.max_q,
            max_k=ctx.max_k,
            dropout_p=0.0,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
            scale=ctx.scale,
        )
        return grad_q, grad_k, grad_v, None, None, None


def usp_attention(query, key, value, ulysses_group=None, ring_group=None):
    """USP self-attention on [B, S_local, H, D] tensors; returns the same layout.

    Ulysses all-to-all temporarily gathers the sequence (sharding heads), ring
    attention covers the remaining split; with no groups this is plain SDPA.
    """
    scale = query.shape[-1] ** -0.5

    if ulysses_group is not None:
        query = ulysses_input_all_to_all(query, ulysses_group)
        key = ulysses_input_all_to_all(key, ulysses_group)
        value = ulysses_input_all_to_all(value, ulysses_group)

    q = query.transpose(1, 2)  # [B, H, S, D]
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    if ring_group is not None and dist.get_world_size(ring_group) > 1:
        out = _RingFlashAttention.apply(q.contiguous(), k.contiguous(), v.contiguous(), ring_group, scale, False)
    else:
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False, scale=scale
        )
    out = out.transpose(1, 2).contiguous()  # [B, S, H, D]

    if ulysses_group is not None:
        out = ulysses_output_all_to_all(out, ulysses_group)
    return out
