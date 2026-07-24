"""Differentiable USP (Ulysses x Ring) attention operators, owned by the trainer.

Layout convention matches sglang-diffusion's USP (heads sharded across the ulysses
group inside attention, sequence sharded outside), so training numerics stay
aligned with rollout; the collectives only move data. Local attention is torch
SDPA by default and may be injected by the model adapter; ring attention uses
torch's ring templates with an aten fused op selected per RING_KERNELS.
"""

import torch
import torch.distributed as dist


class _GatherSequence(torch.autograd.Function):
    """All-gather local shards along dim; backward sums then returns each rank's slice.

    The backward all-reduces the incoming gradient over the sp group first:
    downstream partial grads then carry an sp factor, so FSDP's
    1/(dp*sp) mean over a dp x sp shard mesh restores (1/dp) * sum_dp exactly.
    """

    @staticmethod
    def forward(ctx, x, group, sp_rank, sp_size, dim):
        ctx.group = group
        ctx.sp_rank = sp_rank
        ctx.dim = dim
        ctx.local_size = x.shape[dim]
        parts = [torch.empty_like(x) for _ in range(sp_size)]
        dist.all_gather(parts, x.contiguous(), group=group)
        return torch.cat(parts, dim=dim)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad, group=ctx.group)
        start = ctx.sp_rank * ctx.local_size
        return grad.narrow(ctx.dim, start, ctx.local_size), None, None, None, None


def shard_sequence(x, sp_rank, sp_size, dim=1):
    s = x.shape[dim]
    if s % sp_size:
        raise ValueError(f"sequence length {s} is not divisible by sp_size {sp_size}")
    s_local = s // sp_size
    return x.narrow(dim, sp_rank * s_local, s_local)


def gather_sequence(x, group, sp_rank, sp_size, dim=1):
    return _GatherSequence.apply(x, group, sp_rank, sp_size, dim)


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


# --fsdp-attention-backend values ring attention honors: aten ops returning LSE with a real backward.
RING_KERNELS = {None: "flash", "_native_flash": "flash", "_native_cudnn": "cudnn"}


class _RingAttention(torch.autograd.Function):
    """Ring attention via torch's ring templates (fwd + reverse-ring bwd), aten fused ops.

    q/k/v: [B, H, S, D].
    """

    @staticmethod
    def forward(ctx, query, key, value, group, scale, kernel):
        # torch's private ring templates, at their torch >= 2.11 home (2.9: experimental._attention).
        from torch.distributed.tensor.experimental._context_parallel._attention import _templated_ring_attention

        if kernel == "cudnn":
            op = torch.ops.aten._scaled_dot_product_cudnn_attention
            # cudnn computes LSE only on request; ring merging always needs it
            op_kwargs = {"attn_bias": None, "compute_log_sumexp": True}
        else:
            op = torch.ops.aten._scaled_dot_product_flash_attention
            op_kwargs = {}
        out, lse, cum_q, cum_k, max_q, max_k, philox_seed, philox_offset, _dbg = _templated_ring_attention(
            group,
            2,
            op,
            query=query,
            key=key,
            value=value,
            is_causal=False,
            dropout_p=0.0,
            scale=scale,
            **op_kwargs,
        )
        out = out.to(query.dtype)
        ctx.save_for_backward(query, key, value, out, lse, cum_q, cum_k, philox_seed, philox_offset)
        ctx.group, ctx.scale, ctx.max_q, ctx.max_k = group, scale, max_q, max_k
        ctx.kernel = kernel
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from torch.distributed.tensor.experimental._context_parallel._attention import (
            _templated_ring_attention_backward,
        )

        if ctx.kernel == "cudnn":
            op = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default
            op_kwargs = {"attn_bias": None}
        else:
            op = torch.ops.aten._scaled_dot_product_flash_attention_backward.default
            op_kwargs = {}
        query, key, value, out, lse, cum_q, cum_k, philox_seed, philox_offset = ctx.saved_tensors
        grad_q, grad_k, grad_v, *_ = _templated_ring_attention_backward(
            ctx.group,
            2,
            op,
            grad_out=grad_out.contiguous(),
            grad_out_name="grad_out",
            query=query,
            key=key,
            value=value,
            out=out,
            logsumexp=lse,
            is_causal=False,
            cum_seq_q=cum_q,
            cum_seq_k=cum_k,
            max_q=ctx.max_q,
            max_k=ctx.max_k,
            dropout_p=0.0,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
            scale=ctx.scale,
            **op_kwargs,
        )
        return grad_q, grad_k, grad_v, None, None, None


def usp_attention(query, key, value, ulysses_group=None, ring_group=None, local_attention_fn=None, ring_backend=None):
    """USP self-attention on [B, S_local, H, D] tensors; returns the same layout.

    Ulysses all-to-all temporarily gathers the sequence (sharding heads), ring
    attention covers the remaining split. Without Ring, ``local_attention_fn`` may
    route the gathered tensors through the model's configured attention backend;
    the fallback is plain SDPA. With Ring, ``ring_backend`` picks the local
    kernel per ``RING_KERNELS``.
    """
    if ulysses_group is not None:
        query = ulysses_input_all_to_all(query, ulysses_group)
        key = ulysses_input_all_to_all(key, ulysses_group)
        value = ulysses_input_all_to_all(value, ulysses_group)

    if ring_group is not None:
        if ring_backend not in RING_KERNELS:
            raise ValueError(f"ring attention has no kernel for attention backend {ring_backend!r}")
        scale = query.shape[-1] ** -0.5
        q = query.transpose(1, 2)  # [B, H, S, D]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = _RingAttention.apply(
            q.contiguous(), k.contiguous(), v.contiguous(), ring_group, scale, RING_KERNELS[ring_backend]
        )
        out = out.transpose(1, 2).contiguous()  # [B, S, H, D]
    elif local_attention_fn is not None:
        out = local_attention_fn(query, key, value)
    else:
        scale = query.shape[-1] ** -0.5
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=scale)
        out = out.transpose(1, 2).contiguous()  # [B, S, H, D]

    if ulysses_group is not None:
        out = ulysses_output_all_to_all(out, ulysses_group)
    return out
