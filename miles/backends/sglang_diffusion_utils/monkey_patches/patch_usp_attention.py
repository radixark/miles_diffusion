import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask, softmax_scale: float) -> torch.Tensor:
    # Pass attn_mask through unchanged so PyTorch's SDPA dispatches to flash
    # (mask=None path) or efficient (mask present) backend matching diffusers.
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=softmax_scale,
    ).transpose(1, 2)


def _patched_forward(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask=None,
    num_replicated_prefix: int = 0,
    num_replicated_suffix: int = 0,
    skip_sequence_parallel_override: bool = False,
):
    sp_world = get_sp_world_size()
    effective_skip = self.skip_sequence_parallel or skip_sequence_parallel_override

    if effective_skip or sp_world == 1:
        # sp=1 (and replicated-KV cross attention): identical to the original
        # parity patch — this exact call is what the registered CI standards
        # and the training-side forward were aligned against. Do not change.
        return _sdpa(q, k, v, attn_mask, self.softmax_scale)

    # sp>1: do NOT shard heads (Ulysses all-to-all + per-rank SDPA is not
    # bitwise equal to the sp=1 call: SDPA-flash's split-KV heuristic depends
    # on batch*num_heads, so halving the head count changes the reduction
    # order). Instead every rank all-gathers the full sequence and runs the
    # bit-identical full-head SDPA call that sp=1 runs, then slices its local
    # shard back out. Redundant compute, but this patch only serves RL
    # rollout parity where bitwise SP-invariance is the whole point.
    if num_replicated_prefix > 0 and num_replicated_suffix > 0:
        raise ValueError("USPAttention parity patch: replicated prefix and suffix together unsupported.")

    sp_rank = get_sp_parallel_rank()

    def _gather_joint(t: torch.Tensor):
        if num_replicated_prefix > 0:
            rep, shard = t[:, :num_replicated_prefix], t[:, num_replicated_prefix:]
            full = torch.cat(
                [rep, sequence_model_parallel_all_gather(shard.contiguous(), dim=1)],
                dim=1,
            )
        elif num_replicated_suffix > 0:
            shard, rep = t[:, : -num_replicated_suffix], t[:, -num_replicated_suffix:]
            full = torch.cat(
                [sequence_model_parallel_all_gather(shard.contiguous(), dim=1), rep],
                dim=1,
            )
        else:
            full = sequence_model_parallel_all_gather(t.contiguous(), dim=1)
        return full

    full_mask = attn_mask
    if attn_mask is not None:
        if attn_mask.dim() != 2:
            raise NotImplementedError("USPAttention parity patch: only [B, S_local] key masks supported under SP.")
        full_mask = sequence_model_parallel_all_gather(attn_mask.contiguous(), dim=1)

    full_out = _sdpa(_gather_joint(q), _gather_joint(k), _gather_joint(v), full_mask, self.softmax_scale)

    if num_replicated_prefix > 0:
        shard_len = q.shape[1] - num_replicated_prefix
        start = num_replicated_prefix + sp_rank * shard_len
        return torch.cat(
            [full_out[:, :num_replicated_prefix], full_out[:, start : start + shard_len]],
            dim=1,
        )
    if num_replicated_suffix > 0:
        shard_len = q.shape[1] - num_replicated_suffix
        start = sp_rank * shard_len
        return torch.cat(
            [full_out[:, start : start + shard_len], full_out[:, -num_replicated_suffix:]],
            dim=1,
        )
    shard_len = q.shape[1]
    return full_out[:, sp_rank * shard_len : (sp_rank + 1) * shard_len]


def apply() -> None:
    USPAttention.forward = _patched_forward
