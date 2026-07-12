import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention

_original_forward = USPAttention.forward


class _SDPALocalAttnImpl:
    """Drop-in for USPAttention.attn_impl that runs PyTorch SDPA.

    Matches diffusers' training-side attention numerics (mask=None path
    dispatches to flash, mask path to efficient). Layout contract is the
    same as sgl-d attention impls: [B, S, H, D] in / [B, S, H, D] out.
    """

    def __init__(self, softmax_scale: float):
        self.softmax_scale = softmax_scale

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx_attn_metadata=None) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=self.softmax_scale,
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
    # Keep USPAttention's own control flow (Ulysses all-to-all, replicated
    # prefix/suffix handling) so sp_degree > 1 stays mathematically correct;
    # only the local attention kernel is swapped for SDPA parity. With
    # sp_degree == 1 the original forward reduces to exactly one SDPA call,
    # preserving the previous behavior of this patch.
    original_impl = self.attn_impl
    self.attn_impl = _SDPALocalAttnImpl(self.softmax_scale)
    try:
        return _original_forward(
            self,
            q,
            k,
            v,
            attn_mask=attn_mask,
            num_replicated_prefix=num_replicated_prefix,
            num_replicated_suffix=num_replicated_suffix,
            skip_sequence_parallel_override=skip_sequence_parallel_override,
        )
    finally:
        self.attn_impl = original_impl


def apply() -> None:
    USPAttention.forward = _patched_forward
