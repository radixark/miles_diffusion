import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention


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
    # Pass attn_mask through unchanged so PyTorch's SDPA dispatches to flash
    # (mask=None path) or efficient (mask present) backend matching diffusers.
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=self.softmax_scale,
    ).transpose(1, 2)
    return out


def apply() -> None:
    USPAttention.forward = _patched_forward
