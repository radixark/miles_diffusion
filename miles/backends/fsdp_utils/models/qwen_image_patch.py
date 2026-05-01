"""Opt-in monkey patches that make sglang-diffusion's Qwen-Image DiT
bit-exact with diffusers. Call :func:`apply_qwen_image_diffusers_parity_patches`
before loading the DiT.

Root cause: sglang keeps RMSNorm / LayerNorm intermediates in fp32 and
only rounds to bf16 at the very end; diffusers rounds to bf16 *before*
the weight (or ``(1+scale)``) mul, so that mul runs bf16*bf16. Patches
reproduce diffusers' rounding order on the five hot paths
(RMSNorm / LayerNormScaleShift / ScaleResidualLayerNormScaleShift /
apply_qk_norm_with_optional_rope / MulAdd), plus redirect USPAttention
to ``F.scaled_dot_product_attention`` with ``attn_mask`` passed through
unchanged so the kernel dispatch (flash when mask=None, efficient when
mask is a tensor) matches diffusers' behavior.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_PATCH_APPLIED_ATTR = "_qwen_image_diffusers_parity_applied"


def _ensure_broadcast(mod: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    # Promote [B, D] modulation to [B, 1, D] for broadcasting over seq axis.
    if mod.dim() == ref.dim() - 1:
        return mod.unsqueeze(-2)
    return mod


def _patched_rmsnorm_forward(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
):
    # Matches diffusers.models.normalization.RMSNorm.forward: variance in fp32,
    # rsqrt in fp32, then round back to weight dtype BEFORE weight mul so the
    # weight mul runs bf16*bf16. sglang's default path keeps fp32 through the
    # weight mul and rounds once at the end — off by 1-2 bf16 ULPs.
    if not x.is_contiguous():
        x = x.contiguous()
    orig_dtype = x.dtype

    x_fp32 = x.to(torch.float32)
    if residual is not None:
        x_fp32 = x_fp32 + residual.to(torch.float32)
        residual = x_fp32.to(orig_dtype)

    variance_size_override = getattr(self, "variance_size_override", None)
    x_var = (
        x_fp32
        if variance_size_override is None
        else x_fp32[..., :variance_size_override]
    )
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x_fp32 = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)

    out = x_fp32.to(orig_dtype)
    if self.weight is not None:
        out = out * self.weight

    if residual is None:
        return out
    return out, residual


def _patched_layernorm_scale_shift_forward(
    self,
    x: torch.Tensor,
    shift: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
):
    # Replace sglang's fused kernel with the diffusers bf16 sequence:
    # LayerNorm(x) then (1+scale)*x + shift in bf16 eager.
    normed = self.norm(x)
    if shift is None and scale is None:
        return normed
    scale = _ensure_broadcast(scale, normed)
    shift = _ensure_broadcast(shift, normed)
    return normed * (1 + scale) + shift


def _patched_mul_add_forward(
    self,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    k: int = 0,
):
    # Replace sglang's fp32 fused kernel with the diffusers bf16 equivalent
    # `c + a*(k+b)`; used for the MLP residual at block.forward:1010/1026.
    if b.dim() == 4:
        num_frames = b.shape[1]
        frame_seqlen = a.shape[1] // num_frames
        return c + (
            a.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (k + b)
        ).flatten(1, 2)
    return c + a * (k + b)


def _patched_usp_attention_forward(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask=None,
    num_replicated_prefix: int = 0,
    num_replicated_suffix: int = 0,
    skip_sequence_parallel_override: bool = False,
):
    # Route joint attention through F.scaled_dot_product_attention. Pass
    # attn_mask through unchanged — diffusers only constructs a joint_mask
    # when ``encoder_hidden_states_mask is not None``; in the mask=None
    # path it calls SDPA with attn_mask=None and PyTorch dispatches to the
    # flash-attention backend. Previously we synthesized an all-True mask
    # here, which forced dispatch to the efficient-attention backend and
    # accumulated ~5e-4 bf16 drift per call × 60 blocks into ~5-7% overall
    # (only the ``encoder_hidden_states_mask=not None`` path needs a real
    # mask; rank-0 workloads without padding go through this None path).
    import torch.nn.functional as F

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


def _patched_apply_qk_norm_with_optional_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm,
    k_norm,
    head_dim: int,
    cos_sin_cache=None,
    *,
    is_neox: bool = False,
    positions=None,
    position_offset: int = 0,
    allow_inplace: bool = True,
):
    # Replace sglang's fused qk-norm-rope CUDA kernel (which bypasses the
    # patched RMSNorm.forward) with: patched q_norm/k_norm + diffusers'
    # complex ROPE formula from apply_rotary_emb_qwen(use_real=False).
    q_normed = q_norm(q)
    k_normed = k_norm(k)
    if cos_sin_cache is None:
        return q_normed, k_normed

    # Layout: [cos_half | sin_half] along last dim; each half = head_dim/2.
    half = cos_sin_cache.shape[-1] // 2
    freqs_cis = torch.complex(cos_sin_cache[..., :half], cos_sin_cache[..., half:])

    def _apply(x: torch.Tensor) -> torch.Tensor:
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        f = freqs_cis.unsqueeze(1).to(x.device)
        if f.dim() < x_c.dim():
            f = f.unsqueeze(0)
        return torch.view_as_real(x_c * f).flatten(3).type_as(x)

    return _apply(q_normed), _apply(k_normed)


def _patched_scale_residual_layernorm_scale_shift_forward(
    self,
    residual: torch.Tensor,
    x: torch.Tensor,
    gate,
    shift: torch.Tensor,
    scale: torch.Tensor,
):
    # Replace the fused CUTLASS kernel with the diffusers sequence:
    # residual + gate*x (bf16), then LayerNorm, then (1+scale)*x + shift.
    if isinstance(gate, int):
        assert gate == 1
        residual_out = residual + x
    elif gate.dim() == 4:
        num_frames = gate.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        residual_out = residual + (
            x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate
        ).flatten(1, 2)
    else:
        residual_out = residual + x * gate

    normed = self.norm(residual_out)
    scale = _ensure_broadcast(scale, normed)
    shift = _ensure_broadcast(shift, normed)
    return normed * (1 + scale) + shift, residual_out


def apply_qwen_image_diffusers_parity_patches() -> None:
    """Install diffusers-parity forward replacements. Idempotent."""
    from sglang.multimodal_gen.runtime.layers import layernorm as _layernorm_mod
    from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
    from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
    from sglang.multimodal_gen.runtime.layers.layernorm import (
        LayerNormScaleShift,
        RMSNorm,
        ScaleResidualLayerNormScaleShift,
    )
    from sglang.multimodal_gen.runtime.models.dits import qwen_image as _qi_mod

    if getattr(RMSNorm, _PATCH_APPLIED_ATTR, False):
        return

    RMSNorm.forward = _patched_rmsnorm_forward
    LayerNormScaleShift.forward = _patched_layernorm_scale_shift_forward
    ScaleResidualLayerNormScaleShift.forward = (
        _patched_scale_residual_layernorm_scale_shift_forward
    )
    MulAdd.forward = _patched_mul_add_forward
    USPAttention.forward = _patched_usp_attention_forward

    # Replace the fused qk-norm+rope helper on both the defining module and
    # the dits.qwen_image module that imported it by name.
    _layernorm_mod.apply_qk_norm_with_optional_rope = (
        _patched_apply_qk_norm_with_optional_rope
    )
    _qi_mod.apply_qk_norm_with_optional_rope = _patched_apply_qk_norm_with_optional_rope

    for cls in (
        RMSNorm,
        LayerNormScaleShift,
        ScaleResidualLayerNormScaleShift,
        MulAdd,
        USPAttention,
    ):
        setattr(cls, _PATCH_APPLIED_ATTR, True)

    logger.info("Applied Qwen-Image diffusers-parity patches.")
