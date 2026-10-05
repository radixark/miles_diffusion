"""Qwen-Image rollout patches: make the sgl-d forward bitwise-equal to the diffusers train forward."""

import os

import torch
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.layers import layernorm as layernorm_mod
from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    RMSNorm,
    ScaleResidualLayerNormScaleShift,
)
from sglang.multimodal_gen.runtime.models.dits import qwen_image as qwen_image_mod

_orig_split_seqs = qwen_image_mod.split_seqs


def _rmsnorm_forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
    # diffusers' RMSNorm rounds to weight dtype BEFORE the weight mul; sgl-d keeps fp32 through it.
    if not x.is_contiguous():
        x = x.contiguous()
    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    if residual is not None:
        x_fp32 = x_fp32 + residual.to(torch.float32)
        residual = x_fp32.to(orig_dtype)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_fp32 = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
    out = x_fp32.to(orig_dtype)
    if self.weight is not None:
        out = out * self.weight
    if residual is None:
        return out
    return out, residual


def _ensure_broadcast(mod: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if mod.dim() == ref.dim() - 1:
        return mod.unsqueeze(-2)
    return mod


def _fp32_layer_norm(norm: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # nn.LayerNorm exactly as train-side autocast runs it: fp32 in, fp32 out.
    weight = norm.weight.float() if norm.weight is not None else None
    bias = norm.bias.float() if norm.bias is not None else None
    return F.layer_norm(x.float(), norm.normalized_shape, weight, bias, norm.eps)


def _layernorm_scale_shift_forward(
    self,
    x: torch.Tensor,
    shift: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
):
    normed = _fp32_layer_norm(self.norm, x)
    if shift is None and scale is None:
        return normed.to(x.dtype)
    scale = _ensure_broadcast(scale, normed)
    shift = _ensure_broadcast(shift, normed)
    # (1 + scale) rounds in bf16, the modulation promotes to fp32 -- the train-side autocast semantics.
    out = normed * (1 + scale) + shift
    return out.to(x.dtype)


def _scale_residual_layernorm_scale_shift_forward(
    self,
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
):
    residual_out = residual + x * gate
    normed = _fp32_layer_norm(self.norm, residual_out)
    scale = _ensure_broadcast(scale, normed)
    shift = _ensure_broadcast(shift, normed)
    out = normed * (1 + scale) + shift
    return out.to(x.dtype), residual_out


def _mul_add_forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0):
    # diffusers bf16 equivalent of the fused fp32 kernel.
    return c + a * (k + b)


def _qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm,
    k_norm,
    head_dim: int,
    cos_sin_cache=None,
    freqs_complex=None,
    *,
    is_neox: bool = False,
    positions=None,
    position_offset: int = 0,
    allow_inplace: bool = True,
):
    # Replace the fused qk-norm-rope CUDA kernel with the patched norms + diffusers' complex RoPE.
    q_normed = q_norm(q)
    k_normed = k_norm(k)
    if cos_sin_cache is None:
        return q_normed, k_normed

    half = cos_sin_cache.shape[-1] // 2
    freqs_cis = torch.complex(cos_sin_cache[..., :half], cos_sin_cache[..., half:])

    def _apply(x: torch.Tensor) -> torch.Tensor:
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        f = freqs_cis.unsqueeze(1).to(x.device)
        if f.dim() < x_c.dim():
            f = f.unsqueeze(0)
        return torch.view_as_real(x_c * f).flatten(3).type_as(x)

    return _apply(q_normed), _apply(k_normed)


def _contiguous_split_seqs(joint, prefix_len, local_pad, dim=1):
    # batch>1 split views are strided; contiguize so the out-proj GEMMs match diffusers' flattened GEMM.
    prefix, body = _orig_split_seqs(joint, prefix_len, local_pad, dim=dim)
    return prefix.contiguous(), body.contiguous()


def apply() -> None:
    os.environ["SGLANG_ENABLE_FUSED_QKNORM_ROPE"] = "0"
    RMSNorm.forward = _rmsnorm_forward
    LayerNormScaleShift.forward = _layernorm_scale_shift_forward
    ScaleResidualLayerNormScaleShift.forward = _scale_residual_layernorm_scale_shift_forward
    MulAdd.forward = _mul_add_forward
    layernorm_mod.apply_qk_norm_with_optional_rope = _qk_norm_rope
    qwen_image_mod.apply_qk_norm_with_optional_rope = _qk_norm_rope
    qwen_image_mod.split_seqs = _contiguous_split_seqs
