from typing import Optional

import torch

from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm


def _patched_forward(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
):
    # diffusers' RMSNorm rounds to weight dtype BEFORE the weight mul, so the
    # mul runs bf16*bf16. sgl-d's default keeps fp32 through the weight mul.
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


def apply() -> None:
    RMSNorm.forward = _patched_forward
