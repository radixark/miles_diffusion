from typing import Optional

import torch

from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift

from miles.backends.sglang_diffusion_utils.monkey_patches._common import ensure_broadcast


def _patched_forward(
    self,
    x: torch.Tensor,
    shift: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
):
    # diffusers sequence: LayerNorm(x) then (1+scale)*x + shift in bf16 eager.
    normed = self.norm(x)
    if shift is None and scale is None:
        return normed
    scale = ensure_broadcast(scale, normed)
    shift = ensure_broadcast(shift, normed)
    return normed * (1 + scale) + shift


def apply() -> None:
    LayerNormScaleShift.forward = _patched_forward
