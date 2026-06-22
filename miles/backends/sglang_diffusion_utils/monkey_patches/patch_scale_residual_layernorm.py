import torch
from sglang.multimodal_gen.runtime.layers.layernorm import ScaleResidualLayerNormScaleShift

from miles.backends.sglang_diffusion_utils.monkey_patches._common import ensure_broadcast


def _patched_forward(
    self,
    residual: torch.Tensor,
    x: torch.Tensor,
    gate,
    shift: torch.Tensor,
    scale: torch.Tensor,
):
    # diffusers sequence: residual + gate*x (bf16), then LayerNorm, then
    # (1+scale)*x + shift.
    if isinstance(gate, int):
        assert gate == 1
        residual_out = residual + x
    elif gate.dim() == 4:
        num_frames = gate.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        residual_out = residual + (x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate).flatten(1, 2)
    else:
        residual_out = residual + x * gate

    normed = self.norm(residual_out)
    scale = ensure_broadcast(scale, normed)
    shift = ensure_broadcast(shift, normed)
    return normed * (1 + scale) + shift, residual_out


def apply() -> None:
    ScaleResidualLayerNormScaleShift.forward = _patched_forward
