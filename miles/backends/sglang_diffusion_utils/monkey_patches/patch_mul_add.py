import torch

from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd


def _patched_forward(
    self,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    k: int = 0,
):
    # diffusers bf16 equivalent of the fused fp32 kernel: c + a*(k+b).
    if b.dim() == 4:
        num_frames = b.shape[1]
        frame_seqlen = a.shape[1] // num_frames
        return c + (a.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (k + b)).flatten(1, 2)
    return c + a * (k + b)


def apply() -> None:
    MulAdd.forward = _patched_forward
