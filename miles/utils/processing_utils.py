"""Image and video tensor processing used by diffusion output consumers."""

from __future__ import annotations

import numpy as np
import torch

from miles.utils.types import Sample


def cfhw_to_fhwc(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a per-sample tensor from ``[C, F, H, W]`` to ``[F, H, W, C]``."""
    if tensor.ndim != 4:
        raise ValueError(f"expected a 4D CFHW tensor, got shape {tuple(tensor.shape)}")
    return tensor.permute(1, 2, 3, 0).contiguous()


def fhwc_to_cfhw(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a per-sample tensor from ``[F, H, W, C]`` to ``[C, F, H, W]``."""
    if tensor.ndim != 4:
        raise ValueError(f"expected a 4D FHWC tensor, got shape {tuple(tensor.shape)}")
    return tensor.permute(3, 0, 1, 2).contiguous()


def image_or_video_to_uint8(tensor: torch.Tensor, *, round_normalized: bool = False) -> torch.Tensor:
    """Convert image or video values in ``[0, 1]`` or ``[0, 255]`` to ``torch.uint8``."""
    output = tensor.float()
    if float(output.max()) <= 1.0 + 1e-3:
        output = output * 255.0
        if round_normalized:
            output = output.round()
    return output.clamp(0, 255).to(torch.uint8)


def sample_frame_indices(num_total_frames: int, num_frames: int | None) -> list[int]:
    if num_total_frames <= 0:
        raise ValueError(f"video has no frames: {num_total_frames}")
    if num_frames is None or num_total_frames <= num_frames:
        return list(range(num_total_frames))
    if num_frames == 1:
        return [num_total_frames // 2]
    step = (num_total_frames - 1) / (num_frames - 1)
    return [int(round(i * step)) for i in range(num_frames)]


def sample_to_rgb_hwc_uint8_frames(
    sample: Sample,
    num_frames: int | None,
    *,
    round_normalized: bool = False,
) -> list[np.ndarray]:
    cfhw = sample.generated_output
    if cfhw is None:
        raise ValueError("generated_output is None")

    indices = sample_frame_indices(cfhw.shape[1], num_frames)
    selected = image_or_video_to_uint8(cfhw[:, indices].detach().cpu(), round_normalized=round_normalized)
    fhwc = cfhw_to_fhwc(selected)
    return [np.ascontiguousarray(fhwc[i].numpy()) for i in range(len(indices))]
