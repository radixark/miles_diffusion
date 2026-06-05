"""Video PickScore helpers (LTX / sglang FHWC and trainer [C,F,H,W] layouts)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image


def sample_frame_indices(num_total_frames: int, num_frames: int) -> list[int]:
    if num_total_frames <= 0:
        raise ValueError(f"video has no frames: {num_total_frames}")
    if num_total_frames <= num_frames:
        return list(range(num_total_frames))
    if num_frames == 1:
        return [num_total_frames // 2]
    step = (num_total_frames - 1) / (num_frames - 1)
    return [int(round(i * step)) for i in range(num_frames)]


def generated_output_to_fchw(t: torch.Tensor) -> torch.Tensor:
    """Return ``[F, C, H, W]`` float tensor in ``[0, 1]``."""
    t = t.detach().cpu().float()
    if t.ndim == 3:
        if t.shape[0] not in (1, 3):
            raise ValueError(f"expected [C, H, W] with C in {{1, 3}}, got {tuple(t.shape)}")
        t = t.unsqueeze(0)
    elif t.ndim == 4:
        if t.shape[-1] in (1, 3):
            t = t.permute(0, 3, 1, 2)
        elif t.shape[0] in (1, 3):
            t = t.permute(1, 0, 2, 3)
        elif t.shape[1] not in (1, 3):
            raise ValueError(f"unrecognized 4D video layout: {tuple(t.shape)}")
    elif t.ndim == 5:
        if t.shape[0] == 1 and t.shape[-1] in (1, 3):
            t = t[0].permute(0, 3, 1, 2)
        else:
            raise ValueError(f"unrecognized 5D video layout: {tuple(t.shape)}")
    else:
        raise ValueError(f"generated_output must be 3D–5D, got {tuple(t.shape)}")

    if float(t.max()) > 1.0 + 1e-3:
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def fchw_frame_to_hwc_uint8(frame_chw: torch.Tensor) -> np.ndarray:
    hwc = frame_chw.numpy().transpose(1, 2, 0)
    if float(hwc.max()) <= 1.0 + 1e-3:
        hwc = hwc * 255.0
    return np.ascontiguousarray(hwc.clip(0, 255).astype(np.uint8))


def fchw_to_pil_frames(video_fchw: torch.Tensor, frame_indices: Sequence[int]) -> list[Image.Image]:
    return [
        Image.fromarray(fchw_frame_to_hwc_uint8(video_fchw[idx]))
        for idx in frame_indices
    ]


def is_video_generated_output(t: torch.Tensor) -> bool:
    """True when output carries multiple temporal frames (LTX / sglang video)."""
    fchw = generated_output_to_fchw(t)
    return fchw.shape[0] > 1
