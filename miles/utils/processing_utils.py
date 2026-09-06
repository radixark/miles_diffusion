"""Tensor layout conversions used by diffusion output consumers."""

from __future__ import annotations

import torch


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
