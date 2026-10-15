"""FLUX.3 image attention backend selection: the DiT runs the engine's default flash kernel only."""

from __future__ import annotations

import torch

from .modeling import is_blackwell


def set_attention_backend(model: torch.nn.Module, backend: str) -> None:
    # The engine's default kernel for this GPU: FA3 on Hopper, FA4 on Blackwell.
    expected = "_flash_4" if is_blackwell() else "_flash_3"
    if backend != expected:
        raise ValueError(f"FLUX.3 image on this GPU supports --fsdp-attention-backend {expected} only, got {backend!r}")
