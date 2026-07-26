"""Preserve tensor stride when sgl-d offloads rollout weights to pinned host memory.

sglang #32032 allocates the host buffer with ``torch.empty(t.shape, ...)``, which is always
contiguous; ``ltx_2_vae`` then sees a non-channels_last weight and drops off the cuDNN NDHWC
conv path. Same allocation as ``srt/utils/offloader.py`` (``empty_like`` takes no pin_memory).
"""

from __future__ import annotations

import torch

_APPLIED = False


def _module_to_pinned_cpu(module: torch.nn.Module) -> None:
    for t in list(module.parameters()) + list(module.buffers()):
        if t.device.type == "cuda":
            pin = torch.empty_strided(
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                layout=t.layout,
                device="cpu",
                pin_memory=True,
            )
            pin.copy_(t.data, non_blocking=True)
            t.data = pin


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from sglang.multimodal_gen.runtime.managers.memory_managers import memory_occupation_controller as moc

    moc._module_to_pinned_cpu = _module_to_pinned_cpu
    _APPLIED = True
