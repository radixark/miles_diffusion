"""LTX-2 model-side training behavior."""

from __future__ import annotations

from typing import Any

import torch


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    model.set_gradient_checkpointing(True)


def flash_attention_entrypoints(backend: str) -> list[tuple[str, Any, str]]:
    """Flash kernels MilesModelBackend can patch with ``deterministic=True``."""
    import ltx_core.model.transformer.attention as ltx_attn

    entrypoints: list[tuple[str, Any, str]] = []
    if ltx_attn.flash_attn_interface is not None:
        entrypoints.append(("flash_attention_3", ltx_attn.flash_attn_interface, "flash_attn_func"))
    entrypoints.append(("flash_attention_4", ltx_attn, "flash_attn_4_func"))
    return entrypoints


def required_flash_kernel_label(backend: str) -> str | None:
    if "3" in backend:
        return "flash_attention_3"
    if "4" in backend:
        return "flash_attention_4"
    return None
