"""LTX-2 model-side training behavior and temporary scheduler integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class _SchedulerConfig:
    num_train_timesteps: int = 1000


@dataclass
class _LTXSchedulerHolder:
    sigmas: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    timesteps: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    num_inference_steps: int = 0
    _step_index: int | None = None
    _begin_index: int | None = None
    config: _SchedulerConfig = field(default_factory=_SchedulerConfig)

    def to(self, device):
        self.sigmas = self.sigmas.to(device)
        self.timesteps = self.timesteps.to(device)
        return self


def build_train_scheduler(args):
    """Sigma/timestep holder mirroring the diffusers scheduler surface the trainer touches."""
    from ltx_core.components.schedulers import LTX2Scheduler

    num_steps = int(getattr(args, "diffusion_num_steps", 24))
    sigmas = LTX2Scheduler().execute(steps=num_steps).float()
    return _LTXSchedulerHolder(
        sigmas=sigmas,
        timesteps=sigmas[:num_steps],
        num_inference_steps=num_steps,
    )


def load_scheduler(args):
    return build_train_scheduler(args)


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
