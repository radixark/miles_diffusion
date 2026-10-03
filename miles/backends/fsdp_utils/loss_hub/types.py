"""Shared data types passed into diffusion prepare and loss hooks."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class DiffusionLossContext:
    """Train-side handles for prepare and loss hooks."""

    models: dict[str, torch.nn.Module]
    train_pipeline_config: Any
    sde_backend: Any
    args: Namespace
    forward_dtype: torch.dtype
    device: torch.device
    rollout_id: int = 0
    microbatch_id: int = 0
    dp_rank: int = 0
    reference_models: dict[str, torch.nn.Module] = field(default_factory=dict)
    teacher_models: dict[str, torch.nn.Module] = field(default_factory=dict)


@dataclass(frozen=True)
class FlowNoiseSchedule:
    """Discrete rectified-flow grid; ``sigmas`` may end with one terminal value that is never sampled."""

    timesteps: torch.Tensor
    sigmas: torch.Tensor

    @classmethod
    def shifted(cls, shift: float, num_train_timesteps: int) -> FlowNoiseSchedule:
        """``num_train_timesteps`` evenly spaced sigmas in (0, 1], warped by ``shift * s / (1 + (shift - 1) * s)``."""
        sigmas = torch.linspace(1.0, 1.0 / num_train_timesteps, num_train_timesteps, dtype=torch.float64)
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        return cls(
            (sigmas * num_train_timesteps).to(torch.float32),
            torch.cat([sigmas, sigmas.new_zeros(1)]).to(torch.float32),
        )


@dataclass
class PreparedBatch:
    """Actor-owned DiT forward inputs produced by a prepare hook.

    ``latents`` and both timestep fields are keyed by latent stream name: "visual", plus e.g.
    "audio" for joint audio-video models. The DiT prediction comes back keyed the same way.
    """

    latents: dict[str, torch.Tensor]
    timesteps: dict[str, torch.Tensor]
    timesteps_for_model: dict[str, torch.Tensor]
    model: nn.Module
    component_name: str
    guidance_scale: float
    use_cfg: bool
    cfg_batching: bool
    true_cfg_scale: float | None
    pos_cond: dict | None
    neg_cond: dict | None
    joint_cond: dict | None
    advantage: torch.Tensor
    extras: dict[str, Any] = field(default_factory=dict)
