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
    scheduler: Any
    args: Namespace
    forward_dtype: torch.dtype
    device: torch.device
    rollout_id: int = 0
    microbatch_id: int = 0
    dp_rank: int = 0


@dataclass
class PreparedBatch:
    """Actor-owned DiT forward inputs produced by a prepare hook."""

    latents: torch.Tensor
    timesteps: torch.Tensor
    timesteps_for_model: torch.Tensor
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
