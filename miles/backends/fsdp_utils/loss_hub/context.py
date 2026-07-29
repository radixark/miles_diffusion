"""Shared handles passed into diffusion prepare / loss-formula hooks."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class DiffusionLossContext:
    """Train-side handles for prepare / loss-formula callables.

    Owned by the FSDP actor; kept free of Ray / optim internals so hooks stay
    unit-testable and swappable via ``--custom-*-path``.
    """

    models: dict[str, torch.nn.Module]
    model: torch.nn.Module
    train_pipeline_config: Any
    sde_backend: Any
    scheduler: Any
    args: Namespace
    forward_dtype: torch.dtype
    device: torch.device
    # Optional EMA shadow handle; owned by actor (see ``fsdp_utils.ema``).
    ema_shadow: Any = None


@dataclass
class PreparedBatch:
    """Actor-owned DiT forward inputs produced by a prepare hook.

    ``extras`` carries algorithm-specific tensors for the loss formula
    (e.g. ``next_latents`` / ``log_prob_old`` for Flow-GRPO, ``x0`` for NFT).
    """

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
