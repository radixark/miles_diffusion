"""Shared data types passed into diffusion prepare and loss hooks."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from miles.backends.fsdp_utils.input_dtype_policy import InputDtypes, cast_input


@dataclass(frozen=True, eq=False)
class DiffusionLossContext:
    """Rollout-level, read-only handles for the loss hooks; what changes per micro-batch travels as arguments.

    Compared by identity: nothing hashes a context, and a synthesized hash would fail on the dict fields.
    """

    models: dict[str, torch.nn.Module]
    train_pipeline_config: Any
    # Keyed by latent stream.
    sde_backends: dict[str, Any]
    args: Namespace
    forward_dtype: torch.dtype
    device: torch.device
    rollout_id: int = 0
    dp_rank: int = 0
    reference_models: dict[str, torch.nn.Module] = field(default_factory=dict)
    external_data: dict[str, Any] = field(default_factory=dict)


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
    """One stage's DiT forward input, built by a prepare hook.

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

    def model_inputs(self, input_dtypes: InputDtypes) -> dict[str, Any]:
        """What the forward sends the model: the latents, model timesteps and conds cast per ``input_dtypes``, under
        ``compute_noise_pred``'s kwarg names. The raw fields stay as rollout handed them, for the formulas."""

        def cond(value: dict | None) -> dict | None:
            return None if value is None else {key: cast_input(item, input_dtypes.cond) for key, item in value.items()}

        return {
            "latents_input": {name: cast_input(x, input_dtypes.latents) for name, x in self.latents.items()},
            "timesteps_input": {
                name: cast_input(t, input_dtypes.timestep) for name, t in self.timesteps_for_model.items()
            },
            "pos_cond": cond(self.pos_cond),
            "neg_cond": cond(self.neg_cond),
            "joint_cond": cond(self.joint_cond),
        }


@dataclass(frozen=True, eq=False)
class PreparedBatches:
    """The stages of one micro-batch, as the prepare hook returns them.

    ``actor`` is the training forward's input; the EMA and reference forwards take it too. ``teacher`` is the frozen
    teacher's input with its own CFG fields: the trunk runs the teacher loaded by --teacher-load for its
    ``component_name`` on it (its ``model`` field is not read). ``actor_forward`` is a second, no-grad forward of the
    actor's model. Each stage's prediction reaches the formula next to its batch: ``new_pred``; ``teacher_pred`` and
    ``teacher_batch``; ``actor_forward_pred`` and ``actor_forward_batch``.
    """

    actor: PreparedBatch
    teacher: PreparedBatch | None = None
    actor_forward: PreparedBatch | None = None
