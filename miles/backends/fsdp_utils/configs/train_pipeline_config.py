"""Training-side pipeline config for diffusion models.

Mirrors the spirit of sglang-d's PipelineConfig but only contains the
model-specific logic needed for the GRPO training loop:
  - How to prepare conditioning kwargs from DenoisingEnv
  - How to unpack trajectories
  - How to apply CFG (with or without rescale)
  - How to expand conditioning for timestep batching

Each model (QwenImage, SD3, Flux, ...) subclasses TrainPipelineConfig
and overrides the relevant methods.
"""

from __future__ import annotations

import abc

import torch
from miles.utils.types import CondKwargs, DiTTrajectory


_REGISTRY: dict[str, type[TrainPipelineConfig]] = {}


def register_train_pipeline_config(*model_name_patterns: str):
    """Decorator: register a TrainPipelineConfig subclass for one or more model name patterns."""

    def wrapper(cls):
        for pat in model_name_patterns:
            _REGISTRY[pat.lower()] = cls
        return cls

    return wrapper


def get_train_pipeline_config(model_name: str) -> TrainPipelineConfig:
    """Look up and instantiate a TrainPipelineConfig by matching model_name against registered patterns."""
    name_lower = model_name.lower()
    for pattern, cls in _REGISTRY.items():
        if pattern in name_lower:
            return cls()
    raise ValueError(
        f"No TrainPipelineConfig registered for model '{model_name}'. " f"Known patterns: {list(_REGISTRY.keys())}"
    )


class TrainPipelineConfig(abc.ABC):
    """Base class. Subclass per model family."""

    lora_target_modules: list[str] = ["to_q", "to_k", "to_v", "to_out.0"]
    needs_timestep_scaling: bool = True
    optimizer_state_allowed_missing: list[str] = []

    def prepare_trajectory(
        self,
        traj: DiTTrajectory,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack trajectory into (latents, next_latents, timesteps).

        Default handles the common (T+1, ...) layout. Override for models
        with different trajectory formats.
        """
        all_latents = traj.latents.to(device, dtype=torch.float32)
        latents = all_latents[:-1]
        next_latents = all_latents[1:]
        timesteps = traj.timesteps.to(device, dtype=torch.float32)
        return latents, next_latents, timesteps

    @abc.abstractmethod
    def prepare_cond_kwargs(
        self,
        cond: CondKwargs | None,
        device: torch.device,
    ) -> dict:
        """Convert CondKwargs to model-specific forward() kwargs."""

    def expand_cond_for_timestep_batch(
        self,
        cond_kwargs: dict,
        batch_size: int,
    ) -> dict:
        """Expand per-sample conditioning to a timestep batch."""
        out = {}
        for k, v in cond_kwargs.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.expand(batch_size, *v.shape[1:]) if v.shape[0] == 1 else v
            elif isinstance(v, list):
                out[k] = v * batch_size if len(v) == 1 else v
            else:
                out[k] = v
        return out

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
    ) -> dict:
        """Stack a list of per-sample cond_kwargs (output of prepare_cond_kwargs)
        into a single batched dict suitable for one DiT forward over M samples.

        Model-specific because variable-length text embeds need padding + mask.
        Default: naive concat along batch dim, only valid when shapes match.
        """
        raise NotImplementedError(
            "Must implement collate_cond_for_sample_batch to enable --micro-batch-size-sample in fsdp training"
        )

    @abc.abstractmethod
    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        """Apply classifier-free guidance. Model-specific (e.g. rescale or not)."""

    @abc.abstractmethod
    def preprocess_model_before_fsdp(self, model: torch.nn.Module) -> None:
        """Preprocess the model before FSDP."""
