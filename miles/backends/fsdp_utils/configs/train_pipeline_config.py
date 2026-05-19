"""Training-side pipeline config for diffusion models.

Model-specific logic for the GRPO training forward (after rollout has built
train-pair dicts):
  - prepare_cond_kwargs / collate / expand for DenoisingEnv
  - CFG combine
  - FSDP preprocess hooks

Trajectory unpacking for train-pair construction lives in
``miles.utils.train_data_utils.RolloutTrainDataConverter``.
"""

from __future__ import annotations

import abc

import torch
from miles.utils.types import CondKwargs


_REGISTRY: dict[str, type["TrainPipelineConfig"]] = {}


def register_train_pipeline_config(*model_name_patterns: str):
    """Decorator: register a TrainPipelineConfig subclass for one or more model name patterns."""
    def wrapper(cls):
        for pat in model_name_patterns:
            _REGISTRY[pat.lower()] = cls
        return cls
    return wrapper


def get_train_pipeline_config(model_name: str) -> "TrainPipelineConfig":
    """Look up and instantiate a TrainPipelineConfig by matching model_name against registered patterns."""
    name_lower = model_name.lower()
    for pattern, cls in _REGISTRY.items():
        if pattern in name_lower:
            return cls()
    raise ValueError(
        f"No TrainPipelineConfig registered for model '{model_name}'. "
        f"Known patterns: {list(_REGISTRY.keys())}"
    )


class TrainPipelineConfig(abc.ABC):
    """Base class. Subclass per model family."""

    lora_target_modules: list[str] = ["to_q", "to_k", "to_v", "to_out.0"]
    optimizer_state_allowed_missing: list[str] = []
    update_weight_target_module: str = "transformer"

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
            "Must implement collate_cond_for_sample_batch to enable micro-batch-size > 1 in fsdp training"
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
        pass