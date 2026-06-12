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
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Callable

import torch
from miles.utils.types import CondKwargs, DiTTrajectory


@dataclass
class SdeWindowBatch:
    """One sample's tensors after optional SDE-window slicing."""

    latents: torch.Tensor
    next_latents: torch.Tensor
    timesteps: torch.Tensor
    log_prob_old: torch.Tensor
    advantage: torch.Tensor
    rollout_model_output: torch.Tensor | None
    window_size: int
    step_indices: torch.Tensor | None


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

    is_diffusers_pipeline: bool = True
    supports_cfg: bool = True
    fsdp_wrap_classes: list[str] | None = None
    lora_target_modules: list[str] = ["to_q", "to_k", "to_v", "to_out.0"]
    needs_timestep_scaling: bool = True
    # When set, ``dit_trajectory.timesteps`` are on an AdaLN scale (e.g. σ×1000)
    # but CPS/SDE log_prob expects σ in 0..1. Divide by this for sde_step only.
    sde_timestep_divisor: float | None = None
    optimizer_state_allowed_missing: list[str] = []

    def scale_timesteps_for_sde(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self.sde_timestep_divisor is not None:
            return timesteps / float(self.sde_timestep_divisor)
        return timesteps

    def load_model_and_scheduler(
        self,
        args: Namespace,
        init_context_factory: Callable[[], Any],
    ) -> tuple[torch.nn.Module, Any]:
        """Load DiT + scheduler. Default: diffusers ``DiffusionPipeline`` (transformer only)."""
        from diffusers import DiffusionPipeline

        diffusion_model_id = args.diffusion_model or args.hf_checkpoint
        master_dtype_name = getattr(args, "fsdp_master_dtype", "bf16")
        master_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[master_dtype_name]

        with init_context_factory():
            pipeline = DiffusionPipeline.from_pretrained(
                diffusion_model_id,
                torch_dtype=master_dtype,
                trust_remote_code=True,
                text_encoder=None,
                vae=None,
                tokenizer=None,
            )
            model = pipeline.transformer
            scheduler = pipeline.scheduler
            del pipeline
        return model, scheduler

    def apply_sde_step_window(
        self,
        *,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        timesteps: torch.Tensor,
        log_prob_old: torch.Tensor,
        advantage: torch.Tensor,
        rollout_model_output: torch.Tensor | None,
        sde_step_indices: list[int] | None,
        default_window_size: int,
        device: torch.device,
    ) -> SdeWindowBatch:
        """Slice trajectory/objective tensors to the rollout SDE window."""
        step_indices: torch.Tensor | None = None
        if sde_step_indices is not None:
            step_indices = torch.as_tensor(sde_step_indices, device=device, dtype=torch.long)
            latents = latents[step_indices]
            next_latents = next_latents[step_indices]
            timesteps = timesteps[step_indices]
            log_prob_old = log_prob_old[step_indices]
            advantage = advantage[: step_indices.numel()]
            if rollout_model_output is not None:
                n_mo = int(rollout_model_output.shape[0])
                n_win = int(step_indices.numel())
                if n_mo != n_win:
                    # Full-length debug tensors (legacy): index by global step.
                    rollout_model_output = rollout_model_output[step_indices]
                # else: sglang packs debug outputs in SDE-window order (0..W-1).
            window_size = int(step_indices.numel())
        else:
            window_size = default_window_size

        return SdeWindowBatch(
            latents=latents,
            next_latents=next_latents,
            timesteps=timesteps,
            log_prob_old=log_prob_old,
            advantage=advantage,
            rollout_model_output=rollout_model_output,
            window_size=window_size,
            step_indices=step_indices,
        )

    def resolve_tile_sde_step_indices(
        self,
        grids: dict,
        sample_indices: torch.Tensor,
        tstep_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        """Map a training tile to global denoising step indices."""
        window = grids.get("sde_step_indices_window")
        if window is None:
            return None
        return window[sample_indices][:, tstep_indices].reshape(-1).long()

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

    def build_train_cond_kwargs(
        self,
        cond: CondKwargs | None,
        *,
        latents: torch.Tensor,
        args: Namespace,
        device: torch.device,
    ) -> dict:
        """Build per-sample conditioning for the training forward pass."""
        return self.prepare_cond_kwargs(cond, device)

    def resolve_sigmas_ref(
        self,
        timesteps_ref: torch.Tensor,
        sigmas_snapshot: torch.Tensor | None,
        scheduler: Any,
    ) -> torch.Tensor:
        """Build ``[T+1]`` sigma reference for the training scheduler."""
        device = timesteps_ref.device
        if sigmas_snapshot is not None:
            return sigmas_snapshot.to(device).float()

        sched_config = getattr(scheduler, "config", None)
        num_train_timesteps = (
            int(sched_config.num_train_timesteps) if sched_config is not None else 1000
        )
        if not self.needs_timestep_scaling:
            sigmas_ref = self.scale_timesteps_for_sde(timesteps_ref)
        else:
            sigmas_ref = timesteps_ref / float(num_train_timesteps)
        return torch.cat([sigmas_ref, sigmas_ref.new_zeros(1)])

    def build_sde_extra(
        self,
        scheduler: Any,
        grids: dict,
        sample_indices: torch.Tensor,
        tstep_indices: torch.Tensor,
        args: Namespace,
    ) -> dict | None:
        """Optional per-tile metadata for model-specific SDE log_prob."""
        idx = self.resolve_tile_sde_step_indices(grids, sample_indices, tstep_indices)
        if idx is None:
            return None
        return {"sde_step_indices": idx}

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
            f"Must implement collate_cond_for_sample_batch to enable --micro-batch-size-sample in fsdp training"
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

    def forward_velocity(
        self,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        cond: dict,
    ) -> torch.Tensor:
        return model(
            hidden_states=latents_input,
            timestep=timesteps_input,
            return_dict=False,
            **cond,
        )[0]

    def forward_velocity_cfg_joint(
        self,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        joint_cond: dict,
    ) -> torch.Tensor:
        return model(
            hidden_states=torch.cat([latents_input, latents_input], dim=0),
            timestep=torch.cat([timesteps_input, timesteps_input], dim=0),
            return_dict=False,
            **joint_cond,
        )[0]

    def sde_step(
        self,
        scheduler: Any,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
        prev_sample: torch.Tensor,
        *,
        noise_level: float,
        extra: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from miles.utils.sde_log_prob import sde_step_with_logprob

        prev, log_prob, prev_mean, std_dev_t = sde_step_with_logprob(
            scheduler,
            noise_pred.float(),
            timesteps,
            sample.float(),
            prev_sample=prev_sample.float(),
            noise_level=noise_level,
        )
        return prev, log_prob, prev_mean, std_dev_t