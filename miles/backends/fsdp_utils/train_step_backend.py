"""Training-step backends for diffusion GRPO.

TrainPipelineConfig holds model-specific cond/trajectory/CFG logic.
TrainStepBackend holds trainer lifecycle + forward + SDE paths so actor
stays model-agnostic without ``if is_ltx`` branches.
"""

from __future__ import annotations

import abc
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline

from miles.utils.sde_log_prob import sde_step_with_logprob

if TYPE_CHECKING:
    from miles.backends.fsdp_utils.configs.train_pipeline_config import TrainPipelineConfig


def _pack_cond_for_joint_cfg(pos: dict, neg: dict) -> dict:
    out: dict = {}
    for key, value in pos.items():
        if isinstance(value, torch.Tensor):
            out[key] = torch.cat([value, neg[key]], dim=0)
        elif isinstance(value, list):
            out[key] = value + neg[key]
        else:
            out[key] = value
    return out


class TrainStepBackend(abc.ABC):
    """Orchestrates load / forward / SDE for one model family."""

    supports_cfg_training: bool = True
    fsdp_wrap_classes: list[str] | None = None

    def __init__(self, config: TrainPipelineConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def load_model_and_scheduler(
        self,
        args,
        init_context_factory,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[torch.nn.Module, object]:
        ...

    def apply_gradient_checkpointing(self, model: torch.nn.Module, args) -> None:
        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()

    def get_fsdp_wrap_classes(self) -> list[str] | None:
        return self.fsdp_wrap_classes

    def should_use_cfg(self, args) -> bool:
        if not self.supports_cfg_training:
            return False
        guidance_scale = args.diffusion_guidance_scale
        true_cfg_scale = args.diffusion_true_cfg_scale
        cfg_scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        return cfg_scale > 0

    def resolve_sigmas_ref(
        self,
        timesteps_ref: torch.Tensor,
        sigmas_snapshot: torch.Tensor | None,
        scheduler,
        *,
        num_train_timesteps: int,
    ) -> torch.Tensor:
        if sigmas_snapshot is not None:
            return sigmas_snapshot.to(timesteps_ref.device).float()
        sigmas_ref = timesteps_ref / float(num_train_timesteps)
        return torch.cat([sigmas_ref, sigmas_ref.new_zeros(1)])

    def scale_timesteps_for_sde(self, timesteps_flat: torch.Tensor) -> torch.Tensor:
        return timesteps_flat / float(self.config.sde_timestep_divisor)

    @abc.abstractmethod
    def compute_noise_pred(
        self,
        *,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        pos_cond: dict,
        neg_cond: dict | None,
        use_cfg: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
        fsdp_cfg_batching: bool,
        disable_adapter: bool = False,
    ) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def sde_step_logprob(
        self,
        *,
        scheduler,
        noise_pred: torch.Tensor,
        timesteps_for_sde: torch.Tensor,
        timesteps_flat: torch.Tensor,
        latents_flat: torch.Tensor,
        prev_sample: torch.Tensor,
        noise_level: float,
        grids: dict | None = None,
        sample_indices: torch.Tensor | None = None,
        tstep_indices: torch.Tensor | None = None,
        args=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (log_prob, prev_sample_mean, std_dev_t)."""

    def append_model_output_compare_stats(
        self,
        log_stats: dict[str, list[torch.Tensor]],
        noise_pred: torch.Tensor,
        rollout_mo_flat: torch.Tensor,
    ) -> None:
        pass


class DiffusersTrainStepBackend(TrainStepBackend):
    """Default path: diffusers DiT + generic SDE logprob."""

    def load_model_and_scheduler(
        self,
        args,
        init_context_factory,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[torch.nn.Module, object]:
        with init_context_factory():
            pipeline = DiffusionPipeline.from_pretrained(
                args.hf_checkpoint,
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

    def compute_noise_pred(
        self,
        *,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        pos_cond: dict,
        neg_cond: dict | None,
        use_cfg: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
        fsdp_cfg_batching: bool,
        disable_adapter: bool = False,
    ) -> torch.Tensor:
        def _forward(cond: dict) -> torch.Tensor:
            return model(
                hidden_states=latents_input,
                timestep=timesteps_input,
                return_dict=False,
                **cond,
            )[0]

        adapter_ctx = model.disable_adapter() if disable_adapter else nullcontext()
        with adapter_ctx:
            if not use_cfg:
                return _forward(pos_cond)
            if fsdp_cfg_batching:
                joint_cond = _pack_cond_for_joint_cfg(pos_cond, neg_cond)
                joint_out = model(
                    hidden_states=torch.cat([latents_input, latents_input], dim=0),
                    timestep=torch.cat([timesteps_input, timesteps_input], dim=0),
                    return_dict=False,
                    **joint_cond,
                )[0]
                noise_pred_pos, noise_pred_neg = joint_out.chunk(2, dim=0)
            else:
                noise_pred_pos = _forward(pos_cond)
                noise_pred_neg = _forward(neg_cond)
            return self.config.cfg_combine(
                noise_pred_pos,
                noise_pred_neg,
                guidance_scale,
                true_cfg_scale=true_cfg_scale,
            )

    def sde_step_logprob(
        self,
        *,
        scheduler,
        noise_pred: torch.Tensor,
        timesteps_for_sde: torch.Tensor,
        timesteps_flat: torch.Tensor,
        latents_flat: torch.Tensor,
        prev_sample: torch.Tensor,
        noise_level: float,
        grids: dict | None = None,
        sample_indices: torch.Tensor | None = None,
        tstep_indices: torch.Tensor | None = None,
        args=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, log_prob, prev_mean, std_dev_t = sde_step_with_logprob(
            scheduler,
            noise_pred.float(),
            timesteps_flat,
            latents_flat.float(),
            prev_sample=prev_sample.float(),
            noise_level=noise_level,
        )
        return log_prob, prev_mean, std_dev_t


class LTXTrainStepBackend(TrainStepBackend):
    """LTX-2.3: custom loader, velocity forward, CPS SDE."""

    supports_cfg_training = False
    fsdp_wrap_classes = ["BasicAVTransformerBlock"]

    def load_model_and_scheduler(
        self,
        args,
        init_context_factory,
        *,
        master_dtype: torch.dtype | None = None,
    ) -> tuple[torch.nn.Module, object]:
        from dataclasses import dataclass, field

        from ltx_core.components.schedulers import LTX2Scheduler

        from miles.backends.fsdp_utils.configs.ltx import (
            load_ltx_transformer_for_train,
            resolve_transformer_checkpoint,
        )

        @dataclass
        class _LTXSchedulerHolder:
            sigmas: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
            timesteps: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
            num_inference_steps: int = 0
            _step_index: int | None = None
            _begin_index: int | None = None

            def to(self, device):
                self.sigmas = self.sigmas.to(device)
                self.timesteps = self.timesteps.to(device)
                return self

        master_dtype_name = getattr(args, "fsdp_master_dtype", "bf16")
        resolved_dtype = master_dtype or {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[master_dtype_name]

        checkpoint = resolve_transformer_checkpoint(
            args.diffusion_model,
            explicit_path=getattr(args, "sglang_transformer_weights_path", None),
        )
        model = load_ltx_transformer_for_train(checkpoint, device="cpu", dtype=resolved_dtype)

        num_steps = int(getattr(args, "diffusion_num_steps", 24))
        ltx_sched = LTX2Scheduler()
        sigmas = ltx_sched.execute(steps=num_steps).float()
        scheduler = _LTXSchedulerHolder(
            sigmas=sigmas, timesteps=sigmas[:num_steps], num_inference_steps=num_steps,
        )

        if getattr(args, "gradient_checkpointing", False):
            if hasattr(model, "set_gradient_checkpointing"):
                model.set_gradient_checkpointing(True)
            elif hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing()

        return model, scheduler

    def apply_gradient_checkpointing(self, model: torch.nn.Module, args) -> None:
        # Applied inside load_model_and_scheduler for LTX.
        pass

    def resolve_sigmas_ref(
        self,
        timesteps_ref: torch.Tensor,
        sigmas_snapshot: torch.Tensor | None,
        scheduler,
        *,
        num_train_timesteps: int = 1000,
    ) -> torch.Tensor:
        device = timesteps_ref.device
        if sigmas_snapshot is not None:
            return sigmas_snapshot.to(device).float()
        sigmas_ref = timesteps_ref / float(self.config.sde_timestep_divisor)
        return torch.cat([sigmas_ref, sigmas_ref.new_zeros(1)])

    def compute_noise_pred(
        self,
        *,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        pos_cond: dict,
        neg_cond: dict | None,
        use_cfg: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
        fsdp_cfg_batching: bool,
        disable_adapter: bool = False,
    ) -> torch.Tensor:
        adapter_ctx = model.disable_adapter() if disable_adapter else nullcontext()
        with adapter_ctx:
            return self.config.forward_velocity(model, latents_input, timesteps_input, pos_cond)

    def sde_step_logprob(
        self,
        *,
        scheduler,
        noise_pred: torch.Tensor,
        timesteps_for_sde: torch.Tensor,
        timesteps_flat: torch.Tensor,
        latents_flat: torch.Tensor,
        prev_sample: torch.Tensor,
        noise_level: float,
        grids: dict | None = None,
        sample_indices: torch.Tensor | None = None,
        tstep_indices: torch.Tensor | None = None,
        args=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sde_extra = self.config.build_sde_extra(scheduler, grids, sample_indices, tstep_indices, args)
        _, log_prob, prev_mean, std_dev_t = self.config.sde_step(
            scheduler,
            noise_pred,
            timesteps_for_sde,
            latents_flat,
            prev_sample=prev_sample,
            noise_level=noise_level,
            extra=sde_extra,
        )
        return log_prob, prev_mean, std_dev_t

    def append_model_output_compare_stats(
        self,
        log_stats: dict[str, list[torch.Tensor]],
        noise_pred: torch.Tensor,
        rollout_mo_flat: torch.Tensor,
    ) -> None:
        flat_train = noise_pred.float().reshape(noise_pred.shape[0], -1)
        flat_rollout = rollout_mo_flat.float().reshape(rollout_mo_flat.shape[0], -1)
        log_stats["model_output_cosine_sim"].append(
            F.cosine_similarity(flat_train, flat_rollout, dim=1).mean().detach()
        )
