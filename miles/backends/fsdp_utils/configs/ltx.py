"""LTX-2.3 video diffusion training pipeline config.

Adapts ltx_core's ``LTXModel`` (non-diffusers; Modality-keyed forward, patchified
``[B, T, D]`` token latents, per-token timesteps, custom ``LTX2Scheduler``) into
miles' FSDP GRPO training loop.
"""

from __future__ import annotations

import torch

from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config


@register_train_pipeline_config("ltx")
class LTXTrainPipelineConfig(TrainPipelineConfig):
    """Training-side adapter for LTX-2.3 video DiT."""

    is_diffusers_pipeline = False
    needs_timestep_scaling = False
    # Rollout stores σ×1000 in dit_trajectory.timesteps; CPS uses scheduler σ∈[0,1].
    sde_timestep_divisor = 1000.0
    supports_cfg = False

    fsdp_wrap_classes = ["BasicAVTransformerBlock"]

    lora_target_modules = [
        "to_q", "to_k", "to_v", "to_out.0",
        "net.0.proj", "net.2",
    ]

    def load_model_and_scheduler(self, args, init_context_factory):
        from dataclasses import dataclass, field

        from ltx_core.components.schedulers import LTX2Scheduler
        from miles.backends.model_families.ltx import (
            load_ltx_transformer_for_train,
            resolve_transformer_checkpoint,
        )

        @dataclass
        class _LTXSchedulerHolder:
            sigmas: "torch.Tensor" = field(default_factory=lambda: torch.tensor([]))
            timesteps: "torch.Tensor" = field(default_factory=lambda: torch.tensor([]))
            num_inference_steps: int = 0
            _step_index: int | None = None
            _begin_index: int | None = None

            def to(self, device):
                self.sigmas = self.sigmas.to(device)
                self.timesteps = self.timesteps.to(device)
                return self

        master_dtype_name = getattr(args, "fsdp_master_dtype", "bf16")
        master_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[master_dtype_name]

        from miles.backends.model_families.ltx import resolve_transformer_checkpoint

        checkpoint = resolve_transformer_checkpoint(
            args.diffusion_model,
            explicit_path=getattr(args, "sglang_transformer_weights_path", None),
        )
        model = load_ltx_transformer_for_train(checkpoint, device="cpu", dtype=master_dtype)

        num_steps = int(getattr(args, "diffusion_num_steps", 24))
        ltx_sched = LTX2Scheduler()
        sigmas = ltx_sched.execute(steps=num_steps).float()
        scheduler = _LTXSchedulerHolder(
            sigmas=sigmas, timesteps=sigmas[:num_steps], num_inference_steps=num_steps,
        )
        return model, scheduler

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None:
            return {}
        kwargs: dict = {}
        if cond.encoder_hidden_states:
            ctx = torch.cat(cond.encoder_hidden_states).to(device)
            if ctx.ndim == 2:
                ctx = ctx.unsqueeze(0)
            kwargs["context"] = ctx
        if cond.audio_encoder_hidden_states:
            audio_ctx = torch.cat(cond.audio_encoder_hidden_states).to(device)
            if audio_ctx.ndim == 2:
                audio_ctx = audio_ctx.unsqueeze(0)
            kwargs["audio_context"] = audio_ctx
        if cond.encoder_attention_mask is not None:
            mask = cond.encoder_attention_mask.to(device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            kwargs["context_mask"] = mask
        if cond.audio_encoder_attention_mask is not None:
            audio_mask = cond.audio_encoder_attention_mask.to(device)
            if audio_mask.ndim == 1:
                audio_mask = audio_mask.unsqueeze(0)
            kwargs["audio_context_mask"] = audio_mask
        return kwargs

    def build_train_cond_kwargs(
        self,
        cond: CondKwargs | None,
        *,
        latents: torch.Tensor,
        args,
        device: torch.device,
    ) -> dict:
        """Merge rollout text embeds with locally rebuilt T2V geometry."""
        from miles.backends.fsdp_utils.ltx_geometry import build_ltx_t2v_geometry

        kwargs = self.prepare_cond_kwargs(cond, device)
        if "context" not in kwargs:
            raise ValueError(
                "LTX train requires denoising_env.pos_cond_kwargs.encoder_hidden_states"
            )

        ref = latents[0] if latents.ndim >= 2 else latents
        if ref.ndim == 2:
            batch_size, num_tokens, latent_dim = 1, ref.shape[0], ref.shape[1]
        else:
            batch_size, num_tokens, latent_dim = ref.shape[0], ref.shape[1], ref.shape[2]

        geom = build_ltx_t2v_geometry(
            batch_size=batch_size,
            num_tokens=num_tokens,
            latent_dim=latent_dim,
            height=int(getattr(args, "diffusion_height", 512)),
            width=int(getattr(args, "diffusion_width", 512)),
            num_frames=int(getattr(args, "ltx_frames", 25)),
            fps=float(getattr(args, "ltx_fps", 24.0)),
            device=device,
            dtype=ref.dtype,
        )
        kwargs.update(geom)
        return kwargs

    def build_sde_extra(
        self,
        scheduler,
        grids: dict,
        sample_indices: torch.Tensor,
        tstep_indices: torch.Tensor,
        args,
    ) -> dict | None:
        if grids.get("sde_step_indices_window") is None:
            return None

        idx = grids["sde_step_indices_window"][sample_indices][:, tstep_indices]
        idx = idx.reshape(-1).long()

        return {
            "sigmas": scheduler.sigmas,
            "sde_step_indices": idx,
            "dynamics_type": getattr(args, "ltx_dynamics_type", "cps"),
            "sigma_min_override": getattr(args, "ltx_sigma_min", None),
        }

    def expand_cond_for_timestep_batch(self, cond_kwargs: dict, batch_size: int) -> dict:
        out: dict = {}
        for k, v in cond_kwargs.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.expand(batch_size, *v.shape[1:]) if v.shape[0] == 1 else v
            else:
                out[k] = v
        return out

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
    ) -> dict:
        out: dict = {}
        for key in per_sample_cond_kwargs[0]:
            values = [kw[key] for kw in per_sample_cond_kwargs if key in kw]
            if not values:
                continue
            if isinstance(values[0], torch.Tensor):
                out[key] = torch.cat(values, dim=0).to(device)
            else:
                out[key] = values
        return out

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        if scale == 1.0:
            return noise_pred_pos
        return noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)

    def preprocess_model_before_fsdp(self, model: torch.nn.Module) -> None:
        return None

    @staticmethod
    def _modality_timesteps_for_adaln(per_token_t: torch.Tensor) -> torch.Tensor:
        """Collapse per-token sigma to batch-global AdaLN input when uniform.

        sglang rollout builds temb with shape ``[B, 1, D]`` (scheduler timestep
        is batch-scalar expanded only for masking). ltx_core defaults to
        ``[B, T, D]`` when ``Modality.timesteps`` has length T, which diverges
        in AdaLN even when every active token shares the same sigma.
        """
        if per_token_t.ndim != 2 or per_token_t.shape[1] == 1:
            return per_token_t
        ref = per_token_t[:, :1]
        if torch.allclose(per_token_t, ref.expand_as(per_token_t), rtol=0.0, atol=0.0):
            return ref
        return per_token_t

    def forward_velocity(
        self,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        cond: dict,
    ) -> torch.Tensor:
        from ltx_core.model.transformer.modality import Modality
        from ltx_core.utils import to_denoised

        device = latents_input.device
        dtype = latents_input.dtype
        B = latents_input.shape[0]

        # dit_trajectory.timesteps are σ×1000; ltx_core AdaLN expects σ∈[0,1] and
        # multiplies by timestep_scale_multiplier (1000) internally.
        sigma_scaled = timesteps_input.to(latents_input.dtype)
        sigma_unit = sigma_scaled / float(self.sde_timestep_divisor)
        denoise_mask = cond["denoise_mask"].to(device)
        denoise_mask_2d = denoise_mask.squeeze(-1) if denoise_mask.ndim == 3 else denoise_mask
        denoise_mask_float = denoise_mask_2d.float()

        per_token_t = (sigma_unit.view(B, 1) * denoise_mask_2d).to(dtype)
        adaln_timesteps = self._modality_timesteps_for_adaln(per_token_t)

        video_modality = Modality(
            enabled=True,
            latent=latents_input,
            sigma=sigma_unit.reshape(B),
            timesteps=adaln_timesteps,
            positions=cond["positions"].to(dtype),
            context=cond["context"].to(dtype),
            context_mask=None,
        )
        with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype):
            velocity, _ = model(video=video_modality, audio=None, perturbations=None)

        per_token_t_3d = per_token_t.unsqueeze(-1) if per_token_t.ndim == 2 else per_token_t
        x0_pred = to_denoised(latents_input, velocity, per_token_t_3d).float()

        clean_latent = cond["clean_latent"].to(device).float()
        denoise_mask_3d = denoise_mask_float.unsqueeze(-1) if denoise_mask_float.ndim == 2 else denoise_mask_float
        x0_pred = x0_pred * denoise_mask_3d + clean_latent * (1.0 - denoise_mask_3d)

        sigma_safe = torch.clamp(sigma_unit, min=1e-8).view(B, 1, 1)
        velocity_for_sde = (latents_input.float() - x0_pred) / sigma_safe
        return velocity_for_sde.to(dtype)

    def forward_velocity_cfg_joint(
        self,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        joint_cond: dict,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "LTX trains with guidance_scale=1.0; --fsdp-cfg-batching is not supported."
        )

    def sde_step(
        self,
        scheduler,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
        prev_sample: torch.Tensor,
        *,
        noise_level: float,
        extra: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from miles.utils.sde_log_prob import sde_step_with_logprob_dynamics

        if extra is None or "sigmas" not in extra or "sde_step_indices" not in extra:
            raise ValueError(
                "LTXTrainPipelineConfig.sde_step requires extra={'sigmas','sde_step_indices',...}"
            )
        sigmas = extra["sigmas"].to(sample.device).float()
        step_indices = extra["sde_step_indices"].to(sample.device).long()
        sigma_view = timesteps.float()
        sigma_next = sigmas[torch.clamp(step_indices + 1, max=len(sigmas) - 1)]

        dynamics_type = extra.get("dynamics_type", "cps")
        sigma_min_override = extra.get("sigma_min_override", None)
        if sigma_min_override == 0.0:
            sigma_min_override = None

        prev, log_prob, prev_mean, std_dev_t, _dt_sqrt = sde_step_with_logprob_dynamics(
            model_output=noise_pred.float(),
            sigma=sigma_view,
            sigma_next=sigma_next,
            sample=sample.float(),
            sigmas=sigmas,
            prev_sample=prev_sample.float(),
            sigma_min_override=sigma_min_override,
            noise_level=noise_level,
            dynamics_type=dynamics_type,
        )
        if std_dev_t.ndim > 1:
            std_dev_t = std_dev_t.mean(dim=tuple(range(1, std_dev_t.ndim)))
        return prev, log_prob, prev_mean, std_dev_t
