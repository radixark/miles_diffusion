"""Wan2.2 training pipeline config."""

from __future__ import annotations

import torch
from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config


@register_train_pipeline_config("wan2_2")
class Wan2_2TrainPipelineConfig(TrainPipelineConfig):
    hf_ckpt_name_patterns = ("wan2.2", "wan-2.2")
    # High-noise expert ("transformer") handles t >= boundary, low-noise expert
    # ("transformer_2") the rest.
    boundary_ratio = 0.875
    # Wan DiT expects raw scheduler timesteps (0..num_train_timesteps), no /1000 scaling.
    needs_timestep_scaling = False

    def component_for_timestep(self, timestep: float, num_train_timesteps: int) -> str:
        if timestep >= self.boundary_ratio * num_train_timesteps:
            return "transformer"
        return "transformer_2"

    def select_guidance_scale(
        self,
        timestep: float,
        num_train_timesteps: int,
        guidance_scale: float,
        guidance_scale_2: float | None,
    ) -> float:
        if timestep >= self.boundary_ratio * num_train_timesteps:
            return guidance_scale
        # Rollout backend (sglang-diffusion) uses batch.guidance_scale_2 for low-noise steps with NO fallback;
        # While high-noise and low-noise can be different;
        # A misalignment of guidance_scale_2 between training and rollout would hurt training significantly, so we require it to be set explicitly.
        assert guidance_scale_2 is not None, (
            "Wan2.2 low-noise steps require --diffusion-guidance-scale-2 "
            "(rollout already denoises them with guidance_scale_2)."
        )
        return guidance_scale_2

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None or not cond.encoder_hidden_states:
            return {}
        enc = torch.cat(cond.encoder_hidden_states).to(device)
        if enc.ndim == 2:
            enc = enc.unsqueeze(0)
        return {"encoder_hidden_states": enc}

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
        pad_to_len: int | None = None,  # accepted for interface parity (PR #10); Wan2.2 concats fixed-length T5 embeds
    ) -> dict:
        encs = [kw["encoder_hidden_states"] for kw in per_sample_cond_kwargs]
        return {"encoder_hidden_states": torch.cat(encs, dim=0).to(device)}

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        return noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)

    def preprocess_model_before_fsdp(self, model: torch.nn.Module) -> None:
        return None
