"""Wan2.2 training pipeline config."""

from __future__ import annotations

import torch
from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config


@register_train_pipeline_config("Wan2.2-T2V-A14B", "Wan-AI/Wan2.2-T2V-A14B")
class Wan2_2TrainPipelineConfig(TrainPipelineConfig):
    target_components = ["transformer"]

    def prepare_timesteps_for_model(
        self,
        timesteps: torch.Tensor,
        *,
        num_train_timesteps: int,
    ) -> torch.Tensor:
        return timesteps

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
