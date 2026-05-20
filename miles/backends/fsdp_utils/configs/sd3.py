"""Stable Diffusion 3 training pipeline config."""

from __future__ import annotations

import torch

from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config


@register_train_pipeline_config("stabilityai/stable-diffusion-3")
class SD3TrainPipelineConfig(TrainPipelineConfig):
    """Training-side adapters for diffusers SD3Transformer2DModel."""

    lora_target_modules = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
    ]
    needs_timestep_scaling = False

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None:
            return {}

        kwargs = {}
        if cond.encoder_hidden_states:
            encoder_hidden_states = torch.cat(cond.encoder_hidden_states).to(device)
            if encoder_hidden_states.ndim == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            kwargs["encoder_hidden_states"] = encoder_hidden_states

        if cond.pooled_projections:
            pooled_projections = torch.cat(cond.pooled_projections).to(device)
            if pooled_projections.ndim == 1:
                pooled_projections = pooled_projections.unsqueeze(0)
            kwargs["pooled_projections"] = pooled_projections

        return kwargs

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
    ) -> dict:
        out = {}
        for key in per_sample_cond_kwargs[0]:
            values = [kwargs[key] for kwargs in per_sample_cond_kwargs if key in kwargs]
            if values and isinstance(values[0], torch.Tensor):
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
        return noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)

    def preprocess_model_before_fsdp(self, model: torch.nn.Module) -> None:
        return None
