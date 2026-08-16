"""Cosmos3 training pipeline config."""

from __future__ import annotations

import math

import torch
from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config

# Cosmos3 reuses the Wan2.2 VAE (4x temporal compression).
_VAE_TEMPORAL_FACTOR = 4

# GEN-tower param name fragments; everything else (UND tower, lm_head, unused heads) stays frozen.
_GEN_PARAM_FRAGMENTS = (
    ".add_q_proj.",
    ".add_k_proj.",
    ".add_v_proj.",
    ".to_add_out.",
    ".norm_added_q.",
    ".norm_added_k.",
    ".mlp_moe_gen.",
    ".input_layernorm_moe_gen.",
    ".post_attention_layernorm_moe_gen.",
    ".norm_moe_gen.",
    ".proj_in.",
    ".proj_out.",
    ".time_embedder.",
)


def _is_gen_param(name: str) -> bool:
    dotted = f".{name}"
    return any(fragment in dotted for fragment in _GEN_PARAM_FRAGMENTS)


@register_train_pipeline_config("cosmos3")
class Cosmos3TrainPipelineConfig(TrainPipelineConfig):
    hf_ckpt_name_patterns = ("cosmos3", "cosmos-3")
    # Timesteps stay fp32 (bf16 rounds the karras grid); conds pass through, the packed forward casts its own inputs.
    input_dtype_policy = {"latents": "default", "cond": None, "timestep": "fp32"}
    # The packed forward is single-sample; never batch the CFG branches.
    cfg_batching = False
    lora_target_modules = ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]

    @classmethod
    def validate_args(cls, args) -> None:
        if list(args.update_weight_target_modules) != ["transformer"]:
            raise ValueError("Cosmos3 requires --update-weight-target-module transformer.")

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None or cond.text_ids is None:
            return {}
        return {
            "text_ids": cond.text_ids.to(device),
            "text_mask": cond.text_mask.to(device),
            "fps": cond.fps,
        }

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
        pad_to_len: int | None = None,
    ) -> dict:
        # Packed single-sample forward: keep per-sample kwargs; no batched tensor to build.
        return {"per_sample": per_sample_cond_kwargs}

    def compute_noise_pred(
        self,
        *,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        pos_cond: dict | None,
        neg_cond: dict | None,
        joint_cond: dict | None,
        use_cfg: bool,
        cfg_batching: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
    ) -> torch.Tensor:
        assert not cfg_batching, "Cosmos3 packed forward is single-sample; cfg_batching unsupported"
        config = model.config
        preds = []
        for i, pos in enumerate(pos_cond["per_sample"]):
            latent = latents_input[i : i + 1]
            timestep = float(timesteps_input[i])
            pred = self._packed_forward(model, latent, timestep, pos, config)
            if use_cfg:
                pred_neg = self._packed_forward(model, latent, timestep, neg_cond["per_sample"][i], config)
                pred = self.cfg_combine(pred, pred_neg, guidance_scale, true_cfg_scale=true_cfg_scale)
            preds.append(pred)
        return torch.stack(preds, dim=0)

    def _packed_forward(
        self,
        model: torch.nn.Module,
        latent: torch.Tensor,
        timestep: float,
        cond: dict,
        config,
    ) -> torch.Tensor:
        """One packed (text, video) forward mirroring the diffusers denoising loop (T2V/T2I: all frames noisy)."""
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
            get_3d_mrope_ids_text_tokens,
            get_3d_mrope_ids_vae_tokens,
        )

        device = latent.device
        und_len = int(cond["text_mask"].sum().item())
        input_ids = cond["text_ids"].reshape(-1)[:und_len]

        text_mrope_ids, next_offset = get_3d_mrope_ids_text_tokens(
            num_tokens=und_len,
            temporal_offset=0,
            use_float_positions=config.enable_fps_modulation,
        )
        vision_offset = next_offset + config.unified_3d_mrope_temporal_modality_margin

        p = config.latent_patch_size
        _, _, latent_t, latent_h, latent_w = latent.shape
        patch_h = math.ceil(latent_h / p)
        patch_w = math.ceil(latent_w / p)
        num_vision_tokens = latent_t * patch_h * patch_w

        vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=vision_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=cond["fps"] if config.enable_fps_modulation else None,
            base_fps=float(config.base_fps),
            temporal_compression_factor=_VAE_TEMPORAL_FACTOR,
        )

        sequence_length = und_len + num_vision_tokens
        vision_sequence_indexes = torch.arange(und_len, sequence_length, dtype=torch.long, device=device)
        preds_vision, _, _ = model(
            input_ids=input_ids,
            text_indexes=torch.arange(und_len, dtype=torch.long, device=device),
            position_ids=torch.cat([text_mrope_ids, vision_mrope_ids], dim=1).to(device),
            und_len=und_len,
            sequence_length=sequence_length,
            vision_tokens=[latent],
            vision_token_shapes=[(latent_t, patch_h, patch_w)],
            vision_sequence_indexes=vision_sequence_indexes,
            vision_mse_loss_indexes=vision_sequence_indexes,
            vision_timesteps=torch.full((num_vision_tokens,), timestep, device=device, dtype=torch.float32),
            vision_noisy_frame_indexes=[torch.arange(latent_t, dtype=torch.long, device=device)],
        )
        return preds_vision[0]

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        return noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)

    def postprocess_model_after_materialize(self, model: torch.nn.Module) -> None:
        # The UND tower sits inside the training graph, so it must be explicitly frozen.
        for name, param in model.named_parameters():
            if "lora_" not in name and not _is_gen_param(name):
                param.requires_grad_(False)

        # Cast the timestep sinusoid to the MLP weight dtype before linear_1, exactly as sglang-d does.
        def _cast_to_weight_dtype(module, args):
            dtype = module.linear_1.weight.dtype
            return tuple(a.to(dtype) if torch.is_tensor(a) else a for a in args)

        model.time_embedder.register_forward_pre_hook(_cast_to_weight_dtype)
