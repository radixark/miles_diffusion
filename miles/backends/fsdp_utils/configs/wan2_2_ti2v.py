"""Wan2.2-TI2V-5B single-DiT training configuration."""

from __future__ import annotations

import torch

from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config


@register_train_pipeline_config("wan2_2_ti2v")
class Wan2_2_TI2VTrainPipelineConfig(TrainPipelineConfig):
    """Reproduce Wan2.2 TI2V's masked, sequence-expanded timestep input."""

    hf_ckpt_name_patterns = ("wan2.2-ti2v", "wan2_2_ti2v", "ti2v-5b")
    cfg_batching = False
    lora_target_modules = [
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn2.to_q",
        "attn2.to_k",
        "attn2.to_v",
        "attn2.to_out.0",
        "ffn.net.0.proj",
        "ffn.net.2",
    ]

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None:
            return {}
        result: dict = {}
        if cond.encoder_hidden_states:
            enc = torch.cat(cond.encoder_hidden_states).to(device)
            if enc.ndim == 2:
                enc = enc.unsqueeze(0)
            result["encoder_hidden_states"] = enc
        if cond.wan_ti2v_reserved_frames_mask is not None:
            mask = cond.wan_ti2v_reserved_frames_mask.to(device=device, dtype=torch.float32)
            # The rollout mask is repeated across latent channels. Keep one
            # mask per sample before collating the training batch.
            if mask.ndim == 5:
                mask = mask[:, 0]
            elif mask.ndim == 4:
                mask = mask[:1]
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)
            if mask.ndim != 4 or mask.shape[0] != 1:
                raise ValueError(
                    "Wan TI2V reserved-frames mask must normalize to [1,T,H,W], "
                    f"got {tuple(mask.shape)}"
                )
            result["wan_ti2v_reserved_frames_mask"] = mask
        if cond.wan_ti2v_patch_size is not None:
            result["wan_ti2v_patch_size"] = tuple(cond.wan_ti2v_patch_size)
        return result

    def collate_cond_for_sample_batch(
        self, per_sample_cond_kwargs: list[dict], device: torch.device, pad_to_len: int | None = None
    ) -> dict:
        result: dict = {}
        encs = [kw["encoder_hidden_states"] for kw in per_sample_cond_kwargs]
        result["encoder_hidden_states"] = torch.cat(encs, dim=0).to(device)
        masks = [kw["wan_ti2v_reserved_frames_mask"] for kw in per_sample_cond_kwargs]
        result["wan_ti2v_reserved_frames_mask"] = torch.cat(masks, dim=0).to(device)
        result["wan_ti2v_patch_size"] = per_sample_cond_kwargs[0]["wan_ti2v_patch_size"]
        return result

    def compute_noise_pred(
        self,
        *,
        model,
        latents_input,
        timesteps_input,
        pos_cond,
        neg_cond,
        joint_cond,
        use_cfg,
        cfg_batching,
        guidance_scale,
        true_cfg_scale,
    ) -> torch.Tensor:
        def _forward(cond: dict) -> torch.Tensor:
            cond = dict(cond)
            mask = cond.pop("wan_ti2v_reserved_frames_mask")
            patch_size = cond.pop("wan_ti2v_patch_size")
            _, ph, pw = patch_size
            t = timesteps_input.reshape(-1, 1, 1, 1).to(mask.dtype)
            timestep = (mask[:, :, ::ph, ::pw] * t).flatten(1)
            output = model(
                hidden_states=latents_input,
                timestep=timestep,
                return_dict=False,
                **cond,
            )[0]
            if output.shape != latents_input.shape:
                raise ValueError(
                    "Wan TI2V noise prediction must match the latent layout: "
                    f"output={tuple(output.shape)}, latents={tuple(latents_input.shape)}, "
                    f"timestep={tuple(timestep.shape)}"
                )
            return output

        if not use_cfg:
            return _forward(pos_cond)
        pos = _forward(pos_cond)
        neg = _forward(neg_cond)
        return self.cfg_combine(pos, neg, guidance_scale, true_cfg_scale=true_cfg_scale)

    def cfg_combine(self, noise_pred_pos, noise_pred_neg, guidance_scale, true_cfg_scale=None):
        scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        return noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)
