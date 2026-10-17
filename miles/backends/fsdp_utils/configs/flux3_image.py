"""FLUX.3 image training pipeline config."""

from __future__ import annotations

import torch

from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config


@register_train_pipeline_config("flux3_image")
class Flux3ImageTrainPipelineConfig(TrainPipelineConfig):
    tasks = ("t2i",)
    hf_ckpt_name_patterns = ("flux-3-image-klein-base",)
    model_backend_path = "miles.backends.fsdp_utils.model_backend.MilesModelBackend"
    model_package = "miles.backends.fsdp_utils.models.flux3_image"
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out"]

    @classmethod
    def validate_args(cls, args) -> None:
        # The engine serves one sample per request.
        if args.rollout_microgroup_size != 1:
            raise ValueError("flux3_image rollout serves one sample per request; use --rollout-microgroup-size 1")
        # Text length varies per prompt and the DiT has no padding mask; checked here, not at the first train step.
        if args.micro_batch_size != 1:
            raise ValueError(
                f"flux3_image trains one sample per forward; use --micro-batch-size 1, got {args.micro_batch_size}"
            )
        cfg_scale = args.diffusion_true_cfg_scale
        if cfg_scale is None:
            cfg_scale = args.diffusion_guidance_scale
        # Unpatched, the engine runs the whole CFG pair even at a scale of 1.
        if cfg_scale <= 1.0 and "flux3_image" not in args.rollout_patch_groups:
            raise ValueError("flux3_image CFG off (a CFG scale <= 1) needs --rollout-patch-group flux3_image")

    def process_timestep_as_input(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Trajectory timesteps are sigma * 1000; the DiT takes sigma in [0, 1].
        return timesteps / 1000.0

    def process_sigma_as_timesteps_input(self, sigmas: torch.Tensor) -> torch.Tensor:
        return sigmas

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None:
            return {}
        return {
            "encoder_hidden_states": cond.encoder_hidden_states[0].to(device).unsqueeze(0),
            "flux3_txt_4d_rope_pos_ids": cond.flux3_txt_4d_rope_pos_ids.to(device).unsqueeze(0),
            "flux3_img_4d_rope_pos_ids": cond.flux3_img_4d_rope_pos_ids.to(device).unsqueeze(0),
        }

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
        pad_to_len: int | None = None,
    ) -> dict:
        # Text length varies per prompt; the DiT takes one sample per forward.
        if len(per_sample_cond_kwargs) != 1:
            raise NotImplementedError("flux3_image trains with --micro-batch-size 1")
        return {k: v.to(device) for k, v in per_sample_cond_kwargs[0].items()}

    def compute_noise_pred(
        self,
        *,
        model: torch.nn.Module,
        latents_input: dict[str, torch.Tensor],
        timesteps_input: dict[str, torch.Tensor],
        pos_cond: dict | None,
        neg_cond: dict | None,
        joint_cond: dict | None,
        use_cfg: bool,
        cfg_batching: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
    ) -> dict[str, torch.Tensor]:
        latents = latents_input["visual"]
        sigma = timesteps_input["visual"]

        def _forward(cond: dict) -> torch.Tensor:
            return model(image=latents, sigma=sigma, **cond)

        if not use_cfg:
            return {"visual": _forward(pos_cond)}
        # Two batch-1 forwards; the engine packs the pair into one varlen forward, which matches per sample.
        return {
            "visual": self.cfg_combine(
                _forward(pos_cond), _forward(neg_cond), guidance_scale, true_cfg_scale=true_cfg_scale
            )
        }

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        return noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)
