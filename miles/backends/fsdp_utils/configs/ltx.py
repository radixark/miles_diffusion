"""LTX-2 family config: train pipeline config + family validation."""

from __future__ import annotations

from argparse import Namespace

import torch

from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config


@register_train_pipeline_config("ltx")
class LTXTrainPipelineConfig(TrainPipelineConfig):
    """LTX-2.3 GRPO: unguided velocity forward over ltx_core, video alone or video with audio."""

    tasks = ("t2v",)
    supports_cfg_training = False
    hf_ckpt_name_patterns = ("ltx",)
    model_backend_path = "miles.backends.fsdp_utils.model_backend.MilesModelBackend"
    model_package = "miles.backends.fsdp_utils.models.ltx"
    trainable_extra_streams = ("audio",)
    # A video-only run gives the audio branch no gradient, so it has no optimizer state.
    optimizer_state_allowed_missing = ["audio"]
    # forward_velocity anchors element-wise math on latents.dtype; rollout runs it bf16, so cast at the boundary.
    input_dtype_policy = {"latents": "default", "cond": "default", "timestep": None}

    @classmethod
    def validate_args(cls, args: Namespace) -> None:
        # The engine's audio-video cross-attention must match the forward: off for video alone, on with audio.
        trains_audio = "audio" in args.rollout_extra_streams
        if trains_audio and ("ltx_av" not in args.rollout_patch_groups or "ltx" in args.rollout_patch_groups):
            raise ValueError(
                "LTX --rollout-extra-streams audio trains the joint audio-video forward; set --rollout-patch-group "
                f"ltx_av (not ltx), got {args.rollout_patch_groups}"
            )
        if not trains_audio and "ltx_av" in args.rollout_patch_groups:
            raise ValueError(
                "--rollout-patch-group ltx_av leaves the engine's audio-video cross-attention on, but the forward "
                "trains video alone; add --rollout-extra-streams audio or use --rollout-patch-group ltx"
            )
        # The engine's audio rollout is validated without sequence parallelism. Left unset, the engine's SP degree is
        # the GPUs of an engine that TP and CFG parallelism leave over.
        tp_and_cfg = (args.sglang_tp_size or 1) * (2 if args.sglang_enable_cfg_parallel else 1)
        sp_degree = args.sglang_sp_degree or args.rollout_num_gpus_per_engine // tp_and_cfg
        if trains_audio and sp_degree > 1:
            raise ValueError(
                "LTX audio rollout is not validated under engine sequence parallelism: give every engine GPU to "
                f"--sglang-tp-size, got an SP degree of {sp_degree}"
            )

    def configure(self, args: Namespace) -> None:
        self._height = args.diffusion_height
        self._width = args.diffusion_width
        self._num_frames = args.diffusion_output_num_frames
        self._fps = args.diffusion_fps

    lora_target_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "net.0.proj",
        "net.2",
    ]

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
        if cond.encoder_attention_mask:
            mask = torch.cat(cond.encoder_attention_mask).to(device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            kwargs["context_mask"] = mask
        if cond.audio_encoder_attention_mask:
            audio_mask = torch.cat(cond.audio_encoder_attention_mask).to(device)
            if audio_mask.ndim == 1:
                audio_mask = audio_mask.unsqueeze(0)
            kwargs["audio_context_mask"] = audio_mask
        return kwargs

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
        pad_to_len: int | None = None,
    ) -> dict:
        # Fixed-length embeds: naive concat (pad_to_len accepted and ignored, see base).
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
        # LTX trains unguided (supports_cfg_training=False): one forward, no negative branch.
        latents = latents_input["visual"]
        cond = dict(pos_cond or {})
        if "context" not in cond:
            raise ValueError("LTX train requires denoising_env.pos_cond_kwargs.encoder_hidden_states")
        if "positions" not in cond:
            from miles.backends.fsdp_utils.models.ltx.positions import prepare_video_positions

            batch_size, num_tokens, _ = latents.shape
            cond["positions"] = prepare_video_positions(
                batch_size=batch_size,
                num_tokens=num_tokens,
                height=self._height,
                width=self._width,
                num_frames=self._num_frames,
                fps=self._fps,
                device=latents.device,
                dtype=latents.dtype,
            )
        return self.forward_velocity(model, latents_input, timesteps_input, cond)

    def forward_velocity(
        self,
        model: torch.nn.Module,
        latents_input: dict[str, torch.Tensor],
        timesteps_input: dict[str, torch.Tensor],
        cond: dict,
    ) -> dict[str, torch.Tensor]:
        """Noise prediction of the video stream, and of the audio stream when the input carries it."""
        video_latents = latents_input["visual"]
        video_modality = self._modality(
            video_latents, timesteps_input["visual"], cond["positions"].to(video_latents.dtype), cond["context"]
        )
        audio_modality = None
        if "audio" in latents_input:
            from miles.backends.fsdp_utils.models.ltx.positions import prepare_audio_positions

            audio_latents = latents_input["audio"]
            audio_positions = prepare_audio_positions(
                batch_size=audio_latents.shape[0], num_frames=audio_latents.shape[1], device=audio_latents.device
            )
            audio_modality = self._modality(
                audio_latents, timesteps_input["audio"], audio_positions, cond["audio_context"]
            )
        # Compute dtype comes from the trainer's ambient autocast around compute_noise_pred.
        video_velocity, audio_velocity = model(video=video_modality, audio=audio_modality, perturbations=None)

        noise_pred = {"visual": _velocity_to_noise_pred(video_modality, video_velocity)}
        if audio_modality is not None:
            noise_pred["audio"] = _velocity_to_noise_pred(audio_modality, audio_velocity)
        return noise_pred

    def _modality(
        self, latents: torch.Tensor, timesteps: torch.Tensor, positions: torch.Tensor, context: torch.Tensor
    ):
        """One stream's ltx_core ``Modality``; its ``sigma`` is the stream's sigma in [0, 1]."""
        from ltx_core.model.transformer.modality import Modality

        dtype = latents.dtype
        B = latents.shape[0]
        # Trajectory timesteps are σ×1000; ltx_core AdaLN expects σ∈[0,1] and
        # multiplies by timestep_scale_multiplier (1000) internally.
        sigma_unit = timesteps.to(dtype) / float(self.num_train_timesteps)
        return Modality(
            enabled=True,
            latent=latents,
            sigma=sigma_unit.reshape(B),
            timesteps=sigma_unit.view(B, 1).to(dtype),
            positions=positions,
            context=context.to(dtype),
            context_mask=None,
        )

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


def _velocity_to_noise_pred(modality, velocity: torch.Tensor) -> torch.Tensor:
    """The noise prediction of one stream from the velocity the model returned for its ``modality``."""
    from ltx_core.utils import to_denoised

    latents, sigma_unit = modality.latent, modality.sigma
    dtype = latents.dtype
    B = latents.shape[0]
    # Keep the original fp32 denoised reconstruction path: although this is
    # algebraically an identity for T2V, strict e2e metrics depend on its rounding.
    x0_pred = to_denoised(latents, velocity, sigma_unit.view(B, 1).to(dtype).unsqueeze(-1)).float()
    sigma_safe = torch.clamp(sigma_unit, min=1e-8).view(B, 1, 1)
    return ((latents.float() - x0_pred) / sigma_safe).to(dtype)
