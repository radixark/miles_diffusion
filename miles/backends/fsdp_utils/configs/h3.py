"""MiniMax H3 family config: t2va video-only Flow-GRPO."""

from __future__ import annotations

from argparse import Namespace

import torch

from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config

AUDIO_IN_CHANNELS = 32


def _without_padding_tail(layout: dict, token_tags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Token tags and position ids without the padding tail [cu_seqlens[1], seq_len).

    The engine attends the tail as its own document; the diffusers DiT attends unmasked, so the tail must not reach it.
    """
    used = int(layout["cu_seqlens"][1])
    return token_tags[:used], layout["img_position_ids"][:used]


@register_train_pipeline_config("h3")
class H3TrainPipelineConfig(TrainPipelineConfig):
    """MiniMax H3 t2va video-only GRPO (audio branch frozen / deterministic in rollout)."""

    hf_ckpt_name_patterns = ("minimax-h3", "minimax_h3", "/h3")
    supports_cfg_training = False
    sde_timestep_divisor = 1000.0
    optimizer_state_allowed_missing = ["audio"]

    lora_target_modules = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ]

    @classmethod
    def validate_args(cls, args: Namespace) -> None:
        # H3's rollout DiT fuses Q/K/V and rewrites FFN layout. Those transforms
        # live in sglang-d's lora_merge IPC path; train-side merge / full-weight
        # sync would push dense names the engine drops. SFT (--train-only) has
        # no rollout engine and therefore no sync constraint.
        if not args.train_only and not (args.use_lora and args.lora_ipc_weight_sync):
            raise ValueError("H3 training requires --use-lora with --lora-ipc-weight-sync")

    @classmethod
    def apply_rollout_sampling_params(
        cls,
        args: Namespace,
        sampling_params: dict,
        extra_sampling_params: dict,
    ) -> None:
        extra_sampling_params.update(
            {
                # sgl-d accepts only task=t2va for rollout and short_edge=768 for any
                # H3 request, so neither is exposed as an argument.
                "task": "t2va",
                "conditions": [],
                "target": {
                    "short_edge": 768,
                    "aspect_ratio": str(args.diffusion_h3_aspect_ratio),
                    "duration_seconds": float(args.diffusion_h3_duration_seconds),
                },
                "audio_flow_shift": float(args.diffusion_audio_flow_shift),
            }
        )
        if args.diffusion_flow_shift is not None:
            extra_sampling_params["flow_shift"] = float(args.diffusion_flow_shift)
        # MiniMaxH3SamplingParams marks CFG/canvas fields init=False; canvas comes from target.
        extra_sampling_params.pop("guidance_scale_2", None)
        for key in (
            "guidance_scale",
            "guidance_scale_2",
            "true_cfg_scale",
            "negative_prompt",
            "width",
            "height",
            "num_frames",
            "fps",
        ):
            sampling_params.pop(key, None)

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None:
            return {}
        kwargs: dict = {}
        if cond.encoder_hidden_states:
            enc = torch.cat(cond.encoder_hidden_states).to(device)
            if enc.ndim == 2:
                enc = enc.unsqueeze(0)
            kwargs["encoder_hidden_states"] = enc
        if cond.h3_packed_layout is not None:
            kwargs["h3_packed_layout"] = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in cond.h3_packed_layout.items()
            }
        if cond.h3_token_tags is not None:
            kwargs["h3_token_tags"] = cond.h3_token_tags.to(device)
        return kwargs

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
        pad_to_len: int | None = None,
    ) -> dict:
        if len(per_sample_cond_kwargs) != 1:
            raise NotImplementedError("H3 GRPO currently requires micro-batch-size-sample=1")
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in per_sample_cond_kwargs[0].items()}

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
        del neg_cond, joint_cond, use_cfg, cfg_batching, guidance_scale, true_cfg_scale
        latents = latents_input["visual"]
        cond = dict(pos_cond or {})
        packed = cond.get("h3_packed_layout")
        token_tags = cond.get("h3_token_tags")
        encoder_hidden_states = cond.get("encoder_hidden_states")
        if packed is None or token_tags is None or encoder_hidden_states is None:
            raise ValueError("H3 train requires h3_packed_layout, h3_token_tags, encoder_hidden_states in pos_cond")

        device = latents.device
        dtype = latents.dtype

        # latents: [B, num_video_target_rows, width]
        bsz = latents.shape[0]
        if bsz != 1:
            raise NotImplementedError("H3 packed forward supports batch size 1 for now")

        layout = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in packed.items()}
        sigma = (timesteps_input["visual"].float() / float(self.sde_timestep_divisor)).view(-1)
        timestep = 1.0 - sigma
        token_tags, position_ids = _without_padding_tail(layout, torch.as_tensor(token_tags, device=device))
        width = latents.shape[-1]

        img_pos = layout["img_pos"].view(-1).long().to(device)
        audio_pos = layout["audio_pos"].view(-1).long().to(device)
        update_mask = layout["update_mask"].view(-1).bool().to(device)
        text_pos = layout["text_pos"].view(-1).long().to(device)

        # The transformer takes one row block per modality, each ordered like its
        # ``*_indices``, and scatters them into the packed buffer itself. Only the
        # target rows are replayed; conditioning rows stay zero, as does the audio
        # stream (H3 GRPO trains the video branch only).
        video_hidden = torch.zeros(1, int(img_pos.shape[0]), width, device=device, dtype=dtype)
        video_hidden[0, update_mask] = latents[0].to(dtype)
        audio_hidden = torch.zeros(1, int(audio_pos.shape[0]), AUDIO_IN_CHANNELS, device=device, dtype=dtype)

        out = model(
            hidden_states=video_hidden,
            audio_hidden_states=audio_hidden,
            encoder_hidden_states=encoder_hidden_states.to(dtype),
            timestep=timestep.to(dtype),
            timestep_indices=torch.zeros(token_tags.shape[0], device=device, dtype=torch.long),
            token_tags=token_tags.long(),
            position_ids=position_ids.to(device=device, dtype=torch.float32),
            video_indices=img_pos,
            audio_indices=audio_pos,
            text_indices=text_pos,
        )
        velocity = out[0] if isinstance(out, tuple) else out.sample
        # Rows follow video_indices; keep the target subset and return the
        # diffusers-compatible flow direction (negated H3 velocity).
        return {"visual": (-velocity[:, update_mask]).to(dtype)}

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("H3 distilled CFG into the checkpoint; the forward is unguided")
