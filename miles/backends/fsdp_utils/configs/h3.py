"""MiniMax H3: video-only Flow-GRPO and joint audio/video Ref2VA SFT."""

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


@register_train_pipeline_config("h3", tasks=("t2va", "ref2va"))
class H3TrainPipelineConfig(TrainPipelineConfig):
    """MiniMax H3: t2va video-only GRPO, and joint video/audio SFT of t2va or Ref2VA through one packed forward."""

    hf_ckpt_name_patterns = ("minimax-h3", "minimax_h3", "/h3")
    supports_cfg_training = False
    sde_timestep_divisor = 1000.0
    # No loss reaches the audio head when audio is unsupervised (GRPO, or SFT with --fsdp-supervised-streams visual),
    # so its LoRA may hold no optimizer state.
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
        # Only sglang-d's lora_merge IPC path maps trainer names onto the engine's fused Q/K/V and FFN layout.
        if not args.train_only and not (args.use_lora and args.lora_ipc_weight_sync):
            raise ValueError("H3 training requires --use-lora with --lora-ipc-weight-sync")
        if args.loss_type == "sft_loss" and args.micro_batch_size != 1:
            raise ValueError("H3 packed SFT requires --micro-batch-size 1")
        # The visual shift would mis-set the audio's corruption sigmas and AdaLN conditioning.
        if args.loss_type == "sft_loss" and "audio" not in (args.fsdp_flow_shift or {}):
            raise ValueError("H3 SFT requires --fsdp-flow-shift visual=...,audio=...")
        if args.diffusion_task == "ref2va":
            if args.loss_type != "sft_loss":
                raise ValueError("H3 rollout serves only the t2va task")
            if args.update_weight_target_modules != ["transformer_ref"]:
                raise ValueError("H3 Ref2VA SFT requires --update-weight-target-module transformer_ref")

    @classmethod
    def apply_rollout_sampling_params(
        cls,
        args: Namespace,
        sampling_params: dict,
        extra_sampling_params: dict,
    ) -> None:
        extra_sampling_params.update(
            {
                # sgl-d accepts only short_edge=768 for any H3 request, so it is not exposed as an argument.
                "task": args.diffusion_task,
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
        """Pack the target rows of each stream, plus any reference rows, and run one joint H3 forward.

        Each row group gets its own time through ``timestep_indices``; text keeps the target video time:

            row group         rows                 time                             when
            target video      update_mask          visual                           always
            target audio      audio_update_mask    audio                            latents_input has audio (SFT)
            reference video   ~update_mask         max(visual, reference visual)    pos_cond has references (Ref2VA)
            reference audio   ~audio_update_mask   max(audio, reference audio)      pos_cond has references (Ref2VA)

        RL feeds the visual stream alone, so its audio rows stay zero at the video time.
        """
        del neg_cond, joint_cond, use_cfg, cfg_batching, guidance_scale, true_cfg_scale
        cond = pos_cond
        visual = latents_input["visual"]
        device, dtype = visual.device, visual.dtype
        if visual.shape[0] != 1:
            raise NotImplementedError("H3 packed forward supports batch size 1 for now")

        layout = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in cond["h3_packed_layout"].items()}
        token_tags, position_ids = _without_padding_tail(layout, torch.as_tensor(cond["h3_token_tags"], device=device))
        visual_positions = layout["img_pos"].view(-1).long()
        audio_positions = layout["audio_pos"].view(-1).long()
        visual_mask = layout["update_mask"].view(-1).bool()
        # A t2va layout packs every audio row as a target and carries no audio_update_mask.
        audio_mask = (
            layout.get("audio_update_mask", torch.ones_like(audio_positions, dtype=torch.bool)).view(-1).bool()
        )

        # The transformer takes one row block per modality, each ordered like its ``*_indices``, and scatters them
        # into the packed buffer itself. H3 time runs from t=0 noise to t=1 clean.
        visual_hidden = torch.zeros(1, int(visual_positions.shape[0]), visual.shape[-1], device=device, dtype=dtype)
        visual_hidden[0, visual_mask] = visual[0].to(dtype)
        audio_hidden = torch.zeros(1, int(audio_positions.shape[0]), AUDIO_IN_CHANNELS, device=device, dtype=dtype)
        visual_time = 1.0 - (timesteps_input["visual"].float() / float(self.sde_timestep_divisor)).view(-1)
        times = [visual_time]
        timestep_indices = torch.zeros(token_tags.shape[0], device=device, dtype=torch.long)
        if "audio" in latents_input:
            audio_hidden[0, audio_mask] = latents_input["audio"][0].to(dtype)
            audio_time = 1.0 - (timesteps_input["audio"].float() / float(self.sde_timestep_divisor)).view(-1)
            timestep_indices[audio_positions[audio_mask]] = len(times)
            times.append(audio_time)
        if "h3_reference_visual" in cond:
            visual_hidden[0, ~visual_mask] = cond["h3_reference_visual"].to(device=device, dtype=dtype)
            audio_hidden[0, ~audio_mask] = cond["h3_reference_audio"].to(device=device, dtype=dtype)
            timestep_indices[visual_positions[~visual_mask]] = len(times)
            times.append(visual_time.clamp_min(cond["h3_reference_visual_timestep"]))
            timestep_indices[audio_positions[~audio_mask]] = len(times)
            times.append(audio_time.clamp_min(cond["h3_reference_audio_timestep"]))

        out = model(
            hidden_states=visual_hidden,
            audio_hidden_states=audio_hidden,
            encoder_hidden_states=cond["encoder_hidden_states"].to(device=device, dtype=dtype),
            timestep=torch.cat(times).to(dtype),
            timestep_indices=timestep_indices,
            token_tags=token_tags.long(),
            position_ids=position_ids.to(device=device, dtype=torch.float32),
            video_indices=visual_positions,
            audio_indices=audio_positions,
            text_indices=layout["text_pos"].view(-1).long(),
        )
        # H3 predicts clean - noise; the shared flow objective predicts noise - clean.
        visual_velocity = out[0] if isinstance(out, tuple) else out.sample
        predictions = {"visual": (-visual_velocity[:, visual_mask]).to(dtype)}
        # A silent target's audio rows are noised silence: they keep the packed sequence native but carry no loss.
        if "audio" in latents_input and cond["h3_target_has_soundtrack"]:
            audio_velocity = out[1] if isinstance(out, tuple) else out.audio_sample
            predictions["audio"] = (-audio_velocity[:, audio_mask]).to(dtype)
        return predictions

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("H3 distilled CFG into the checkpoint; the forward is unguided")
