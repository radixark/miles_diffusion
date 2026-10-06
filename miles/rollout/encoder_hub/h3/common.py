"""MiniMax H3 encode pieces every task shares: the frozen encoders, the target clip's rows and the train pair.

The recipes are imported from the SGLang engine rather than ported, so cached train pairs match the engine's rows.
"""

from __future__ import annotations

import json
from argparse import Namespace

import torch

from miles.utils.types import Sample

H3_FPS = 24.0
H3_TEXT_HIDDEN_DIM = 5120
H3_QWEN3VL_SELECTED_LM_LAYER = 50  # MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
# One 800-sample hop of the audio VAE at 32 kHz, i.e. one 40 Hz audio latent frame.
AUDIO_DURATION_TOLERANCE_SECONDS = 800 / 32000


def validate_frame_grid_and_stride(args: Namespace) -> None:
    """The temporal constraints every H3 encode shares."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
        minimax_h3_align_frame_count,
    )

    num_frames = int(args.diffusion_output_num_frames)
    if minimax_h3_align_frame_count(num_frames) != num_frames:
        raise ValueError(f"--diffusion-output-num-frames must sit on H3's 17n+5 grid, got {num_frames}")
    if int(args.sft_frame_stride) != 1:
        # A stride would make audio_t (frame_count / 24) describe unkept pixels.
        raise ValueError(f"H3 encode requires --sft-frame-stride 1, got {args.sft_frame_stride}")


# ---------------------------------------------------------------------------
# Frozen encoders
# ---------------------------------------------------------------------------


def _load_video_vae(ckpt_dir: str, device: torch.device):
    from safetensors.torch import load_file
    from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
        MiniMaxH3VideoVAEArchConfig,
        MiniMaxH3VideoVAEConfig,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE

    # source/ holds the native-key weights the engine loads; the repo-root vae/
    # is a diffusers re-export with incompatible key names.
    vae_dir = f"{ckpt_dir}/FL2VA/video_vae"
    with open(f"{vae_dir}/config.json") as f:
        vae_json = json.load(f)
    arch = MiniMaxH3VideoVAEArchConfig(
        latents_mean=vae_json["latents_mean"],
        latents_std=vae_json["latents_std"],
    )
    from sglang.multimodal_gen.runtime.server_args.server_args import (
        ServerArgs,
        get_global_server_args,
        set_global_server_args,
    )

    # VAE construction reads the engine's global server args; this plain encode
    # actor is no engine, so set single-GPU defaults once.
    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args(ServerArgs(model_path="MiniMaxAI/MiniMax-H3"))

    # Encode actors are independent single-device workers; no SGLang replica
    # process group exists for the VAE or shared reference-frame collectives.
    config = MiniMaxH3VideoVAEConfig(arch_config=arch, use_parallel_decode=False, use_parallel_tiling=False)
    config.post_init()
    vae = MiniMaxH3VideoVAE(config)

    state = load_file(f"{vae_dir}/source/model.safetensors")
    missing, unexpected = vae.load_state_dict(state, strict=False)
    # Decoder weights are unused; every encoder-side weight must load.
    missing_encoder = [k for k in missing if not k.startswith("decoder.")]
    if missing_encoder or unexpected:
        raise ValueError(f"H3 video VAE load mismatch: missing={missing_encoder[:5]} unexpected={unexpected[:5]}")
    return vae.to(device=device, dtype=torch.float32).eval(), arch


def load_audio_vae(ckpt_dir: str, device: torch.device):
    from safetensors.torch import load_file
    from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import (
        MiniMaxH3AudioVAEArchConfig,
        MiniMaxH3AudioVAEConfig,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3AudioVAE

    # Native keys; the repo-root audio_vae/ is a diffusers re-export with incompatible key names.
    vae_dir = f"{ckpt_dir}/FL2VA/audio_vae"
    with open(f"{vae_dir}/config.json") as f:
        vae_json = json.load(f)
    arch = MiniMaxH3AudioVAEArchConfig(latents_mean=vae_json["latents_mean"], latents_std=vae_json["latents_std"])
    config = MiniMaxH3AudioVAEConfig(arch_config=arch)
    config.post_init()
    vae = MiniMaxH3AudioVAE(config)
    state = load_file(f"{vae_dir}/{vae_json['source_safetensors_path']}")
    vae.load_state_dict(state, strict=True)
    return vae.to(device=device, dtype=torch.float32).eval(), arch


def _load_text_encoder(ckpt_dir: str, device: torch.device):
    import torch.nn as nn
    from transformers import AutoConfig, Qwen3VLModel

    # The checkpoint ships 64 layers; build the engine's 50-layer trim (the
    # rest, lm_head, and the final norm are unconsumed).
    config = AutoConfig.from_pretrained(f"{ckpt_dir}/text_encoder")
    config.text_config.num_hidden_layers = H3_QWEN3VL_SELECTED_LM_LAYER
    encoder = Qwen3VLModel.from_pretrained(f"{ckpt_dir}/text_encoder", config=config, dtype=torch.bfloat16)
    # H3 reads the unnormalized layer-50 output, as MiniMaxH3Qwen3VLEncoder does.
    encoder.language_model.norm = nn.Identity()
    return encoder.to(device).eval()


def checkpoint_dir(checkpoint: str, allow_patterns: list[str]) -> str:
    """A local checkpoint as is; an HF name downloads only the files matching allow_patterns."""
    import os

    from huggingface_hub import snapshot_download

    return checkpoint if os.path.isdir(checkpoint) else snapshot_download(checkpoint, allow_patterns=allow_patterns)


def load_video_and_text_encoders(args: Namespace, device: torch.device) -> dict:
    from transformers import AutoTokenizer

    ckpt_dir = checkpoint_dir(args.sft_encoder_checkpoint, ["FL2VA/video_vae/*", "text_encoder/*", "tokenizer/*"])
    tokenizer = AutoTokenizer.from_pretrained(f"{ckpt_dir}/tokenizer")
    vae, vae_arch = _load_video_vae(ckpt_dir, device)
    text_encoder = _load_text_encoder(ckpt_dir, device)
    return {
        "device": device,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "vae_arch": vae_arch,
    }


# ---------------------------------------------------------------------------
# Target: the clip's frames and soundtrack
# ---------------------------------------------------------------------------


def encode_target_video(encoder: dict, media_clip: dict) -> tuple[torch.Tensor, int, int, int]:
    """The target's x0 rows [rows, 96] and its latent (t, h, w) grid."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
        minimax_h3_encode_reference_video_rows,
    )

    frames = media_clip["video"].permute(1, 2, 3, 0).numpy()
    return minimax_h3_encode_reference_video_rows(encoder["vae"], frames, encoder["vae_arch"])


def target_duration_seconds(media_clip: dict) -> float:
    """The target length H3 generates: its frames at 24 fps, the cadence each task's encode_sample requires."""
    return media_clip["video"].shape[1] / H3_FPS


def encode_target_audio(encoder: dict, sample: Sample, media_clip: dict) -> torch.Tensor:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
        minimax_h3_encode_reference_audio_rows,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
        minimax_h3_audio_latent_t,
    )

    if "audio" in sample.target:
        raise ValueError("H3 trains the target video's own soundtrack; mux the target audio into the video")
    path = sample.target["visual"]
    duration = target_duration_seconds(media_clip)
    audio_t = minimax_h3_audio_latent_t(duration)
    # A silent target gets zero latent rows: the trainer noises them like any stream, so the packed sequence stays
    # native, but supervises only the video (see h3_target_has_soundtrack).
    if media_clip["audio_sample_rate"] is None:
        return torch.zeros(2 * audio_t, 32)
    # The soundtrack shares the frames' file timeline, so it is cropped from the first frame's time. The "audio"
    # chain decodes it at the native rate and resamples once onto the VAE's 32 kHz; the video_audio chain would pass
    # through 44.1 kHz first.
    payload = minimax_h3_encode_reference_audio_rows(
        encoder["audio_vae"],
        path,
        encoder["audio_vae_arch"],
        material_chain="audio",
        max_duration_seconds=duration,
        start_time_seconds=float(media_clip["frame_times_seconds"][0]),
        source_sample_rate=media_clip["audio_sample_rate"],
    )
    # The audio VAE pads to its 800-sample hop, so a soundtrack ending less than one hop before the last frame still
    # fills every latent frame; drop the one trailing padding token. Anything shorter is an error.
    if float(payload["duration_seconds"]) + AUDIO_DURATION_TOLERANCE_SECONDS < duration:
        raise ValueError(
            f"H3 target audio is shorter than the selected video interval: "
            f"{payload['duration_seconds']:.6f}s < {duration:.6f}s ({path})"
        )
    actual_t = int(payload["ref_audio_t"])
    rows = payload["rows"]
    if actual_t not in {audio_t, audio_t + 1}:
        raise ValueError(f"H3 target audio has {actual_t} latent frames, not {audio_t}")
    return rows.reshape(2, actual_t, 32)[:, :audio_t].reshape(2 * audio_t, 32).contiguous().cpu()


# ---------------------------------------------------------------------------
# Train pair
# ---------------------------------------------------------------------------


def train_pair(
    sample: Sample,
    target_rows: dict[str, torch.Tensor],
    text_hidden_states: torch.Tensor,
    packed_layout: dict,
    token_tags: torch.Tensor,
    **task_cond_kwargs,
) -> dict:
    """The cached pair: fp32 x0 rows per stream, plus the H3 config's compute_noise_pred inputs."""
    return {
        "latent": {stream: rows.to(torch.float16).cpu() for stream, rows in target_rows.items()},
        "cond_kwargs": {
            "encoder_hidden_states": text_hidden_states.to(dtype=torch.bfloat16, device="cpu"),
            "h3_packed_layout": packed_layout,
            "h3_token_tags": token_tags,
            **task_cond_kwargs,
        },
        "prompt": sample.prompt,
    }
