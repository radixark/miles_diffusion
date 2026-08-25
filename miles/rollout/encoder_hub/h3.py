"""MiniMax H3 frozen encoders (Qwen3-VL-32B layer-50 + video VAE) for offline t2va SFT.

Every encode recipe is imported from the sglang engine rather than ported, so
cached train pairs stay bit-compatible with the engine's rollout conditioning:

- packed layout: ``minimax_h3_packed_sequence``;
- x0 rows: ``minimax_h3_encode_reference_video_rows`` (fp32 VAE, seed-42
  posterior sample, mean/std normalization, [1,2,2] patchify -> [rows, 96]);
- text ids: ``minimax_h3_text_only_ids`` (verbatim prompt, no special tokens).
"""

from __future__ import annotations

import json
from argparse import Namespace

import torch


H3_FPS = 24.0
H3_SHORT_EDGE = 768
H3_TEXT_HIDDEN_DIM = 5120


def validate_args(args: Namespace) -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
        minimax_h3_align_frame_count,
    )

    num_frames = int(args.diffusion_output_num_frames)
    if minimax_h3_align_frame_count(num_frames) != num_frames:
        raise ValueError(f"--diffusion-output-num-frames must sit on H3's 17n+5 grid, got {num_frames}")
    height, width = int(args.diffusion_height), int(args.diffusion_width)
    if min(height, width) != H3_SHORT_EDGE:
        raise ValueError(f"H3 pins short_edge={H3_SHORT_EDGE}, got {height}x{width}")
    if height % 32 or width % 32:
        # /16 VAE latent grid then /2 patchify: both spatial dims must survive.
        raise ValueError(f"H3 canvas dims must be multiples of 32, got {height}x{width}")
    if int(args.sft_frame_stride) != 1:
        # A stride would make audio_t (frame_count / 24) describe unkept pixels.
        raise ValueError(f"H3 encode requires --sft-frame-stride 1, got {args.sft_frame_stride}")


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

    config = MiniMaxH3VideoVAEConfig(arch_config=arch)
    config.post_init()
    vae = MiniMaxH3VideoVAE(config)

    state = load_file(f"{vae_dir}/source/model.safetensors")
    missing, unexpected = vae.load_state_dict(state, strict=False)
    # Decoder weights are unused; every encoder-side weight must load.
    missing_encoder = [k for k in missing if not k.startswith("decoder.")]
    if missing_encoder or unexpected:
        raise ValueError(f"H3 video VAE load mismatch: missing={missing_encoder[:5]} unexpected={unexpected[:5]}")
    return vae.to(device=device, dtype=torch.float32).eval(), arch


H3_QWEN3VL_SELECTED_LM_LAYER = 50  # MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER


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


def load_encoder(args: Namespace, device: torch.device) -> dict:
    import os

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    ckpt = args.sft_encoder_checkpoint
    if os.path.isdir(ckpt):
        ckpt_dir = ckpt
    else:
        ckpt_dir = snapshot_download(
            ckpt,
            allow_patterns=["FL2VA/video_vae/*", "text_encoder/*", "tokenizer/*"],
        )
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


@torch.no_grad()
def encode_sample(encoder: dict, media_clip: dict, prompt: str, generator: torch.Generator) -> dict:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
        minimax_h3_text_only_ids,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
        minimax_h3_encode_reference_video_rows,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
        minimax_h3_audio_latent_t,
        minimax_h3_video_latent_t,
    )

    del generator  # the engine encode recipe pins its own VAE sample seed (42)
    device = encoder["device"]

    # audio_t below is frame_count / H3_FPS: any other fps packs a wrong duration.
    if media_clip["fps"] is not None and abs(media_clip["fps"] - H3_FPS) > 0.01:
        raise ValueError(f'H3 encode requires {H3_FPS:g} fps clips, got {media_clip["fps"]:g}')

    # Engine input is uint8 frames: invert the reader's [-1, 1] normalization (exact roundtrip).
    frames = ((media_clip["video"].permute(1, 2, 3, 0) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()

    rows, latent_t, latent_h, latent_w = minimax_h3_encode_reference_video_rows(
        encoder["vae"], frames, encoder["vae_arch"]
    )
    expected_latent_t = minimax_h3_video_latent_t(frames.shape[0])
    if latent_t != expected_latent_t:
        raise ValueError(f"VAE produced latent_t={latent_t}, engine geometry expects {expected_latent_t}")

    text_ids = minimax_h3_text_only_ids(encoder["tokenizer"], prompt).to(device)
    hidden = encoder["text_encoder"](
        input_ids=text_ids[None],
        attention_mask=torch.ones_like(text_ids)[None],
        use_cache=False,
    ).last_hidden_state.to(torch.bfloat16)
    if list(hidden.shape) != [1, int(text_ids.shape[0]), H3_TEXT_HIDDEN_DIM]:
        raise ValueError(f"unexpected text hidden shape {list(hidden.shape)}")

    audio_t = minimax_h3_audio_latent_t(frames.shape[0] / H3_FPS)
    packed = minimax_h3_packed_sequence(
        text_len=int(text_ids.shape[0]),
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        include_keyframe_cond=False,
    )
    token_tags = packed["token_tags"]

    return {
        # fp16, not bf16: the recipe's use_fp16_latent already quantized these
        # values to fp16, so this cast is lossless and keeps engine bit-compat.
        "latent": rows.to(torch.float16).cpu(),
        "cond_kwargs": {
            "encoder_hidden_states": hidden.cpu(),
            "h3_packed_layout": packed,
            "h3_token_tags": token_tags,
        },
        "prompt": prompt,
    }
