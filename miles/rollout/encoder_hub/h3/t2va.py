"""MiniMax H3 offline t2va SFT encode.

Every encode recipe is imported from the sglang engine rather than ported, so
cached train pairs stay bit-compatible with the engine's rollout conditioning:

- packed layout: ``minimax_h3_packed_sequence``;
- x0 rows: ``minimax_h3_encode_reference_video_rows`` (fp32 VAE, seed-42
  posterior sample, mean/std normalization, [1,2,2] patchify -> [rows, 96]);
- audio x0 rows: the video's own soundtrack over its frames' interval (``common.encode_target_audio``);
- text ids: ``minimax_h3_text_only_ids`` (verbatim prompt, no special tokens).
"""

from __future__ import annotations

from argparse import Namespace

import torch

from miles.rollout.encoder_hub.h3 import common
from miles.rollout.encoder_hub.h3.common import H3_TEXT_HIDDEN_DIM
from miles.utils.types import Sample

H3_SHORT_EDGE = 768
H3_VAE_SPATIAL_STRIDE = 16
# The uncond the CFG distilled into H3 is guessed as a single space prompt.
H3_UNCOND_PROMPT = " "


# ---------------------------------------------------------------------------
# Encoder hub interface: the entry points sft_rollout calls
# ---------------------------------------------------------------------------


def validate_args(args: Namespace) -> None:
    common.validate_frame_grid_and_stride(args)
    height, width = int(args.diffusion_height), int(args.diffusion_width)
    if min(height, width) != H3_SHORT_EDGE:
        raise ValueError(f"H3 pins short_edge={H3_SHORT_EDGE}, got {height}x{width}")
    if height % 32 or width % 32:
        # /16 VAE latent grid then /2 patchify: both spatial dims must survive.
        raise ValueError(f"H3 canvas dims must be multiples of 32, got {height}x{width}")


def load_encoder(args: Namespace, device: torch.device) -> dict:
    return common.load_encoders(args, device)


@torch.no_grad()
def encode_sample(
    encoder: dict, sample: Sample, media_clip: dict, generator: torch.Generator, args: Namespace
) -> dict:
    if sample.conditions:
        raise ValueError("H3 t2va SFT takes no reference conditions")

    del generator  # the engine encode recipe pins its own VAE sample seed (42)
    target_audio_rows = common.encode_target_audio(encoder, sample, media_clip)
    rows, latent_t, latent_h, latent_w = common.encode_target_video(encoder, media_clip)
    hidden, packed = _encode_prompt_layout(
        encoder,
        sample.prompt,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=target_audio_rows.shape[0] // 2,
    )
    return common.train_pair(
        sample,
        {"visual": rows, "audio": target_audio_rows},
        hidden,
        packed,
        packed["token_tags"],
        h3_target_has_soundtrack=media_clip["audio_sample_rate"] is not None,
    )


@torch.no_grad()
def encode_uncond(encoder: dict, args: Namespace) -> dict:
    """The uncond cond kwargs every pair shares: ``H3_UNCOND_PROMPT`` packed on the run's fixed target grid.

    ``read_media_clip`` admits only targets of exactly this frame count and canvas, so no sample is needed.
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
        minimax_h3_audio_latent_t,
        minimax_h3_video_latent_t,
    )

    num_frames = int(args.diffusion_output_num_frames)
    hidden, packed = _encode_prompt_layout(
        encoder,
        H3_UNCOND_PROMPT,
        latent_t=minimax_h3_video_latent_t(num_frames),
        latent_h=int(args.diffusion_height) // H3_VAE_SPATIAL_STRIDE,
        latent_w=int(args.diffusion_width) // H3_VAE_SPATIAL_STRIDE,
        audio_t=minimax_h3_audio_latent_t(num_frames / common.H3_FPS),
    )
    return {
        "encoder_hidden_states": hidden.to(dtype=torch.bfloat16, device="cpu"),
        "h3_packed_layout": packed,
        "h3_token_tags": packed["token_tags"],
    }


def _encode_prompt_layout(
    encoder: dict, prompt: str, *, latent_t: int, latent_h: int, latent_w: int, audio_t: int
) -> tuple[torch.Tensor, dict]:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
        minimax_h3_text_only_ids,
    )

    device = encoder["device"]
    text_ids = minimax_h3_text_only_ids(encoder["tokenizer"], prompt).to(device)
    hidden = encoder["text_encoder"](
        input_ids=text_ids[None],
        attention_mask=torch.ones_like(text_ids)[None],
        use_cache=False,
    ).last_hidden_state
    if list(hidden.shape) != [1, int(text_ids.shape[0]), H3_TEXT_HIDDEN_DIM]:
        raise ValueError(f"unexpected text hidden shape {list(hidden.shape)}")

    packed = minimax_h3_packed_sequence(
        text_len=int(text_ids.shape[0]),
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        include_keyframe_cond=False,
    )
    return hidden, packed
