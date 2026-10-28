"""FLUX.3 Image SFT pairs from an sglang-diffusion encode-only server (--sft-encode-url)."""

from __future__ import annotations

import torch

from miles.utils.types import Sample


def validate_args(args) -> None:
    if args.sft_encode_url is None:
        raise ValueError("flux3_image SFT has no local encoder; pass --sft-encode-url")


def pair_from_encoded(encoded: dict, sample: Sample) -> dict:
    # The server returns the prompt branch only; a CFG pair means it is not encode-only.
    ((text, text_ids),) = encoded["flux3_text"]
    target = encoded["encode_target"]["visual"]
    image, image_ids = target["latents"], target["latent_ids"]
    return {
        "latent": {"visual": image.to(torch.bfloat16)},
        "cond_kwargs": {
            "encoder_hidden_states": text.unsqueeze(0),
            "flux3_txt_4d_rope_pos_ids": text_ids.unsqueeze(0),
            "flux3_img_4d_rope_pos_ids": image_ids.unsqueeze(0),
        },
        "prompt": sample.prompt,
    }
