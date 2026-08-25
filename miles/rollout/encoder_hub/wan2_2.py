"""Wan2.2 frozen encoders (UMT5 text encoder + VAE) for offline SFT encoding."""

from __future__ import annotations

import torch


def validate_args(args) -> None:
    if (args.diffusion_output_num_frames - 1) % 4 != 0:
        raise ValueError("--diffusion-output-num-frames must be 4k+1 for the Wan VAE temporal stride")


def load_encoder(args, device: torch.device) -> dict:
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel

    ckpt = args.sft_encoder_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(ckpt, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(ckpt, subfolder="text_encoder", torch_dtype=torch.float32).to(
        device
    )
    vae = AutoencoderKLWan.from_pretrained(ckpt, subfolder="vae", torch_dtype=torch.float32).to(device)
    view = (1, vae.config.z_dim, 1, 1, 1)
    return {
        "device": device,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "latents_mean": torch.tensor(vae.config.latents_mean).view(view).to(device),
        "latents_std": torch.tensor(vae.config.latents_std).view(view).to(device),
    }


@torch.no_grad()
def encode_sample(encoder: dict, media_clip: dict, prompt: str, generator: torch.Generator) -> dict:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    device = encoder["device"]
    pixels = media_clip["video"].unsqueeze(0).to(device, torch.float32) / 127.5 - 1.0
    latent = encoder["vae"].encode(pixels).latent_dist.sample(generator)
    latent = (latent - encoder["latents_mean"]) / encoder["latents_std"]

    inputs = encoder["tokenizer"](
        [prompt_clean(prompt)],
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    embeds = encoder["text_encoder"](inputs.input_ids.to(device), inputs.attention_mask.to(device)).last_hidden_state
    embeds[:, int(inputs.attention_mask[0].sum()) :] = 0

    return {
        "latent": latent[0].to(torch.bfloat16).cpu(),
        "cond_kwargs": {"encoder_hidden_states": embeds.to(torch.bfloat16).cpu()},
        "prompt": prompt,
    }
