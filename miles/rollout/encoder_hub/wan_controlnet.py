"""Encode aligned RGB targets and pose-video conditions with the frozen Wan VAE."""

import torch

from .wan2_2 import encode_sample as encode_target
from .wan2_2 import load_encoder, validate_args  # noqa: F401


@torch.no_grad()
def encode_sample(encoder, media_clip, prompt, generator):
    if "control_video" not in media_clip:
        raise ValueError("wan_controlnet requires an aligned control_video for every RGB target")
    if media_clip["video"].shape != media_clip["control_video"].shape:
        raise ValueError("RGB target and control video must have identical C,T,H,W")
    pair = encode_target(encoder, media_clip, prompt, generator)
    pixels = media_clip["control_video"].unsqueeze(0).to(encoder["device"], torch.float32) / 127.5 - 1.0
    # Deterministic conditions: posterior noise belongs to the target, not the pose.
    latent = encoder["vae"].encode(pixels).latent_dist.mode()
    latent = (latent - encoder["latents_mean"]) / encoder["latents_std"]
    pair["cond_kwargs"]["control_latents"] = latent.to(torch.bfloat16).cpu()
    return pair
