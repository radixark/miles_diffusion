"""Generate a pose-controlled video with the original Wan base plus an exported branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--controlnet-checkpoint", type=Path, required=True)
    parser.add_argument("--pose-video", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.width % 32 or args.height % 32 or args.frames < 1 or (args.frames - 1) % 4:
        parser.error("require dimensions divisible by 32 and a 4k+1 frame count")
    if args.output.exists() and not args.overwrite:
        parser.error("output exists; choose a new path or pass --overwrite")
    if args.output.suffix.lower() != ".mp4":
        parser.error("output must be an MP4 file")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import cv2
    import numpy as np
    import torch
    import diffusers
    import transformers
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
    from transformers import UMT5EncoderModel
    from miles.backends.fsdp_utils.models.wan_controlnet import WanPoseControlTransformer

    cap = cv2.VideoCapture(str(args.pose_video))
    if not cap.isOpened():
        raise ValueError(f"cannot open pose video: {args.pose_video}")
    if abs(cap.get(cv2.CAP_PROP_FPS) - args.fps) > 1e-3:
        cap.release()
        raise ValueError("pose FPS differs from requested output; preprocess explicitly")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != args.frames or any(frame.shape != (args.height, args.width, 3) for frame in frames):
        raise ValueError("pose frame count/size must exactly match output; no implicit resampling or cropping")

    torch.set_grad_enabled(False)
    vae = AutoencoderKLWan.from_pretrained(args.base_model, subfolder="vae", torch_dtype=torch.float32)
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.base_model, subfolder="text_encoder", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(args.base_model, vae=vae, text_encoder=text_encoder, torch_dtype=torch.bfloat16)
    # Keep model_index.json's expand_timesteps setting and the official UniPC flow scheduler.
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift, use_flow_sigmas=True
    )
    wrapper = WanPoseControlTransformer.from_controlnet(pipe.transformer, args.controlnet_checkpoint)
    pipe.transformer = wrapper.eval()
    pipe.to(args.device)
    pipe.vae.enable_tiling() if args.vae_tiling else pipe.vae.disable_tiling()
    pixels = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
    pixels = pixels.to(device=args.device, dtype=torch.float32) / 127.5 - 1
    pose_latents = pipe.vae.encode(pixels).latent_dist.mode()
    shape = (1, -1, 1, 1, 1)
    mean = torch.tensor(pipe.vae.config.latents_mean, device=args.device).view(shape)
    std = torch.tensor(pipe.vae.config.latents_std, device=args.device).view(shape)
    pose_latents = ((pose_latents - mean) / std).to(wrapper.dtype)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    with wrapper.control_condition(pose_latents, control_scale=args.control_scale):
        generated = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="np",
        ).frames[0]
    if generated.shape != (args.frames, args.height, args.width, 3):
        raise ValueError(f"unexpected generated shape: {generated.shape}")
    generated = np.round(np.clip(generated, 0, 1) * 255).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{args.width}x{args.height}",
            "-r",
            str(args.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        input=generated.tobytes(),
        check=True,
    )
    temporary.replace(args.output)
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update(
        scheduler=type(pipe.scheduler).__name__,
        scheduler_config=dict(pipe.scheduler.config),
        torch_version=torch.__version__,
        diffusers_version=diffusers.__version__,
        transformers_version=transformers.__version__,
        expand_timesteps=bool(pipe.config.expand_timesteps),
        conditioning="normalized Wan VAE posterior mode",
        controlnet_config=json.loads((args.controlnet_checkpoint / "controlnet_config.json").read_text()),
    )
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
