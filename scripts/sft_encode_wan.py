"""Pre-encode a (video, prompt) jsonl dataset into Wan SFT train pairs.

Each output ``{index:08d}.pt`` holds ``{"latent": [C,T,H,W] fp16 (VAE-normalized),
"cond_kwargs": {"encoder_hidden_states": [1, 512, D] bf16}, "prompt": str}``,
the pair schema consumed by miles.backends.fsdp_utils.loss_hub.sft.
"""

import argparse
import json
from pathlib import Path

import ray
import torch


def read_video_clip(path: str, *, height: int, width: int, num_frames: int, frame_stride: int) -> torch.Tensor:
    import torchvision

    frames, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")
    span = (num_frames - 1) * frame_stride + 1
    if frames.shape[0] < span:
        raise ValueError(f"{path} has {frames.shape[0]} frames, need {span}")
    start = (frames.shape[0] - span) // 2
    frames = frames[start : start + span : frame_stride].float() / 127.5 - 1.0

    scale = max(height / frames.shape[2], width / frames.shape[3])
    new_h = max(height, round(frames.shape[2] * scale))
    new_w = max(width, round(frames.shape[3] * scale))
    frames = torch.nn.functional.interpolate(frames, size=(new_h, new_w), mode="bilinear", antialias=True)
    top = (new_h - height) // 2
    left = (new_w - width) // 2
    return frames[:, :, top : top + height, left : left + width].permute(1, 0, 2, 3)


@ray.remote(num_gpus=1)
class WanEncodeActor:
    def __init__(self, checkpoint: str):
        from diffusers import AutoencoderKLWan
        from transformers import AutoTokenizer, UMT5EncoderModel

        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            checkpoint, subfolder="text_encoder", torch_dtype=torch.bfloat16
        ).to(self.device)
        self.vae = AutoencoderKLWan.from_pretrained(checkpoint, subfolder="vae", torch_dtype=torch.float32).to(
            self.device
        )
        view = (1, self.vae.config.z_dim, 1, 1, 1)
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(view).to(self.device)
        self.latents_std = torch.tensor(self.vae.config.latents_std).view(view).to(self.device)

    @torch.no_grad()
    def encode(
        self, items: list[dict], output_dir: str, height: int, width: int, num_frames: int, frame_stride: int
    ) -> int:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        done = 0
        for item in items:
            out_path = Path(output_dir) / f"{item['index']:08d}.pt"
            if out_path.exists():
                continue
            video = read_video_clip(
                item["video"], height=height, width=width, num_frames=num_frames, frame_stride=frame_stride
            )
            latent = self.vae.encode(video.unsqueeze(0).to(self.device)).latent_dist.sample()
            latent = (latent - self.latents_mean) / self.latents_std

            inputs = self.tokenizer(
                [prompt_clean(item["prompt"])],
                padding="max_length",
                max_length=512,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            embeds = self.text_encoder(
                inputs.input_ids.to(self.device), inputs.attention_mask.to(self.device)
            ).last_hidden_state
            embeds[:, int(inputs.attention_mask[0].sum()) :] = 0

            torch.save(
                {
                    "latent": latent[0].to(torch.float16).cpu(),
                    "cond_kwargs": {"encoder_hidden_states": embeds.to(torch.bfloat16).cpu()},
                    "prompt": item["prompt"],
                },
                out_path,
            )
            done += 1
        return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--data-path", required=True, help="jsonl with one {video, prompt} object per line")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-key", default="video")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    if (args.num_frames - 1) % 4 != 0:
        parser.error("--num-frames must be 4k+1 for the Wan VAE temporal stride")

    items = []
    with open(args.data_path) as f:
        for index, line in enumerate(f):
            row = json.loads(line)
            items.append({"index": index, "video": row[args.video_key], "prompt": row[args.prompt_key]})

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ray.init()
    actors = [WanEncodeActor.remote(args.hf_checkpoint) for _ in range(args.num_gpus)]
    done = ray.get(
        [
            actor.encode.remote(
                items[i :: args.num_gpus], args.output_dir, args.height, args.width, args.num_frames, args.frame_stride
            )
            for i, actor in enumerate(actors)
        ]
    )
    print(f"encoded {sum(done)} new samples into {args.output_dir} ({len(items)} total)")


if __name__ == "__main__":
    main()
