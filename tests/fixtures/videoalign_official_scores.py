"""Run the OFFICIAL VideoReward (old transformers/peft, in the overlay) on frame lists.

Used by the videoalign alignment test. Run in a subprocess with the overlay first on PYTHONPATH so
the official repo + its pinned transformers/peft shadow the main environment; torch and CUDA come
from the main env. Writes the official scores and input tensors (hashes) to --output for the test
to compare against the port.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

OFFICIAL_REVISION = "219ab9db64c045e5181a2202d11f686439351292"

SAMPLES = [
    ((256, 256, 8), 17, "a red ball bouncing on a wooden floor"),
    ((512, 512, 4), 53, "a slow camera pan across a mountain lake at sunrise"),
    ((768, 512, 8), 91, "a dog runs across a grassy field as the camera follows it"),
]


def make_video(width, height, frames, offset):
    video = []
    for t in range(frames):
        y, x = np.indices((height, width))
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = (3 * x + y + offset + 17 * t) % 256
        rgb[..., 1] = (x + 5 * y + 2 * offset + 13 * t) % 256
        rgb[..., 2] = (7 * x + 11 * y + 3 * offset + 7 * t) % 256
        video.append(Image.fromarray(rgb))
    return video


def _sha256(tensor):
    import hashlib

    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.repo_dir)

    import torch

    revision = subprocess.check_output(["git", "-C", args.repo_dir, "rev-parse", "HEAD"], text=True).strip()
    if revision != OFFICIAL_REVISION:
        raise ValueError(f"Expected VideoAlign revision {OFFICIAL_REVISION}, got {revision}")

    import inference
    from prompt_template import build_prompt
    from vision_process import process_vision_info

    original_create = inference.create_model_and_processor

    def create_local_model(*, model_config, peft_lora_config, training_args):
        model_config.model_name_or_path = args.model_path
        training_args.disable_flash_attn2 = True
        return original_create(
            model_config=model_config, peft_lora_config=peft_lora_config, training_args=training_args
        )

    inference.create_model_and_processor = create_local_model
    inferencer = inference.VideoVLMRewardInference(
        load_from_pretrained=args.checkpoint_dir, device="cuda", dtype=torch.bfloat16
    )
    processor = inferencer.processor
    model = inferencer.model
    data_config = inferencer.data_config

    samples = []
    chats = []
    for (w, h, f), offset, prompt in SAMPLES:
        frames = make_video(w, h, f, offset)
        chat = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frames,
                            "max_pixels": data_config.max_frame_pixels,
                            "fps": data_config.fps,
                            "sample_type": data_config.sample_type,
                        },
                        {
                            "type": "text",
                            "text": build_prompt(prompt, data_config.eval_dim, data_config.prompt_template_type),
                        },
                    ],
                }
            ]
        ]
        chats.extend(chat)
        image_inputs, video_inputs = process_vision_info(chat)
        texts = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        batch = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(return_dict=True, **batch)["logits"]  # [1, 3]
        samples.append(
            {
                "width": w,
                "height": h,
                "frames": f,
                "offset": offset,
                "prompt": prompt,
                "VQ": float(logits[0][0]),
                "MQ": float(logits[0][1]),
                "TA": float(logits[0][2]),
                "input_ids": batch["input_ids"][0].tolist(),
                "video_grid_thw": batch["video_grid_thw"][0].tolist(),
                "pixel_values_videos_shape": list(batch["pixel_values_videos"].shape),
                "pixel_values_videos_sha256": _sha256(batch["pixel_values_videos"]),
            }
        )

    image_inputs, video_inputs = process_vision_info(chats)
    texts = processor.apply_chat_template(chats, tokenize=False, add_generation_prompt=True)
    batch = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}
    with torch.no_grad():
        batch_scores = model(return_dict=True, **batch)["logits"].float().cpu().tolist()
    Path(args.output).write_text(
        json.dumps({"revision": revision, "samples": samples, "batch_scores": batch_scores}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
