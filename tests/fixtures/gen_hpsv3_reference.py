"""Regenerate ``hpsv3_reference.json`` with the official HPSv3 inferencer.

The official package pins ``transformers==4.45.2``, which cannot share an environment with
miles, so the reference is recorded once in a separate venv and checked in:

    python -m venv /tmp/hpsv3ref && source /tmp/hpsv3ref/bin/activate
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
    pip install transformers==4.45.2 tokenizers==0.20.3 accelerate==1.8.0 peft==0.10.0 trl==0.8.6 \
        datasets==3.6.0 fire matplotlib pandas omegaconf safetensors tensorboard qwen-vl-utils==0.0.11 rich prettytable
    pip install --no-deps hpsv3==1.0.0     # its declared deps would pull deepspeed
    python tests/fixtures/gen_hpsv3_reference.py [--checkpoint-path HPSv3.safetensors]

Without flash-attn installed the official model runs on sdpa, the same backend the alignment
test pins. Module level imports only numpy and PIL so the test can reuse ``make_image``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# (width, height): square, landscape and portrait sizes hit different smart_resize branches
SAMPLES = [
    ((512, 512), 17, "a colorful geometric landscape"),
    ((1024, 768), 53, "an abstract portrait with vivid lines"),
    ((768, 1344), 91, 'a neon sign that reads "HPS" above a rainy street at night'),
    (
        (1024, 1024),
        137,
        "a highly detailed oil painting of an old lighthouse on a cliff during a thunderstorm, "
        "crashing waves, dramatic lighting, seagulls circling, warm light from the windows, "
        "intricate brush strokes, trending on artstation, masterpiece",
    ),
]


def make_image(width: int, height: int, offset: int) -> Image.Image:
    y, x = np.indices((height, width))
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = (3 * x + y + offset) % 256
    rgb[..., 1] = (x + 5 * y + 2 * offset) % 256
    rgb[..., 2] = (7 * x + 11 * y + 3 * offset) % 256
    return Image.fromarray(rgb)


def _sha256(tensor) -> str:
    import hashlib

    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    import torch
    import transformers
    from hpsv3 import HPSv3RewardInferencer

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output", default=str(Path(__file__).with_name("hpsv3_reference.json")))
    parser.add_argument("--debug-output", default=None, help="Optional .pt with full inputs and reward-token states")
    args = parser.parse_args()

    inferencer = HPSv3RewardInferencer(checkpoint_path=args.checkpoint_path, device="cuda")
    model = inferencer.model
    images = [make_image(w, h, offset) for (w, h), offset, _ in SAMPLES]
    prompts = [prompt for _, _, prompt in SAMPLES]

    captured = {}
    model.rm_head.register_forward_hook(lambda _m, inputs, output: captured.update(hidden=inputs[0], logits=output))

    # keywords: the PyPI 1.0.0 reward() takes (image_paths, prompts), the GitHub one (prompts, image_paths);
    # PyPI also drops its @torch.inference_mode()
    def reward(prompts, images):
        with torch.inference_mode():
            return inferencer.reward(prompts=prompts, image_paths=images)

    batch = inferencer.prepare_batch(image_paths=images, prompts=prompts)
    batched = reward(prompts, images)
    reward_token_id = model.special_token_ids[0]
    reward_hidden = captured["hidden"][batch["input_ids"] == reward_token_id]

    samples = []
    for i, (((width, height), offset, prompt), image) in enumerate(zip(SAMPLES, images, strict=True)):
        single = inferencer.prepare_batch(image_paths=[image], prompts=[prompt])
        single_logits = reward([prompt], [image])
        samples.append(
            {
                "width": width,
                "height": height,
                "offset": offset,
                "prompt": prompt,
                "input_ids": single["input_ids"][0].tolist(),
                "image_grid_thw": single["image_grid_thw"][0].tolist(),
                "pixel_values_shape": list(single["pixel_values"].shape),
                "pixel_values_sha256": _sha256(single["pixel_values"]),
                "pixel_values_sum": float(single["pixel_values"].double().sum()),
                "mu": float(single_logits[0, 0]),
                "sigma": float(single_logits[0, 1]),
                "mu_batched": float(batched[i, 0]),
                "sigma_batched": float(batched[i, 1]),
            }
        )

    runtime = {
        "model_dtype": str(model.dtype),
        "rm_head_dtype": str(next(model.rm_head.parameters()).dtype),
        "reward_hidden_dtype": str(captured["hidden"].dtype),
        "logits_dtype": str(batched.dtype),
        "attn_implementation": model.config._attn_implementation,
        "reward_token_id": int(reward_token_id),
        "vocab_size": int(model.get_input_embeddings().weight.shape[0]),
        "gpu": torch.cuda.get_device_name(),
    }

    # fp32 pass over the same (bf16-rounded) weights and rotary buffers: free of kernel noise, so the
    # alignment test can hold the port to a tight tolerance here and to the bf16 noise band above
    model.float()
    for sample, image in zip(samples, images, strict=True):
        sample["mu_fp32"] = float(reward([sample["prompt"]], [image])[0, 0])

    reference = {
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "runtime": runtime,
        "samples": samples,
    }
    Path(args.output).write_text(json.dumps(reference, indent=1) + "\n")
    print(json.dumps({k: v for k, v in reference.items() if k != "samples"}, indent=1))
    for sample in samples:
        print(
            sample["width"],
            sample["height"],
            len(sample["input_ids"]),
            sample["mu"],
            sample["mu_batched"],
            sample["mu_fp32"],
        )

    if args.debug_output:
        torch.save(
            {
                "batch": {k: v.cpu() for k, v in batch.items()},
                "reward_hidden": reward_hidden.float().cpu(),
                "batched_logits": batched.float().cpu(),
            },
            args.debug_output,
        )


if __name__ == "__main__":
    main()
