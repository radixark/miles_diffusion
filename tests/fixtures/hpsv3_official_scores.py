"""Score the HPSv3 alignment samples with the official implementation, for tests/fast-gpu/test_hpsv3_alignment.py.

The official package needs ``transformers==4.45.2``, which cannot share miles' environment. The test installs
``OFFICIAL_PINS`` into a directory of their own (``pip install --no-deps --target``, nothing else changes) and
runs this file in a subprocess with that directory first on PYTHONPATH, so the official code runs on miles'
own torch and CUDA and its scores can be compared bit for bit. To run it by hand:

    pip install --no-deps --target /tmp/hpsv3_official <OFFICIAL_PINS>
    PYTHONPATH=/tmp/hpsv3_official python tests/fixtures/hpsv3_official_scores.py --output scores.json

The rest of the official import chain (pandas, matplotlib, omegaconf, fire, datasets, accelerate, ...) comes from
miles' own environment. Module level imports only numpy and PIL so the test can import ``make_image`` and the pins.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# hpsv3's declared deps would pull deepspeed; these are the ones its inference import chain needs on top of
# miles' environment (huggingface_hub < 1.0 for transformers 4.45)
OFFICIAL_PINS = [
    "transformers==4.45.2",
    "tokenizers==0.20.3",
    "huggingface_hub==0.25.2",
    "hpsv3==1.0.0",
    "trl==0.8.6",
    "peft==0.10.0",
]

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
    import sys
    from pathlib import Path

    # the official code switches to flash-attn 2 whenever `import flash_attn` succeeds, and miles' image ships
    # an FA3 `flash_attn` namespace; the port runs sdpa, so make the import fail
    sys.modules["flash_attn"] = None

    import torch
    import transformers
    from hpsv3 import HPSv3RewardInferencer

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    inferencer = HPSv3RewardInferencer(checkpoint_path=args.checkpoint_path, device="cuda")
    model = inferencer.model
    images = [make_image(w, h, offset) for (w, h), offset, _ in SAMPLES]
    prompts = [prompt for _, _, prompt in SAMPLES]

    # keywords: the PyPI 1.0.0 reward() takes (image_paths, prompts), the GitHub one (prompts, image_paths);
    # PyPI also drops its @torch.inference_mode()
    def mu(prompts, images) -> list[float]:
        with torch.inference_mode():
            return [float(v) for v in inferencer.reward(prompts=prompts, image_paths=images)[:, 0]]

    samples = []
    for ((width, height), offset, prompt), image in zip(SAMPLES, images, strict=True):
        single = inferencer.prepare_batch(image_paths=[image], prompts=[prompt])
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
            }
        )

    # bf16 as deployed, then fp32 over the same bf16-rounded weights and rotary buffers
    for precision in ("bf16", "fp32"):
        if precision == "fp32":
            model.float()
        suffix = "" if precision == "bf16" else "_fp32"
        for sample, prompt, image in zip(samples, prompts, images, strict=True):
            sample["mu" + suffix] = mu([prompt], [image])[0]
        for sample, value in zip(samples, mu(prompts, images), strict=True):
            sample["mu_batched" + suffix] = value

    scores = {
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "attn_implementation": model.config._attn_implementation,
        "reward_token_id": int(model.special_token_ids[0]),
        "samples": samples,
    }
    Path(args.output).write_text(json.dumps(scores, indent=1) + "\n")


if __name__ == "__main__":
    main()
