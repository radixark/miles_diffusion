"""Smoke e2e: Qwen-Image OCR GRPO, 2 GPUs, 2 rollout steps.

Runs the example script itself with NUM_ROLLOUT=2, so CI verifies the exact
artifact users run — the example can't drift from what the test covers. Two
truncated rollouts still exercise the full colocate loop (sglang rollout ->
OCR reward -> GRPO -> FSDP LoRA update -> weight sync) within the 1800 s
per-file budget; model/dataset/paddle caches are host-mounted on the CI node.
"""

import os
import subprocess
from pathlib import Path

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=1500,
    suite="stage-b-2-gpu-h200",
)

ROOT_DIR = Path(__file__).resolve().parents[3]
SCRIPT = ROOT_DIR / "scripts" / "run-diffusion-grpo-ocr-2gpu-flowgrpo-aligned.sh"


def execute():
    env = dict(os.environ)
    env["NUM_ROLLOUT"] = "2"
    subprocess.run(["bash", str(SCRIPT)], check=True, cwd=ROOT_DIR, env=env)


if __name__ == "__main__":
    execute()
