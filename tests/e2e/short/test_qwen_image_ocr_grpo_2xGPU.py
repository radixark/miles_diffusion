"""Smoke e2e: Qwen-Image OCR GRPO, 2 GPUs, 2 rollout steps.

Runs the example script itself with NUM_ROLLOUT=2 — the example can't drift
from what CI verifies. Two truncated rollouts exercise the full colocate
loop (sglang rollout -> OCR reward -> GRPO -> FSDP LoRA update -> weight
sync) within the 1800 s per-file budget, then the recorded metric series
must match the registered standard in tests/ci/fixtures/e2e_standards/
step by step. Re-register via the update-e2e-metrics workflow (or
`e2e_metrics_registry.py register`) when a change is intentional.
"""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=1500,
    suite="stage-b-2-gpu-h200",
    script="scripts/run-diffusion-grpo-ocr-2gpu-flowgrpo-aligned.sh",
    env={"NUM_ROLLOUT": "2"},
    metrics=[
        "rollout/reward/raw_mean",
        "rollout/reward/raw_std",
        "rollout/reward/group_mean_avg",
    ],
)
