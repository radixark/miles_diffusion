"""E2E: Qwen-Image OCR GRPO, 4 GPUs, 2 rollouts — runs the example script
itself and checks its metric series against the registered standard
(tests/ci/fixtures/e2e_standards/). Runs with --deterministic-mode
(torch deterministic algorithms + NCCL/CUBLAS determinism), so every metric
is compared strictly, bit for bit."""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=1500,
    suite="stage-c-5-gpu-h200",
    script="scripts/run-diffusion-grpo-ocr-4gpu-flowgrpo-aligned.sh",
    env={"NUM_ROLLOUT": "2", "DETERMINISTIC_MODE": "1"},
    metrics=[
        "rollout/reward/raw_num_samples",
        "rollout/reward/raw_mean",
        "rollout/reward/raw_median",
        "rollout/reward/raw_std",
        "train/log_prob_old_idx_0",
        "train/log_prob_new_idx_0",
        "train/log_prob_mean_abs_diff",
        "train/grad_norm",
    ],
)
