"""E2E: SD3.5-medium OCR GRPO, 2-GPU colocate (FSDP DP=2 + 2 engines), 2
rollouts — runs the example script itself and checks its metric series bitwise
against the registered standard (tests/ci/fixtures/e2e_standards/);
--deterministic-mode makes every series strictly comparable. HF_HUB_OFFLINE=1:
SD3.5 is gated, CI serves it from the pre-staged hf_cache mount."""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=1200,
    suite="stage-c-3-gpu-h200",
    script="scripts/run-diffusion-grpo-sd3-ocr-sglang.sh",
    env={"NUM_ROLLOUT": "2", "DETERMINISTIC_MODE": "1", "HF_HUB_OFFLINE": "1"},
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
