"""E2E: MiniMax H3 t2va PickScore Flow-GRPO, 2-GPU colocate (FSDP DP=2 + one
tp2 sglang rollout engine, CPU PickScore), 2 rollouts of 8 videos and no eval
(the recipe's own sizes take hours on one engine) — runs the example script
itself and checks its metric series against the registered standard
(tests/ci/fixtures/e2e_standards/). Runs with --deterministic-mode (torch
deterministic algorithms + NCCL/CUBLAS determinism), so every metric is
compared strictly, bit for bit. Also the only e2e exercising the SDE-window
trajectory transport (H3 requests filtered latents with step-index provenance)."""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=3000,
    suite="stage-c-3-gpu-h200",
    script="scripts/run_diffusion_grpo_h3_t2va_2gpu.py",
    args=["--num-rollout", "2", "--n-samples-per-prompt", "4", "--eval-interval", "0"],
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
