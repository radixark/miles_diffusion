"""E2E: Qwen-Image pickscore GRPO, 4 train + 1 reward GPUs, 2 rollouts — runs
the example script itself and checks its metric series bitwise against the
registered standard (tests/ci/fixtures/e2e_standards/); --deterministic-mode
makes every series strictly comparable."""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=2100,
    suite="stage-c-5-gpu-h200",
    labels=["e2e"],
    script="scripts/run-diffusion-grpo-pickscore-5gpu-flowgrpo-aligned.sh",
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
