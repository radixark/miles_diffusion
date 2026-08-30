"""E2E: Cosmos3-Nano T2I PickScore GRPO, 4-GPU fully colocated (train, rollout
and PickScore reward share the same 4 GPUs) — runs the canonical recipe's real
configuration and checks its metric series against the registered standard
(tests/ci/fixtures/e2e_standards/). Comparison is strict, bit for bit.

Only --num-rollout is cut down, 10000 -> 2: one weight-sync round trip is
enough to catch drift in the post-update rollout and the second optimizer step.

--extra-args --deterministic-mode is what makes the strict comparison viable:
the recipe ships --diffusion-debug-mode but, unlike the CI-only recipes, does
not pin determinism itself, and cosmos3 leaves --fsdp-attention-backend unset,
so SDPA is covered by torch's global flag.

train/model_output_mean_abs_diff and train/model_output_rel_max compare the raw
DiT outputs between engine and trainer; the recipe recomputes old log-probs at
ingestion, so the log_prob series guard the trainer-side pipeline rather than
cross-side parity.
"""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=1200,
    suite="stage-c-5-gpu-h200",
    script="scripts/run_diffusion_grpo_cosmos3_pickscore_t2i_4gpu.py",
    args=["--num-rollout", "2", "--extra-args", "--deterministic-mode"],
    labels=["e2e"],
    metrics=[
        "rollout/reward/raw_num_samples",
        "rollout/reward/raw_mean",
        "rollout/reward/raw_median",
        "rollout/reward/raw_std",
        "train/log_prob_old_idx_0",
        "train/log_prob_new_idx_0",
        "train/log_prob_mean_abs_diff",
        "train/model_output_mean_abs_diff",
        "train/model_output_rel_max",
        "train/grad_norm",
    ],
)
