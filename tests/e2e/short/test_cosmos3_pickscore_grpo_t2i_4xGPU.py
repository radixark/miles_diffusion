"""E2E: Cosmos3-Nano T2I PickScore GRPO, 4-GPU fully colocated (train, rollout
and PickScore reward share the same 4 GPUs) — runs the bitwise recipe's real
configuration and checks its metric series against the registered standard
(tests/ci/fixtures/e2e_standards/). Runs with --deterministic-mode, so every
metric is compared strictly, bit for bit.

Only --num-rollout is cut down, 10000 -> 2: one weight-sync round trip is
enough to catch drift in the post-update rollout and the second optimizer step.
--cuda-visible-devices "" unpins the recipe's default 0,1,2,3 so it inherits
the runner's GPU set (the 5gpu runner exposes GPUs 3-7).

What this test uniquely guards is the bitwise train<->rollout parity the
cosmos3_bitwise rollout patch group + --sglang-lora-merge-mode dynamic
establish: train/model_output_mean_abs_diff / train/model_output_rel_max
compare the raw DiT outputs between engine and trainer — with parity both are
exactly 0. The recipe recomputes old log-probs at ingestion, so the log_prob
series guard the trainer-side pipeline rather than cross-side parity.
"""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=4800,
    suite="stage-c-5-gpu-h200",
    script="scripts/run_diffusion_grpo_cosmos3_pickscore_t2i_4gpu_bitwise.py",
    args=["--num-rollout", "2", "--cuda-visible-devices", ""],
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
