"""E2E: Wan2.2-A14B full-finetune video PickScore GRPO, the 16-GPU multi-node recipe
capped to 1 node x 4 GPUs via --four-gpu-ci (batch /4, FSDP shard 4, SP kept, one
4-GPU rollout engine, reward colocated one worker per GPU), 2 rollouts — runs the
example script itself and checks its metric series against the registered standard
(tests/ci/fixtures/e2e_standards/). Runs with --deterministic-mode, so every metric
is compared strictly, bit for bit — including train/model_output_mean_abs_diff, which
must be exactly 0.0: the true-on-policy guarantee (importance ratio = 1) is itself
under test."""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=2400,
    suite="stage-c-5-gpu-h200",
    script="scripts/run_diffusion_grpo_wan22_pickscore_16gpu_multinode.py",
    args=["--num-rollout", "2", "--four-gpu-ci", "--cuda-visible-devices", ""],
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
