"""E2E: Qwen-Image on-policy train<->rollout parity, 4 GPUs (train + rollout +
colocated pickscore reward), 2 rollouts. The recipe's whole point is that every
optimizer step is on-policy, so the training forward must reproduce the rollout
snapshot exactly: the standard pins `train/model_output_*_abs_diff` at 0.0, and
any patch-group or collate regression shows up there first. Runs with
--deterministic-mode, so every metric is compared strictly, bit for bit."""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=2400,
    suite="stage-c-5-gpu-h200",
    labels=["e2e"],
    script="scripts/run_diffusion_grpo_qwen_image_max_alignment_4gpu.py",
    # Empty on purpose: the runner already pins CUDA_VISIBLE_DEVICES=3,4,5,6,7,
    # and the recipe's own "4,5,6,7" default would be read as physical ids.
    args=["--num-rollout", "2", "--cuda-visible-devices", ""],
    metrics=[
        "rollout/reward/raw_num_samples",
        "rollout/reward/raw_mean",
        "rollout/reward/raw_median",
        "rollout/reward/raw_std",
        "train/model_output_max_abs_diff",
        "train/model_output_mean_abs_diff",
        "train/log_prob_old_idx_0",
        "train/log_prob_new_idx_0",
        "train/log_prob_mean_abs_diff",
        "train/grad_norm",
    ],
)
