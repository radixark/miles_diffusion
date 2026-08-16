"""Single-node 4-GPU proxy for the 17-GPU Wan2.2 full-finetune GRPO recipe.

`--four-gpu-ci` preserves the production 2-GPU engine shape and true-on-policy
invariant while reducing FSDP replication, rollout batch size, and reward placement.
"""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=2400,
    suite="stage-c-5-gpu-h200",
    script="scripts/run_diffusion_grpo_wan22_pickscore_17gpu_multinode.py",
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
