"""E2E: SD3.5-medium OCR GRPO with Flow-GRPO score centering, 2-GPU colocate.

Same recipe, topology and --deterministic-mode as test_sd3_ocr_grpo_2xGPU (FSDP
DP=2 + 2 sglang rollout engines, 2 rollouts), plus:

    --diffusion-score-centering         subtract the expected score under the rollout's transition
    --diffusion-debug-mode              the rollout returns its per-step transition means
    --diffusion-recompute-old-log-prob  keep the PPO ratio about policy movement only

so the run covers the rollout -> trainer path for rollout_prev_sample_means and the
correction term end to end. train/prev_mean_diff_over_noise_std is how far the rollout's
transition mean sits from the trainer's; the other series pin the training trajectory
bit for bit against the registered standard.
"""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=1200,
    suite="stage-c-3-gpu-h200",
    script="scripts/run_diffusion_grpo_sd3_ocr_sglang.py",
    args=[
        "--num-rollout",
        "2",
        "--extra-args",
        "--diffusion-score-centering --diffusion-debug-mode --diffusion-recompute-old-log-prob",
    ],
    labels=["e2e"],
    metrics=[
        "rollout/reward/raw_num_samples",
        "rollout/reward/raw_mean",
        "rollout/reward/raw_median",
        "rollout/reward/raw_std",
        "train/log_prob_old_idx_0",
        "train/log_prob_new_idx_0",
        "train/log_prob_mean_abs_diff",
        "train/prev_mean_diff_over_noise_std",
        "train/model_output_mean_abs_diff",
        "train/grad_norm",
    ],
)
