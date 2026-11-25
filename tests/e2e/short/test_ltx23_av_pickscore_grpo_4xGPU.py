"""E2E: LTX-2.3 audio-video PickScore GRPO (--audio), 4 train GPUs + 1 pickscore GPU, 2 rollouts
— the engine rolls out the audio stream next to the video one with audio-video cross-attention on, and
the trainer replays the joint forward. Besides the joint log-prob gap, each stream's own gap is checked.
Runs with --deterministic-mode, so every metric is compared strictly, bit for bit."""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=1800,
    suite="stage-c-5-gpu-h200",
    labels=["e2e"],
    script="scripts/run_diffusion_grpo_ltx23_sglang.py",
    args=[
        "--audio",
        "--num-rollout",
        "2",
        "--extra-args",
        "--use-output-store --output-store-mooncake-master 127.0.0.1:50064 --output-store-protocol rdma",
    ],
    metrics=[
        "rollout/reward/raw_num_samples",
        "rollout/reward/raw_mean",
        "rollout/reward/raw_median",
        "rollout/reward/raw_std",
        "train/log_prob_old_idx_0",
        "train/log_prob_new_idx_0",
        "train/log_prob_mean_abs_diff",
        "train/log_prob_mean_abs_diff_visual",
        "train/log_prob_mean_abs_diff_audio",
        "train/grad_norm",
    ],
)
