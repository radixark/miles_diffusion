"""E2E: SD3.5-medium PickScore DiffusionNFT, 2 train GPUs colocated with 2 sglang
rollout engines + 1 dedicated PickScore GPU, 2 rollouts — runs the example script
itself and checks its metric series against the registered standard
(tests/ci/fixtures/e2e_standards/). Runs with --deterministic-mode (torch
deterministic algorithms + NCCL/CUBLAS determinism), so every metric is compared
strictly, bit for bit.

DiffusionNFT is a dual-policy x0-MSE objective, not a ratio/log-prob one, so the
train-side metrics here are the NFT branch losses rather than the Flow-GRPO
`log_prob_*` series (which nft_loss_formula never emits). nft_pos_loss vs
nft_neg_loss is the NFT analogue of the log-prob drift check: it tracks how far
the trained policy has moved from the EMA reference.

train/grad_norm is deliberately absent: fp16 forward enables ShardedGradScaler,
whose initial scale overflows on NFT's much larger loss, so step 1 logs nan and
nan == nan is False for any standard, with or without a tolerance.
"""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=600,
    suite="stage-c-3-gpu-h200",
    script="scripts/run-diffusion-nft-sd3-pickscore.sh",
    env={
        "NUM_ROLLOUT": "2",
        "CUDA_VISIBLE_DEVICES": "0,1,2",
    },
    metrics=[
        "rollout/reward/raw_num_samples",
        "rollout/reward/raw_mean",
        "rollout/reward/raw_median",
        "rollout/reward/raw_std",
        "train/nft_loss",
        "train/nft_loss_per_pair",
        "train/nft_pos_loss",
        "train/nft_neg_loss",
        "train/nft_r_mean",
        "train/nft_adv_mean",
        "train/adv_abs_mean",
        "train/nft_t_mean",
    ],
)
