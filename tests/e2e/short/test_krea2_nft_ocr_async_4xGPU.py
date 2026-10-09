"""E2E: Krea-2-Raw async DiffusionNFT with OCR, 3 FSDP training GPUs and one separate
single-GPU rollout engine that receives the LoRA adapters over NCCL. This is the async
recipe's 6 + 2 layout at half scale: 3 prompts x 8 samples keep its per-GPU load
(8 samples per training rank, 24 per engine). --num-rollout is cut from 100 to 3, which
also skips the recipe's every-25-steps eval. The recipe enables --deterministic-mode, so
every metric is compared strictly.

Batches 0 and 1 are both sampled with the initial EMA, since batch 1 is prefetched while
rollout 0 trains. Rollout 1 trains that batch against the latest EMA, one step ahead of
its sampler, and batch 2 is the first sampled after an NCCL weight push. Three rollouts
cover the overlap, the NCCL LoRA sync and the latest-EMA pi_old.
"""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=1500,
    suite="stage-c-5-gpu-h200",
    script="scripts/run_diffusion_nft_krea2_async.py",
    args=["--train-gpus", "3", "--rollout-gpus", "1", "--rollout-batch-size", "3", "--num-rollout", "3"],
    labels=["e2e"],
    metrics=[
        "rollout/reward/raw_num_samples",
        "rollout/reward/raw_mean",
        "rollout/reward/raw_median",
        "rollout/reward/raw_std",
        "train/grad_norm",
        "train/nft_loss",
        "train/nft_pos_loss",
        "train/nft_neg_loss",
        "train/nft_r_mean",
        "train/nft_t_mean",
    ],
)
