"""E2E: SD3.5-medium DiffusionNFT, 2-GPU colocate (FSDP DP=2 + 2 sglang rollout
engines), 2 rollouts — runs the example script's smoke path (OCR reward, tiny
batch) and checks its metric series against the registered standard
(tests/ci/fixtures/e2e_standards/). Runs with --deterministic-mode, so every
metric is compared strictly, bit for bit.

RECORDING BRANCH — cut from c0bf98d, the commit that merged DiffusionNFT (#63),
so the standard captures NFT as it behaved the day it landed. Do not merge this
branch; only its recorded standard is carried forward.
"""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=1200,
    suite="stage-c-3-gpu-h200",
    script="scripts/run-diffusion-nft-sd3-pickscore.sh",
    env={
        "SMOKE": "1",
        "NUM_ROLLOUT": "2",
        "CUDA_VISIBLE_DEVICES": "0,1",
    },
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
