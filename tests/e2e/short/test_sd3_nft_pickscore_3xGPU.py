"""E2E: SD3.5-medium DiffusionNFT with PickScore, 3-GPU (2 colocated FSDP DP=2 + sglang
rollout engines, 1 dedicated reward GPU) — runs the example script's real configuration,
not a reduced one, and checks its metric series against the registered standard
(tests/ci/fixtures/e2e_standards/). Runs with --deterministic-mode, so every metric is
compared strictly, bit for bit.

Only NUM_ROLLOUT is cut down: the recipe's 100 rollouts become 8, which is the shortest
run that still reaches the behaviour worth guarding. The fp16 grad scaler starts at 65536
and halves on each overflow, so the first optimizer steps are skipped -- until one lands,
the weights never move, the EMA reference stays identical to the policy, and NFT's two
loss branches are algebraically the same value. The tail of the run is where pos and neg
actually separate.

RECORDING BRANCH — cut from c0bf98d, the commit that merged DiffusionNFT (#63), so the
standard captures NFT as it behaved the day it landed. Do not merge this branch; only its
recorded standard is carried forward.
"""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=3600,
    suite="stage-c-3-gpu-h200",
    script="scripts/run-diffusion-nft-sd3-pickscore.sh",
    env={
        "NUM_ROLLOUT": "8",
        "CUDA_VISIBLE_DEVICES": "0,1,2",
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
