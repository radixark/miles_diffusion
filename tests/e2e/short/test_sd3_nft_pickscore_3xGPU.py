"""E2E: SD3.5-medium DiffusionNFT with PickScore, 3-GPU (2 colocated FSDP DP=2 + sglang
rollout engines, 1 dedicated reward GPU) — runs the example script's real configuration,
not a reduced one, and checks its metric series against the registered standard
(tests/ci/fixtures/e2e_standards/). Runs with --deterministic-mode, so every metric is
compared strictly, bit for bit.

Only --num-rollout is cut down, 100 -> 4, which is the shortest run that still reaches the
behaviour worth guarding. Step 1 overflows the fp16 grad scaler's 65536 init scale and is
skipped, so the weights do not move; step 2 is the first that lands, so through it the EMA
reference is still identical to the policy and NFT's two loss branches are algebraically
one value. They separate at step 3. Two rollouts would exercise none of that.

The NFT-side metrics are what the GRPO e2e cannot cover: NFT has no log-prob ratio, its
loss is the two-branch x0-MSE, and nft_t_mean tracks the sigma grid the prepare hook feeds
the DiT.
"""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=900,
    suite="stage-c-3-gpu-h200",
    script="scripts/run_diffusion_nft_sd3_pickscore.py",
    args=["--num-rollout", "4", "--cuda-visible-devices", "0,1,2"],
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
