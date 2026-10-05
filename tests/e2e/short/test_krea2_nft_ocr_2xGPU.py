"""E2E: Krea-2-Raw DiffusionNFT with OCR, 2-GPU colocate (FSDP DP=2 sharing the GPUs
with two single-GPU sglang rollout engines) — runs the recipe's real configuration
and checks its metric series against the registered standard
(tests/ci/fixtures/e2e_standards/). The recipe enables --deterministic-mode, so
every metric is compared strictly, bit for bit.

Only --num-rollout is cut down, 100 -> 4. Rollout 0 is generated under the initial
EMA, which still equals the policy, so NFT's two loss branches are algebraically one
value at step 1; the LoRA IPC weight sync then publishes the updated EMA before each
later rollout and the branches separate. This guards the synchronous colocate path
for the Krea-2 family; the asynchronous variant has its own standard.
"""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=2400,
    suite="stage-c-3-gpu-h200",
    script="scripts/run_diffusion_nft_krea2.py",
    args=["--num-rollout", "4"],
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
