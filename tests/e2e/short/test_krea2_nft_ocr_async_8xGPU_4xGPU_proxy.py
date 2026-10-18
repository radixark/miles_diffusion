"""4-GPU proxy for the 8-GPU Krea-2-Raw async DiffusionNFT OCR recipe.

`--four-gpu-ci` halves both pools and the batch: 3 training GPUs (FSDP DP=3) and 1
single-GPU sglang rollout engine on separate pools, 3 prompts x 8 samples per step, so each
training GPU keeps the production micro-batches. The recipe runs train_diffusion_async.py
under --deterministic-mode, so every metric is compared strictly, bit for bit against the
registered standard (tests/ci/fixtures/e2e_standards/).

--num-rollout is also cut down, 100 -> 4. This guards the NCCL LoRA weight sync and the
one-step-off pipeline: rollouts 0 and 1 both sample the initial EMA (rollout 1 is
generated while step 1 trains), and rollouts 2 and 3 sample the EMA one step behind.
"""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=2400,
    suite="stage-c-5-gpu-h200",
    script="scripts/run_diffusion_nft_krea2_async.py",
    args=["--num-rollout", "4", "--four-gpu-ci"],
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
