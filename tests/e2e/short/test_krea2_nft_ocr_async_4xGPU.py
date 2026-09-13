"""E2E: Krea-2-Raw DiffusionNFT with OCR, 2 FSDP training GPUs and 2 separate
single-GPU rollout engines. Run the full recipe with only --num-rollout reduced
from 100 to 4; the recipe already enables --deterministic-mode, so comparison
against the registered metric standard is strict.

The async pipeline generates its first two batches with the initial EMA. The
third batch is the first generated after a weight update; four rollouts cover
two updated rollout batches and training against their lagged EMA references.
"""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=1800,
    suite="stage-c-5-gpu-h200",
    script="scripts/run_diffusion_nft_krea2_async.py",
    args=["--reward", "ocr", "--num-rollout", "4"],
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
