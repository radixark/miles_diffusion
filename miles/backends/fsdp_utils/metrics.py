"""What the FSDP train loop logs, and how each metric reduces across ranks."""

from collections.abc import Collection

import torch

from miles.utils.metric_buffer import MetricBuffer, MetricReduce

# Declaring metrics here also fixes the cross-rank layout the reduction packs into.
SCHEMA = {
    "loss": MetricReduce.MEAN,
    "loss_visual": MetricReduce.MEAN,
    "loss_audio": MetricReduce.MEAN,
    "policy_loss": MetricReduce.MEAN,
    "kl_loss": MetricReduce.MEAN,
    "loss_abs_mean": MetricReduce.MEAN,
    "adv_abs_mean": MetricReduce.MEAN,
    "ratio_abs_minus_1": MetricReduce.MEAN,
    "approx_kl": MetricReduce.MEAN,
    "clipfrac": MetricReduce.MEAN,
    "log_prob_new_idx_0": MetricReduce.MEAN,
    "log_prob_old_idx_0": MetricReduce.MEAN,
    "log_prob_mean_abs_diff": MetricReduce.MEAN,
    "model_output_mean_abs_diff": MetricReduce.MEAN,
    "model_output_max_abs_diff": MetricReduce.MAX,
    "model_output_rel_max": MetricReduce.MAX,
    "prev_mean_diff_over_noise_std": MetricReduce.MEAN,
    "score_center_expected_clipped_ratio": MetricReduce.MEAN,
    "score_center_density_abs_diff": MetricReduce.MEAN,
    "grad_norm": MetricReduce.REPLICATED,
    "nft_loss": MetricReduce.MEAN,
    "nft_loss_per_pair": MetricReduce.MEAN,
    "nft_r_mean": MetricReduce.MEAN,
    "nft_pos_loss": MetricReduce.MEAN,
    "nft_neg_loss": MetricReduce.MEAN,
    "nft_adv_mean": MetricReduce.MEAN,
    "nft_t_mean": MetricReduce.MEAN,
    "nft_num_timesteps": MetricReduce.MEAN,
    "dmd2_grad_abs_mean": MetricReduce.MEAN,
    "dmd2_score_gap_abs_mean": MetricReduce.MEAN,
    "dmd2_teacher_dist_abs_mean": MetricReduce.MEAN,
    "dmd2_x0_mean": MetricReduce.MEAN,
    "dmd2_x0_std": MetricReduce.MEAN,
    "x0_mean_abs_diff": MetricReduce.MEAN,
    "x0_max_abs_diff": MetricReduce.MAX,
    "x0_rel_max": MetricReduce.MAX,
    "peak_memory_allocated_gb": MetricReduce.MAX,
    "device_memory_used_gb": MetricReduce.MAX,
    "pdd_next_latent_mae": MetricReduce.MEAN,
    "pdd_update_relative_error": MetricReduce.MEAN,
}

# Also logged per rollout denoising step, under step_metric_key; a step no pair trained on is not logged.
STEP_SCHEMA = {
    "dmd2_grad_abs_mean": MetricReduce.MEAN,
    "dmd2_score_gap_abs_mean": MetricReduce.MEAN,
    "dmd2_teacher_dist_abs_mean": MetricReduce.MEAN,
    "x0_mean_abs_diff": MetricReduce.MEAN,
}


def step_metric_key(key: str, step_index: int) -> str:
    return f"{key}_step{step_index}"


def sigma_bucket_key(stream_name: str, bucket: int, num_buckets: int) -> str:
    # Keep integer edges as floats (0.0, 1.0) so key names stay stable across bucket counts.
    def edge(value: float) -> str:
        text = f"{value:g}"
        return text if "." in text else f"{value:.1f}"

    return f"loss_{stream_name}_sigma_{edge(bucket / num_buckets)}_{edge((bucket + 1) / num_buckets)}"


def new_metric_buffer(
    group, device: torch.device, components: Collection[str], sigma_buckets: int = 0, num_rollout_steps: int = 0
) -> MetricBuffer:
    """Buffer for one optimizer step, reducing over the DP group."""
    schema = dict(SCHEMA)
    for key, reduce in STEP_SCHEMA.items():
        for step_index in range(num_rollout_steps):
            schema[step_metric_key(key, step_index)] = reduce
    # SFT loss by sigma bucket: velocity-MSE magnitude varies strongly with the
    # corruption level, so only bucketed curves are comparable across steps.
    for stream_name in ("visual", "audio"):
        for bucket in range(sigma_buckets):
            schema[sigma_bucket_key(stream_name, bucket, sigma_buckets)] = MetricReduce.MEAN
    for component in components if len(components) > 1 else ():
        schema[f"log_prob_mean_abs_diff_{component}"] = MetricReduce.MEAN
        schema[f"model_output_mean_abs_diff_{component}"] = MetricReduce.MEAN
    return MetricBuffer(group=group, device=device, schema=schema)


def record_prev_mean_diff_over_noise_std(
    metrics: MetricBuffer,
    rollout_prev_mean: torch.Tensor,
    prev_mean: torch.Tensor,
    noise_std: torch.Tensor,
) -> None:
    """Record the trainer-vs-rollout SDE transition-mean gap per pair, RMS in units of the step's noise std."""
    bsz = prev_mean.shape[0]
    with torch.no_grad():
        # flattened per pair like model_output, so logging never depends on the debug layout
        noise_std = noise_std.reshape(bsz, -1)
        mean_diff = rollout_prev_mean.reshape(bsz, -1) - prev_mean.reshape(bsz, -1)
        diff_over_std = torch.where(noise_std > 0, mean_diff / noise_std, 0.0)
        metrics.emit_mean(
            "prev_mean_diff_over_noise_std", total=diff_over_std.pow(2).mean(dim=1).sqrt().sum(), count=bsz
        )


def record_rollout_train_abs_diff(
    metrics: MetricBuffer,
    prefix: str,
    train: torch.Tensor,
    rollout: torch.Tensor,
    component: str | None = None,
) -> None:
    """Record the train-vs-rollout deviation, also split by `component` when given."""
    bsz = train.shape[0]
    reference = rollout.reshape(bsz, -1).float()
    diff = (train.reshape(bsz, -1).float() - reference).abs()
    total, worst, count = diff.sum(), diff.max(), diff.numel()
    metrics.emit_max(f"{prefix}_max_abs_diff", worst)
    metrics.emit_mean(f"{prefix}_mean_abs_diff", total=total, count=count)
    metrics.emit_max(f"{prefix}_rel_max", worst / (reference.abs().max() + 1e-30))
    if component is not None:
        metrics.emit_mean(f"{prefix}_mean_abs_diff_{component}", total=total, count=count)
