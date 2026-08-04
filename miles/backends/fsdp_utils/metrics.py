"""What the FSDP train loop logs, and how each metric reduces across ranks."""

from collections.abc import Collection

import torch

from miles.utils.metric_buffer import MetricBuffer, MetricReduce

# Declaring metrics here also fixes the cross-rank layout the reduction packs into.
SCHEMA = {
    "loss": MetricReduce.MEAN,
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
    "grad_norm": MetricReduce.REPLICATED,
    "nft_loss": MetricReduce.MEAN,
    "nft_loss_per_pair": MetricReduce.MEAN,
    "nft_r_mean": MetricReduce.MEAN,
    "nft_pos_loss": MetricReduce.MEAN,
    "nft_neg_loss": MetricReduce.MEAN,
    "nft_adv_mean": MetricReduce.MEAN,
    "nft_t_mean": MetricReduce.MEAN,
    "nft_num_timesteps": MetricReduce.MEAN,
}


def new_metric_buffer(group, device: torch.device, components: Collection[str]) -> MetricBuffer:
    """Buffer for one optimizer step, reducing over the DP group."""
    schema = dict(SCHEMA)
    for component in components if len(components) > 1 else ():
        schema[f"log_prob_mean_abs_diff_{component}"] = MetricReduce.MEAN
        schema[f"model_output_mean_abs_diff_{component}"] = MetricReduce.MEAN
    return MetricBuffer(group=group, device=device, schema=schema)


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
