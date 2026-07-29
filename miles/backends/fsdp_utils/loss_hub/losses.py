"""Default Flow-GRPO loss formula (actor owns DiT forward).

Custom algorithms swap ``--custom-loss-function-path`` (formula only: receives
``new_pred`` / ``ref_pred``). Batch preparation lives in ``prepare.py``.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

import torch

from miles.backends.fsdp_utils.loss_hub.context import DiffusionLossContext, PreparedBatch
from miles.backends.fsdp_utils.metrics import record_rollout_train_abs_diff
from miles.utils.metric_buffer import MetricBuffer
from miles.utils.misc import load_function
from miles.utils.train_data_utils import stack_train_pair_rollout_debug

LossFormulaFn = Callable[..., torch.Tensor | None]


def flow_grpo_loss_formula(
    ctx: DiffusionLossContext,
    batch: list[dict],
    prepared: PreparedBatch,
    *,
    new_pred: torch.Tensor,
    ref_pred: torch.Tensor | None,
    metrics: MetricBuffer,
    write_old_log_prob: bool = False,
    old_log_prob_from_new: bool = False,
) -> torch.Tensor | None:
    """SDE log-prob + PPO-clip (+ optional KL vs ``ref_pred``). Actor owns DiT forward."""
    args = ctx.args
    clip_range = args.diffusion_clip_range
    noise_level = args.diffusion_noise_level
    kl_beta = float(args.diffusion_kl_beta)

    next_latents = prepared.extras["next_latents"]
    next_timesteps = prepared.extras["next_timesteps"]
    log_prob_old_rollout = prepared.extras["log_prob_old"]

    _, log_prob_new, prev_sample_mean_new, std_dev_t_new = ctx.sde_backend.sde_step_logprob(
        new_pred.float(),
        prepared.timesteps,
        next_timesteps,
        prepared.latents.float(),
        prev_sample=next_latents.float(),
        noise_level=noise_level,
    )

    if write_old_log_prob:
        for pair, log_prob in zip(batch, log_prob_new, strict=True):
            pair["log_prob_old"] = log_prob.cpu()
        return None

    log_prob_old = log_prob_new.detach() if old_log_prob_from_new else log_prob_old_rollout
    ratio = torch.exp(log_prob_new - log_prob_old)
    unclipped = -prepared.advantage * ratio
    clipped = -prepared.advantage * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    per_pair_loss = torch.maximum(unclipped, clipped)
    loss_sum = per_pair_loss.sum()
    bsz = len(batch)

    kl_sum = loss_sum.new_zeros(())
    if kl_beta > 0:
        if ref_pred is None:
            raise ValueError("Flow-GRPO KL requires a reference DiT forward (actor ref_mode=lora_base)")
        _, _, prev_sample_mean_ref, _ = ctx.sde_backend.sde_step_logprob(
            ref_pred.float(),
            prepared.timesteps,
            next_timesteps,
            prepared.latents.float(),
            prev_sample=next_latents.float(),
            noise_level=noise_level,
        )
        kl_per_pair = ((prev_sample_mean_new - prev_sample_mean_ref) ** 2).mean(
            dim=tuple(range(1, prev_sample_mean_new.ndim)),
            keepdim=True,
        ) / (2 * std_dev_t_new**2)
        loss_sum = loss_sum + kl_beta * kl_per_pair.sum()
        kl_sum = kl_per_pair.sum()

    with torch.no_grad():
        metrics.emit_mean("loss", total=loss_sum, count=bsz)
        metrics.emit_mean("policy_loss", total=per_pair_loss.sum(), count=bsz)
        metrics.emit_mean("kl_loss", total=kl_sum, count=bsz)
        metrics.emit_mean("loss_abs_mean", total=per_pair_loss.abs().sum(), count=bsz)
        metrics.emit_mean("adv_abs_mean", total=prepared.advantage.abs().sum(), count=bsz)
        metrics.emit_mean("ratio_abs_minus_1", total=(ratio - 1.0).abs().sum(), count=bsz)
        metrics.emit_mean("approx_kl", total=0.5 * ((log_prob_new - log_prob_old) ** 2).sum(), count=bsz)
        metrics.emit_mean("clipfrac", total=(torch.abs(ratio - 1.0) > clip_range).float().sum(), count=bsz)
        metrics.emit_mean("log_prob_new_idx_0", total=log_prob_new[0], count=1)
        metrics.emit_mean("log_prob_old_idx_0", total=log_prob_old[0], count=1)
        log_prob_abs_diff_sum = torch.abs(log_prob_new - log_prob_old).sum()
        metrics.emit_mean("log_prob_mean_abs_diff", total=log_prob_abs_diff_sum, count=bsz)
        if len(ctx.models) > 1:
            metrics.emit_mean(
                f"log_prob_mean_abs_diff_{prepared.component_name}",
                total=log_prob_abs_diff_sum,
                count=bsz,
            )

        rollout_model_output = stack_train_pair_rollout_debug(batch, "rollout_step_model_output")
        if rollout_model_output is not None:
            record_rollout_train_abs_diff(
                metrics,
                "model_output",
                new_pred.float(),
                rollout_model_output.to(device=ctx.device, dtype=torch.float32),
                component=prepared.component_name if len(ctx.models) > 1 else None,
            )

    return loss_sum


def resolve_loss_formula_fn(args: Namespace) -> LossFormulaFn:
    """Loss *formula* only — DiT forward stays in the actor.

    Custom path defaults (e.g. NFT) are assigned in ``arguments.py``. When the
    path is unset, Flow-GRPO is the default implementation.
    """
    path = getattr(args, "custom_loss_function_path", None)
    if path:
        fn = load_function(path)
        if fn is None:
            raise ValueError(f"Failed to load custom loss formula from {path!r}")
        return fn
    return flow_grpo_loss_formula
