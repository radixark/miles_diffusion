"""Closed-form center of the clipped Flow-GRPO ratio under the rollout Gaussian.

Rollout and trainer transitions are q = N(mu_q, Sigma) and p = N(mu, Sigma) with
the same fixed diagonal covariance. Flow-GRPO's per-element mean log-density
makes the PPO ratio r = (p/q)^(1/D); under q, log r is normal with mean
-kappa/(2D) and variance kappa/D^2, where kappa = |mu - mu_q|^2 / sigma^2.
E_q[clip_A(r)] therefore has a closed form, and adding A * (center - center.detach())
to the clipped PPO loss subtracts the expected weighted score from its gradient.
Only first derivatives are implemented; scalar work is done in float64.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from miles.utils.metric_buffer import MetricBuffer
from miles.utils.train_data_utils import stack_train_pair_rollout_debug

if TYPE_CHECKING:
    from miles.backends.fsdp_utils.sde_step_backend import SdeStepOutput


def _exp_difference(log_left: torch.Tensor, log_right: torch.Tensor) -> torch.Tensor:
    """exp(log_left) - exp(log_right), keeping accuracy when the two nearly cancel."""
    left_larger = log_left >= log_right
    large = torch.maximum(log_left, log_right)
    small = torch.minimum(log_left, log_right)
    # Both can be -inf in Gaussian tails; avoid the indeterminate inf - inf.
    delta = torch.where(torch.isfinite(large), small - large, torch.zeros_like(large))
    magnitude = torch.exp(large) * (-torch.expm1(delta))
    return torch.where(left_larger, magnitude, -magnitude)


class _ClippedRatioCenter(torch.autograd.Function):
    """E_q[clip_A(r)] with an exact d/dkappa, since autograd through sqrt(kappa) gives 0 * inf at kappa == 0."""

    @staticmethod
    def forward(ctx, kappa, advantage, dimension, clip_range):
        ctx.input_dtype = kappa.dtype
        kappa = kappa.to(dtype=torch.float64)
        alpha = 1.0 / dimension
        log_overlap = -0.5 * alpha * (1.0 - alpha) * kappa
        at_zero = kappa == 0

        # The safe branch is evaluated at 1, then replaced by the exact kappa == 0 limit.
        safe_kappa = torch.where(at_zero, torch.ones_like(kappa), kappa)
        sqrt_kappa = torch.sqrt(safe_kappa)
        positive = torch.broadcast_to(advantage.detach().to(device=kappa.device) >= 0, kappa.shape)
        log_clip = torch.where(
            positive,
            torch.full_like(kappa, math.log1p(clip_range)),
            torch.full_like(kappa, math.log1p(-clip_range)),
        )
        # Under q: log r ~ N(-alpha * kappa / 2, alpha^2 * kappa). Direct formulas avoid
        # subtracting two close standardized values.
        clip_z = log_clip / (alpha * sqrt_kappa) + 0.5 * sqrt_kappa
        tilted_clip_z = log_clip / (alpha * sqrt_kappa) + (0.5 - alpha) * sqrt_kappa
        log_cdf_below_tilted = torch.special.log_ndtr(tilted_clip_z)
        log_cdf_above_tilted = torch.special.log_ndtr(-tilted_clip_z)
        log_unclipped_term = log_overlap + torch.where(positive, log_cdf_below_tilted, log_cdf_above_tilted)
        log_clipped_term = log_clip + torch.special.log_ndtr(torch.where(positive, -clip_z, clip_z))
        value = torch.exp(torch.logaddexp(log_unclipped_term, log_clipped_term))

        log_pdf_over_sqrt_kappa = (
            log_overlap - 0.5 * tilted_clip_z.square() - 0.5 * math.log(2.0 * math.pi) - 0.5 * torch.log(safe_kappa)
        )
        if alpha == 1.0:
            log_bulk = torch.full_like(kappa, -math.inf)
        else:
            log_bulk = (
                log_overlap + math.log1p(-alpha) + torch.where(positive, log_cdf_below_tilted, log_cdf_above_tilted)
            )
        upper_derivative = -0.5 * alpha * torch.exp(torch.logaddexp(log_bulk, log_pdf_over_sqrt_kappa))
        lower_derivative = 0.5 * alpha * _exp_difference(log_pdf_over_sqrt_kappa, log_bulk)
        derivative = torch.where(positive, upper_derivative, lower_derivative)
        value = torch.where(at_zero, torch.ones_like(value), value)
        derivative = torch.where(at_zero, torch.full_like(derivative, -0.5 * alpha * (1.0 - alpha)), derivative)

        ctx.save_for_backward(derivative)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        (derivative,) = ctx.saved_tensors
        grad_kappa = (grad_output.to(torch.float64) * derivative).to(ctx.input_dtype)
        return grad_kappa, None, None, None


def clipped_ratio_center(
    kappa: torch.Tensor,
    dimension: int,
    advantage: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    """E_q[clip_A((p/q)^(1/D))] per batch item, differentiable in ``kappa``.

    ``kappa`` is the full squared mean displacement over the Gaussian variance (not
    its per-dimension mean); ``advantage`` only contributes its detached sign, which
    picks the clip side (``1 + clip_range`` for A >= 0, ``1 - clip_range`` below).
    """
    return _ClippedRatioCenter.apply(kappa, advantage, dimension, clip_range)


def gaussian_kappa(prev_mean: torch.Tensor, rollout_prev_mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """|prev_mean - rollout_prev_mean|^2 / variance per batch item; only ``prev_mean`` carries gradient."""
    delta = prev_mean.to(torch.float64) - rollout_prev_mean.detach().to(torch.float64)
    return (delta.square() / variance.detach()).flatten(1).sum(1)


def gaussian_mean_log_ratio(
    next_latents: torch.Tensor,
    prev_mean: torch.Tensor,
    rollout_prev_mean: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    """(log p - log q) / D at ``next_latents``, computed without subtracting two large log-densities."""
    delta = prev_mean.to(torch.float64) - rollout_prev_mean.detach().to(torch.float64)
    residual = next_latents.detach().to(torch.float64) - rollout_prev_mean.detach().to(torch.float64)
    return ((delta * residual - 0.5 * delta.square()) / variance.detach()).flatten(1).mean(1)


def gaussian_score_center(
    *,
    prev_mean: torch.Tensor,
    rollout_prev_mean: torch.Tensor,
    noise_std: torch.Tensor,
    rollout_noise_std: torch.Tensor,
    next_latents: torch.Tensor,
    log_ratio: torch.Tensor,
    advantage: torch.Tensor,
    clip_range: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Center for one micro-batch of SDE transitions, plus the diagnostics it was computed from.

    Args:
        prev_mean: trainer transition mean, ``[B, ...]``, carries the gradient.
        rollout_prev_mean: the engine's transition mean for the same step, same shape.
        noise_std: trainer transition std, broadcastable to ``prev_mean``.
        rollout_noise_std: the engine's transition std, one value per batch item.
        next_latents: the latent the engine actually sampled, same shape as ``prev_mean``.
        log_ratio: the loss's ``log_prob_new - log_prob_old`` per batch item.
        advantage: per-item advantage; only its sign is used.
        clip_range: the PPO clip range.

    Returns:
        ``(center, metrics)``: the float64 ``[B]`` center and per-item diagnostic tensors.
    """
    if rollout_prev_mean.shape != prev_mean.shape:
        raise ValueError(
            f"Rollout prev-sample means {tuple(rollout_prev_mean.shape)} are laid out unlike the trainer's "
            f"{tuple(prev_mean.shape)}; score centering does not support this model family"
        )
    batch_size = prev_mean.shape[0]
    rollout_noise_std = rollout_noise_std.float().reshape(batch_size, *([1] * (prev_mean.ndim - 1)))
    if not torch.allclose(noise_std.float(), rollout_noise_std, rtol=2e-5, atol=1e-8):
        raise ValueError(
            "Score centering needs the trainer and rollout SDE steps to share one transition covariance; "
            "their noise stds differ (different --diffusion-noise-level or sigma schedule?)"
        )
    variance = rollout_noise_std.double().square()
    dimension = prev_mean[0].numel()
    kappa = gaussian_kappa(prev_mean, rollout_prev_mean, variance)
    center = clipped_ratio_center(kappa, dimension, advantage, clip_range)
    # log_prob_old must be q's density at next_latents; no-const log-probs or misaligned
    # debug tensors would otherwise center the wrong objective silently.
    analytic_log_ratio = gaussian_mean_log_ratio(next_latents, prev_mean, rollout_prev_mean, variance)
    density_abs_diff = (analytic_log_ratio.detach() - log_ratio.detach().double()).abs()
    if not torch.isfinite(density_abs_diff).all() or density_abs_diff.max() > 5e-5:
        raise ValueError(
            "Rollout log-probs are not the Gaussian density of the rollout prev-sample means at the sampled "
            f"latents (max |diff| {density_abs_diff.max().item():.3e}); score centering cannot use them"
        )
    metrics = {
        "score_center_expected_clipped_ratio": center.detach(),
        "score_center_density_abs_diff": density_abs_diff,
    }
    return center, metrics


def score_centering_term(
    batch: list[dict],
    step_out: SdeStepOutput,
    rollout_prev_mean: torch.Tensor | None,
    *,
    next_latents: torch.Tensor,
    log_ratio: torch.Tensor,
    advantage: torch.Tensor,
    clip_range: float,
    loss_dtype: torch.dtype,
    metrics: MetricBuffer,
) -> torch.Tensor:
    """Per-pair term to add to the clipped PPO loss for --diffusion-score-centering.

    It is zero-valued, so the logged loss and the clip mask are unchanged; only the gradient moves.
    ``rollout_prev_mean`` is the stacked ``rollout_step_prev_sample_mean``, already on ``step_out``'s
    device in float32.
    """
    rollout_noise_std = stack_train_pair_rollout_debug(batch, "rollout_step_noise_std_dev")
    if rollout_prev_mean is None or rollout_noise_std is None:
        raise ValueError(
            "--diffusion-score-centering needs the rollout's per-step prev-sample means and noise stds; "
            "these rollouts carry none (run with --diffusion-debug-mode)"
        )
    center, centering_metrics = gaussian_score_center(
        prev_mean=step_out.prev_mean,
        rollout_prev_mean=rollout_prev_mean,
        noise_std=step_out.noise_std,
        rollout_noise_std=rollout_noise_std.to(step_out.prev_mean.device),
        next_latents=next_latents,
        log_ratio=log_ratio,
        advantage=advantage,
        clip_range=clip_range,
    )
    with torch.no_grad():
        for key, values in centering_metrics.items():
            metrics.emit_mean(key, total=values.sum(), count=len(batch))
    return advantage * (center - center.detach()).to(loss_dtype)
