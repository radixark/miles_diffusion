"""Gaussian score centering inside the production Flow-GRPO loss.

    engine: x' ~ N(mu_q, s^2)  --rollout debug tensors-->  trainer: r = exp(log N(x'|mu, s^2) - log_prob_old)
                                                                     loss = max(-A r, -A clip(r))
    --diffusion-score-centering:  loss += A * (center - center.detach()),   center = E_q[clip_A(r)]

    production gradient   ==  float64 oracle: autograd through -A * (clip(r) - center)   (closed form)
    logged loss           ==  plain PPO loss (the added term is zero-valued)
    clip-masked pairs     -->  zero gradient without centering, non-zero with it
    mu_q == mu            -->  gradient bit-identical to plain PPO
    no debug tensors / stds differ / log_prob_old is not q's density  -->  ValueError
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils.loss_hub.flow_grpo import flow_grpo_loss_formula
from miles.backends.fsdp_utils.sde_step_backend import FlowSdeStepBackend

NOISE_LEVEL = 0.7
CLIP_RANGE = 0.2
VIEW = (-1, 1, 1, 1)


class _MetricSink:
    def emit_mean(self, key, *, total, count):
        pass

    def emit_max(self, key, value):
        pass


@pytest.fixture
def transition():
    """Four 2x2 transitions whose trainer mean sits 1.2 noise stds from the rollout mean along one coordinate.

    The sampled latents are placed so the PPO ratios are (1.6, 1.0, 0.6, 1.1) under advantages
    (+1, +2, -1, -2): pairs 0 and 2 are clip-masked, pairs 1 and 3 are not.
    """
    backend = FlowSdeStepBackend()
    args = SimpleNamespace(
        diffusion_clip_range=CLIP_RANGE,
        diffusion_noise_level=NOISE_LEVEL,
        diffusion_kl_beta=0.0,
        diffusion_score_centering=True,
    )
    ctx = SimpleNamespace(args=args, sde_backend=backend, models={"dit": object()}, device=torch.device("cpu"))
    latent = torch.zeros((4, 1, 2, 2), dtype=torch.float32)
    sigmas = torch.tensor([0.8, 0.55, 0.8, 0.55])
    next_sigmas = torch.tensor([0.55, 0.25, 0.55, 0.25])
    sigma_max = torch.full((4,), 0.8)
    sigma, sigma_next = sigmas.view(VIEW), next_sigmas.view(VIEW)
    rollout_prev_mean, noise_std, std_dev_t = backend.prev_sample_mean_and_std(
        torch.zeros_like(latent), latent, sigma, sigma_next, sigma_max=sigma_max.view(VIEW), noise_level=NOISE_LEVEL
    )
    displacement = torch.zeros_like(latent)
    displacement[:, 0, 0, 0] = 1.2 * noise_std.flatten()
    pred_coefficient = (1 + std_dev_t.square() * (1 - sigma) / (2 * sigma)) * (sigma_next - sigma)
    new_pred = (displacement / pred_coefficient).detach().requires_grad_(True)
    target_ratio = torch.tensor([1.6, 1.0, 0.6, 1.1])
    z = torch.tensor([[0.0, 0.3, -0.7, 0.2], [0.0, -0.2, 0.1, 0.5], [0.0, 0.8, -0.4, 0.3], [0.0, -0.1, 0.4, -0.2]])
    z[:, 0] = (4 * target_ratio.log() + 0.5 * 1.2**2) / 1.2
    next_latents = rollout_prev_mean + noise_std * z.reshape_as(latent)
    batch = [
        {
            "rollout_debug_tensors": {
                "rollout_step_prev_sample_mean": rollout_prev_mean[i].clone(),
                "rollout_step_noise_std_dev": noise_std[i].flatten().clone(),
            }
        }
        for i in range(4)
    ]
    prepared = SimpleNamespace(
        latents={"visual": latent},
        advantage=torch.tensor([1.0, 2.0, -1.0, -2.0]),
        component_name="dit",
        extras={
            "next_latents": next_latents,
            "sigmas": sigmas,
            "next_sigmas": next_sigmas,
            "sigma_max": sigma_max,
            "log_prob_old": backend.log_prob(next_latents, rollout_prev_mean, noise_std).detach(),
        },
    )
    return SimpleNamespace(
        ctx=ctx,
        batch=batch,
        prepared=prepared,
        new_pred=new_pred,
        rollout_prev_mean=rollout_prev_mean,
        noise_std=noise_std,
    )


def _loss(t, new_pred):
    return flow_grpo_loss_formula(
        t.ctx, t.batch, t.prepared, new_pred={"visual": new_pred}, old_pred=None, ref_pred=None, metrics=_MetricSink()
    )


def _grad(t, new_pred, *, score_centering):
    t.ctx.args.diffusion_score_centering = score_centering
    pred = new_pred.detach().clone().requires_grad_(True)
    loss = _loss(t, pred)
    (grad,) = torch.autograd.grad(loss, pred)
    return loss.item(), grad


def test_gradient_matches_the_closed_form_oracle_and_the_loss_value_is_unchanged(transition):
    """A sign, side, scale or detach mistake in the correction term, or a changed logged loss, fails here."""
    t = transition
    production_loss, production_grad = _grad(t, t.new_pred, score_centering=True)

    # Independent float64 objective -A * (clip(r) - center), with ordinary autograd
    # through the closed-form lognormal expectation (every kappa is strictly > 0).
    oracle_pred = t.new_pred.detach().clone().requires_grad_(True)
    prev_mean, _, _ = t.ctx.sde_backend.prev_sample_mean_and_std(
        oracle_pred,
        t.prepared.latents["visual"],
        t.prepared.extras["sigmas"].view(VIEW),
        t.prepared.extras["next_sigmas"].view(VIEW),
        sigma_max=t.prepared.extras["sigma_max"].view(VIEW),
        noise_level=NOISE_LEVEL,
    )
    delta = prev_mean.double() - t.rollout_prev_mean.double()
    variance = t.noise_std.double().square()
    residual = t.prepared.extras["next_latents"].double() - t.rollout_prev_mean.double()
    alpha = 1 / prev_mean[0].numel()
    ratio = ((delta * residual - 0.5 * delta.square()) / variance).flatten(1).mean(1).exp()
    kappa = (delta.square() / variance).flatten(1).sum(1)
    root = kappa.sqrt()
    overlap = (-0.5 * alpha * (1 - alpha) * kappa).exp()
    advantage = t.prepared.advantage.double()
    upper, lower = 1 + CLIP_RANGE, 1 - CLIP_RANGE
    clip = torch.where(advantage >= 0, upper, lower)
    clip_z = clip.log() / (alpha * root) + 0.5 * root
    tilted_clip_z = clip.log() / (alpha * root) + (0.5 - alpha) * root
    center_positive = overlap * torch.special.ndtr(tilted_clip_z) + clip * torch.special.ndtr(-clip_z)
    center_negative = overlap * torch.special.ndtr(-tilted_clip_z) + clip * torch.special.ndtr(clip_z)
    center = torch.where(advantage >= 0, center_positive, center_negative)
    clipped = torch.where(advantage >= 0, ratio.clamp(max=upper), ratio.clamp(min=lower))
    oracle_loss = (-advantage * (clipped - center)).sum()
    (oracle_grad,) = torch.autograd.grad(oracle_loss, oracle_pred)
    torch.testing.assert_close(production_grad, oracle_grad, rtol=4e-5, atol=2e-6)

    baseline_loss, baseline_grad = _grad(t, t.new_pred, score_centering=False)
    assert production_loss == baseline_loss
    assert (production_grad - baseline_grad).abs().max().item() > 1e-3
    # Clip-masked pairs (0 and 2) move only through the center.
    assert baseline_grad[0].abs().sum().item() == 0.0 and baseline_grad[2].abs().sum().item() == 0.0
    assert production_grad[0].abs().sum().item() > 0.0 and production_grad[2].abs().sum().item() > 0.0


def test_identical_means_leave_the_gradient_bit_identical(transition):
    """At kappa == 0 the center must be an exact no-op, not a NaN or a rounding-sized perturbation."""
    t = transition
    aligned_pred = torch.zeros_like(t.new_pred)
    _, centered_grad = _grad(t, aligned_pred, score_centering=True)
    _, baseline_grad = _grad(t, aligned_pred, score_centering=False)
    assert torch.isfinite(centered_grad).all()
    torch.testing.assert_close(centered_grad, baseline_grad, rtol=0, atol=0)


def test_rollouts_without_debug_tensors_are_rejected(transition):
    """Without the check, a missing debug field would surface as an AttributeError on None."""
    del transition.batch[0]["rollout_debug_tensors"]["rollout_step_prev_sample_mean"]
    with pytest.raises(ValueError, match="these rollouts carry none"):
        _loss(transition, transition.new_pred)


def test_differing_transition_stds_are_rejected(transition):
    """A rollout std off by 10% breaks the equal-covariance closed form; it must raise, not center silently."""
    transition.batch[0]["rollout_debug_tensors"]["rollout_step_noise_std_dev"] *= 1.1
    with pytest.raises(ValueError, match="share one transition covariance"):
        _loss(transition, transition.new_pred)


def test_log_prob_old_that_is_not_the_rollout_density_is_rejected(transition):
    """A log_prob_old shifted by 0.01 (no-const or misaligned log-probs) fails the density rebuild check."""
    transition.prepared.extras["log_prob_old"] = transition.prepared.extras["log_prob_old"] + 0.01
    with pytest.raises(ValueError, match="not the Gaussian density"):
        _loss(transition, transition.new_pred)
