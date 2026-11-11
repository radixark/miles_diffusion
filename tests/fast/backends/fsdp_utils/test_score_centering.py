"""The closed-form center E_q[clip_A(r)] and its exact derivative.

    q = N(mu_q, s^2 I), p = N(mu, s^2 I), r = (p/q)^(1/D)      alpha = 1/D, kappa = |mu - mu_q|^2 / s^2
    under q:  log r ~ N(-alpha * kappa / 2, alpha^2 * kappa)   --> center(kappa) = E_q[clip_A(r)] in closed form

    center(kappa)         --gradcheck-->   finite differences             (both clip sides, D from 1 to 65536)
    kappa in [0, 1e6]     -->              finite value and gradient, exact kappa == 0 limit (center 1)
    z ~ N(0, 1) samples   -->              E[clip(r)] and E[1{unclipped} r score] match center, dcenter/dmu
    gaussian_mean_log_ratio / gaussian_kappa  ==  the same quantities from torch Normal log-densities
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import math

import pytest
import torch

from miles.backends.fsdp_utils.loss_hub.score_centering import (
    clipped_ratio_center,
    gaussian_kappa,
    gaussian_mean_log_ratio,
)


@pytest.mark.parametrize("dimension", [1, 2, 8, 65536])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_center_gradient_matches_finite_differences(dimension, sign):
    """A wrong hand-written d/dkappa (the custom backward) fails here on at least one clip side."""
    kappa_value = 0.7 if dimension < 100 else 2.0 * dimension  # tiny alpha with realistic log-ratio spread
    kappa = torch.tensor(kappa_value, dtype=torch.float64, requires_grad=True)
    advantage = torch.tensor(sign)

    def center(value):
        return clipped_ratio_center(value, dimension, advantage, 0.15)

    assert torch.autograd.gradcheck(center, (kappa,), eps=1e-4, atol=1e-8, rtol=3e-5)


@pytest.mark.parametrize("dimension", [1, 4, 65536])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_center_is_finite_from_zero_kappa_to_far_tails(dimension, sign):
    """Autograd through sqrt(kappa) at 0, or exp of a far tail, would give NaN or inf here."""
    kappa = torch.tensor([0.0, 1e-300, 1e-16, 1e-8, 1.0, 1e6], dtype=torch.float64, requires_grad=True)
    center = clipped_ratio_center(kappa, dimension, torch.full_like(kappa, sign), 1e-4)
    (grad,) = torch.autograd.grad(center.sum(), kappa)
    assert torch.isfinite(center).all() and torch.isfinite(grad).all()
    alpha = 1 / dimension
    assert center[0].item() == 1.0
    assert grad[0].item() == -0.5 * alpha * (1 - alpha)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_center_and_its_gradient_match_monte_carlo(sign):
    """The closed form (clip side, tilt, truncation) is checked against sampling on both advantage signs."""
    # All dependence is along the mean displacement, so this 1-D projection is an
    # exact reduction of any equal-covariance D-dimensional pair.
    generator = torch.Generator().manual_seed(173)
    z_half = torch.randn(200000, generator=generator, dtype=torch.float64)
    z = torch.cat((z_half, -z_half))
    dimension, displacement, clip_range = 2, 1.6, 0.2
    alpha = 1 / dimension
    ratio = torch.exp(alpha * (displacement * z - 0.5 * displacement**2))
    score = alpha * (z - displacement)

    kappa = torch.tensor(displacement**2, dtype=torch.float64, requires_grad=True)
    center = clipped_ratio_center(kappa, dimension, torch.tensor(sign), clip_range)
    center_grad = 2 * displacement * torch.autograd.grad(center, kappa)[0]
    if sign > 0:
        transformed = ratio.clamp(max=1 + clip_range)
        effective_score = ratio * score * (ratio < 1 + clip_range)
    else:
        transformed = ratio.clamp(min=1 - clip_range)
        effective_score = ratio * score * (ratio > 1 - clip_range)
    tolerance = 6 / math.sqrt(z.numel())
    assert abs(transformed.mean().item() - center.item()) < tolerance * transformed.std().item() + 1e-4
    assert abs(effective_score.mean().item() - center_grad.item()) < tolerance * effective_score.std().item() + 1e-4
    # The uncentered weighted score drifts; subtracting the analytic center removes it.
    assert abs(effective_score.mean().item()) > 0.015
    centered = effective_score - center_grad
    assert abs(centered.mean().item()) < tolerance * centered.std().item() + 1e-4


def test_mean_log_ratio_and_kappa_match_the_gaussian_densities():
    """An algebra slip in the cancellation-free log ratio or in kappa shows up against torch's Normal."""
    generator = torch.Generator().manual_seed(29)
    rollout_prev_mean = torch.randn((4, 2, 3), generator=generator, dtype=torch.float64)
    prev_mean = rollout_prev_mean + 0.3 * torch.randn(
        rollout_prev_mean.shape, generator=generator, dtype=torch.float64
    )
    variance = torch.tensor([0.1, 0.2, 0.7, 1.4], dtype=torch.float64).reshape(4, 1, 1)
    next_latents = rollout_prev_mean + variance.sqrt() * torch.randn(
        rollout_prev_mean.shape, generator=generator, dtype=torch.float64
    )
    p = torch.distributions.Normal(prev_mean, variance.sqrt())
    q = torch.distributions.Normal(rollout_prev_mean, variance.sqrt())
    expected = (p.log_prob(next_latents) - q.log_prob(next_latents)).flatten(1).mean(1)
    actual = gaussian_mean_log_ratio(next_latents, prev_mean, rollout_prev_mean, variance)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    expected_kappa = ((prev_mean - rollout_prev_mean).square() / variance).flatten(1).sum(1)
    torch.testing.assert_close(gaussian_kappa(prev_mean, rollout_prev_mean, variance), expected_kappa)
