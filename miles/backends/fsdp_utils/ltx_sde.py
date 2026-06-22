"""LTX-specific SDE dynamics (schedule-decoupled CPS / Flow-SDE / ODE paths)."""

from __future__ import annotations

import math
from typing import Optional

import torch

CANONICAL_DYNAMICS_TYPES = ("sde", "flow_sde", "cps", "ode", "dance_sde")


def normalize_dynamics_type(name: str) -> str:
    """Map a dynamics-type alias (CLI / legacy casing) to its canonical name."""
    key = str(name).strip().lower().replace("-", "_")
    if key not in CANONICAL_DYNAMICS_TYPES:
        raise ValueError(
            f"Unknown dynamics_type {name!r}; expected one of {CANONICAL_DYNAMICS_TYPES}"
        )
    return key


def sde_step_with_logprob_dynamics(
    model_output: torch.FloatTensor,
    sigma: torch.FloatTensor,
    sigma_next: torch.FloatTensor,
    sample: torch.FloatTensor,
    sigmas: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    sigma_min_override: Optional[float] = None,
    noise_level: float = 0.8,
    dynamics_type: str = "flow_sde",
):
    """Schedule-decoupled SDE step with log-prob for LTX-2.3 and similar models."""
    dynamics_type = normalize_dynamics_type(dynamics_type)
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    ndim = sample.ndim
    sigma_view = sigma.float()
    sigma_next_view = sigma_next.float()
    while sigma_view.ndim < ndim:
        sigma_view = sigma_view.unsqueeze(-1)
    while sigma_next_view.ndim < ndim:
        sigma_next_view = sigma_next_view.unsqueeze(-1)

    dt = sigma_next_view - sigma_view

    sigma_max = sigmas[0].float().item()
    if sigma_min_override is not None:
        sigma_min = sigma_min_override
    else:
        sigma_min = max(sigmas[-2].float().item(), 1e-4) if len(sigmas) > 1 else 1e-4

    if dynamics_type == "ode":
        prev_sample_mean = sample + dt * model_output
        std_dev_t = torch.zeros_like(sigma_view)
        if prev_sample is None:
            prev_sample = prev_sample_mean
        log_prob = torch.zeros(sample.shape[0], dtype=sample.dtype, device=sample.device)

    elif dynamics_type == "flow_sde":
        std_dev_t = (sigma_min + (sigma_max - sigma_min) * sigma_view) * noise_level
        sigma_safe = torch.clamp(sigma_view, min=1e-8)

        drift_sample = 1.0 + std_dev_t**2 / (2.0 * sigma_safe) * dt
        drift_model = (1.0 + std_dev_t**2 * (1.0 - sigma_view) / (2.0 * sigma_safe)) * dt
        prev_sample_mean = sample * drift_sample + model_output * drift_model

        noise_scale = std_dev_t * torch.sqrt(torch.clamp(-dt, min=1e-12))

        if prev_sample is None:
            if deterministic:
                prev_sample = sample + dt * model_output
            else:
                variance_noise = torch.randn(
                    sample.shape, generator=generator, device=sample.device, dtype=sample.dtype,
                )
                prev_sample = prev_sample_mean + noise_scale * variance_noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2.0 * noise_scale**2 + 1e-12)
            - torch.log(noise_scale + 1e-12)
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    elif dynamics_type == "cps":
        angle = torch.tensor(noise_level, dtype=sigma_next_view.dtype, device=sigma_next_view.device) * torch.pi / 2
        std_dev_t = sigma_next_view * torch.sin(angle)

        x0 = sample - sigma_view * model_output
        x1 = sample + model_output * (1.0 - sigma_view)
        sqrt_term = torch.sqrt(torch.clamp(sigma_next_view**2 - std_dev_t**2, min=1e-12))
        prev_sample_mean = x0 * (1.0 - sigma_next_view) + x1 * sqrt_term

        if prev_sample is None:
            if deterministic:
                prev_sample = prev_sample_mean
            else:
                variance_noise = torch.randn(
                    sample.shape, generator=generator, device=sample.device, dtype=sample.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * variance_noise

        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    elif dynamics_type == "dance_sde":
        sigma_safe = torch.clamp(sigma_view, min=1e-8)
        x0_pred = sample - sigma_safe * model_output
        std_dev_t = torch.as_tensor(noise_level, dtype=sample.dtype, device=sample.device)
        log_term = 0.5 * noise_level**2 * (sample - x0_pred * (1.0 - sigma_view)) / (sigma_safe**2)
        prev_sample_mean = sample + (model_output + log_term) * dt
        noise_scale = std_dev_t * torch.sqrt(torch.clamp(-dt, min=1e-12))

        if prev_sample is None:
            if deterministic:
                prev_sample = sample + dt * model_output
            else:
                variance_noise = torch.randn(
                    sample.shape, generator=generator, device=sample.device, dtype=sample.dtype,
                )
                prev_sample = prev_sample_mean + noise_scale * variance_noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2.0 * noise_scale**2 + 1e-12)
            - torch.log(noise_scale + 1e-12)
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    else:
        raise ValueError(
            f"dynamics_type {dynamics_type!r} is not supported by the "
            "schedule-decoupled path; use flow_sde / cps / ode / dance_sde."
        )

    dt_sqrt = torch.sqrt(torch.clamp(-dt, min=1e-12))
    return prev_sample, log_prob, prev_sample_mean, std_dev_t, dt_sqrt
