"""SDE step with log probability for flow matching schedulers.

Adapted from flow_grpo/diffusers_patch/sd3_sde_with_logprob.py.
"""

from __future__ import annotations

import math

import torch


def _broadcast_sigma(sigma: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    sigma_view = sigma.float()
    while sigma_view.ndim < sample.ndim:
        sigma_view = sigma_view.unsqueeze(-1)
    return sigma_view


def sde_step_with_logprob(
    scheduler,
    model_output: torch.FloatTensor,
    timestep: float | torch.FloatTensor,
    sample: torch.FloatTensor,
    prev_sample: torch.FloatTensor,
    noise_level: float = 0.7,
    *,
    sde_type: str = "sde",
    sigma: torch.FloatTensor | None = None,
    sigma_prev: torch.FloatTensor | None = None,
):
    """Compute the log probability of `prev_sample` under one reverse-SDE step.

    Args:
        scheduler: A flow-matching scheduler with `sigmas` and `index_for_timestep`.
            Ignored when ``sigma`` and ``sigma_prev`` are both provided.
        model_output: Predicted velocity from DiT.
        timestep: Current timestep(s), shape (B,). Used only for scheduler lookup.
        sample: Current latent.
        prev_sample: Recorded next-step latent to score under the SDE.
        noise_level: SDE noise scaling factor (eta).
        sde_type: ``"sde"`` (default, SD3 flow-SDE) or ``"cps"``.
        sigma: Optional current sigma(s), shape (B,). Bypasses scheduler lookup.
        sigma_prev: Optional next sigma(s), shape (B,). Required with ``sigma``.

    Returns:
        (prev_sample, log_prob, prev_sample_mean, std_dev_t)
    """
    model_output = model_output.float()
    sample = sample.float()
    prev_sample = prev_sample.float()
    sde_type = str(sde_type).strip().lower()

    if sigma is not None and sigma_prev is not None:
        sigma = _broadcast_sigma(sigma, sample)
        sigma_prev = _broadcast_sigma(sigma_prev, sample)
    else:
        step_index = [scheduler.index_for_timestep(t) for t in timestep]
        prev_step_index = [s + 1 for s in step_index]
        sigma = scheduler.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
        sigma_prev = scheduler.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))

    if sde_type == "cps":
        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)
        pred_original_sample = sample - sigma * model_output
        noise_estimate = sample + model_output * (1.0 - sigma)
        prev_sample_mean = pred_original_sample * (1.0 - sigma_prev) + noise_estimate * torch.sqrt(
            torch.clamp(sigma_prev**2 - std_dev_t**2, min=1e-12)
        )
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
    elif sde_type == "sde":
        sigma_max = scheduler.sigmas[1].item()
        dt = sigma_prev - sigma

        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level

        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
            - torch.log(std_dev_t * torch.sqrt(-1 * dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
    else:
        raise ValueError(f"Unsupported sde_type {sde_type!r}; expected 'sde' or 'cps'.")

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob, prev_sample_mean, std_dev_t
