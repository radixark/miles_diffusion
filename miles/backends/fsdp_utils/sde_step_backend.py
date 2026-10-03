"""SDE step backend: scores a recorded (x_t -> x_{t+1}) transition for training.

Layered to mirror sgl-d's ``flow_sde_sampling`` stages, so the dynamics kernel
can eventually be shared with the rollout engine (SAMPLE mode draws noise
around the same mean; SCORE mode — this trainer — scores the recorded
residual):

  - ``prev_sample_mean_and_std``: the deterministic dynamics kernel
  - ``log_prob``: Gaussian scoring of ``prev_sample - prev_mean``
  - ``sde_step_logprob``: the SCORE-mode composition the trainer calls

Every transition carries the sigmas the engine stepped with (sigma, next_sigma,
sigma_max), so scoring needs neither a scheduler nor the timestep values.

Selected via ``--sde-step-backend-path`` (miles custom-function style); a
model family with its own dynamics overrides the two layers, not the trainer.
"""

from __future__ import annotations

import abc
import math

import torch


class SdeStepBackend(abc.ABC):
    @abc.abstractmethod
    def prev_sample_mean_and_std(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_prev: torch.Tensor,
        *,
        sigma_max: torch.Tensor,
        noise_level: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(prev_mean, noise_std, std_dev_t)``.

        ``noise_std`` is the Gaussian std of the transition (drives log_prob);
        ``std_dev_t`` is the diffusion-scale factor reported to the trainer
        (KL denominator) — flow-SDE keeps them distinct, CPS has them equal.
        """

    def log_prob(
        self,
        prev_sample: torch.Tensor,
        prev_mean: torch.Tensor,
        noise_std: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian log-density of the recorded transition, mean over non-batch dims."""
        log_prob = (
            -((prev_sample.detach() - prev_mean) ** 2) / (2 * noise_std**2)
            - torch.log(noise_std)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    def sde_step_logprob(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        *,
        prev_sample: torch.Tensor,
        noise_level: float,
        sigmas: torch.Tensor,
        next_sigmas: torch.Tensor,
        sigma_max: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model_output = model_output.float()
        sample = sample.float()
        prev_sample = prev_sample.float()
        view = (-1, *([1] * (sample.ndim - 1)))
        prev_mean, noise_std, std_dev_t = self.prev_sample_mean_and_std(
            model_output,
            sample,
            sigmas.view(view),
            next_sigmas.view(view),
            sigma_max=sigma_max.view(view),
            noise_level=noise_level,
        )
        return prev_sample, self.log_prob(prev_sample, prev_mean, noise_std), prev_mean, std_dev_t


class DiffusersSdeStepBackend(SdeStepBackend):
    """Flow-matching SDE (matches sgl-d flow_sde_sampling rollout_sde_type="sde"); current default."""

    def prev_sample_mean_and_std(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_prev: torch.Tensor,
        *,
        sigma_max: torch.Tensor,
        noise_level: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = sigma_prev - sigma

        # sigma_max (the rollout grid's second sigma, sigmas[1]) stands in near sigma == 1, where
        # sigma / (1 - sigma) diverges; isclose, not ==, is the engine's test.
        at_one = torch.isclose(sigma, sigma.new_tensor(1.0))
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(at_one, sigma_max, sigma))) * noise_level
        prev_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )
        return prev_mean, std_dev_t * torch.sqrt(-1 * dt), std_dev_t


class CpsSdeStepBackend(SdeStepBackend):
    """CPS dynamics (matches sgl-d flow_sde_sampling rollout_sde_type="cps")."""

    def prev_sample_mean_and_std(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_prev: torch.Tensor,
        *,
        sigma_max: torch.Tensor,
        noise_level: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)
        pred_original = sample - sigma * model_output
        noise_estimate = sample + model_output * (1.0 - sigma)
        prev_mean = pred_original * (1.0 - sigma_prev) + noise_estimate * torch.sqrt(
            torch.clamp(sigma_prev**2 - std_dev_t**2, min=1e-12)
        )
        return prev_mean, std_dev_t, std_dev_t

    def log_prob(
        self,
        prev_sample: torch.Tensor,
        prev_mean: torch.Tensor,
        noise_std: torch.Tensor,
    ) -> torch.Tensor:
        # Drops constants — pairs with rollout_log_prob_no_const=True on the engine side.
        log_prob = -((prev_sample.detach() - prev_mean) ** 2)
        return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
