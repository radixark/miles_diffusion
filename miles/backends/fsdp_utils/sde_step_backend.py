"""SDE step backend: scores a recorded (x_t -> x_{t+1}) transition for training.

Layered to mirror sgl-d's ``flow_sde_sampling`` stages, so the dynamics kernel
can eventually be shared with the rollout engine (SAMPLE mode draws noise
around the same mean; SCORE mode — this trainer — scores the recorded
residual):

  - ``prev_sample_mean_and_std``: the deterministic dynamics kernel
  - ``log_prob``: Gaussian scoring of ``prev_sample - prev_mean``
  - ``sde_step_logprob``: the SCORE-mode composition the trainer calls

Selected via ``--sde-step-backend-path`` (miles custom-function style); a
model family with its own dynamics overrides the two layers, not the trainer.
"""

from __future__ import annotations

import abc
import math

import torch


class SdeStepBackend(abc.ABC):
    def __init__(self, scheduler=None, *, sde_timestep_divisor: float = 1.0):
        # Primitive params only (no train pipeline config) so the rollout process — which has
        # no train pipeline — can load and construct the same backend for shared stepping.
        # The dynamics is encoded by the concrete subclass, not passed in.
        self.scheduler = scheduler
        self.sde_timestep_divisor = sde_timestep_divisor

    @abc.abstractmethod
    def resolve_sigmas(
        self, timesteps: torch.Tensor, next_timesteps: torch.Tensor, *, ndim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map the pair's own (timestep, next_timestep) — the actual rollout values, not a
        positional index — to (sigma, sigma_next), broadcast to ndim. Families with a linear
        timestep<->σ relation resolve from the values directly, off the scheduler."""

    @abc.abstractmethod
    def prev_sample_mean_and_std(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_prev: torch.Tensor,
        *,
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
        timesteps: torch.Tensor,
        next_timesteps: torch.Tensor,
        sample: torch.Tensor,
        *,
        prev_sample: torch.Tensor,
        noise_level: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model_output = model_output.float()
        sample = sample.float()
        prev_sample = prev_sample.float()
        sigma, sigma_prev = self.resolve_sigmas(timesteps, next_timesteps, ndim=sample.ndim)
        prev_mean, noise_std, std_dev_t = self.prev_sample_mean_and_std(
            model_output, sample, sigma, sigma_prev, noise_level=noise_level
        )
        return prev_sample, self.log_prob(prev_sample, prev_mean, noise_std), prev_mean, std_dev_t


class DiffusersSdeStepBackend(SdeStepBackend):
    """Flow-matching SDE over diffusers scheduler sigmas (current default)."""

    def resolve_sigmas(
        self, timesteps: torch.Tensor, next_timesteps: torch.Tensor, *, ndim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Diffusers σ needs the scheduler's (shifted) timestep->σ map; look it up by the
        # actual rollout timestep value, then take the neighbouring σ (scheduler is filled
        # from the rollout snapshot, so +1 lands on the recorded next step / terminal 0).
        step_index = [self.scheduler.index_for_timestep(t) for t in timesteps]
        prev_step_index = [s + 1 for s in step_index]
        view = (-1, *([1] * (ndim - 1)))
        return self.scheduler.sigmas[step_index].view(view), self.scheduler.sigmas[prev_step_index].view(view)

    def prev_sample_mean_and_std(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_prev: torch.Tensor,
        *,
        noise_level: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma_max = self.scheduler.sigmas[1].item()
        dt = sigma_prev - sigma

        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
        prev_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )
        return prev_mean, std_dev_t * torch.sqrt(-1 * dt), std_dev_t


class CpsSdeStepBackend(SdeStepBackend):
    """CPS dynamics; σ = timestep/divisor resolved straight from the rollout values."""

    def resolve_sigmas(
        self, timesteps: torch.Tensor, next_timesteps: torch.Tensor, *, ndim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Linear timestep<->σ (timesteps are σ×divisor), so σ/σ_next come directly from the
        # carried rollout values — no scheduler, no positional-alignment assumption.
        divisor = float(self.sde_timestep_divisor)
        view = (-1, *([1] * (ndim - 1)))
        return (timesteps.float() / divisor).view(view), (next_timesteps.float() / divisor).view(view)

    def prev_sample_mean_and_std(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_prev: torch.Tensor,
        *,
        noise_level: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # CPS kernel (matches sgl-d flow_sde_sampling rollout_sde_type="cps").
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
