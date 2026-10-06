"""Flow-GRPO score centering: cancelling the train/rollout drift in the policy gradient.

    rollout draws x' ~ N(mu_q, s²), trainer scores log N(x' | mu_theta, s²)
    mu_q != mu_theta --> E_q[score] = grad log p(mu_q) != 0 --> drift = E[A] * E_q[score]
    --diffusion-score-centering subtracts E_q[score] --> a constant advantage moves nothing
    mu_q == mu_theta --> exact no-op; clip-masked pairs stay masked; logged loss unchanged
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils.loss_hub.flow_grpo import flow_grpo_loss_formula
from miles.backends.fsdp_utils.sde_step_backend import CpsSdeStepBackend, DiffusersSdeStepBackend

NOISE_LEVEL = 0.7


class _FakeScheduler:
    def __init__(self, num_steps=8):
        self.sigmas = torch.linspace(1.0, 0.0, num_steps + 1)

    def index_for_timestep(self, t):
        return int(torch.argmin((self.sigmas[:-1] - t).abs()))


class _RecordingMetrics:
    def __init__(self):
        self.means = {}

    def emit_mean(self, key, total, count):
        self.means[key] = float(total) / count

    def emit_max(self, key, value):
        pass


def _diffusers_case():
    sched = _FakeScheduler()
    return DiffusersSdeStepBackend(sched), sched.sigmas[[2, 5, 6]], sched.sigmas[[3, 6, 7]]


def _cps_case():
    timesteps, next_timesteps = torch.tensor([700.0, 400.0, 200.0]), torch.tensor([600.0, 300.0, 100.0])
    return CpsSdeStepBackend(None, sde_timestep_divisor=1000.0), timesteps, next_timesteps


BACKEND_CASES = {"diffusers": _diffusers_case, "cps": _cps_case}


class _Transition:
    """Three pairs at a fixed trainer velocity; the rollout's mean is offset by ``offset_scale`` noise stds."""

    def __init__(self, case: str, offset_scale: float):
        torch.manual_seed(0)
        self.backend, self.timesteps, self.next_timesteps = BACKEND_CASES[case]()
        self.latents = torch.randn(3, 4, 6)
        self.velocity = torch.randn(3, 4, 6)
        with torch.no_grad():
            score = self.backend.sde_step_logprob(
                self.velocity,
                self.timesteps,
                self.next_timesteps,
                self.latents,
                prev_sample=self.latents,
                noise_level=NOISE_LEVEL,
            )
        self.trainer_prev_mean, self.noise_std = score.prev_mean, score.noise_std
        self.rollout_prev_mean = self.trainer_prev_mean + offset_scale * self.noise_std * torch.randn(3, 4, 6)

    def rollout_sample(self, noise: torch.Tensor) -> torch.Tensor:
        return self.rollout_prev_mean + self.noise_std * noise

    def loss_and_grad(
        self,
        next_latents: torch.Tensor,
        *,
        advantage: torch.Tensor,
        score_centering: bool,
        log_prob_old: torch.Tensor | None = None,
        rollout_prev_means: list[torch.Tensor] | None = None,
        with_debug_tensors: bool = True,
    ):
        velocity = self.velocity.clone().requires_grad_(True)
        args = Namespace(
            diffusion_clip_range=1e-4,
            diffusion_noise_level=NOISE_LEVEL,
            diffusion_kl_beta=0.0,
            diffusion_score_centering=score_centering,
        )
        ctx = SimpleNamespace(
            args=args, sde_backend=self.backend, models={"transformer": None}, device=torch.device("cpu")
        )
        prepared = SimpleNamespace(
            latents=self.latents,
            timesteps=self.timesteps,
            advantage=advantage,
            component_name="transformer",
            extras={"next_latents": next_latents, "next_timesteps": self.next_timesteps, "log_prob_old": log_prob_old},
        )
        means = list(self.rollout_prev_mean) if rollout_prev_means is None else rollout_prev_means
        batch = (
            [{"rollout_debug_tensors": {"rollout_step_prev_sample_mean": mean}} for mean in means]
            if with_debug_tensors
            else [{} for _ in means]
        )
        metrics = _RecordingMetrics()
        loss = flow_grpo_loss_formula(
            ctx,
            batch,
            prepared,
            new_pred=velocity,
            old_pred=None,
            ref_pred=None,
            metrics=metrics,
            old_log_prob_from_new=log_prob_old is None,
        )
        (grad,) = torch.autograd.grad(loss, velocity)
        return loss.detach(), grad, metrics


@pytest.mark.parametrize("case", sorted(BACKEND_CASES))
class TestScoreCentering:
    def test_constant_advantage_moves_nothing_in_expectation(self, case):
        # the score is linear in x', so the pair x' = mu_q ± s·eps averages it exactly to E_q[score]
        transition = _Transition(case, offset_scale=0.3)
        advantage = torch.ones(3)
        noise = torch.randn(3, 4, 6)

        def summed_grad(score_centering):
            _, grad_pos, _ = transition.loss_and_grad(
                transition.rollout_sample(noise), advantage=advantage, score_centering=score_centering
            )
            _, grad_neg, _ = transition.loss_and_grad(
                transition.rollout_sample(-noise), advantage=advantage, score_centering=score_centering
            )
            return grad_pos + grad_neg

        velocity = transition.velocity.clone().requires_grad_(True)
        trainer_prev_mean = transition.backend.sde_step_logprob(
            velocity,
            transition.timesteps,
            transition.next_timesteps,
            transition.latents,
            prev_sample=transition.latents,
            noise_level=NOISE_LEVEL,
        ).prev_mean
        expected_score = transition.backend.expected_log_prob(
            transition.rollout_prev_mean, trainer_prev_mean, transition.noise_std
        ).sum()
        (drift,) = torch.autograd.grad(-2.0 * expected_score, velocity)
        assert drift.abs().max() > 0

        tol = 1e-5 * float(drift.abs().max())
        torch.testing.assert_close(summed_grad(False), drift, rtol=0.0, atol=tol)
        torch.testing.assert_close(summed_grad(True), torch.zeros_like(drift), rtol=0.0, atol=tol)

    def test_exact_noop_when_rollout_matches_trainer(self, case):
        transition = _Transition(case, offset_scale=0.0)
        next_latents = transition.rollout_sample(torch.randn(3, 4, 6))
        advantage = torch.tensor([1.0, -0.5, 2.0])
        loss_off, grad_off, _ = transition.loss_and_grad(next_latents, advantage=advantage, score_centering=False)
        loss_on, grad_on, _ = transition.loss_and_grad(next_latents, advantage=advantage, score_centering=True)
        torch.testing.assert_close(loss_on, loss_off, rtol=0.0, atol=0.0)
        torch.testing.assert_close(grad_on, grad_off, rtol=0.0, atol=0.0)

    def test_logged_loss_is_unchanged_under_mismatch(self, case):
        transition = _Transition(case, offset_scale=0.3)
        next_latents = transition.rollout_sample(torch.randn(3, 4, 6))
        advantage = torch.tensor([1.0, -0.5, 2.0])
        loss_off, grad_off, _ = transition.loss_and_grad(next_latents, advantage=advantage, score_centering=False)
        loss_on, grad_on, _ = transition.loss_and_grad(next_latents, advantage=advantage, score_centering=True)
        torch.testing.assert_close(loss_on, loss_off, rtol=0.0, atol=0.0)
        assert not torch.equal(grad_on, grad_off)

    def test_clip_masked_pairs_get_no_correction(self, case):
        # A > 0 with ratio e > 1 + clip: PPO zeroes these pairs, and the correction must too.
        transition = _Transition(case, offset_scale=0.3)
        next_latents = transition.rollout_sample(torch.randn(3, 4, 6))
        with torch.no_grad():
            log_prob_new = transition.backend.sde_step_logprob(
                transition.velocity,
                transition.timesteps,
                transition.next_timesteps,
                transition.latents,
                prev_sample=next_latents,
                noise_level=NOISE_LEVEL,
            ).log_prob
        _, grad, _ = transition.loss_and_grad(
            next_latents, advantage=torch.ones(3), score_centering=True, log_prob_old=log_prob_new - 1.0
        )
        torch.testing.assert_close(grad, torch.zeros_like(grad), rtol=0.0, atol=0.0)

    def test_prev_mean_diff_is_reported_in_noise_std_units(self, case):
        transition = _Transition(case, offset_scale=0.0)
        transition.rollout_prev_mean = transition.trainer_prev_mean + 0.5 * transition.noise_std
        _, _, metrics = transition.loss_and_grad(
            transition.rollout_sample(torch.randn(3, 4, 6)), advantage=torch.ones(3), score_centering=False
        )
        assert metrics.means["prev_mean_diff_over_noise_std"] == pytest.approx(0.5, rel=1e-5)


class TestScoreCenteringInputs:
    def test_requires_rollout_debug_means(self):
        transition = _Transition("diffusers", offset_scale=0.3)
        next_latents = transition.rollout_sample(torch.randn(3, 4, 6))
        _, _, metrics = transition.loss_and_grad(
            next_latents, advantage=torch.ones(3), score_centering=False, with_debug_tensors=False
        )
        assert "prev_mean_diff_over_noise_std" not in metrics.means
        with pytest.raises(ValueError, match="--diffusion-debug-mode"):
            transition.loss_and_grad(
                next_latents, advantage=torch.ones(3), score_centering=True, with_debug_tensors=False
            )

    def test_mean_layout_must_match_only_when_centering(self):
        # only the correction needs the trainer's exact layout; logging flattens per pair
        transition = _Transition("diffusers", offset_scale=0.3)
        next_latents = transition.rollout_sample(torch.randn(3, 4, 6))
        flat = [mean.reshape(-1) for mean in transition.rollout_prev_mean]
        _, _, metrics = transition.loss_and_grad(
            next_latents, advantage=torch.ones(3), score_centering=False, rollout_prev_means=flat
        )
        assert "prev_mean_diff_over_noise_std" in metrics.means
        with pytest.raises(ValueError, match="laid out unlike"):
            transition.loss_and_grad(
                next_latents, advantage=torch.ones(3), score_centering=True, rollout_prev_means=flat
            )
