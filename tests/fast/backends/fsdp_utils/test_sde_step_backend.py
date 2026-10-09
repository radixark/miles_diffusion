"""SDE step backends score one recorded rollout transition per train pair.

    rollout grid (one sample)     sigmas = [1.0, s1, s2, ..., 0.0]      sigma_max = s1
                                            ▲    ▲
    train pair at step i          sigma = sigmas[i], next_sigma = sigmas[i + 1]

    Diffusers: the pair's carried (sigma, next_sigma, sigma_max) must score bit-identically to
               the old path that looked sigma up in a scheduler by timestep value, and a first
               sigma within isclose of 1 (FlowUniPC: 0.999993) takes sigma_max as the engine does.
    CPS:       the mean/std kernel over the carried sigmas must match sgl-d's rollout_sde_type="cps".
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import math

import torch

from miles.backends.fsdp_utils.sde_step_backend import CpsSdeStepBackend, DiffusersSdeStepBackend
from miles.utils.sde_log_prob import sde_step_with_logprob


class _DiffusersFlowScheduler:
    """The diffusers FlowMatch lookup the trainer used before pairs carried their sigmas."""

    def __init__(self, timesteps, sigmas):
        self.timesteps = timesteps
        self.sigmas = sigmas

    def index_for_timestep(self, timestep):
        indices = (self.timesteps == timestep).nonzero()
        return indices[1 if len(indices) > 1 else 0].item()


class TestDiffusersSdeStepBackend:
    def test_carried_sigmas_match_scheduler_lookup_bitwise(self):
        torch.manual_seed(0)
        shift, num_steps = 3.0, 10
        base = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
        sigmas = torch.cat([shift * base / (1 + (shift - 1) * base), torch.zeros(1)])
        timesteps = sigmas * 1000
        step = torch.tensor([0, 4, 9])  # first step (sigma == 1), a middle step, the step into sigma == 0
        v, x, nxt = (torch.randn(3, 4, 6) for _ in range(3))

        got = DiffusersSdeStepBackend().sde_step_logprob(
            v,
            x,
            prev_sample=nxt,
            noise_level=0.7,
            sigmas=sigmas[step],
            next_sigmas=sigmas[step + 1],
            sigma_max=sigmas[1].expand(3),
        )
        want = sde_step_with_logprob(
            _DiffusersFlowScheduler(timesteps, sigmas), v, timesteps[step], x, prev_sample=nxt, noise_level=0.7
        )
        for g, w in zip(got, want, strict=True):
            torch.testing.assert_close(g, w, rtol=0.0, atol=0.0)
        assert got[1].shape == (3,)

    def test_sigma_just_below_one_takes_sigma_max_like_the_engine(self):
        # FlowUniPC at shift 12 starts at 0.999993, not 1.0; sgl-d's isclose still swaps in
        # sigma_max there, where an exact == would leave sigma / (1 - sigma) ~ 1.4e5.
        sigmas = torch.tensor([0.999993, 0.990818, 0.0])
        x = torch.zeros(1, 4)
        _, _, _, std_dev_t = DiffusersSdeStepBackend().sde_step_logprob(
            torch.zeros(1, 4),
            x,
            prev_sample=x,
            noise_level=0.7,
            sigmas=sigmas[:1],
            next_sigmas=sigmas[1:2],
            sigma_max=sigmas[1:2],
        )
        expected = torch.sqrt(sigmas[0] / (1 - sigmas[1])) * 0.7
        torch.testing.assert_close(std_dev_t.view(()), expected, rtol=0.0, atol=0.0)


class TestCpsSdeStepBackend:
    def test_cps_kernel_matches_reference(self):
        torch.manual_seed(0)
        sigmas = torch.tensor([0.7, 0.3])
        next_sigmas = torch.tensor([0.6, 0.0])  # terminal σ_next = 0
        x, v, nxt = (torch.randn(2, 128, 8) for _ in range(3))
        _, log_prob, mean, std = CpsSdeStepBackend().sde_step_logprob(
            v,
            x,
            prev_sample=nxt,
            noise_level=0.8,
            sigmas=sigmas,
            next_sigmas=next_sigmas,
            sigma_max=torch.full((2,), float("nan")),  # CPS has no sigma == 1 special case
        )

        sigma, sigma_next = sigmas.view(-1, 1, 1), next_sigmas.view(-1, 1, 1)
        std_t = sigma_next * math.sin(0.8 * math.pi / 2)
        expected_mean = (x - sigma * v) * (1 - sigma_next) + (x + v * (1 - sigma)) * torch.sqrt(
            torch.clamp(sigma_next**2 - std_t**2, min=1e-12)
        )
        torch.testing.assert_close(mean, expected_mean, rtol=0.0, atol=0.0)
        # no-const log_prob = -(prev - mean)^2 mean over non-batch dims
        torch.testing.assert_close(log_prob, (-((nxt - expected_mean) ** 2)).mean(dim=(1, 2)), rtol=0.0, atol=0.0)
        assert log_prob.shape == (2,)
