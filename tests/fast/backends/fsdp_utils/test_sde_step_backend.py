from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import math

import torch

from miles.backends.fsdp_utils.sde_step_backend import CpsSdeStepBackend, DiffusersSdeStepBackend
from miles.utils.sde_log_prob import sde_step_with_logprob


class _FakeScheduler:
    def __init__(self, num_steps=8):
        self.sigmas = torch.linspace(1.0, 0.0, num_steps + 1)

    def index_for_timestep(self, t):
        return int(torch.argmin((self.sigmas[:-1] - t).abs()))


class TestDiffusersSdeStepBackend:
    # The layered backend (σ resolution + mean/std kernel + Gaussian log_prob) must
    # reproduce the monolithic sde_step_with_logprob bit-for-bit — the refactor contract.
    def test_matches_monolithic_reference(self):
        torch.manual_seed(0)
        sched = _FakeScheduler()
        t = sched.sigmas[[2, 5]]  # per-pair timestep
        nt = sched.sigmas[[3, 6]]  # per-pair next timestep (ignored by the diffusers +1 path)
        v, x, nxt = (torch.randn(2, 4, 6) for _ in range(3))

        backend = DiffusersSdeStepBackend(sched)
        got = backend.sde_step_logprob(v, t, nt, x, prev_sample=nxt, noise_level=0.7)
        want = sde_step_with_logprob(sched, v, t, x, prev_sample=nxt, noise_level=0.7)
        for g, w in zip(got, want, strict=True):
            torch.testing.assert_close(g, w, rtol=0.0, atol=0.0)
        assert got[1].shape == (2,)


class TestCpsSdeStepBackend:
    # LTX uses CPS dynamics: σ = timestep/divisor straight from the rollout values
    # (no scheduler), and the CPS mean/std kernel must match sgl-d's rollout_sde_type="cps".
    def test_cps_kernel_matches_reference(self):
        torch.manual_seed(0)
        sb = CpsSdeStepBackend(None, sde_timestep_divisor=1000.0)  # no scheduler / no config
        t = torch.tensor([700.0, 300.0])
        nt = torch.tensor([600.0, 0.0])  # terminal σ_next = 0
        x, v, nxt = (torch.randn(2, 128, 8) for _ in range(3))
        _, log_prob, mean, std = sb.sde_step_logprob(v, t, nt, x, prev_sample=nxt, noise_level=0.8)

        sigma, sigma_next = (t / 1000).view(-1, 1, 1), (nt / 1000).view(-1, 1, 1)
        std_t = sigma_next * math.sin(0.8 * math.pi / 2)
        expected_mean = (x - sigma * v) * (1 - sigma_next) + (x + v * (1 - sigma)) * torch.sqrt(
            torch.clamp(sigma_next**2 - std_t**2, min=1e-12)
        )
        torch.testing.assert_close(mean, expected_mean, rtol=0.0, atol=0.0)
        # no-const log_prob = -(prev - mean)^2 mean over non-batch dims
        torch.testing.assert_close(log_prob, (-((nxt - expected_mean) ** 2)).mean(dim=(1, 2)), rtol=0.0, atol=0.0)
        assert log_prob.shape == (2,)
