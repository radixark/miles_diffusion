"""Train-vs-rollout log-prob drift is also logged per rollout resolution.

    --diffusion-resolution-buckets 512x512,672x384  ─► schema declares ..._res_512x512, ..._res_672x384
    a micro-batch of 672x384 pairs                  ─► feeds ..._res_672x384 (and the overall metric)
    a resolution the buckets do not list            ─► only the overall metric (custom strategies)
    a fixed-resolution run                          ─► no per-resolution keys at all
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils.loss_hub.flow_grpo import flow_grpo_loss_formula
from miles.backends.fsdp_utils.metrics import new_metric_buffer
from miles.rollout.resolution_strategy_hub import declared_resolutions

BUCKETS = "512x512,672x384"
OVERALL = "log_prob_mean_abs_diff"


class _FixedLogProbBackend:
    """sde_step_logprob stub: the 'new' log-prob is whatever the test put in the prediction."""

    def sde_step_logprob(self, pred, latents, *, prev_sample, noise_level, sigmas, next_sigmas, sigma_max):
        log_prob = pred.reshape(pred.shape[0], -1)[:, 0]
        return None, log_prob, pred, torch.ones_like(log_prob)


def _strategy_args(**overrides):
    values = dict(
        diffusion_resolution_strategy_path="miles.rollout.resolution_strategy_hub.weighted_buckets",
        diffusion_resolution_buckets=BUCKETS,
        diffusion_resolution_weights=None,
    )
    values.update(overrides)
    return Namespace(**values)


def _run_loss(metrics, resolution, log_prob_new, log_prob_old):
    bsz = len(log_prob_new)
    ctx = SimpleNamespace(
        args=Namespace(diffusion_clip_range=1e-4, diffusion_noise_level=1.0, diffusion_kl_beta=0.0),
        sde_backend=_FixedLogProbBackend(),
        models={"transformer": None},
        device=torch.device("cpu"),
    )
    prepared = SimpleNamespace(
        latents={"visual": torch.zeros(bsz, 1)},
        advantage=torch.ones(bsz),
        component_name="transformer",
        extras={
            "next_latents": torch.zeros(bsz, 1),
            "log_prob_old": torch.tensor(log_prob_old),
            "sigmas": torch.zeros(bsz),
            "next_sigmas": torch.zeros(bsz),
            "sigma_max": torch.ones(bsz),
        },
    )
    batch = [{"resolution": resolution} for _ in range(bsz)]
    flow_grpo_loss_formula(
        ctx,
        batch,
        prepared,
        new_pred={"visual": torch.tensor(log_prob_new).reshape(bsz, 1)},
        old_pred=None,
        ref_pred=None,
        metrics=metrics,
    )


def _buffer(args):
    return new_metric_buffer(None, torch.device("cpu"), ["transformer"], resolutions=declared_resolutions(args))


def test_buckets_are_declared_height_first():
    assert declared_resolutions(_strategy_args()) == ((512, 512), (672, 384))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(diffusion_resolution_strategy_path=None),  # fixed resolution
        dict(diffusion_resolution_buckets=None),  # custom strategy without a bucket list
    ],
)
def test_nothing_is_declared_without_a_bucket_list(overrides):
    assert declared_resolutions(_strategy_args(**overrides)) == ()


def test_drift_is_split_by_resolution():
    metrics = _buffer(_strategy_args())
    _run_loss(metrics, (672, 384), log_prob_new=[1.0, 1.0], log_prob_old=[1.5, 0.5])  # |diff| 0.5 each
    _run_loss(metrics, (512, 512), log_prob_new=[2.0], log_prob_old=[1.9])  # |diff| 0.1

    out = metrics.reduce()

    assert out["log_prob_mean_abs_diff_res_672x384"] == pytest.approx(0.5)
    assert out["log_prob_mean_abs_diff_res_512x512"] == pytest.approx(0.1)
    assert out[OVERALL] == pytest.approx((0.5 * 2 + 0.1) / 3)


def test_an_undeclared_resolution_only_feeds_the_overall_metric():
    metrics = _buffer(_strategy_args())
    _run_loss(metrics, (384, 672), log_prob_new=[1.0], log_prob_old=[0.75])

    out = metrics.reduce()

    assert out[OVERALL] == pytest.approx(0.25)
    assert not [key for key in out if "_res_" in key]


def test_a_fixed_resolution_run_logs_no_per_resolution_keys():
    metrics = _buffer(_strategy_args(diffusion_resolution_strategy_path=None))
    _run_loss(metrics, None, log_prob_new=[1.0], log_prob_old=[0.5])

    out = metrics.reduce()

    assert out[OVERALL] == pytest.approx(0.5)
    assert not [key for key in out if "_res_" in key]
