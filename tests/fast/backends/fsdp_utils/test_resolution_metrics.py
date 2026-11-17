"""Train-vs-rollout log-prob drift is also logged per rollout resolution.

    args.diffusion_resolution_bucket_list           ─► schema declares ..._res_512x512, ..._res_672x384
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
from miles.backends.fsdp_utils.sde_step_backend import SdeStepOutput

OVERALL = "log_prob_mean_abs_diff"


class _FixedLogProbBackend:
    """sde_step_logprob stub: the 'new' log-prob is whatever the test put in the prediction."""

    def sde_step_logprob(self, pred, latents, *, prev_sample, noise_level, sigmas, next_sigmas, sigma_max):
        log_prob = pred.reshape(pred.shape[0], -1)[:, 0]
        ones = torch.ones_like(log_prob)
        return SdeStepOutput(
            prev_sample=prev_sample,
            log_prob=log_prob,
            prev_mean=pred,
            std_dev_t=ones,
            noise_std=ones,
        )


def _declared(*buckets):
    """What arg validation leaves on args.diffusion_resolution_bucket_list."""
    return Namespace(diffusion_resolution_bucket_list=tuple(buckets))


def _run_loss(metrics, resolution, log_prob_new, log_prob_old, log_prob_rollout=None, **kwargs):
    bsz = len(log_prob_new)
    ctx = SimpleNamespace(
        args=Namespace(
            diffusion_clip_range=1e-4,
            diffusion_noise_level=1.0,
            diffusion_kl_beta=0.0,
            diffusion_score_centering=False,
        ),
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
            "log_prob_rollout": torch.tensor(log_prob_rollout if log_prob_rollout is not None else log_prob_old),
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
        optim_step_idx=0,
        train_output_inputs=[],
        **kwargs,
    )


def _buffer(args):
    return new_metric_buffer(
        None, torch.device("cpu"), ["transformer"], resolutions=args.diffusion_resolution_bucket_list
    )


def test_drift_is_split_by_resolution():
    metrics = _buffer(_declared((512, 512), (672, 384)))
    _run_loss(metrics, (672, 384), log_prob_new=[1.0, 1.0], log_prob_old=[1.5, 0.5])  # |diff| 0.5 each
    _run_loss(metrics, (512, 512), log_prob_new=[2.0], log_prob_old=[1.9])  # |diff| 0.1

    out = metrics.reduce()

    assert out["log_prob_mean_abs_diff_res_672x384"] == pytest.approx(0.5)
    assert out["log_prob_mean_abs_diff_res_512x512"] == pytest.approx(0.1)
    assert out[OVERALL] == pytest.approx((0.5 * 2 + 0.1) / 3)


def test_an_undeclared_resolution_only_feeds_the_overall_metric():
    metrics = _buffer(_declared((512, 512), (672, 384)))
    _run_loss(metrics, (384, 672), log_prob_new=[1.0], log_prob_old=[0.75])

    out = metrics.reduce()

    assert out[OVERALL] == pytest.approx(0.25)
    assert not [key for key in out if "_res_" in key]


def test_a_fixed_resolution_run_logs_no_per_resolution_keys():
    metrics = _buffer(_declared())
    _run_loss(metrics, None, log_prob_new=[1.0], log_prob_old=[0.5])

    out = metrics.reduce()

    assert out[OVERALL] == pytest.approx(0.5)
    assert not [key for key in out if "_res_" in key]


def test_the_drift_measures_replay_even_when_old_log_probs_are_recomputed():
    """--diffusion-recompute-old-log-prob makes log_prob_old a train-side value -- on the first
    optimizer step, log_prob_new itself. Reading it would show a flat zero, which looks like
    perfect alignment rather than like no measurement."""
    metrics = _buffer(_declared((512, 512)))

    _run_loss(
        metrics,
        (512, 512),
        log_prob_new=[1.0],
        log_prob_old=[1.0],  # what recompute leaves behind
        log_prob_rollout=[0.7],  # what the rollout actually reported
        old_log_prob_from_new=True,
    )

    out = metrics.reduce()
    assert out["log_prob_mean_abs_diff_res_512x512"] == pytest.approx(0.3)
    assert out[OVERALL] == pytest.approx(0.0)  # the overall metric keeps its existing meaning
