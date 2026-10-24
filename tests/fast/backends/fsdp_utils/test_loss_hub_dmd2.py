"""DMD2 critic-group hooks: the critic (fake score) and the frozen teacher share this group; the actor holds the student.

    rollout pair: x0 = the student's x0 prediction at its stop step i, denoising_env = the rollout cond
                │
    prepare_critic_batch ──► prepare_sft_batch on {"latent": {"visual": x0}, "cond_kwargs": prepared pos cond}
                             (the critic's K steps are the SFT loss on the student's samples; the last step,
                              whose samples get scored, draws grid points in [2%, 98%] of the train timesteps)
                │
    critic_loss_formula ──► sft_loss_formula; on the last step it also keeps, per micro-batch,
                            x_s = that step's noisy input, sigma, and x0_fake = x_s - sigma v_fake from the
                            training forward, i.e. the critic before the last update
                │
    score_last_step_samples, without grad after the critic's steps, per kept micro-batch:
      teacher, CFG --dmd2-teacher-guidance-scale over the rollout's neg cond ──► x0_real = x_s - sigma v_real
      ──► {sample_index: {"x0_real", "x0_fake"}}; the teacher is resharded so sleep moves its shards

What each test pins:
  test_critic_batch_is_the_sft_batch_on_the_student_x0   earlier steps: inputs and targets == SFT on (x0, pos cond)
  test_last_step_draws_from_the_scoring_range           last step: the same SFT draw over grid points [20, 980]
  test_critic_loss_is_the_sft_loss                      loss and metrics == sft_loss_formula on every step
  test_only_the_last_step_keeps_scoring_inputs          earlier steps keep nothing, so only the last step is scored
  test_scores_match_separate_forwards_at_the_kept_input fused scores == critic and teacher forwards run apart at x_s
  test_teacher_runs_cfg_and_the_critic_runs_unguided    CFG combine for the teacher only
  test_teacher_cfg_needs_the_rollout_negative_cond      rejected instead of a forward without text
  test_scoring_reshards_the_teacher                     teacher resharded after its no-grad forwards
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest
import torch
import torch.nn as nn

import miles.backends.fsdp_utils.loss_hub.dmd2 as dmd2
from miles.backends.fsdp_utils.configs.sd3 import SD3TrainPipelineConfig
from miles.backends.fsdp_utils.loss_hub.sft import prepare_sft_batch, sft_loss_formula
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, FlowNoiseSchedule
from miles.utils.types import CondKwargs, DenoisingEnv

FLOW_SHIFT = 3.0
LATENT_SHAPE = (2, 4, 4)
CRITIC_STEPS = 3


class _CondVelocity(nn.Module):
    """v = scale * mean(text embedding) per sample, so a prediction reveals which cond it saw."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states, timestep, return_dict, encoder_hidden_states, pooled_projections):
        text = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1)
        return (self.scale * text.expand_as(hidden_states),)


class _NoisyInputVelocity(nn.Module):
    """v = x_s + timestep, so x0 = x_s - sigma v depends on both inputs."""

    def forward(self, hidden_states, timestep, return_dict, **cond):
        return (hidden_states + timestep.view(-1, 1, 1, 1),)


class _Metrics:
    def __init__(self):
        self.seen = {}

    def emit_mean(self, key, *, total, count):
        prev_total, prev_count = self.seen.get(key, (0.0, 0))
        self.seen[key] = (prev_total + float(total), prev_count + count)


def _cond(value):
    return CondKwargs(encoder_hidden_states=[torch.full((1, 3, 8), value)], pooled_projections=[torch.zeros(1, 8)])


def _pairs(num_pairs, with_neg=True):
    return [
        {
            "x0": torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(n)),
            "denoising_env": DenoisingEnv(
                pos_cond_kwargs=_cond(1.0 + n), neg_cond_kwargs=_cond(-1.0) if with_neg else None
            ),
            "sample_index": 100 + n,
        }
        for n in range(num_pairs)
    ]


def _ctx(teacher_guidance_scale=3.0, critic_scale=1.0, teacher_scale=1.0):
    return DiffusionLossContext(
        models={"transformer": _CondVelocity(critic_scale)},
        teacher_models={"transformer": _CondVelocity(teacher_scale)},
        train_pipeline_config=SD3TrainPipelineConfig(),
        sde_backend=None,
        args=Namespace(
            seed=7,
            micro_batch_size=2,
            num_steps_per_rollout=CRITIC_STEPS,
            fsdp_flow_shift={"visual": FLOW_SHIFT},
            fsdp_supervised_streams=["visual"],
            log_loss_sigma_bucket=0,
            dmd2_teacher_guidance_scale=teacher_guidance_scale,
        ),
        forward_dtype=torch.float32,
        device=torch.device("cpu"),
        rollout_id=4,
        dp_rank=1,
    )


def _critic_step(ctx, pairs, optim_step_idx, metrics=None):
    """One critic optimizer step as _train_core runs it: prepare, the critic's training forward, the loss hook."""
    ctx.optim_step_idx = optim_step_idx
    steps = []
    for start in range(0, len(pairs), ctx.args.micro_batch_size):
        batch = pairs[start : start + ctx.args.micro_batch_size]
        ctx.microbatch_id = optim_step_idx * 10 + start
        prepared = dmd2.prepare_critic_batch(ctx, batch)
        new_pred = ctx.train_pipeline_config.compute_noise_pred(
            model=prepared.model,
            latents_input=prepared.latents,
            timesteps_input=prepared.timesteps_for_model,
            pos_cond=prepared.pos_cond,
            neg_cond=None,
            joint_cond=None,
            use_cfg=prepared.use_cfg,
            cfg_batching=prepared.cfg_batching,
            guidance_scale=prepared.guidance_scale,
            true_cfg_scale=None,
        )
        loss = dmd2.critic_loss_formula(
            ctx, batch, prepared, new_pred=new_pred, old_pred=None, ref_pred=None, metrics=metrics or _Metrics()
        )
        steps.append((batch, prepared, new_pred, loss))
    return steps


def _sft_pairs(ctx, pairs):
    config = ctx.train_pipeline_config
    return [
        {
            "latent": {"visual": pair["x0"]},
            "cond_kwargs": config.prepare_cond_kwargs(pair["denoising_env"].pos_cond_kwargs, ctx.device),
        }
        for pair in pairs
    ]


def test_critic_batch_is_the_sft_batch_on_the_student_x0():
    ctx = _ctx()
    pairs = _pairs(3)

    for optim_step_idx in range(CRITIC_STEPS - 1):
        ctx.optim_step_idx = optim_step_idx
        critic_batch = dmd2.prepare_critic_batch(ctx, pairs)
        sft_batch = prepare_sft_batch(ctx, _sft_pairs(ctx, pairs))

        assert critic_batch.model is ctx.models["transformer"]
        for field in ("latents", "timesteps", "timesteps_for_model"):
            assert torch.equal(getattr(critic_batch, field)["visual"], getattr(sft_batch, field)["visual"])
        assert torch.equal(critic_batch.extras["target"]["visual"], sft_batch.extras["target"]["visual"])
        assert torch.equal(critic_batch.pos_cond["encoder_hidden_states"], sft_batch.pos_cond["encoder_hidden_states"])
        assert not critic_batch.use_cfg


def test_last_step_draws_from_the_scoring_range():
    ctx = _ctx()
    pairs = _pairs(256)
    ctx.optim_step_idx = CRITIC_STEPS - 1

    critic_batch = dmd2.prepare_critic_batch(ctx, pairs)
    sft_batch = prepare_sft_batch(ctx, _sft_pairs(ctx, pairs), grid_index_range=(20, 980))

    assert torch.equal(critic_batch.latents["visual"], sft_batch.latents["visual"])
    assert torch.equal(critic_batch.extras["target"]["visual"], sft_batch.extras["target"]["visual"])
    grid = FlowNoiseSchedule.shifted(FLOW_SHIFT, 1000).sigmas
    sigmas = critic_batch.extras["sigmas"]["visual"]
    assert set(sigmas.tolist()) <= set(grid[20:981].tolist())
    # The same draw without the range leaves it, so the restriction is exercised.
    assert not set(prepare_sft_batch(ctx, _sft_pairs(ctx, pairs)).extras["sigmas"]["visual"].tolist()) <= set(
        grid[20:981].tolist()
    )


def test_critic_loss_is_the_sft_loss():
    for optim_step_idx in range(CRITIC_STEPS):
        ctx = _ctx()
        metrics = _Metrics()
        sft_metrics = _Metrics()
        for batch, prepared, new_pred, loss in _critic_step(ctx, _pairs(3), optim_step_idx, metrics):
            sft_loss = sft_loss_formula(
                ctx, batch, prepared, new_pred=new_pred, old_pred=None, ref_pred=None, metrics=sft_metrics
            )
            assert torch.equal(loss, sft_loss)
        assert metrics.seen == sft_metrics.seen


def test_only_the_last_step_keeps_scoring_inputs():
    ctx = _ctx()
    pairs = _pairs(9)

    for optim_step_idx in range(CRITIC_STEPS - 1):
        _critic_step(ctx, pairs[optim_step_idx * 3 : optim_step_idx * 3 + 3], optim_step_idx)
    assert ctx.train_output_inputs == []

    _critic_step(ctx, pairs[6:], CRITIC_STEPS - 1)
    scores = dmd2.score_last_step_samples(ctx, pairs)

    assert list(scores) == [106, 107, 108]


def test_scores_match_separate_forwards_at_the_kept_input():
    ctx = _ctx(teacher_guidance_scale=3.0)
    ctx.models["transformer"] = _NoisyInputVelocity()
    pairs = _pairs(5)

    steps = _critic_step(ctx, pairs, CRITIC_STEPS - 1)
    scores = dmd2.score_last_step_samples(ctx, pairs)

    assert list(scores) == [100, 101, 102, 103, 104]
    for batch, prepared, _, _ in steps:
        x_s = prepared.latents["visual"]
        timesteps = prepared.timesteps["visual"]
        sigma = prepared.extras["sigmas"]["visual"].view(-1, 1, 1, 1)
        fake = dmd2._predict_velocity(ctx, ctx.models["transformer"], batch, x_s, timesteps, 1.0)
        real = dmd2._predict_velocity(ctx, ctx.teacher_models["transformer"], batch, x_s, timesteps, 3.0)
        for n, pair in enumerate(batch):
            assert torch.equal(scores[pair["sample_index"]]["x0_fake"], x_s[n] - sigma[n] * fake[n])
            assert torch.equal(scores[pair["sample_index"]]["x0_real"], x_s[n] - sigma[n] * real[n])


def test_teacher_runs_cfg_and_the_critic_runs_unguided():
    ctx = _ctx(teacher_guidance_scale=3.0, teacher_scale=1.0, critic_scale=2.0)
    pairs = _pairs(2)

    [(_, prepared, _, _)] = _critic_step(ctx, pairs, CRITIC_STEPS - 1)
    scores = dmd2.score_last_step_samples(ctx, pairs)

    x_s = prepared.latents["visual"]
    sigma = prepared.extras["sigmas"]["visual"].view(-1, 1, 1, 1)
    for n, pair in enumerate(pairs):
        pos, neg = 1.0 + n, -1.0
        torch.testing.assert_close(
            scores[pair["sample_index"]]["x0_real"], x_s[n] - sigma[n] * (neg + 3.0 * (pos - neg))
        )
        torch.testing.assert_close(scores[pair["sample_index"]]["x0_fake"], x_s[n] - sigma[n] * 2.0 * pos)


def test_teacher_cfg_needs_the_rollout_negative_cond():
    ctx = _ctx(teacher_guidance_scale=3.0)
    _critic_step(ctx, _pairs(2, with_neg=False), CRITIC_STEPS - 1)
    with pytest.raises(ValueError, match="needs the rollout's negative cond"):
        dmd2.score_last_step_samples(ctx, _pairs(2, with_neg=False))
    # Unguided, the teacher reads only the positive cond.
    ctx = _ctx(teacher_guidance_scale=1.0)
    _critic_step(ctx, _pairs(2, with_neg=False), CRITIC_STEPS - 1)
    dmd2.score_last_step_samples(ctx, _pairs(2, with_neg=False))


def test_scoring_reshards_the_teacher(monkeypatch):
    resharded = []
    monkeypatch.setattr(dmd2, "reshard_model", resharded.append)
    ctx = _ctx()
    _critic_step(ctx, _pairs(2), CRITIC_STEPS - 1)

    dmd2.score_last_step_samples(ctx, _pairs(2))

    assert resharded == [ctx.teacher_models["transformer"]]
