"""DMD2 hooks: the critic (fake score) and the frozen teacher share the critic group; the actor holds the student.

    rollout pair: x0 = the student's x0 prediction at its stop step i, denoising_env = the rollout cond
                │
    prepare_critic_batch ──► sft_batch on {"latent": {"visual": x0}, "cond_kwargs": prepared pos cond}
                             (the critic's K steps are the SFT loss on the student's samples; the last step,
                              whose samples get scored, draws grid points whose unshifted sigma lies in
                              --dmd2-score-sigma-range, default [0.02, 0.98])
                │
                             the last step also carries the teacher's stage, PreparedBatches.teacher, the same
                             noisy latents at --dmd2-teacher-guidance-scale with the rollout's neg cond for its CFG)
                │
    the trunk runs the teacher without grad at x_s before the critic's training forward ──► v_real
                │
    critic_loss_formula ──► sft_loss_formula on v, or on x0 = x_s - sigma v under --dmd2-critic-loss-space x0;
                            on the last step it also keeps, per sample, x0_fake = x_s - sigma v_fake from the
                            training forward, i.e. the critic before the last update, and x0_real = x_s - sigma v_real
                │
    score_last_step_samples, after the critic's steps ──► {sample_index: {"x0_real", "x0_fake"}} on the host
                │  ctx.external_data["critic"] on the paired actor rank
    select_scored_pairs ──► the actor's train-input hook keeps only the scored pairs, i.e. the critic's last step
    prepare_generator_batch ──► the student's unguided step i on the rollout input x_i
    dmd2_loss_formula:
      x0 = x_i - sigma_i v_student                       (with grad; the rollout's x0_i only feeds a metric)
      g  = nan_to_num((x0_fake - x0_real) / mean|x0 - x0_real|)  per pair
      loss = 0.5 mean((x0 - sg(x0 - g))^2)  ──► d/dx0 = g / numel, the reference implementation's gradient

What each test pins:
  test_critic_batch_is_the_sft_batch_on_the_student_x0   earlier steps: inputs and targets == SFT on (x0, pos cond)
  test_last_step_draws_from_the_scoring_range           last step: the same SFT draw over grid points [20, 980]
  test_score_sigma_range_bounds_the_unshifted_sigma     a custom range bounds the unshifted sigma, not 1 - sigma
  test_critic_loss_is_the_sft_loss                      loss and metrics == sft_loss_formula on every step
  test_x0_critic_loss_weights_the_velocity_mse          x0 loss == x0 MSE == sigma^2 * velocity MSE, input intact
  test_only_the_last_step_keeps_scoring_inputs          earlier steps keep nothing, so only the last step is scored
  test_earlier_steps_ask_for_no_teacher                 no teacher scale, no negative cond before the last step
  test_last_step_asks_for_the_guided_teacher            teacher scale and the neg cond collated per CFG batching
  test_unguided_teacher_needs_no_negative_cond          scale 1: the teacher reads only the positive cond
  test_teacher_cfg_needs_the_rollout_negative_cond      rejected at prepare instead of a forward without text
  test_scores_are_x0_at_the_kept_input                  x0_fake and x0_real == x_s - sigma v of each forward
  test_teacher_runs_cfg_and_the_critic_runs_unguided    CFG combine for the teacher only; kept x0 carry no grad
  test_actor_keeps_the_critic_last_step_pairs           same shard ──► the actor trains on the scored 1/K only
  test_scores_outside_the_shard_are_rejected            a broken rank pairing fails instead of shrinking the batch
  test_generator_batch_is_the_student_step_on_its_input  x_i, t_i, sigma_i and the positive cond, no CFG
  test_generator_needs_a_single_component                multi-DiT families are rejected
  test_surrogate_gradient_matches_the_reference          same gradient as the reference's batch-mean MSE
  test_critic_matching_the_teacher_gives_no_gradient     x0_fake == x0_real ──► zero loss and gradient
  test_gaussian_student_moves_toward_the_teacher         analytic scores: sign and normalization move mu
  test_health_metrics_are_recorded_overall_and_per_step  score gap, teacher distance, x0 stats, split by stop step

--dmd2-critic-scoring original: the critic's last step is plain SFT and carries the scoring batch, a fresh draw in the
scoring range, as the teacher's and the critic's actor-forward input; the formula keeps the x0 of those forwards:
  test_under_original_only_the_last_step_carries_a_scoring_batch_and_it_stays_plain_sft
  test_the_scoring_batch_is_the_reuse_last_step_batch_at_its_own_draw
  test_under_original_the_formula_keeps_the_x0_of_the_actor_forward_batch
  test_either_mode_scores_the_last_step_once             by its training forward or before it, never both or neither
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import dataclasses
from argparse import Namespace

import pytest
import torch
import torch.nn as nn

import miles.backends.fsdp_utils.loss_hub.dmd2 as dmd2
from miles.backends.fsdp_utils.configs.sd3 import SD3TrainPipelineConfig
from miles.backends.fsdp_utils.loss_hub.sft import sft_batch, sft_loss_formula
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, FlowNoiseSchedule
from miles.backends.fsdp_utils.metrics import new_metric_buffer
from miles.utils.train_data_utils import build_microbatch_schedule
from miles.utils.types import CondKwargs, DenoisingEnv

FLOW_SHIFT = 3.0
LATENT_SHAPE = (2, 4, 4)
CRITIC_STEPS = 3


class _CondVelocity(nn.Module):
    """v = scale * mean(text embedding) per sample, so a prediction reveals which cond it saw."""

    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

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


def _ctx(teacher_guidance_scale=3.0, critic_scale=1.0):
    return DiffusionLossContext(
        models={"transformer": _CondVelocity(critic_scale)},
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
            dmd2_score_sigma_range=[0.02, 0.98],
            dmd2_critic_loss_space="velocity",
            dmd2_critic_scoring="reuse-critic-step",
        ),
        forward_dtype=torch.float32,
        device=torch.device("cpu"),
        rollout_id=4,
        dp_rank=1,
    )


def _forward(config, model, prepared, *, use_cfg, cfg_batching, guidance_scale):
    return config.compute_noise_pred(
        model=model,
        latents_input=prepared.latents,
        timesteps_input=prepared.timesteps_for_model,
        pos_cond=prepared.pos_cond,
        neg_cond=prepared.neg_cond,
        joint_cond=prepared.joint_cond,
        use_cfg=use_cfg,
        cfg_batching=cfg_batching,
        guidance_scale=guidance_scale,
        true_cfg_scale=None,
    )


def _critic_step(ctx, pairs, optim_step_idx, train_output_inputs, metrics=None, teacher=None):
    """One critic optimizer step as _train_core runs it: prepare, the teacher on its batch, the critic's no-grad
    forward on the actor-forward batch, the training forward, the loss hook."""
    config = ctx.train_pipeline_config
    teacher = _CondVelocity(1.0) if teacher is None else teacher

    def forward(model, prepared):
        return _forward(
            config,
            model,
            prepared,
            use_cfg=prepared.use_cfg,
            cfg_batching=prepared.cfg_batching,
            guidance_scale=prepared.guidance_scale,
        )

    steps = []
    for start in range(0, len(pairs), ctx.args.micro_batch_size):
        batch = pairs[start : start + ctx.args.micro_batch_size]
        prepared_batches = dmd2.prepare_critic_batch(
            ctx, batch, microbatch_id=optim_step_idx * 10 + start, optim_step_idx=optim_step_idx
        )
        prepared = prepared_batches.actor
        teacher_pred = None
        if prepared_batches.teacher is not None:
            with torch.no_grad():
                teacher_pred = forward(teacher, prepared_batches.teacher)
        actor_forward_pred = None
        if prepared_batches.actor_forward is not None:
            with torch.no_grad():
                actor_forward_pred = forward(prepared.model, prepared_batches.actor_forward)
        new_pred = forward(prepared.model, prepared)
        loss = dmd2.critic_loss_formula(
            ctx,
            batch,
            prepared,
            new_pred=new_pred,
            old_pred=None,
            ref_pred=None,
            teacher_pred=teacher_pred,
            teacher_batch=prepared_batches.teacher,
            actor_forward_pred=actor_forward_pred,
            actor_forward_batch=prepared_batches.actor_forward,
            metrics=metrics or _Metrics(),
            optim_step_idx=optim_step_idx,
            train_output_inputs=train_output_inputs,
        )
        steps.append((batch, prepared_batches, new_pred, loss))
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
        critic_batch = dmd2.prepare_critic_batch(ctx, pairs, microbatch_id=0, optim_step_idx=optim_step_idx).actor
        expected = sft_batch(ctx, _sft_pairs(ctx, pairs), microbatch_id=0, optim_step_idx=optim_step_idx)

        assert critic_batch.model is ctx.models["transformer"]
        for field in ("latents", "timesteps", "timesteps_for_model"):
            assert torch.equal(getattr(critic_batch, field)["visual"], getattr(expected, field)["visual"])
        assert torch.equal(critic_batch.extras["target"]["visual"], expected.extras["target"]["visual"])
        assert torch.equal(critic_batch.pos_cond["encoder_hidden_states"], expected.pos_cond["encoder_hidden_states"])
        assert not critic_batch.use_cfg


def test_last_step_draws_from_the_scoring_range():
    ctx = _ctx()
    pairs = _pairs(256)
    last_step = CRITIC_STEPS - 1

    critic_batch = dmd2.prepare_critic_batch(ctx, pairs, microbatch_id=0, optim_step_idx=last_step).actor
    expected = sft_batch(
        ctx, _sft_pairs(ctx, pairs), microbatch_id=0, optim_step_idx=last_step, grid_index_range=(20, 980)
    )

    assert torch.equal(critic_batch.latents["visual"], expected.latents["visual"])
    assert torch.equal(critic_batch.extras["target"]["visual"], expected.extras["target"]["visual"])
    grid = FlowNoiseSchedule.shifted(FLOW_SHIFT, 1000).sigmas
    sigmas = critic_batch.extras["sigmas"]["visual"]
    assert set(sigmas.tolist()) <= set(grid[20:981].tolist())
    # The same draw without the range leaves it, so the restriction is exercised.
    unrestricted = sft_batch(ctx, _sft_pairs(ctx, pairs), microbatch_id=0, optim_step_idx=last_step)
    assert not set(unrestricted.extras["sigmas"]["visual"].tolist()) <= set(grid[20:981].tolist())


def test_score_sigma_range_bounds_the_unshifted_sigma():
    ctx = _ctx()
    ctx.args.dmd2_score_sigma_range = [0.1, 0.5]
    pairs = _pairs(256)
    last_step = CRITIC_STEPS - 1

    critic_batch = dmd2.prepare_critic_batch(ctx, pairs, microbatch_id=0, optim_step_idx=last_step).actor
    # Grid index i carries the unshifted sigma 1 - i / 1000, so [0.1, 0.5] is grid points [500, 900].
    expected = sft_batch(
        ctx, _sft_pairs(ctx, pairs), microbatch_id=0, optim_step_idx=last_step, grid_index_range=(500, 900)
    )

    sigmas = critic_batch.extras["sigmas"]["visual"]
    assert torch.equal(sigmas, expected.extras["sigmas"]["visual"])
    unshifted = sigmas / (FLOW_SHIFT - (FLOW_SHIFT - 1) * sigmas)
    assert unshifted.min() >= 0.1 - 1e-6 and unshifted.max() <= 0.5 + 1e-6


def test_critic_loss_is_the_sft_loss():
    for optim_step_idx in range(CRITIC_STEPS):
        ctx = _ctx()
        metrics = _Metrics()
        sft_metrics = _Metrics()
        for batch, prepared_batches, new_pred, loss in _critic_step(ctx, _pairs(3), optim_step_idx, [], metrics):
            sft_loss = sft_loss_formula(
                ctx,
                batch,
                prepared_batches.actor,
                new_pred=new_pred,
                old_pred=None,
                ref_pred=None,
                metrics=sft_metrics,
                optim_step_idx=optim_step_idx,
                train_output_inputs=[],
            )
            assert torch.equal(loss, sft_loss)
        assert metrics.seen == sft_metrics.seen


def test_x0_critic_loss_weights_the_velocity_mse():
    for optim_step_idx in range(CRITIC_STEPS):
        ctx = _ctx()
        ctx.models["transformer"] = _NoisyInputVelocity()
        ctx.args.dmd2_critic_loss_space = "x0"
        for batch, prepared_batches, new_pred, loss in _critic_step(ctx, _pairs(3), optim_step_idx, []):
            prepared = prepared_batches.actor
            x_s = prepared.latents["visual"]
            sigma = prepared.extras["sigmas"]["visual"].view(-1, 1, 1, 1)
            x0 = torch.stack([pair["x0"] for pair in batch])
            x0_mse = (x_s - sigma * new_pred["visual"] - x0).square().flatten(1).mean(dim=1)
            velocity_mse = (new_pred["visual"] - prepared.extras["target"]["visual"]).square().flatten(1).mean(dim=1)
            torch.testing.assert_close(loss, x0_mse.sum())
            torch.testing.assert_close(loss, (sigma.flatten() ** 2 * velocity_mse).sum())


def test_only_the_last_step_keeps_scoring_inputs():
    ctx = _ctx()
    pairs = _pairs(9)

    inputs = []
    for optim_step_idx in range(CRITIC_STEPS - 1):
        _critic_step(ctx, pairs[optim_step_idx * 3 : optim_step_idx * 3 + 3], optim_step_idx, inputs)
    assert inputs == []

    _critic_step(ctx, pairs[6:], CRITIC_STEPS - 1, inputs)
    assert [index for index, _, _ in inputs] == [106, 107, 108]
    scores = dmd2.score_last_step_samples(ctx, pairs, inputs)

    assert list(scores) == [106, 107, 108]


def test_earlier_steps_ask_for_no_teacher():
    ctx = _ctx(teacher_guidance_scale=3.0)
    for optim_step_idx in range(CRITIC_STEPS - 1):
        prepared_batches = dmd2.prepare_critic_batch(ctx, _pairs(2), microbatch_id=0, optim_step_idx=optim_step_idx)
        assert prepared_batches.teacher is None
        assert prepared_batches.actor.neg_cond is None and prepared_batches.actor.joint_cond is None


@pytest.mark.parametrize("cfg_batching", [False, True])
def test_last_step_asks_for_the_guided_teacher(cfg_batching):
    ctx = _ctx(teacher_guidance_scale=3.0)
    config = ctx.train_pipeline_config
    config.cfg_batching = cfg_batching
    pairs = _pairs(2)

    prepared_batches = dmd2.prepare_critic_batch(ctx, pairs, microbatch_id=0, optim_step_idx=CRITIC_STEPS - 1)
    prepared = prepared_batches.actor

    # The training batch stays unguided; the teacher's input carries the CFG and the negative cond.
    assert not prepared.use_cfg and not prepared.cfg_batching
    assert prepared.neg_cond is None and prepared.joint_cond is None
    teacher_batch = prepared_batches.teacher
    assert torch.equal(teacher_batch.latents["visual"], prepared.latents["visual"])
    assert (teacher_batch.use_cfg, teacher_batch.cfg_batching, teacher_batch.guidance_scale) == (
        True,
        cfg_batching,
        3.0,
    )
    assert teacher_batch.true_cfg_scale is None
    pos_list = [config.prepare_cond_kwargs(pair["denoising_env"].pos_cond_kwargs, ctx.device) for pair in pairs]
    neg_list = [config.prepare_cond_kwargs(pair["denoising_env"].neg_cond_kwargs, ctx.device) for pair in pairs]
    pos = config.collate_cond_for_sample_batch(pos_list, ctx.device)["encoder_hidden_states"]
    assert torch.equal(teacher_batch.pos_cond["encoder_hidden_states"], pos)
    if cfg_batching:
        joint = config.collate_cond_for_sample_batch(pos_list + neg_list, ctx.device)["encoder_hidden_states"]
        assert teacher_batch.neg_cond is None
        assert torch.equal(teacher_batch.joint_cond["encoder_hidden_states"], joint)
    else:
        neg = config.collate_cond_for_sample_batch(neg_list, ctx.device)["encoder_hidden_states"]
        assert teacher_batch.joint_cond is None
        assert torch.equal(teacher_batch.neg_cond["encoder_hidden_states"], neg)


def test_unguided_teacher_needs_no_negative_cond():
    ctx = _ctx(teacher_guidance_scale=1.0)
    prepared_batches = dmd2.prepare_critic_batch(
        ctx, _pairs(2, with_neg=False), microbatch_id=0, optim_step_idx=CRITIC_STEPS - 1
    )
    teacher_batch = prepared_batches.teacher
    assert (teacher_batch.use_cfg, teacher_batch.cfg_batching, teacher_batch.guidance_scale) == (False, False, 1.0)
    assert teacher_batch.neg_cond is None and teacher_batch.joint_cond is None


def test_teacher_cfg_needs_the_rollout_negative_cond():
    ctx = _ctx(teacher_guidance_scale=3.0)
    with pytest.raises(ValueError, match="needs the rollout's negative cond"):
        dmd2.prepare_critic_batch(ctx, _pairs(2, with_neg=False), microbatch_id=0, optim_step_idx=CRITIC_STEPS - 1)
    # Earlier steps run no teacher, so they take pairs without a negative cond.
    dmd2.prepare_critic_batch(ctx, _pairs(2, with_neg=False), microbatch_id=0, optim_step_idx=0)


def test_scores_are_x0_at_the_kept_input():
    ctx = _ctx(teacher_guidance_scale=1.0)
    ctx.models["transformer"] = _NoisyInputVelocity()
    pairs = _pairs(5)

    inputs = []
    steps = _critic_step(ctx, pairs, CRITIC_STEPS - 1, inputs, teacher=_NoisyInputVelocity())
    scores = dmd2.score_last_step_samples(ctx, pairs, inputs)

    assert list(scores) == [100, 101, 102, 103, 104]
    for batch, prepared_batches, _, _ in steps:
        prepared = prepared_batches.actor
        x_s = prepared.latents["visual"]
        velocity = x_s + prepared.timesteps_for_model["visual"].view(-1, 1, 1, 1)
        sigma = prepared.extras["sigmas"]["visual"].view(-1, 1, 1, 1)
        for n, pair in enumerate(batch):
            assert torch.equal(scores[pair["sample_index"]]["x0_fake"], x_s[n] - sigma[n] * velocity[n])
            assert torch.equal(scores[pair["sample_index"]]["x0_real"], x_s[n] - sigma[n] * velocity[n])


def test_teacher_runs_cfg_and_the_critic_runs_unguided():
    ctx = _ctx(teacher_guidance_scale=3.0, critic_scale=2.0)
    teacher = _CondVelocity(1.0)
    pairs = _pairs(2)

    inputs = []
    [(_, prepared_batches, _, loss)] = _critic_step(ctx, pairs, CRITIC_STEPS - 1, inputs, teacher=teacher)
    scores = dmd2.score_last_step_samples(ctx, pairs, inputs)

    prepared = prepared_batches.actor
    x_s = prepared.latents["visual"]
    sigma = prepared.extras["sigmas"]["visual"].view(-1, 1, 1, 1)
    for n, pair in enumerate(pairs):
        pos, neg = 1.0 + n, -1.0
        torch.testing.assert_close(
            scores[pair["sample_index"]]["x0_real"], x_s[n] - sigma[n] * (neg + 3.0 * (pos - neg))
        )
        torch.testing.assert_close(scores[pair["sample_index"]]["x0_fake"], x_s[n] - sigma[n] * 2.0 * pos)
        assert not scores[pair["sample_index"]]["x0_real"].requires_grad
        assert not scores[pair["sample_index"]]["x0_fake"].requires_grad
    loss.backward()
    assert ctx.models["transformer"].scale.grad is not None
    assert teacher.scale.grad is None


def test_actor_keeps_the_critic_last_step_pairs():
    ctx = _ctx()
    pairs = _pairs(9)
    schedule = build_microbatch_schedule(
        num_pairs_per_optim_step=3, num_optim_steps_per_rollout=CRITIC_STEPS, micro_batch_size=3
    )
    inputs = []
    for optim_step_idx, [(pair_lo, pair_hi)] in enumerate(schedule):
        _critic_step(ctx, pairs[pair_lo:pair_hi], optim_step_idx, inputs)

    actor_ctx = _ctx()
    actor_ctx.external_data["critic"] = dmd2.score_last_step_samples(ctx, pairs, inputs)

    assert dmd2.select_scored_pairs(actor_ctx, pairs) == pairs[6:]


def test_scores_outside_the_shard_are_rejected():
    pairs = _pairs(4)
    ctx = _ctx()
    ctx.external_data["critic"] = {pair["sample_index"]: {} for pair in pairs[2:]}
    with pytest.raises(ValueError, match="only 1 of them"):
        dmd2.select_scored_pairs(ctx, pairs[:3])


def _generator_pairs(num_pairs):
    return [
        {
            "latent": torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(n)).to(torch.bfloat16),
            "x0": torch.zeros(LATENT_SHAPE),
            "timestep": 750.0 - 250.0 * n,
            "sigma": 0.75 - 0.25 * n,
            "denoising_env": DenoisingEnv(pos_cond_kwargs=_cond(1.0 + n), neg_cond_kwargs=None),
            "sample_index": 100 + n,
            "step_index": n,
        }
        for n in range(num_pairs)
    ]


def _metrics():
    return new_metric_buffer(None, torch.device("cpu"), ["transformer"], num_rollout_steps=4)


def _loss(ctx, pairs, x0):
    """dmd2_loss_formula with the student predicting ``x0`` from each pair's rollout input."""
    prepared = dmd2.generator_batch(ctx, pairs, microbatch_id=0, optim_step_idx=0)
    sigmas = prepared.extras["sigmas"].view(-1, 1, 1, 1)
    new_pred = {"visual": (prepared.latents["visual"] - x0) / sigmas}
    metrics = _metrics()
    return (
        dmd2.dmd2_loss_formula(
            ctx,
            pairs,
            prepared,
            new_pred=new_pred,
            old_pred=None,
            ref_pred=None,
            metrics=metrics,
            optim_step_idx=0,
            train_output_inputs=[],
        ),
        metrics,
    )


def _scores(pairs, x0_real, x0_fake):
    return {
        pair["sample_index"]: {"x0_real": real, "x0_fake": fake}
        for pair, real, fake in zip(pairs, x0_real, x0_fake, strict=True)
    }


def test_generator_batch_is_the_student_step_on_its_input():
    ctx = _ctx()
    pairs = _generator_pairs(2)

    prepared = dmd2.generator_batch(ctx, pairs, microbatch_id=0, optim_step_idx=0)

    assert prepared.model is ctx.models["transformer"]
    assert torch.equal(prepared.latents["visual"], torch.stack([pair["latent"] for pair in pairs]).float())
    assert prepared.timesteps["visual"].tolist() == [750.0, 500.0]
    assert prepared.extras["sigmas"].tolist() == [0.75, 0.5]
    assert prepared.pos_cond["encoder_hidden_states"][:, 0, 0].tolist() == [1.0, 2.0]
    assert not prepared.use_cfg and prepared.neg_cond is None


def test_generator_needs_a_single_component():
    ctx = _ctx()
    ctx.models["transformer_2"] = ctx.models["transformer"]
    with pytest.raises(ValueError, match="single DiT component"):
        dmd2.prepare_generator_batch(ctx, _generator_pairs(1), microbatch_id=0, optim_step_idx=0)


def test_surrogate_gradient_matches_the_reference():
    ctx = _ctx()
    pairs = _generator_pairs(2)
    x0 = torch.randn(2, *LATENT_SHAPE, requires_grad=True)
    x0_real, x0_fake = torch.randn(2, *LATENT_SHAPE), torch.randn(2, *LATENT_SHAPE)
    ctx = dataclasses.replace(ctx, external_data={"critic": _scores(pairs, x0_real, x0_fake)})

    loss_sum, _ = _loss(ctx, pairs, x0)
    (loss_sum / len(pairs)).backward()

    # tianweiy/DMD2 compute_distribution_matching_loss, on the same x0 and predictions.
    reference_x0 = x0.detach().clone().requires_grad_()
    with torch.no_grad():
        p_real, p_fake = reference_x0 - x0_real, reference_x0 - x0_fake
        grad = torch.nan_to_num((p_real - p_fake) / p_real.abs().mean(dim=[1, 2, 3], keepdim=True))
    torch.nn.functional.mse_loss(reference_x0, (reference_x0 - grad).detach(), reduction="mean").mul(0.5).backward()

    torch.testing.assert_close(x0.grad, reference_x0.grad)


def test_critic_matching_the_teacher_gives_no_gradient():
    ctx = _ctx()
    pairs = _generator_pairs(2)
    x0 = torch.randn(2, *LATENT_SHAPE, requires_grad=True)
    x0_pred = torch.randn(2, *LATENT_SHAPE)
    ctx = dataclasses.replace(ctx, external_data={"critic": _scores(pairs, x0_pred, x0_pred)})

    loss_sum, metrics = _loss(ctx, pairs, x0)
    loss_sum.backward()

    assert float(loss_sum) == 0.0
    assert torch.count_nonzero(x0.grad) == 0
    assert metrics.reduce()["dmd2_grad_abs_mean"] == 0.0


def test_gaussian_student_moves_toward_the_teacher():
    # Data N(mu, 1) per element, noised as x_s = a x + b eps: E[x | x_s] = mu + a (x_s - a mu) / (a^2 + b^2).
    def posterior_mean(mu, x_s, a, b):
        return mu + a * (x_s - a * mu) / (a**2 + b**2)

    ctx = _ctx()
    pairs = _generator_pairs(2)
    teacher_mu, student_mu = 1.0, torch.tensor(-1.0, requires_grad=True)
    optimizer = torch.optim.SGD([student_mu], lr=0.5)
    generator = torch.Generator().manual_seed(0)
    gaps = []
    for _ in range(20):
        optimizer.zero_grad()
        x0 = student_mu + torch.randn(2, *LATENT_SHAPE, generator=generator)
        a, b = 0.5, 0.5
        x_s = a * x0.detach() + b * torch.randn(x0.shape, generator=generator)
        scores = _scores(pairs, posterior_mean(teacher_mu, x_s, a, b), posterior_mean(float(student_mu), x_s, a, b))
        ctx = dataclasses.replace(ctx, external_data={"critic": scores})
        loss_sum, _ = _loss(ctx, pairs, x0)
        (loss_sum / len(pairs)).backward()
        optimizer.step()
        gaps.append(abs(teacher_mu - float(student_mu)))

    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 0.5 * gaps[0]


def test_trainer_and_rollout_x0_diff_is_recorded():
    ctx = _ctx()
    pairs = _generator_pairs(2)
    x0 = torch.stack([pair["x0"] for pair in pairs]) + 0.25
    ctx = dataclasses.replace(ctx, external_data={"critic": _scores(pairs, x0, x0)})

    _, metrics = _loss(ctx, pairs, x0)

    reduced = metrics.reduce()
    assert reduced["x0_mean_abs_diff"] == pytest.approx(0.25)
    assert reduced["x0_max_abs_diff"] == pytest.approx(0.25)


def test_health_metrics_are_recorded_overall_and_per_step():
    ctx = _ctx()
    pairs = _generator_pairs(2)
    # Pair n stopped at step n: x0 = 0.5 + n against a teacher x0 of 0 and a critic x0 of 0.25.
    x0 = torch.stack([pair["x0"] for pair in pairs]) + torch.tensor([0.5, 1.5]).view(2, 1, 1, 1)
    ctx = dataclasses.replace(
        ctx,
        external_data={"critic": _scores(pairs, torch.zeros(2, *LATENT_SHAPE), torch.full((2, *LATENT_SHAPE), 0.25))},
    )

    _, metrics = _loss(ctx, pairs, x0)

    reduced = metrics.reduce()
    assert reduced["dmd2_score_gap_abs_mean"] == pytest.approx(0.25)
    assert reduced["dmd2_teacher_dist_abs_mean"] == pytest.approx(1.0)
    assert reduced["dmd2_x0_mean"] == pytest.approx(1.0)
    assert reduced["dmd2_x0_std"] == pytest.approx(0.0, abs=1e-6)
    for step, distance in enumerate([0.5, 1.5]):
        assert reduced[f"dmd2_teacher_dist_abs_mean_step{step}"] == pytest.approx(distance)
        assert reduced[f"dmd2_grad_abs_mean_step{step}"] == pytest.approx(0.25 / distance)
        assert reduced[f"dmd2_score_gap_abs_mean_step{step}"] == pytest.approx(0.25)
        assert reduced[f"x0_mean_abs_diff_step{step}"] == pytest.approx(distance)
    # No pair stopped at steps 2 or 3, so they log nothing.
    assert not any(key.endswith(("_step2", "_step3")) for key in reduced)


def _original_ctx(teacher_guidance_scale=3.0):
    ctx = _ctx(teacher_guidance_scale)
    ctx.args.dmd2_critic_scoring = "original"
    return ctx


def test_under_original_only_the_last_step_carries_a_scoring_batch_and_it_stays_plain_sft():
    ctx = _original_ctx()
    pairs = _pairs(256)
    last_step = CRITIC_STEPS - 1

    critic_batches = dmd2.prepare_critic_batch(ctx, pairs, microbatch_id=6, optim_step_idx=last_step)

    expected = sft_batch(ctx, _sft_pairs(ctx, pairs), microbatch_id=6, optim_step_idx=last_step)
    assert torch.equal(critic_batches.actor.latents["visual"], expected.latents["visual"])
    assert torch.equal(critic_batches.actor.extras["target"]["visual"], expected.extras["target"]["visual"])
    assert critic_batches.actor.neg_cond is None and critic_batches.actor.joint_cond is None
    # The teacher and the critic's actor forward both run on the scoring draw.
    assert torch.equal(critic_batches.teacher.latents["visual"], critic_batches.actor_forward.latents["visual"])
    first_step = dmd2.prepare_critic_batch(ctx, pairs, microbatch_id=0, optim_step_idx=0)
    assert first_step.teacher is None and first_step.actor_forward is None


@pytest.mark.parametrize("cfg_batching", [False, True])
def test_the_scoring_batch_is_the_reuse_last_step_batch_at_its_own_draw(cfg_batching):
    pairs = _pairs(4)
    ctx = _ctx(teacher_guidance_scale=3.0)
    ctx.train_pipeline_config.cfg_batching = cfg_batching
    reuse = dmd2.prepare_critic_batch(ctx, pairs, microbatch_id=9, optim_step_idx=CRITIC_STEPS - 1)

    original_ctx = _original_ctx()
    original_ctx.train_pipeline_config.cfg_batching = cfg_batching
    # The same micro-batch id: the scoring draw is named, not numbered past the training ids.
    original = dmd2.prepare_critic_batch(original_ctx, pairs, microbatch_id=9, optim_step_idx=CRITIC_STEPS - 1)
    scoring = original.actor_forward

    expected = sft_batch(
        original_ctx,
        _sft_pairs(original_ctx, pairs),
        microbatch_id=9,
        optim_step_idx=CRITIC_STEPS - 1,
        grid_index_range=dmd2._score_grid_index_range(original_ctx),
        draw="scoring",
    )
    for field in ("latents", "timesteps", "timesteps_for_model"):
        assert torch.equal(getattr(scoring, field)["visual"], getattr(expected, field)["visual"])
    assert torch.equal(scoring.extras["sigmas"]["visual"], expected.extras["sigmas"]["visual"])
    # Its noise is neither the training forward's nor the reuse mode's at this id.
    assert not torch.equal(scoring.latents["visual"], original.actor.latents["visual"])
    assert not torch.equal(scoring.latents["visual"], reuse.actor.latents["visual"])
    assert original.teacher.guidance_scale == reuse.teacher.guidance_scale == 3.0
    for cond in ("pos_cond", "neg_cond", "joint_cond"):
        reuse_cond, scoring_cond = getattr(reuse.teacher, cond), getattr(original.teacher, cond)
        if reuse_cond is None:
            assert scoring_cond is None
        else:
            assert torch.equal(scoring_cond["encoder_hidden_states"], reuse_cond["encoder_hidden_states"])


def test_under_original_the_formula_keeps_the_x0_of_the_actor_forward_batch():
    ctx = _original_ctx(teacher_guidance_scale=1.0)
    ctx.models["transformer"] = _NoisyInputVelocity()
    pairs = _pairs(2)

    inputs = []
    [(_, prepared_batches, new_pred, loss)] = _critic_step(
        ctx, pairs, CRITIC_STEPS - 1, inputs, teacher=_CondVelocity(2.0)
    )
    scores = dmd2.score_last_step_samples(ctx, pairs, inputs)

    assert list(scores) == [100, 101]
    scoring = prepared_batches.actor_forward
    x_s = scoring.latents["visual"]
    sigma = scoring.extras["sigmas"]["visual"].view(-1, 1, 1, 1)
    critic_velocity = x_s + scoring.timesteps_for_model["visual"].view(-1, 1, 1, 1)
    for n, pair in enumerate(pairs):
        assert torch.equal(scores[pair["sample_index"]]["x0_fake"], x_s[n] - sigma[n] * critic_velocity[n])
        torch.testing.assert_close(scores[pair["sample_index"]]["x0_real"], x_s[n] - sigma[n] * 2.0 * (1.0 + n))
    # The loss is still the SFT loss of the training forward.
    sft_loss = sft_loss_formula(
        ctx,
        pairs,
        prepared_batches.actor,
        new_pred=new_pred,
        old_pred=None,
        ref_pred=None,
        metrics=_Metrics(),
        optim_step_idx=CRITIC_STEPS - 1,
        train_output_inputs=[],
    )
    assert torch.equal(loss, sft_loss)


@pytest.mark.parametrize("critic_scoring", ["reuse-critic-step", "original"])
def test_either_mode_scores_the_last_step_once(critic_scoring):
    """A step scored twice or not at all leaves the actor the wrong scores, or none, without an error."""
    ctx = _ctx()
    ctx.args.dmd2_critic_scoring = critic_scoring

    for step in range(CRITIC_STEPS):
        by_training_forward = dmd2._is_scoring_step(ctx, step)
        before_training = dmd2._is_scored_before_training(ctx, step)
        assert by_training_forward + before_training == (step == CRITIC_STEPS - 1)
    assert dmd2._is_scored_before_training(ctx, CRITIC_STEPS - 1) == (critic_scoring == "original")
