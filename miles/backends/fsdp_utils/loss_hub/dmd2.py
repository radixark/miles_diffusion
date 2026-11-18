"""DMD2 hooks: the critic group learns and scores the student's samples; the actor follows the scores.

--dmd2-critic-scoring picks how the critic scores its last step's samples: ``reuse-critic-step`` scores them with
that step's training forward (the critic hooks' last step is the scoring batch), ``original`` scores them with a
separate forward-only pass (the ``critic-scoring`` hooks, run by ``FSDPTrainRayActor.compute``) before the critic
trains that step on them, as tianweiy/DMD2 does.
"""

from __future__ import annotations

import dataclasses

import torch

from miles.backends.fsdp_utils.loss_hub.sft import prepare_sft_batch, sft_loss_formula
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, PreparedBatch
from miles.backends.fsdp_utils.metrics import record_rollout_train_abs_diff, step_metric_key
from miles.utils.metric_buffer import MetricBuffer


def prepare_critic_batch(
    ctx: DiffusionLossContext,
    batch: list[dict],
    *,
    microbatch_id: int,
    optim_step_idx: int,
    pad_to_len: int | None = None,
) -> PreparedBatch:
    """SFT flow matching on the student's x0 predictions, conditioned on their rollout prompts.

    Under --dmd2-critic-scoring reuse-critic-step the last step's samples are scored at their noisy inputs, so that
    step prepares the scoring batch: grid points from the scoring range, the teacher asked for.
    """
    if _is_scoring_step(ctx, optim_step_idx):
        return prepare_scoring_batch(
            ctx, batch, microbatch_id=microbatch_id, optim_step_idx=optim_step_idx, pad_to_len=pad_to_len
        )
    return prepare_sft_batch(
        ctx, _sft_batch(ctx, batch), microbatch_id=microbatch_id, optim_step_idx=optim_step_idx, pad_to_len=pad_to_len
    )


def prepare_scoring_batch(
    ctx: DiffusionLossContext,
    batch: list[dict],
    *,
    microbatch_id: int,
    optim_step_idx: int,
    pad_to_len: int | None = None,
) -> PreparedBatch:
    """The critic's SFT draw on the student's x0 at grid points in --dmd2-score-sigma-range, with the teacher asked
    for at the same noisy inputs."""
    prepared = prepare_sft_batch(
        ctx,
        _sft_batch(ctx, batch),
        microbatch_id=microbatch_id,
        optim_step_idx=optim_step_idx,
        pad_to_len=pad_to_len,
        grid_index_range=_score_grid_index_range(ctx),
    )
    return _request_teacher_forward(ctx, batch, prepared, pad_to_len=pad_to_len)


def _sft_batch(ctx: DiffusionLossContext, batch: list[dict]) -> list[dict]:
    config = ctx.train_pipeline_config
    return [
        {
            "latent": {"visual": pair["x0"]},
            "cond_kwargs": config.prepare_cond_kwargs(pair["denoising_env"].pos_cond_kwargs, ctx.device),
        }
        for pair in batch
    ]


def _score_grid_index_range(ctx: DiffusionLossContext) -> tuple[int, int]:
    min_sigma, max_sigma = ctx.args.dmd2_score_sigma_range
    num_train_timesteps = ctx.train_pipeline_config.num_train_timesteps
    # Grid index i carries the unshifted sigma 1 - i / num_train_timesteps.
    return round(num_train_timesteps * (1 - max_sigma)), round(num_train_timesteps * (1 - min_sigma))


def _request_teacher_forward(
    ctx: DiffusionLossContext, batch: list[dict], prepared: PreparedBatch, *, pad_to_len: int | None
) -> PreparedBatch:
    """Ask the trunk for the teacher at --dmd2-teacher-guidance-scale on this batch, with the rollout's negative
    cond collated for its CFG when the scale needs one."""
    config = ctx.train_pipeline_config
    guidance_scale = ctx.args.dmd2_teacher_guidance_scale
    if guidance_scale <= 1.0:
        return dataclasses.replace(prepared, teacher_guidance_scale=guidance_scale)
    envs = [pair["denoising_env"] for pair in batch]
    if any(env.neg_cond_kwargs is None for env in envs):
        raise ValueError(
            "--dmd2-teacher-guidance-scale > 1 needs the rollout's negative cond, "
            "which the engine returns only for requests that run CFG"
        )
    pos_list = [config.prepare_cond_kwargs(env.pos_cond_kwargs, ctx.device) for env in envs]
    neg_list = [config.prepare_cond_kwargs(env.neg_cond_kwargs, ctx.device) for env in envs]
    if config.cfg_batching:
        joint_cond = config.collate_cond_for_sample_batch(pos_list + neg_list, ctx.device, pad_to_len=pad_to_len)
        return dataclasses.replace(prepared, joint_cond=joint_cond, teacher_guidance_scale=guidance_scale)
    neg_cond = config.collate_cond_for_sample_batch(neg_list, ctx.device, pad_to_len=pad_to_len)
    return dataclasses.replace(prepared, neg_cond=neg_cond, teacher_guidance_scale=guidance_scale)


def critic_loss_formula(
    ctx: DiffusionLossContext,
    batch: list[dict],
    prepared: PreparedBatch,
    *,
    new_pred: dict[str, torch.Tensor],
    old_pred: dict[str, torch.Tensor] | None,
    ref_pred: dict[str, torch.Tensor] | None,
    teacher_pred: dict[str, torch.Tensor] | None = None,
    metrics: MetricBuffer,
    optim_step_idx: int,
    train_output_inputs: list,
    write_old_log_prob: bool = False,
    old_log_prob_from_new: bool = False,
) -> torch.Tensor:
    """The SFT loss in --dmd2-critic-loss-space.

    On the last step it also keeps the critic's and the teacher's x0 at each noisy input, before that step's update.
    """
    if _is_scoring_step(ctx, optim_step_idx):
        keep_scores(batch, prepared, new_pred, teacher_pred, train_output_inputs)
    if ctx.args.dmd2_critic_loss_space == "x0":
        latents = prepared.latents["visual"]
        sigmas = prepared.extras["sigmas"]["visual"].view(len(batch), *([1] * (latents.ndim - 1)))
        # x0 = x_sigma - sigma v, so the MSE on x0 weights the velocity MSE by sigma^2.
        x0_target = latents - sigmas * prepared.extras["target"]["visual"]
        prepared = dataclasses.replace(prepared, extras={**prepared.extras, "target": {"visual": x0_target}})
        new_pred = {"visual": latents - sigmas * new_pred["visual"].float()}
    return sft_loss_formula(
        ctx,
        batch,
        prepared,
        new_pred=new_pred,
        old_pred=old_pred,
        ref_pred=ref_pred,
        metrics=metrics,
        optim_step_idx=optim_step_idx,
        train_output_inputs=train_output_inputs,
        write_old_log_prob=write_old_log_prob,
        old_log_prob_from_new=old_log_prob_from_new,
    )


def scoring_formula(
    ctx: DiffusionLossContext,
    batch: list[dict],
    prepared: PreparedBatch,
    *,
    new_pred: dict[str, torch.Tensor],
    old_pred: dict[str, torch.Tensor] | None,
    ref_pred: dict[str, torch.Tensor] | None,
    teacher_pred: dict[str, torch.Tensor] | None = None,
    metrics: MetricBuffer,
    optim_step_idx: int,
    train_output_inputs: list,
    **kwargs,
) -> None:
    """Keeps the critic's and the teacher's x0 at each noisy input and computes no loss.

    The forward-only scoring of --dmd2-critic-scoring original, before the critic trains its last step on these samples.
    """
    keep_scores(batch, prepared, new_pred, teacher_pred, train_output_inputs)
    return None


def keep_scores(
    batch: list[dict],
    prepared: PreparedBatch,
    new_pred: dict[str, torch.Tensor],
    teacher_pred: dict[str, torch.Tensor] | None,
    train_output_inputs: list,
) -> None:
    """Per sample, x0_fake = x_s - sigma v_critic and x0_real = x_s - sigma v_teacher, for score_last_step_samples."""
    latents = prepared.latents["visual"]
    sigmas = prepared.extras["sigmas"]["visual"].view(len(batch), *([1] * (latents.ndim - 1)))
    x0_fake = latents - sigmas * new_pred["visual"].detach().float()
    x0_real = latents - sigmas * teacher_pred["visual"].float()
    for pair, real, fake in zip(batch, x0_real, x0_fake, strict=True):
        train_output_inputs.append((pair["sample_index"], real, fake))


def score_last_step_samples(
    ctx: DiffusionLossContext, train_pairs: list[dict], train_output_inputs: list
) -> dict[int, dict]:
    """The teacher's and the critic's x0 kept at each scored sample's noisy input, by sample."""
    return {index: {"x0_real": real.cpu(), "x0_fake": fake.cpu()} for index, real, fake in train_output_inputs}


def last_step_pairs(ctx: DiffusionLossContext, train_pairs: list[dict]) -> list[dict]:
    """The last 1/K of the rank's pairs, K the critic's steps per rollout: its last step's pairs, the ones it scores."""
    return train_pairs[len(train_pairs) - len(train_pairs) // ctx.args.num_steps_per_rollout :]


def _is_scoring_step(ctx: DiffusionLossContext, optim_step_idx: int) -> bool:
    """Under reuse-critic-step the last step's training forward is the critic's score; under original no step is."""
    return ctx.args.dmd2_critic_scoring == "reuse-critic-step" and optim_step_idx == ctx.args.num_steps_per_rollout - 1


def select_scored_pairs(ctx: DiffusionLossContext, train_pairs: list[dict]) -> list[dict]:
    """The pairs of the critic's last step, which it scored; the actor trains on these only."""
    scores = ctx.external_data["critic"]
    scored = [pair for pair in train_pairs if pair["sample_index"] in scores]
    if len(scored) != len(scores):
        raise ValueError(
            f"the critic scored {len(scores)} samples, but only {len(scored)} of them are in this rank's train pairs"
        )
    return scored


def prepare_generator_batch(
    ctx: DiffusionLossContext,
    batch: list[dict],
    *,
    microbatch_id: int,
    optim_step_idx: int,
    pad_to_len: int | None = None,
) -> PreparedBatch:
    """The student's step ``i`` on its own rollout input ``x_i``, unguided.

    The pairs carry their own noise, so ``microbatch_id`` and ``optim_step_idx`` are part of every prepare hook's
    signature and unused here.
    """
    if len(ctx.models) != 1:
        raise ValueError("DMD2 currently supports a single DiT component")
    device = ctx.device
    config = ctx.train_pipeline_config
    component_name, model = next(iter(ctx.models.items()))
    latents = torch.stack([pair["latent"] for pair in batch]).to(device=device, dtype=torch.float32)
    timesteps = torch.tensor([pair["timestep"] for pair in batch], device=device, dtype=torch.float32)
    sigmas = torch.tensor([pair["sigma"] for pair in batch], device=device, dtype=torch.float32)
    pos_list = [config.prepare_cond_kwargs(pair["denoising_env"].pos_cond_kwargs, device) for pair in batch]
    # Rollout trajectories carry only the visual stream.
    return PreparedBatch(
        latents={"visual": latents},
        timesteps={"visual": timesteps},
        timesteps_for_model={"visual": config.process_timestep_as_input(timesteps)},
        model=model,
        component_name=component_name,
        guidance_scale=1.0,
        use_cfg=False,
        cfg_batching=False,
        true_cfg_scale=None,
        pos_cond=config.collate_cond_for_sample_batch(pos_list, device, pad_to_len=pad_to_len),
        neg_cond=None,
        joint_cond=None,
        advantage=torch.ones(len(batch), device=device, dtype=torch.float32),
        extras={"sigmas": sigmas},
    )


def dmd2_loss_formula(
    ctx: DiffusionLossContext,
    batch: list[dict],
    prepared: PreparedBatch,
    *,
    new_pred: dict[str, torch.Tensor],
    old_pred: dict[str, torch.Tensor] | None,
    ref_pred: dict[str, torch.Tensor] | None,
    teacher_pred: dict[str, torch.Tensor] | None = None,
    metrics: MetricBuffer,
    optim_step_idx: int,
    train_output_inputs: list,
    write_old_log_prob: bool = False,
    old_log_prob_from_new: bool = False,
) -> torch.Tensor:
    """Distribution matching: x0's gradient is ``g / numel``, ``g = (x0_fake - x0_real) / mean|x0 - x0_real|``."""
    latents = prepared.latents["visual"]
    sigmas = prepared.extras["sigmas"].view(len(batch), *([1] * (latents.ndim - 1)))
    x0 = latents - sigmas * new_pred["visual"].float()
    scores = ctx.external_data["critic"]
    x0_real = torch.stack([scores[pair["sample_index"]]["x0_real"] for pair in batch]).to(x0.device)
    x0_fake = torch.stack([scores[pair["sample_index"]]["x0_fake"] for pair in batch]).to(x0.device)
    reduce_dims = tuple(range(1, x0.ndim))
    with torch.no_grad():
        grad = torch.nan_to_num((x0_fake - x0_real) / (x0 - x0_real).abs().mean(dim=reduce_dims, keepdim=True))
    per_pair = 0.5 * (x0 - (x0 - grad).detach()).square().mean(dim=reduce_dims)
    loss_sum = per_pair.sum()

    with torch.no_grad():
        bsz = len(batch)
        metrics.emit_mean("loss", total=loss_sum, count=bsz)
        rollout_x0 = torch.stack([pair["x0"] for pair in batch]).to(device=x0.device, dtype=torch.float32)
        record_rollout_train_abs_diff(metrics, "x0", x0, rollout_x0)
        per_pair = {
            "dmd2_grad_abs_mean": grad.abs().mean(dim=reduce_dims),
            "dmd2_score_gap_abs_mean": (x0_fake - x0_real).abs().mean(dim=reduce_dims),
            "dmd2_teacher_dist_abs_mean": (x0 - x0_real).abs().mean(dim=reduce_dims),
            "x0_mean_abs_diff": (x0 - rollout_x0).abs().mean(dim=reduce_dims),
        }
        for key in ("dmd2_grad_abs_mean", "dmd2_score_gap_abs_mean", "dmd2_teacher_dist_abs_mean"):
            metrics.emit_mean(key, total=per_pair[key].sum(), count=bsz)
        metrics.emit_mean("dmd2_x0_mean", total=x0.flatten(1).mean(dim=1).sum(), count=bsz)
        metrics.emit_mean("dmd2_x0_std", total=x0.flatten(1).std(dim=1).sum(), count=bsz)
        for index, pair in enumerate(batch):
            for key, values in per_pair.items():
                metrics.emit_mean(step_metric_key(key, pair["step_index"]), total=values[index], count=1)
    return loss_sum
