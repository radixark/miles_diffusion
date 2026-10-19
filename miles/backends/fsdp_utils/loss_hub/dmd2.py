"""DMD2 critic-group hooks: the critic (fake score) learns the student's samples and scores those of its last step."""

from __future__ import annotations

import dataclasses

import torch

from miles.backends.fsdp_utils.ema import reshard_model
from miles.backends.fsdp_utils.input_dtype_policy import apply_input_dtype_policy
from miles.backends.fsdp_utils.loss_hub.sft import prepare_sft_batch, sft_loss_formula
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, PreparedBatch
from miles.utils.metric_buffer import MetricBuffer


def prepare_critic_batch(
    ctx: DiffusionLossContext,
    batch: list[dict],
    *,
    pad_to_len: int | None = None,
) -> PreparedBatch:
    """SFT flow matching on the student's x0 predictions, conditioned on their rollout prompts.

    The last step's samples are scored at their noisy inputs, so its grid points come from the scoring range.
    """
    config = ctx.train_pipeline_config
    sft_batch = [
        {
            "latent": {"visual": pair["x0"]},
            "cond_kwargs": config.prepare_cond_kwargs(pair["denoising_env"].pos_cond_kwargs, ctx.device),
        }
        for pair in batch
    ]
    grid_index_range = None
    if _is_scoring_step(ctx):
        min_sigma, max_sigma = ctx.args.dmd2_score_sigma_range
        num_train_timesteps = config.num_train_timesteps
        # Grid index i carries the unshifted sigma 1 - i / num_train_timesteps.
        grid_index_range = (round(num_train_timesteps * (1 - max_sigma)), round(num_train_timesteps * (1 - min_sigma)))
    return prepare_sft_batch(ctx, sft_batch, pad_to_len=pad_to_len, grid_index_range=grid_index_range)


def critic_loss_formula(
    ctx: DiffusionLossContext,
    batch: list[dict],
    prepared: PreparedBatch,
    *,
    new_pred: dict[str, torch.Tensor],
    old_pred: dict[str, torch.Tensor] | None,
    ref_pred: dict[str, torch.Tensor] | None,
    metrics: MetricBuffer,
    write_old_log_prob: bool = False,
    old_log_prob_from_new: bool = False,
) -> torch.Tensor:
    """The SFT loss in --dmd2-critic-loss-space.

    On the last step it also keeps the critic's x0 at each noisy input, before that step's update.
    """
    latents = prepared.latents["visual"]
    sigmas = prepared.extras["sigmas"]["visual"].view(len(batch), *([1] * (latents.ndim - 1)))
    if _is_scoring_step(ctx):
        x0_fake = latents - sigmas * new_pred["visual"].detach().float()
        ctx.train_output_inputs.append(
            (batch, prepared.component_name, latents, prepared.timesteps["visual"], sigmas, x0_fake)
        )
    if ctx.args.dmd2_critic_loss_space == "x0":
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
        write_old_log_prob=write_old_log_prob,
        old_log_prob_from_new=old_log_prob_from_new,
    )


def score_last_step_samples(ctx: DiffusionLossContext, train_pairs: list[dict]) -> dict[int, dict]:
    """The teacher's x0 at each scored sample's noisy input, next to the critic's x0 kept there, by sample."""
    predictions = {}
    for batch, component_name, latents, timesteps, sigmas, x0_fake in ctx.train_output_inputs:
        teacher = ctx.teacher_models[component_name]
        real_pred = _predict_velocity(ctx, teacher, batch, latents, timesteps, ctx.args.dmd2_teacher_guidance_scale)
        for pair, x0_real, fake in zip(batch, latents - sigmas * real_pred, x0_fake, strict=True):
            predictions[pair["sample_index"]] = {"x0_real": x0_real.cpu(), "x0_fake": fake.cpu()}

    # A no-grad forward leaves FSDP's root parameters gathered; sleep must move the shards.
    for model in ctx.teacher_models.values():
        reshard_model(model)
    return predictions


def _is_scoring_step(ctx: DiffusionLossContext) -> bool:
    return ctx.optim_step_idx == ctx.args.num_steps_per_rollout - 1


def _predict_velocity(
    ctx: DiffusionLossContext,
    model: torch.nn.Module,
    batch: list[dict],
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """One velocity prediction, with CFG over the rollout's negative cond when ``guidance_scale > 1``."""
    config = ctx.train_pipeline_config
    use_cfg = guidance_scale > 1.0
    envs = [pair["denoising_env"] for pair in batch]
    if use_cfg and any(env.neg_cond_kwargs is None for env in envs):
        raise ValueError(
            "--dmd2-teacher-guidance-scale > 1 needs the rollout's negative cond, "
            "which the engine returns only for requests that run CFG"
        )
    pos_list = [config.prepare_cond_kwargs(env.pos_cond_kwargs, ctx.device) for env in envs]
    neg_list = [config.prepare_cond_kwargs(env.neg_cond_kwargs, ctx.device) for env in envs] if use_cfg else []
    cfg_batching = use_cfg and config.cfg_batching
    pos_cond = neg_cond = joint_cond = None
    if cfg_batching:
        joint_cond = config.collate_cond_for_sample_batch(pos_list + neg_list, ctx.device)
    else:
        pos_cond = config.collate_cond_for_sample_batch(pos_list, ctx.device)
        if use_cfg:
            neg_cond = config.collate_cond_for_sample_batch(neg_list, ctx.device)

    latents_in, timesteps_in, (pos_in, neg_in, joint_in) = apply_input_dtype_policy(
        config.input_dtype_policy,
        latents={"visual": latents},
        timesteps={"visual": config.process_timestep_as_input(timesteps)},
        conds=(pos_cond, neg_cond, joint_cond),
        default_dtype=ctx.forward_dtype,
    )
    with torch.autocast(
        "cuda",
        dtype=ctx.forward_dtype,
        enabled=ctx.forward_dtype != torch.float32 and config.enable_autocast,
    ):
        pred = config.compute_noise_pred(
            model=model,
            latents_input=latents_in,
            timesteps_input=timesteps_in,
            pos_cond=pos_in,
            neg_cond=neg_in,
            joint_cond=joint_in,
            use_cfg=use_cfg,
            cfg_batching=cfg_batching,
            guidance_scale=guidance_scale,
            true_cfg_scale=None,
        )
    return pred["visual"].float()
