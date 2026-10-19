"""DMD2 hooks: the critic group learns and scores the student's samples; the actor follows the scores."""

from __future__ import annotations

import dataclasses

import torch

from miles.backends.fsdp_utils.ema import reshard_model
from miles.backends.fsdp_utils.input_dtype_policy import apply_input_dtype_policy
from miles.backends.fsdp_utils.loss_hub.sft import prepare_sft_batch, sft_loss_formula
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, PreparedBatch
from miles.backends.fsdp_utils.metrics import record_rollout_train_abs_diff
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
    pad_to_len: int | None = None,
) -> PreparedBatch:
    """The student's step ``i`` on its own rollout input ``x_i``, unguided."""
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
    metrics: MetricBuffer,
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
        metrics.emit_mean("dmd2_grad_abs_mean", total=grad.abs().mean(dim=reduce_dims).sum(), count=bsz)
        rollout_x0 = torch.stack([pair["x0"] for pair in batch]).to(device=x0.device, dtype=torch.float32)
        record_rollout_train_abs_diff(metrics, "x0", x0, rollout_x0)
    return loss_sum
