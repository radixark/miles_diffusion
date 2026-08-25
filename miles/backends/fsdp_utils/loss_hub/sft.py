"""SFT batch preparation and loss formula (rectified-flow velocity MSE)."""

from __future__ import annotations

import torch
import torch.nn as nn

from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, PreparedBatch
from miles.backends.fsdp_utils.metrics import sigma_bucket_key
from miles.utils.hash_utils import stable_hash
from miles.utils.metric_buffer import MetricBuffer


def sample_grid_indices(
    ctx: DiffusionLossContext,
    bsz: int,
    *,
    generator: torch.Generator,
) -> tuple[str, nn.Module, torch.Tensor]:
    """Pick one rank-aligned DiT component, then per-sample grid indices."""
    num_grid = len(ctx.scheduler.timesteps)
    config = ctx.train_pipeline_config
    if len(ctx.models) == 1:
        component_name, model = next(iter(ctx.models.items()))
        component_for_timestep = getattr(config, "component_for_timestep", None)
        if component_for_timestep is None:
            pool = torch.arange(num_grid, device=generator.device)
        else:
            num_train_timesteps = int(ctx.scheduler.config.num_train_timesteps)
            pool = torch.tensor(
                [
                    i
                    for i, timestep in enumerate(ctx.scheduler.timesteps)
                    if component_for_timestep(float(timestep), num_train_timesteps) == component_name
                ],
                device=generator.device,
            )
    else:
        num_train_timesteps = int(ctx.scheduler.config.num_train_timesteps)
        components = [config.component_for_timestep(float(t), num_train_timesteps) for t in ctx.scheduler.timesteps]
        expert_generator = torch.Generator().manual_seed(
            stable_hash("expert", int(ctx.args.seed), ctx.rollout_id, ctx.microbatch_id)
        )
        component_name = components[int(torch.randint(num_grid, (1,), generator=expert_generator))]
        model = ctx.models[component_name]
        pool = torch.tensor(
            [i for i, name in enumerate(components) if name == component_name],
            device=generator.device,
        )

    draw = torch.randint(len(pool), (bsz,), device=generator.device, generator=generator)
    return component_name, model, pool[draw]


def prepare_sft_batch(
    ctx: DiffusionLossContext,
    batch: list[dict],
    *,
    pad_to_len: int | None = None,
) -> PreparedBatch:
    """Corrupt cached clean latents at sampled grid sigmas; CFG-free cached cond."""
    device = ctx.device
    config = ctx.train_pipeline_config
    bsz = len(batch)

    x0 = torch.stack([pair["latent"] for pair in batch]).to(device=device, dtype=torch.float32)
    sample_generator = torch.Generator(device=device).manual_seed(
        stable_hash("sample", int(ctx.args.seed), ctx.rollout_id, ctx.microbatch_id, ctx.dp_rank)
    )
    component_name, model, idx = sample_grid_indices(ctx, bsz, generator=sample_generator)
    timesteps = ctx.scheduler.timesteps[idx].to(dtype=torch.float32)
    sigmas = ctx.scheduler.sigmas[idx].to(dtype=torch.float32)

    noise = torch.randn(x0.shape, device=device, dtype=torch.float32, generator=sample_generator)
    sigma_exp = sigmas.view(bsz, *([1] * (x0.ndim - 1)))
    latents = (1.0 - sigma_exp) * x0 + sigma_exp * noise

    cond_list = [{key: value.to(device) for key, value in pair["cond_kwargs"].items()} for pair in batch]
    pos_cond = config.collate_cond_for_sample_batch(cond_list, device, pad_to_len=pad_to_len)

    return PreparedBatch(
        latents=latents,
        timesteps=timesteps,
        timesteps_for_model=config.process_timestep_as_input(timesteps),
        model=model,
        component_name=component_name,
        guidance_scale=0.0,
        use_cfg=False,
        cfg_batching=False,
        true_cfg_scale=None,
        pos_cond=pos_cond,
        neg_cond=None,
        joint_cond=None,
        advantage=torch.ones(bsz, device=device, dtype=torch.float32),
        extras={"target": noise - x0, "sigmas": sigmas},
    )


def sft_loss_formula(
    ctx: DiffusionLossContext,
    batch: list[dict],
    prepared: PreparedBatch,
    *,
    new_pred: torch.Tensor,
    ref_pred: torch.Tensor | None,
    metrics: MetricBuffer,
    write_old_log_prob: bool = False,
    old_log_prob_from_new: bool = False,
) -> torch.Tensor:
    """Velocity-target MSE: ``||pred - (eps - x0)||^2`` averaged per pair."""
    target = prepared.extras["target"]
    per_pair = ((new_pred.float() - target) ** 2).mean(dim=tuple(range(1, target.ndim)))
    loss_sum = per_pair.sum()

    with torch.no_grad():
        metrics.emit_mean("loss", total=loss_sum, count=len(batch))
        num_buckets = ctx.args.log_loss_sigma_bucket
        for pair_loss, sigma in zip(per_pair, prepared.extras["sigmas"], strict=True):
            bucket = min(int(float(sigma) * num_buckets), num_buckets - 1)
            metrics.emit_mean(sigma_bucket_key(bucket, num_buckets), total=pair_loss, count=1)
    return loss_sum
