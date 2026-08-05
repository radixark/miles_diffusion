"""SFT batch preparation and loss formula (rectified-flow velocity MSE)."""

from __future__ import annotations

import torch
import torch.nn as nn

from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, PreparedBatch
from miles.backends.fsdp_utils.loss_hub.utils import cast_cond_to_dtype
from miles.utils.metric_buffer import MetricBuffer


def sample_grid_indices(ctx: DiffusionLossContext, bsz: int) -> tuple[str, nn.Module, torch.Tensor]:
    """Pick one DiT component per micro-batch (phase-pure), then grid indices within its range."""
    num_grid = len(ctx.scheduler.timesteps)
    if len(ctx.models) == 1:
        component_name, model = next(iter(ctx.models.items()))
        return component_name, model, torch.randint(num_grid, (bsz,))

    # A uniform anchor index picks each expert with probability equal to its share of the
    # grid, keeping the marginal over indices uniform while the micro-batch stays single-expert.
    num_train_timesteps = int(ctx.scheduler.config.num_train_timesteps)
    config = ctx.train_pipeline_config
    components = [config.component_for_timestep(float(t), num_train_timesteps) for t in ctx.scheduler.timesteps]
    component_name = components[int(torch.randint(num_grid, (1,)))]
    pool = torch.tensor([i for i, name in enumerate(components) if name == component_name])
    return component_name, ctx.models[component_name], pool[torch.randint(len(pool), (bsz,))]


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
    component_name, model, idx = sample_grid_indices(ctx, bsz)
    idx = idx.to(device)
    timesteps = ctx.scheduler.timesteps[idx].to(dtype=torch.float32)
    sigmas = ctx.scheduler.sigmas[idx].to(dtype=torch.float32)

    noise = torch.randn(x0.shape, device=device, dtype=torch.float32)
    sigma_exp = sigmas.view(bsz, *([1] * (x0.ndim - 1)))
    latents = (1.0 - sigma_exp) * x0 + sigma_exp * noise

    num_train_timesteps = int(ctx.scheduler.config.num_train_timesteps)
    if config.needs_timestep_scaling:
        timesteps_for_model = timesteps / float(num_train_timesteps)
    else:
        timesteps_for_model = timesteps

    cond_list = [{key: value.to(device) for key, value in pair["cond_kwargs"].items()} for pair in batch]
    pos_cond = cast_cond_to_dtype(
        config.collate_cond_for_sample_batch(cond_list, device, pad_to_len=pad_to_len),
        ctx.forward_dtype,
    )

    return PreparedBatch(
        latents=latents,
        timesteps=timesteps,
        timesteps_for_model=timesteps_for_model,
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
        extras={"target": noise - x0},
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
    return loss_sum
