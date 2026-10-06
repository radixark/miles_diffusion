"""SFT batch preparation and loss formula (rectified-flow velocity MSE)."""

from __future__ import annotations

import torch
import torch.nn as nn

from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, FlowNoiseSchedule, PreparedBatch
from miles.backends.fsdp_utils.metrics import sigma_bucket_key
from miles.utils.hash_utils import stable_hash
from miles.utils.metric_buffer import MetricBuffer


def select_component(
    ctx: DiffusionLossContext, noise_schedule: FlowNoiseSchedule, device: torch.device
) -> tuple[str, nn.Module, torch.Tensor]:
    """Pick one rank-aligned DiT component and the indices of the grid points it serves."""
    config = ctx.train_pipeline_config
    timesteps = noise_schedule.timesteps.tolist()
    component_for_timestep = getattr(config, "component_for_timestep", None)
    if len(ctx.models) == 1:
        component_name, model = next(iter(ctx.models.items()))
        if component_for_timestep is None:
            return component_name, model, torch.arange(len(timesteps), device=device)
    else:
        # dp_rank stays out of this seed: every rank must forward the same expert.
        expert_generator = torch.Generator().manual_seed(
            stable_hash("expert", int(ctx.args.seed), ctx.rollout_id, ctx.microbatch_id)
        )
        drawn = int(torch.randint(len(timesteps), (1,), generator=expert_generator))
        component_name = component_for_timestep(timesteps[drawn])
        model = ctx.models[component_name]
    served = [i for i, timestep in enumerate(timesteps) if component_for_timestep(timestep) == component_name]
    return component_name, model, torch.tensor(served, device=device)


def prepare_sft_batch(
    ctx: DiffusionLossContext,
    batch: list[dict],
    *,
    pad_to_len: int | None = None,
) -> PreparedBatch:
    """Corrupt every latent stream at its own sampled grid sigma; CFG-free cached cond.

    The visual grid picks the DiT component; every stream then draws uniformly from the grid points it may use.
    """
    device = ctx.device
    config = ctx.train_pipeline_config
    bsz = len(batch)
    stream_names = set(batch[0]["latent"])
    if any(set(pair["latent"]) != stream_names for pair in batch):
        raise ValueError(f"Every sample in an SFT microbatch must have the latent streams {sorted(stream_names)}")

    noise_schedules = {
        stream_name: FlowNoiseSchedule.shifted(ctx.args.fsdp_flow_shift[stream_name], config.num_train_timesteps)
        for stream_name in stream_names
    }
    component_name, model, visual_pool = select_component(ctx, noise_schedules["visual"], device)

    latents, timesteps, sigmas, targets = {}, {}, {}, {}
    for stream_name in sorted(stream_names):
        x0 = torch.stack([pair["latent"][stream_name] for pair in batch]).to(device=device, dtype=torch.float32)
        noise_schedule = noise_schedules[stream_name]
        # The visual stream keeps the single-stream seed, so single-stream runs draw as before; other streams add
        # their name, so adding a stream never changes another stream's draws.
        seed_parts = ("sample", int(ctx.args.seed), ctx.rollout_id, ctx.microbatch_id, ctx.dp_rank)
        generator = torch.Generator(device=device).manual_seed(
            stable_hash(*seed_parts) if stream_name == "visual" else stable_hash(*seed_parts, stream_name)
        )
        # Only the visual stream is held to the chosen component's grid points.
        pool = visual_pool if stream_name == "visual" else torch.arange(len(noise_schedule.timesteps), device=device)
        idx = pool[torch.randint(len(pool), (bsz,), device=device, generator=generator)]
        timesteps[stream_name] = noise_schedule.timesteps.to(device=device, dtype=torch.float32)[idx]
        sigmas[stream_name] = noise_schedule.sigmas.to(device=device, dtype=torch.float32)[idx]

        noise = torch.randn(x0.shape, device=device, dtype=torch.float32, generator=generator)
        sigma_exp = sigmas[stream_name].view(bsz, *([1] * (x0.ndim - 1)))
        latents[stream_name] = (1.0 - sigma_exp) * x0 + sigma_exp * noise
        targets[stream_name] = noise - x0

    # Only top-level tensors move here. A family whose cond values nest tensors
    # inside dicts moves them itself in its collate/forward.
    cond_list = [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in pair["cond_kwargs"].items()
        }
        for pair in batch
    ]
    pos_cond = config.collate_cond_for_sample_batch(cond_list, device, pad_to_len=pad_to_len)

    return PreparedBatch(
        latents=latents,
        timesteps=timesteps,
        timesteps_for_model={stream_name: config.process_timestep_as_input(t) for stream_name, t in timesteps.items()},
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
        extras={"target": targets, "sigmas": sigmas},
    )


def sft_loss_formula(
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
    """Per pair, sum each stream's own velocity MSE, so a stream's weight does not depend on its size.

    A stream is supervised when the config predicts it and --diffusion-supervised-streams names it; any other stream
    is still noised and fed to the DiT.
    """
    loss_sum = 0.0
    per_pair_losses = {}
    for stream_name, target in prepared.extras["target"].items():
        if stream_name not in new_pred or stream_name not in ctx.args.diffusion_supervised_streams:
            continue
        # Shapes must match exactly: broadcasting would silently average the wrong rows.
        if new_pred[stream_name].shape != target.shape:
            raise ValueError(
                f"Stream {stream_name!r} prediction shape {tuple(new_pred[stream_name].shape)} "
                f"does not match target shape {tuple(target.shape)}"
            )
        per_pair_losses[stream_name] = (
            (new_pred[stream_name].float() - target).square().reshape(len(batch), -1).mean(dim=1)
        )
        loss_sum = loss_sum + per_pair_losses[stream_name].sum()

    with torch.no_grad():
        metrics.emit_mean("loss", total=loss_sum, count=len(batch))
        num_buckets = ctx.args.log_loss_sigma_bucket
        # Buckets follow the visual stream, whose sigma also picks the DiT component; every stream counts in loss.
        if num_buckets > 0:
            for pair_loss, sigma in zip(per_pair_losses["visual"], prepared.extras["sigmas"]["visual"], strict=True):
                bucket = min(int(float(sigma) * num_buckets), num_buckets - 1)
                metrics.emit_mean(sigma_bucket_key(bucket, num_buckets), total=pair_loss, count=1)
    return loss_sum
