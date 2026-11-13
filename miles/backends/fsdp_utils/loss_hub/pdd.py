"""PDD prepare/loss hooks. The FSDP actor owns every DiT forward: the student's, and the
teacher's midpoint solve from the rollout's interval state, requested by prepare.

Every latent stream of the family moves on its own grid; one
sampled interval and one joint teacher call cover all streams.
"""

import torch

from miles.distillation.integration import pdd_family
from miles.distillation.pdd import block_update, interval_loss, teacher_mean_velocity
from miles.utils.hash_utils import stable_hash

from .types import PreparedBatch


def prepare_pdd_batch(ctx, batch, *, microbatch_id, optim_step_idx, pad_to_len=None):
    if len(batch) != 1 or pad_to_len is not None:
        raise ValueError("PDD supports one unpadded sample per microbatch")
    pair = batch[0]
    args = ctx.args
    family = pdd_family(args)
    start, lane = pair["pdd_start"], pair["sample_index"]
    if start not in range(0, args.pdd_num_intervals, args.pdd_block_size):
        raise ValueError("Invalid PDD rollout block index")
    grids = {stream: grid.to(ctx.device) for stream, grid in family.grids(args).items()}
    generator = torch.Generator().manual_seed(stable_hash("pdd-interval", args.seed, ctx.rollout_id, lane, start))
    selected = int(torch.randint(start, start + args.pdd_block_size, (), generator=generator))
    # The block and the teacher both start from fixed-weight SGLang rollout states (indexed by interval).
    trajectory = pair["pdd_trajectory"]
    if set(trajectory) != set(family.heads):
        raise ValueError(f"PDD rollout streams {sorted(trajectory)} differ from {sorted(family.heads)}")
    states, x_selected = (
        {stream: trajectory[stream][i].to(ctx.device).unsqueeze(0).float() for stream in family.heads}
        for i in (start, selected)
    )
    if not all(torch.isfinite(state).all() for state in states.values()):
        raise ValueError("Invalid PDD rollout latent values")
    condition = ctx.train_pipeline_config.prepare_cond_kwargs(pair["denoising_env"].pos_cond_kwargs, ctx.device)
    # FP32 conditioning boundary; autocast controls matmuls.
    condition = {
        k: (
            (v.to(ctx.device).float() if v.is_floating_point() else v.to(ctx.device))
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in condition.items()
    }
    condition.update(family.cond_extras)

    def time(time_fn, index):
        return {stream: time_fn(args, stream, grid[index].reshape(1)).to(ctx.device) for stream, grid in grids.items()}

    def solve(forward):
        def teacher(x, sigma):
            timesteps = {
                stream: family.teacher_time(args, stream, value.reshape(1)) for stream, value in sigma.items()
            }
            pred = forward(x, timesteps)
            return {stream: pred[stream].float() for stream in x}

        sigma = {stream: grid[selected] for stream, grid in grids.items()}
        delta = {stream: grid.diff()[selected] for stream, grid in grids.items()}
        return teacher_mean_velocity(teacher, x_selected, sigma, delta, solver="midpoint")

    return PreparedBatch(
        latents=states,
        timesteps={stream: grid[start].reshape(1) for stream, grid in grids.items()},
        timesteps_for_model=time(family.model_time, start),
        model=ctx.models["transformer"],
        component_name="transformer",
        guidance_scale=1.0,
        use_cfg=False,
        cfg_batching=False,
        true_cfg_scale=1.0,
        pos_cond=condition,
        neg_cond=None,
        joint_cond=None,
        advantage=torch.ones(1, device=ctx.device),
        extras={"start": start, "selected": selected, "lane": lane, "grids": grids},
        teacher_guidance_scale=1.0,
        teacher_solver=solve,
    )


def pdd_loss_formula(
    ctx,
    batch,
    prepared,
    *,
    new_pred,
    old_pred,
    ref_pred,
    teacher_pred=None,
    metrics,
    optim_step_idx,
    train_output_inputs,
    write_old_log_prob=False,
    old_log_prob_from_new=False,
):
    if old_pred is not None or ref_pred is not None or write_old_log_prob or old_log_prob_from_new:
        raise ValueError("PDD does not consume policy/reference predictions")
    if teacher_pred is None:
        raise ValueError("PDD needs the teacher target its prepare hook requested")
    n = ctx.args.pdd_num_intervals
    start, selected, grids = (prepared.extras[k] for k in ("start", "selected", "grids"))
    end = start + ctx.args.pdd_block_size
    per_sample_loss = 0
    for stream, x in prepared.latents.items():
        raw = new_pred[stream]
        if raw.shape != (*x.shape[:-1], n * x.shape[-1]):
            raise ValueError(f"PDD student {stream} output shape does not match the parallel heads")
        predictions = raw.reshape(*x.shape[:-1], n, x.shape[-1]).permute(2, 0, 1, 3)
        stream_loss = interval_loss(predictions, selected, teacher_pred[stream])
        if not torch.isfinite(stream_loss).all():
            raise FloatingPointError(f"nonfinite PDD {stream} loss")
        per_sample_loss = per_sample_loss + stream_loss
        with torch.no_grad():
            following = block_update(x, predictions, grids[stream], start, end).detach()
            if not torch.isfinite(following).all():
                raise FloatingPointError(f"nonfinite PDD {stream} trajectory")
            # Trainer vs. engine next state; later blocks also include rollout staleness.
            expected = batch[0]["pdd_trajectory"][stream][end].to(ctx.device).unsqueeze(0).float()
            error = (following - expected).abs()
            prefix = "pdd" if stream == "visual" else f"pdd_{stream}"
            metrics.emit_mean(f"{prefix}_next_latent_mae", total=error.sum(), count=error.numel())
            relative = error.mean() / (expected - x).abs().mean().clamp_min(1e-6)
            metrics.emit_mean(f"{prefix}_update_relative_error", total=relative, count=1)
            if len(prepared.latents) > 1:
                metrics.emit_mean(f"pdd_{stream}_loss", total=stream_loss.sum(), count=len(batch))
    with torch.no_grad():
        metrics.emit_mean("loss", total=per_sample_loss.sum().detach(), count=len(batch))
    return per_sample_loss.sum()
