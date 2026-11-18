"""PDD prepare/loss hooks. The FSDP actor owns every DiT forward: the student's, and the
teacher's midpoint solve from the rollout's interval state, requested by prepare.
"""

import torch

from miles.distillation.integration import model_time, pdd_grid
from miles.distillation.pdd import block_update, interval_loss, teacher_mean_velocity
from miles.utils.hash_utils import stable_hash

from .types import PreparedBatch

STREAM = "visual"


def prepare_pdd_batch(ctx, batch, *, microbatch_id, optim_step_idx, pad_to_len=None):
    if len(batch) != 1 or pad_to_len is not None:
        raise ValueError("PDD supports one unpadded sample per microbatch")
    pair = batch[0]
    args = ctx.args
    start, lane = pair["pdd_start"], pair["sample_index"]
    if start not in range(0, args.pdd_num_intervals, args.pdd_block_size):
        raise ValueError("Invalid PDD rollout block index")
    grid = pdd_grid(args).to(ctx.device)
    generator = torch.Generator().manual_seed(stable_hash("pdd-interval", args.seed, ctx.rollout_id, lane, start))
    selected = int(torch.randint(start, start + args.pdd_block_size, (), generator=generator))
    # The block and the teacher both start from fixed-weight SGLang rollout states (indexed by interval).
    trajectory = pair["pdd_trajectory"]
    x_start, x_selected = (trajectory[i].to(ctx.device).unsqueeze(0).float() for i in (start, selected))
    if not torch.isfinite(x_start).all():
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

    def solve(forward):
        def teacher(x, sigma):
            return forward({STREAM: x}, {STREAM: sigma.reshape(1)})[STREAM].float()

        target = teacher_mean_velocity(teacher, x_selected, grid[selected], grid.diff()[selected], solver="midpoint")
        return {STREAM: target}

    return PreparedBatch(
        latents={STREAM: x_start},
        timesteps={STREAM: grid[start].reshape(1)},
        timesteps_for_model={STREAM: model_time(grid[start].reshape(1)).to(ctx.device)},
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
        extras={"start": start, "selected": selected, "lane": lane, "grid": grid},
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
    start, selected, grid = (prepared.extras[k] for k in ("start", "selected", "grid"))
    end = start + ctx.args.pdd_block_size
    x, raw = prepared.latents[STREAM], new_pred[STREAM]
    if raw.shape != (*x.shape[:-1], n * x.shape[-1]):
        raise ValueError("PDD student output shape does not match the parallel heads")
    # [B, S, N*C] -> [N, B, S, C]
    predictions = raw.reshape(*x.shape[:-1], n, x.shape[-1]).permute(2, 0, 1, 3)
    per_sample_loss = interval_loss(predictions, selected, teacher_pred[STREAM])
    if not torch.isfinite(per_sample_loss).all():
        raise FloatingPointError("nonfinite PDD loss")
    with torch.no_grad():
        following = block_update(x, predictions, grid, start, end)
        if not torch.isfinite(following).all():
            raise FloatingPointError("nonfinite PDD trajectory")
        # Trainer vs. engine next state; later blocks also include rollout staleness.
        expected = batch[0]["pdd_trajectory"][end].to(ctx.device).unsqueeze(0).float()
        error = (following - expected).abs()
        metrics.emit_mean("pdd_next_latent_mae", total=error.sum(), count=error.numel())
        relative = error.mean() / (expected - x).abs().mean().clamp_min(1e-6)
        metrics.emit_mean("pdd_update_relative_error", total=relative, count=1)
        metrics.emit_mean("loss", total=per_sample_loss.sum(), count=len(batch))
    return per_sample_loss.sum()
