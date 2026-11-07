"""PDD prepare/loss hooks. The normal FSDP actor owns student forward/backward."""

import torch

from miles.distillation.integration import embedded_sigmas, engine_timesteps, make_grid
from miles.distillation.pdd import block_update, pdd_loss
from miles.utils.hash_utils import stable_hash

from .types import PreparedBatch


def prepare_pdd_batch(ctx, batch, *, pad_to_len=None):
    if len(batch) != 1 or pad_to_len is not None:
        raise ValueError("PDD supports one unpadded sample per microbatch")
    pair = batch[0]
    start, lane = pair["pdd_start"], pair["sample_index"]
    runtime = ctx.algorithm_state
    if "grid" not in runtime:
        runtime["grid"] = make_grid(
            ctx.args.hf_checkpoint, ctx.args.pdd_num_intervals, (ctx.args.diffusion_height // 16) ** 2, ctx.device
        )
    grid = runtime["grid"]
    model = ctx.models["transformer"]
    if start not in range(0, ctx.args.pdd_num_intervals, ctx.args.pdd_block_size):
        raise ValueError("Invalid PDD rollout block index")
    # Every block starts from the fixed-weight SGLang rollout state.
    state = pair["pdd_trajectory"][start // ctx.args.pdd_block_size].to(ctx.device).unsqueeze(0).float()
    shape = (1, (ctx.args.diffusion_height // 16) ** 2, model.config.in_channels)
    if state.shape != shape or not torch.isfinite(state).all():
        raise ValueError("Invalid PDD rollout latent shape or values")
    generator = torch.Generator().manual_seed(stable_hash("pdd-interval", ctx.args.seed, ctx.rollout_id, lane, start))
    selected = int(torch.randint(start, start + ctx.args.pdd_block_size, (), generator=generator))
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
    # The student sees the time value SGLang embedded; the grid dynamics stay unchanged.
    model_sigma = embedded_sigmas(engine_timesteps(grid[start].reshape(1))).to(ctx.device)
    return PreparedBatch(
        latents={"visual": state},
        timesteps={"visual": grid[start].reshape(1)},
        timesteps_for_model={"visual": model_sigma},
        model=model,
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
    )


def pdd_loss_formula(
    ctx,
    batch,
    prepared,
    *,
    new_pred,
    old_pred,
    ref_pred,
    metrics,
    write_old_log_prob=False,
    old_log_prob_from_new=False,
):
    if old_pred is not None or ref_pred is not None or write_old_log_prob or old_log_prob_from_new:
        raise ValueError("PDD does not consume policy/reference predictions")
    x = prepared.latents["visual"]
    n = ctx.args.pdd_num_intervals
    raw = new_pred["visual"]
    if raw.shape != (*x.shape[:-1], n * x.shape[-1]):
        raise ValueError("PDD student output shape does not match the parallel heads")
    predictions = raw.reshape(*x.shape[:-1], n, x.shape[-1]).permute(2, 0, 1, 3)
    teacher = ctx.teacher_models["transformer"]

    def teacher_forward(state, sigma):
        with torch.autocast(ctx.device.type, dtype=ctx.forward_dtype, enabled=ctx.forward_dtype != torch.float32):
            return ctx.train_pipeline_config.compute_noise_pred(
                model=teacher,
                latents_input={"visual": state.bfloat16()},
                timesteps_input={"visual": sigma.expand(state.shape[0])},
                pos_cond={
                    k: v.bfloat16() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                    for k, v in prepared.pos_cond.items()
                },
                neg_cond=None,
                joint_cond=None,
                use_cfg=False,
                cfg_batching=False,
                guidance_scale=1.0,
                true_cfg_scale=1.0,
            )["visual"].float()

    start, selected, grid = (prepared.extras[k] for k in ("start", "selected", "grid"))
    with torch.autocast(ctx.device.type, dtype=ctx.forward_dtype, enabled=ctx.forward_dtype != torch.float32):
        per_sample_loss = pdd_loss(predictions, x, grid, start, selected, teacher_forward, solver="midpoint")
    if not torch.isfinite(per_sample_loss).all():
        raise FloatingPointError("nonfinite PDD loss")
    end = start + ctx.args.pdd_block_size
    with torch.no_grad():
        following = block_update(x, predictions, grid, start, end).detach()
        if not torch.isfinite(following).all():
            raise FloatingPointError("nonfinite PDD trajectory")
        # Trainer vs. engine next state; later blocks also include rollout staleness.
        expected = batch[0]["pdd_trajectory"][end // ctx.args.pdd_block_size].to(ctx.device).unsqueeze(0).float()
        error = (following - expected).abs()
        metrics.emit_mean("pdd_next_latent_mae", total=error.sum(), count=error.numel())
        relative = error.mean() / (expected - x).abs().mean().clamp_min(1e-6)
        metrics.emit_mean("pdd_update_relative_error", total=relative, count=1)
        loss_sum = per_sample_loss.sum()
        metrics.emit_mean("loss", total=loss_sum.detach(), count=len(batch))
    return per_sample_loss.sum()
