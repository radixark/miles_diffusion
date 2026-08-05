"""DiffusionNFT batch preparation and loss formula."""

from __future__ import annotations

import torch

from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, PreparedBatch
from miles.utils.metric_buffer import MetricBuffer


def sample_noise(like: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(like.shape, device=like.device, dtype=like.dtype, generator=generator)


def corrupt(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Linear flow: ``x_t = (1 - t) x_0 + t ε``."""
    while t.ndim < x0.ndim:
        t = t.unsqueeze(-1)
    return (1.0 - t) * x0 + t * eps


def prepare_nft_batch(
    ctx: DiffusionLossContext,
    batch: list[dict],
    *,
    pad_to_len: int | None = None,
) -> PreparedBatch:
    """Corrupt clean x0 at each pair's sigma; CFG-free cond."""
    if len(ctx.models) != 1:
        raise ValueError("DiffusionNFT currently supports a single DiT component (SD3)")
    device = ctx.device
    config = ctx.train_pipeline_config
    bsz = len(batch)
    x0 = torch.stack([pair["x0"] for pair in batch]).to(device=device, dtype=torch.float32)
    t = torch.tensor([float(pair["timestep"]) for pair in batch], device=device, dtype=torch.float32)
    advantage = torch.tensor([float(pair["advantage"]) for pair in batch], device=device, dtype=torch.float32)

    component_name, model = next(iter(ctx.models.items()))
    pos_list = [config.prepare_cond_kwargs(batch[i]["denoising_env"].pos_cond_kwargs, device) for i in range(bsz)]
    # Cond dtypes are set at the model boundary by the family input_dtype_policy (see actor).
    pos_cond = config.collate_cond_for_sample_batch(pos_list, device, pad_to_len=pad_to_len)

    num_train_timesteps = ctx.scheduler.config.num_train_timesteps
    if config.needs_timestep_scaling:
        timesteps_for_model = t.to(dtype=torch.float32)
    else:
        timesteps_for_model = t * float(num_train_timesteps)

    xt = corrupt(x0, t, sample_noise(x0))
    return PreparedBatch(
        latents=xt,
        timesteps=t,
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
        advantage=advantage,
        extras={"x0": x0},
    )


def nft_r_from_advantages(advantages: torch.Tensor, *, adv_clip_max: float) -> torch.Tensor:
    clip = float(adv_clip_max)
    adv_clipped = torch.clamp(advantages, -clip, clip)
    r = (adv_clipped / clip) / 2.0 + 0.5
    return torch.clamp(r, 0.0, 1.0)


def nft_branch_losses(
    *,
    x0: torch.Tensor,
    xt: torch.Tensor,
    t_exp: torch.Tensor,
    new_pred: torch.Tensor,
    old_pred: torch.Tensor,
    beta: float,
    use_adaptive: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduce_dims = tuple(range(1, x0.ndim))
    positive_pred = beta * new_pred + (1.0 - beta) * old_pred
    negative_pred = (1.0 + beta) * old_pred - beta * new_pred
    x0_pos = xt.to(dtype=new_pred.dtype) - t_exp.to(dtype=new_pred.dtype) * positive_pred
    x0_neg = xt.to(dtype=new_pred.dtype) - t_exp.to(dtype=new_pred.dtype) * negative_pred
    x0_tgt = x0.to(dtype=new_pred.dtype)
    if use_adaptive:
        with torch.no_grad():
            weight_pos = (
                (x0_pos.detach().double() - x0_tgt.double()).abs().mean(dim=reduce_dims, keepdim=True).clamp(min=1e-5)
            ).to(dtype=new_pred.dtype)
            weight_neg = (
                (x0_neg.detach().double() - x0_tgt.double()).abs().mean(dim=reduce_dims, keepdim=True).clamp(min=1e-5)
            ).to(dtype=new_pred.dtype)
        pos_loss = ((x0_pos - x0_tgt) ** 2 / weight_pos).mean(dim=reduce_dims)
        neg_loss = ((x0_neg - x0_tgt) ** 2 / weight_neg).mean(dim=reduce_dims)
    else:
        pos_loss = ((x0_pos - x0_tgt) ** 2).mean(dim=reduce_dims)
        neg_loss = ((x0_neg - x0_tgt) ** 2).mean(dim=reduce_dims)
    return pos_loss, neg_loss


def nft_loss_formula(
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
    """Dual-policy x0-MSE. Actor must supply ``ref_pred`` (EMA / LoRA-base)."""
    if ref_pred is None:
        raise ValueError("NFT loss formula requires a reference prediction from the actor")

    args = ctx.args
    beta = args.diffusion_nft_beta
    adv_clip_max = args.diffusion_nft_adv_clip_max
    use_adaptive = args.diffusion_nft_adaptive_weight

    x0 = prepared.extras["x0"]
    t = prepared.timesteps
    t_exp = t.view(len(batch), *([1] * (x0.ndim - 1)))
    r = nft_r_from_advantages(prepared.advantage, adv_clip_max=adv_clip_max)
    pos_loss, neg_loss = nft_branch_losses(
        x0=x0,
        xt=prepared.latents,
        t_exp=t_exp,
        new_pred=new_pred,
        old_pred=ref_pred,
        beta=beta,
        use_adaptive=use_adaptive,
    )
    r_b = r.to(dtype=pos_loss.dtype)
    per_pair = (r_b * pos_loss / beta + (1.0 - r_b) * neg_loss / beta) * adv_clip_max
    loss_sum = per_pair.sum()

    with torch.no_grad():
        num_timesteps = batch[0]["nft_num_timesteps"]
        per_pair_total = per_pair.sum()
        bsz = len(batch)
        metrics.emit_mean("loss", total=per_pair_total * num_timesteps, count=bsz)
        metrics.emit_mean("nft_loss", total=per_pair_total * num_timesteps, count=bsz)
        metrics.emit_mean("nft_loss_per_pair", total=per_pair_total, count=bsz)
        metrics.emit_mean("nft_r_mean", total=r.sum(), count=bsz)
        metrics.emit_mean("nft_pos_loss", total=pos_loss.sum(), count=bsz)
        metrics.emit_mean("nft_neg_loss", total=neg_loss.sum(), count=bsz)
        metrics.emit_mean("nft_adv_mean", total=prepared.advantage.sum(), count=bsz)
        metrics.emit_mean("nft_t_mean", total=t.sum(), count=bsz)
        metrics.emit_mean(
            "nft_num_timesteps",
            total=torch.tensor(float(num_timesteps), device=ctx.device, dtype=torch.float32),
            count=1,
        )
        metrics.emit_mean("adv_abs_mean", total=prepared.advantage.abs().sum(), count=bsz)

    return loss_sum
