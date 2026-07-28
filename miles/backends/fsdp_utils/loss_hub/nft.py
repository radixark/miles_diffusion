"""DiffusionNFT plugin: custom convert + loss *formula*.

Prepare hook lives in ``prepare.py`` (``prepare_nft_batch``). Actor still owns
DiT forward (+ EMA/LoRA-base reference forward).
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

import torch

from miles.backends.fsdp_utils.loss_hub.advantages import grpo_normalize_rewards
from miles.backends.fsdp_utils.loss_hub.context import DiffusionLossContext, PreparedBatch
from miles.utils.metric_buffer import MetricBuffer
from miles.utils.types import Sample

# ---------------------------------------------------------------------------
# Forward-process math (used by prepare_nft_batch in prepare.py)
# ---------------------------------------------------------------------------


def sample_noise(like: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(like.shape, device=like.device, dtype=like.dtype, generator=generator)


def corrupt(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Linear flow: ``x_t = (1 - t) x_0 + t ε``."""
    while t.ndim < x0.ndim:
        t = t.unsqueeze(-1)
    return (1.0 - t) * x0 + t * eps


def resolve_nft_sigmas(
    sigmas_or_scheduler,
    *,
    training_timestep_fraction: float = 0.99,
) -> torch.Tensor:
    if torch.is_tensor(sigmas_or_scheduler):
        ts = sigmas_or_scheduler.detach().float().flatten()
    else:
        raw = getattr(sigmas_or_scheduler, "sigmas", None)
        if raw is None:
            raise ValueError("NFT needs scheduler.sigmas (or a sigma tensor)")
        ts = raw.detach().float().flatten()
    if ts.numel() == 0:
        raise ValueError("scheduler.sigmas is empty")
    if ts.numel() > 1 and torch.isclose(ts[-1], torch.zeros((), dtype=ts.dtype), atol=1e-8):
        ts = ts[:-1]
    frac = float(training_timestep_fraction)
    if frac < 1.0 and ts.numel() > 1:
        keep = max(1, int(ts.numel() * frac))
        ts = ts[:keep]
    if ts.numel() == 0:
        raise ValueError("No training timesteps left after NFT sigma filtering")
    return ts


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


# ---------------------------------------------------------------------------
# Convert (K-expanded pairs)
# ---------------------------------------------------------------------------


def _clean_x0_from_sample(sample: Sample) -> torch.Tensor:
    traj = sample.dit_trajectory
    if traj is None or traj.latents is None or traj.latents.shape[0] < 1:
        raise ValueError(
            f"sample {sample.index} missing dit_trajectory.latents; "
            "NFT needs the final clean latent x0 from rollout"
        )
    return traj.latents[-1].detach().cpu().float()


def convert_samples_to_nft_train_data(args: Namespace, samples: list[Sample]) -> dict[str, Any]:
    """Expand every sample into K ``(x0, t)`` train pairs (sample-major)."""
    raw_rewards, advantages = grpo_normalize_rewards(args, samples)
    if not samples:
        raise ValueError("NFT convert received empty samples")
    first_traj = samples[0].dit_trajectory
    if first_traj is None:
        raise ValueError("sample 0 missing dit_trajectory")
    if first_traj.timesteps is None:
        raise ValueError("NFT needs dit_trajectory.timesteps from rollout")
    num_train_timesteps = int(getattr(args, "diffusion_num_train_timesteps", 1000) or 1000)
    if first_traj.sigmas is not None:
        scheduler_sigmas = first_traj.sigmas.detach().cpu().float()
    else:
        # Match scheduler_meta_from_rollout when sglang omits sigmas (e.g. ODE rollout).
        ts = first_traj.timesteps.detach().cpu().float()
        scheduler_sigmas = torch.cat([ts / float(num_train_timesteps), ts.new_zeros(1)])
    scheduler_meta = {
        "scheduler_timesteps": first_traj.timesteps.detach().cpu().float(),
        "scheduler_sigmas": scheduler_sigmas,
    }
    frac = float(getattr(args, "diffusion_nft_timestep_fraction", 0.99) or 0.99)
    shuffle_t = bool(getattr(args, "diffusion_nft_shuffle_timesteps", True))
    sigmas = resolve_nft_sigmas(scheduler_meta["scheduler_sigmas"], training_timestep_fraction=frac)
    num_timesteps = int(sigmas.numel())

    train_data: list[dict[str, Any]] = []
    for sample, adv, raw in zip(samples, advantages, raw_rewards, strict=True):
        if sample.denoising_env is None:
            raise ValueError(f"sample {sample.index} missing denoising_env")
        x0 = _clean_x0_from_sample(sample)
        sample_sigmas = sigmas[torch.randperm(num_timesteps)] if shuffle_t else sigmas
        for t in sample_sigmas.tolist():
            train_data.append(
                {
                    "x0": x0,
                    "timestep": float(t),
                    "denoising_env": sample.denoising_env,
                    "advantage": float(adv),
                    "raw_reward": float(raw),
                    "sample_index": sample.index,
                    "prompt": sample.prompt,
                    "nft_num_timesteps": num_timesteps,
                }
            )
    return {"train_data": train_data, **scheduler_meta}


# ---------------------------------------------------------------------------
# Loss formula (receives actor's new_pred / ref_pred)
# ---------------------------------------------------------------------------


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
) -> torch.Tensor | None:
    """Dual-policy x0-MSE. Actor must supply ``ref_pred`` (EMA / LoRA-base)."""
    if write_old_log_prob:
        return None
    if old_log_prob_from_new:
        raise ValueError("DiffusionNFT has no PPO log-prob; old_log_prob_from_new is unsupported")
    if ref_pred is None:
        raise ValueError("NFT loss formula requires a reference prediction from the actor")

    args = ctx.args
    beta = float(getattr(args, "diffusion_nft_beta", 1.0) or 1.0)
    if beta <= 0:
        raise ValueError(f"--diffusion-nft-beta must be > 0, got {beta}")
    adv_clip_max = float(getattr(args, "diffusion_nft_adv_clip_max", 5.0) or 5.0)
    use_adaptive = bool(getattr(args, "diffusion_nft_adaptive_weight", True))

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
        num_timesteps = int(batch[0].get("nft_num_timesteps", 0) or 0)
        per_pair_total = per_pair.sum()
        bsz = len(batch)
        metrics.emit_mean("loss", total=per_pair_total * float(max(num_timesteps, 1)), count=bsz)
        metrics.emit_mean("nft_loss", total=per_pair_total * float(max(num_timesteps, 1)), count=bsz)
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


# Actor: always run a reference DiT forward for NFT (EMA preferred).
nft_loss_formula.ref_mode = "ema"
# Same-sample K pairs must stay in one optimizer window.
nft_loss_formula.requires_sample_aligned_windows = True
