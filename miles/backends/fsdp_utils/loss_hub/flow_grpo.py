"""Flow-GRPO batch preparation and loss formula."""

from __future__ import annotations

import torch

from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, PreparedBatch
from miles.backends.fsdp_utils.metrics import record_rollout_train_abs_diff
from miles.utils.metric_buffer import MetricBuffer
from miles.utils.train_data_utils import stack_train_pair_rollout_debug


def _stack_pair_field(batch: list[dict], key: str, device: torch.device) -> torch.Tensor:
    return torch.stack([pair[key] for pair in batch]).to(device=device, dtype=torch.float32)


def prepare_flow_grpo_batch(
    ctx: DiffusionLossContext,
    batch: list[dict],
    *,
    pad_to_len: int | None = None,
) -> PreparedBatch:
    """Stack SDE-pair fields and build CFG conditioning (guidance from args)."""
    args = ctx.args
    device = ctx.device
    config = ctx.train_pipeline_config
    num_train_timesteps = int(ctx.scheduler.config.num_train_timesteps)
    bsz = len(batch)

    latents = _stack_pair_field(batch, "latent", device)
    next_latents = _stack_pair_field(batch, "next_latent", device)
    timesteps = _stack_pair_field(batch, "timestep", device)
    next_timesteps = _stack_pair_field(batch, "next_timestep", device)
    log_prob_old = _stack_pair_field(batch, "log_prob_old", device)
    advantage = torch.tensor(
        [float(pair["advantage"]) for pair in batch],
        device=device,
        dtype=torch.float32,
    )
    advantage = torch.clamp(advantage, -args.diffusion_adv_clip_max, args.diffusion_adv_clip_max)

    guidance_scale = args.diffusion_guidance_scale
    true_cfg_scale = args.diffusion_true_cfg_scale
    cfg_scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
    use_cfg = cfg_scale > 1.0  # matches sglang do_cfg: guidance<=1 runs single-branch

    if len(ctx.models) == 1:
        component_name, model = next(iter(ctx.models.items()))
    else:
        components = {config.component_for_timestep(t, num_train_timesteps) for t in timesteps.tolist()}
        if len(components) > 1:
            raise ValueError(
                f"Micro-batch mixes denoising phases {sorted(components)}; set "
                "--micro-batch-size 1 so each forward is phase-pure (one DiT, one CFG scale)."
            )
        component_name = components.pop()
        model = ctx.models[component_name]
        guidance_scale = config.select_guidance_scale(
            float(timesteps[0]),
            num_train_timesteps,
            guidance_scale,
            args.diffusion_guidance_scale_2,
        )

    timesteps_for_model = config.process_timestep_as_input(timesteps)

    pos_list = [config.prepare_cond_kwargs(batch[i]["denoising_env"].pos_cond_kwargs, device) for i in range(bsz)]
    neg_list = (
        [config.prepare_cond_kwargs(batch[i]["denoising_env"].neg_cond_kwargs, device) for i in range(bsz)]
        if use_cfg
        else None
    )
    cfg_batching = use_cfg and config.cfg_batching
    joint_cond = pos_cond = neg_cond = None
    if cfg_batching:
        joint_cond = config.collate_cond_for_sample_batch(pos_list + neg_list, device, pad_to_len=pad_to_len)
    else:
        pos_cond = config.collate_cond_for_sample_batch(pos_list, device, pad_to_len=pad_to_len)
        if use_cfg and neg_list is not None:
            neg_cond = config.collate_cond_for_sample_batch(neg_list, device, pad_to_len=pad_to_len)

    return PreparedBatch(
        latents=latents,
        timesteps=timesteps,
        timesteps_for_model=timesteps_for_model,
        model=model,
        component_name=component_name,
        guidance_scale=guidance_scale,
        use_cfg=use_cfg,
        cfg_batching=cfg_batching,
        true_cfg_scale=true_cfg_scale if use_cfg else None,
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        joint_cond=joint_cond,
        advantage=advantage,
        extras={
            "next_latents": next_latents,
            "next_timesteps": next_timesteps,
            "log_prob_old": log_prob_old,
        },
    )


def flow_grpo_loss_formula(
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
    """SDE log-prob + PPO-clip (+ optional KL vs ``ref_pred``). Actor owns DiT forward."""
    args = ctx.args
    clip_range = args.diffusion_clip_range
    noise_level = args.diffusion_noise_level
    kl_beta = float(args.diffusion_kl_beta)

    next_latents = prepared.extras["next_latents"]
    next_timesteps = prepared.extras["next_timesteps"]
    log_prob_old_rollout = prepared.extras["log_prob_old"]

    _, log_prob_new, prev_sample_mean_new, std_dev_t_new = ctx.sde_backend.sde_step_logprob(
        new_pred.float(),
        prepared.timesteps,
        next_timesteps,
        prepared.latents.float(),
        prev_sample=next_latents.float(),
        noise_level=noise_level,
    )

    if write_old_log_prob:
        for pair, log_prob in zip(batch, log_prob_new, strict=True):
            pair["log_prob_old"] = log_prob.cpu()
        return None

    log_prob_old = log_prob_new.detach() if old_log_prob_from_new else log_prob_old_rollout
    ratio = torch.exp(log_prob_new - log_prob_old)
    unclipped = -prepared.advantage * ratio
    clipped = -prepared.advantage * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    per_pair_loss = torch.maximum(unclipped, clipped)
    loss_sum = per_pair_loss.sum()
    bsz = len(batch)

    kl_sum = loss_sum.new_zeros(())
    if kl_beta > 0:
        if ref_pred is None:
            raise ValueError("Flow-GRPO KL requires a reference DiT forward (actor ref_mode=lora_base)")
        _, _, prev_sample_mean_ref, _ = ctx.sde_backend.sde_step_logprob(
            ref_pred.float(),
            prepared.timesteps,
            next_timesteps,
            prepared.latents.float(),
            prev_sample=next_latents.float(),
            noise_level=noise_level,
        )
        kl_per_pair = ((prev_sample_mean_new - prev_sample_mean_ref) ** 2).mean(
            dim=tuple(range(1, prev_sample_mean_new.ndim)),
            keepdim=True,
        ) / (2 * std_dev_t_new**2)
        loss_sum = loss_sum + kl_beta * kl_per_pair.sum()
        kl_sum = kl_per_pair.sum()

    with torch.no_grad():
        metrics.emit_mean("loss", total=loss_sum, count=bsz)
        metrics.emit_mean("policy_loss", total=per_pair_loss.sum(), count=bsz)
        metrics.emit_mean("kl_loss", total=kl_sum, count=bsz)
        metrics.emit_mean("loss_abs_mean", total=per_pair_loss.abs().sum(), count=bsz)
        metrics.emit_mean("adv_abs_mean", total=prepared.advantage.abs().sum(), count=bsz)
        metrics.emit_mean("ratio_abs_minus_1", total=(ratio - 1.0).abs().sum(), count=bsz)
        metrics.emit_mean("approx_kl", total=0.5 * ((log_prob_new - log_prob_old) ** 2).sum(), count=bsz)
        metrics.emit_mean("clipfrac", total=(torch.abs(ratio - 1.0) > clip_range).float().sum(), count=bsz)
        metrics.emit_mean("log_prob_new_idx_0", total=log_prob_new[0], count=1)
        metrics.emit_mean("log_prob_old_idx_0", total=log_prob_old[0], count=1)
        log_prob_abs_diff_sum = torch.abs(log_prob_new - log_prob_old).sum()
        metrics.emit_mean("log_prob_mean_abs_diff", total=log_prob_abs_diff_sum, count=bsz)
        if len(ctx.models) > 1:
            metrics.emit_mean(
                f"log_prob_mean_abs_diff_{prepared.component_name}",
                total=log_prob_abs_diff_sum,
                count=bsz,
            )

        rollout_model_output = stack_train_pair_rollout_debug(batch, "rollout_step_model_output")
        if rollout_model_output is not None:
            record_rollout_train_abs_diff(
                metrics,
                "model_output",
                new_pred.float(),
                rollout_model_output.to(device=ctx.device, dtype=torch.float32),
                component=prepared.component_name if len(ctx.models) > 1 else None,
            )

    return loss_sum
