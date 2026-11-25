"""Flow-GRPO batch preparation and loss formula."""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from miles.backends.fsdp_utils.loss_hub.score_centering import score_centering_term
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, PreparedBatch, PreparedBatches
from miles.backends.fsdp_utils.metrics import record_prev_mean_diff_over_noise_std, record_rollout_train_abs_diff
from miles.utils.metric_buffer import MetricBuffer
from miles.utils.train_data_utils import stack_train_pair_rollout_debug

if TYPE_CHECKING:
    from miles.backends.fsdp_utils.sde_step_backend import SdeStepOutput


def _stack_pair_field(batch: list[dict], key: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Stack one per-stream pair field across the micro-batch, keyed by latent stream name."""
    return {
        stream_name: torch.stack([pair[key][stream_name] for pair in batch]).to(device=device, dtype=torch.float32)
        for stream_name in batch[0][key]
    }


def _stream_weights(latents: dict[str, torch.Tensor]) -> dict[str, float]:
    """Each stream's share of a pair's latent elements; a lone stream's is exactly 1.0."""
    element_counts = {stream_name: latent[0].numel() for stream_name, latent in latents.items()}
    total = sum(element_counts.values())
    return {stream_name: count / total for stream_name, count in element_counts.items()}


def _combine_streams(value_by_stream: dict[str, torch.Tensor], stream_weights: dict[str, float]) -> torch.Tensor:
    """Weight each stream's value by its share of the elements and add them."""
    # Weight first, then add: multiplying by 1.0 keeps a lone stream's values bit for bit.
    return functools.reduce(
        torch.add, (weight * value_by_stream[stream_name] for stream_name, weight in stream_weights.items())
    )


def flow_grpo_batch(
    ctx: DiffusionLossContext,
    batch: list[dict],
    *,
    microbatch_id: int,
    optim_step_idx: int,
    pad_to_len: int | None = None,
) -> PreparedBatch:
    """Stack each latent stream's SDE-pair fields and build CFG conditioning (guidance from args).

    The pairs carry their own noise, so ``microbatch_id`` and ``optim_step_idx`` are part of every prepare hook's
    signature and unused here.
    """
    args = ctx.args
    device = ctx.device
    config = ctx.train_pipeline_config
    bsz = len(batch)

    latents = _stack_pair_field(batch, "latent", device)
    next_latents = _stack_pair_field(batch, "next_latent", device)
    timesteps = _stack_pair_field(batch, "timestep", device)
    sigmas = _stack_pair_field(batch, "sigma", device)
    next_sigmas = _stack_pair_field(batch, "next_sigma", device)
    sigma_max = _stack_pair_field(batch, "sigma_max", device)
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
        # Only multi-DiT families route timesteps to a phase; the primary stream's timesteps pick it.
        phase_timesteps = timesteps[args.primary_stream]
        components = {config.component_for_timestep(t) for t in phase_timesteps.tolist()}
        if len(components) > 1:
            raise ValueError(
                f"Micro-batch mixes denoising phases {sorted(components)}; set "
                "--micro-batch-size 1 so each forward is phase-pure (one DiT, one CFG scale)."
            )
        component_name = components.pop()
        model = ctx.models[component_name]
        guidance_scale = config.select_guidance_scale(
            float(phase_timesteps[0]),
            guidance_scale,
            args.diffusion_guidance_scale_2,
        )

    timesteps_for_model = {
        stream_name: config.process_timestep_as_input(stream_timesteps)
        for stream_name, stream_timesteps in timesteps.items()
    }

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
            "sigmas": sigmas,
            "next_sigmas": next_sigmas,
            "sigma_max": sigma_max,
            "log_prob_old": log_prob_old,
        },
    )


def prepare_flow_grpo_batch(
    ctx: DiffusionLossContext,
    batch: list[dict],
    *,
    microbatch_id: int,
    optim_step_idx: int,
    pad_to_len: int | None = None,
) -> PreparedBatches:
    """The Flow-GRPO hook: one stage, the training forward on the stacked SDE pairs."""
    return PreparedBatches(
        actor=flow_grpo_batch(
            ctx, batch, microbatch_id=microbatch_id, optim_step_idx=optim_step_idx, pad_to_len=pad_to_len
        )
    )


def flow_grpo_loss_formula(
    ctx: DiffusionLossContext,
    batch: list[dict],
    prepared: PreparedBatch,
    *,
    new_pred: dict[str, torch.Tensor],
    old_pred: dict[str, torch.Tensor] | None,
    ref_pred: dict[str, torch.Tensor] | None,
    teacher_pred: dict[str, torch.Tensor] | None = None,
    teacher_batch: PreparedBatch | None = None,
    actor_forward_pred: dict[str, torch.Tensor] | None = None,
    actor_forward_batch: PreparedBatch | None = None,
    metrics: MetricBuffer,
    optim_step_idx: int,
    train_output_inputs: list,
    write_old_log_prob: bool = False,
    old_log_prob_from_new: bool = False,
) -> torch.Tensor | None:
    """SDE log-prob + PPO-clip (+ optional KL vs ``ref_pred``). Actor owns DiT forward.

    One step's action is every scored stream's next latent; given the current latents the streams' transitions are
    independent, so the action's per-element mean log-prob is the streams' means weighted by element count. An ode
    stream steps without noise: it conditions the forward and carries no log-prob, so it is not scored.
    """
    args = ctx.args
    clip_range = args.diffusion_clip_range
    positive_kl_betas = {stream_name: beta for stream_name, beta in args.diffusion_kl_beta.items() if beta > 0}

    next_latents = prepared.extras["next_latents"]
    scored_latents = {
        stream_name: latents
        for stream_name, latents in prepared.latents.items()
        if stream_name in args.sde_step_backend_path
    }
    stream_weights = _stream_weights(scored_latents)

    def sde_step_logprob_by_stream(
        pred: dict[str, torch.Tensor], stream_names: Iterable[str]
    ) -> dict[str, SdeStepOutput]:
        return {
            stream_name: ctx.sde_backends[stream_name].sde_step_logprob(
                pred[stream_name].float(),
                scored_latents[stream_name].float(),
                prev_sample=next_latents[stream_name].float(),
                noise_level=args.diffusion_noise_level[stream_name],
                sigmas=prepared.extras["sigmas"][stream_name],
                next_sigmas=prepared.extras["next_sigmas"][stream_name],
                sigma_max=prepared.extras["sigma_max"][stream_name],
            )
            for stream_name in stream_names
        }

    step_outs = sde_step_logprob_by_stream(new_pred, scored_latents)
    log_prob_new_by_stream = {stream_name: step_out.log_prob for stream_name, step_out in step_outs.items()}
    log_prob_new = _combine_streams(log_prob_new_by_stream, stream_weights)

    if write_old_log_prob:
        per_pair_log_probs = zip(*log_prob_new_by_stream.values(), strict=True)
        for pair, log_probs in zip(batch, per_pair_log_probs, strict=True):
            pair["log_prob_old"] = {
                stream_name: log_prob.cpu()
                for stream_name, log_prob in zip(log_prob_new_by_stream, log_probs, strict=True)
            }
        return None

    log_prob_old_by_stream = (
        {stream_name: log_prob.detach() for stream_name, log_prob in log_prob_new_by_stream.items()}
        if old_log_prob_from_new
        else prepared.extras["log_prob_old"]
    )
    log_prob_old = _combine_streams(log_prob_old_by_stream, stream_weights)
    ratio = torch.exp(log_prob_new - log_prob_old)
    unclipped = -prepared.advantage * ratio
    clipped = -prepared.advantage * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    per_pair_loss = torch.maximum(unclipped, clipped)

    rollout_prev_means = {}
    for stream_name, step_out in step_outs.items():
        rollout_prev_mean = stack_train_pair_rollout_debug(batch, "rollout_step_prev_sample_mean", stream_name)
        if rollout_prev_mean is not None:
            rollout_prev_means[stream_name] = rollout_prev_mean.to(
                device=step_out.prev_mean.device, dtype=torch.float32
            )
    if args.diffusion_score_centering:
        # The closed form is one stream's Gaussian.
        ((stream_name, step_out),) = step_outs.items()
        per_pair_loss = per_pair_loss + score_centering_term(
            batch,
            step_out,
            rollout_prev_means.get(stream_name),
            stream_name=stream_name,
            next_latents=next_latents[stream_name],
            log_ratio=log_prob_new - log_prob_old,
            advantage=prepared.advantage,
            clip_range=clip_range,
            loss_dtype=per_pair_loss.dtype,
            metrics=metrics,
        )
    loss_sum = per_pair_loss.sum()
    bsz = len(batch)

    kl_sum = loss_sum.new_zeros(())
    kl_sum_by_stream = {}
    if positive_kl_betas:
        if ref_pred is None:
            raise ValueError("Flow-GRPO KL requires a reference DiT forward; set --ref-mode lora_base or ref")
        ref_step_outs = sde_step_logprob_by_stream(
            ref_pred, [stream_name for stream_name in positive_kl_betas if stream_name in scored_latents]
        )
        for stream_name in positive_kl_betas:
            step_out = step_outs.get(stream_name)
            if step_out is not None:
                kl_per_pair = ((step_out.prev_mean - ref_step_outs[stream_name].prev_mean) ** 2).mean(
                    dim=tuple(range(1, step_out.prev_mean.ndim)),
                    keepdim=True,
                ) / (2 * step_out.std_dev_t**2)
            else:
                # An ode stream steps without noise, so its transition has no Gaussian to compare.
                velocity_diff = new_pred[stream_name].float() - ref_pred[stream_name].float()
                kl_per_pair = (velocity_diff**2).mean(dim=tuple(range(1, velocity_diff.ndim)))
            kl_sum_by_stream[stream_name] = kl_per_pair.sum()
        kl_sum = functools.reduce(torch.add, kl_sum_by_stream.values())
        loss_sum = loss_sum + functools.reduce(
            torch.add, (beta * kl_sum_by_stream[stream_name] for stream_name, beta in positive_kl_betas.items())
        )

    with torch.no_grad():
        metrics.emit_mean("loss", total=loss_sum, count=bsz)
        metrics.emit_mean("policy_loss", total=per_pair_loss.sum(), count=bsz)
        metrics.emit_mean("kl_loss", total=kl_sum, count=bsz)
        if len(prepared.latents) > 1:
            for stream_name, stream_kl_sum in kl_sum_by_stream.items():
                metrics.emit_mean(f"kl_loss_{stream_name}", total=stream_kl_sum, count=bsz)
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

        # Every stream reports under the same keys: a mean over the streams' pairs or elements, a max over all.
        multi_stream = len(prepared.latents) > 1
        for stream_name in prepared.latents:
            step_out = step_outs.get(stream_name)
            if step_out is not None and multi_stream:
                stream_log_prob_abs_diff = torch.abs(step_out.log_prob - log_prob_old_by_stream[stream_name])
                metrics.emit_mean(
                    f"log_prob_mean_abs_diff_{stream_name}", total=stream_log_prob_abs_diff.sum(), count=bsz
                )

            if stream_name in rollout_prev_means:
                record_prev_mean_diff_over_noise_std(
                    metrics, rollout_prev_means[stream_name], step_out.prev_mean, step_out.noise_std
                )

            rollout_model_output = stack_train_pair_rollout_debug(batch, "rollout_step_model_output", stream_name)
            if rollout_model_output is not None:
                record_rollout_train_abs_diff(
                    metrics,
                    "model_output",
                    new_pred[stream_name].float(),
                    rollout_model_output.to(device=ctx.device, dtype=torch.float32),
                    component=prepared.component_name if len(ctx.models) > 1 else None,
                    stream=stream_name if multi_stream else None,
                )

    return loss_sum
