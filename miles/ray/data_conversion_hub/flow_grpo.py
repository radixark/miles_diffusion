"""Convert Flow-GRPO rollout samples into flat train-pair payloads."""

from typing import Any

import torch

from miles.utils.train_data_utils import scheduler_meta_from_samples
from miles.utils.types import RolloutDebugTensors, Sample


def expand_samples_to_train_pairs(
    args,
    samples: list[Sample],
    rewards: list[float],
    raw_rewards: list[float],
) -> dict[str, Any]:
    train_data, scheduler_meta = _expand_samples_to_train_pairs(samples, rewards, raw_rewards)
    return {"train_data": train_data, **scheduler_meta}


def _expand_samples_to_train_pairs(
    samples: list[Sample],
    rewards: list[float],
    raw_rewards: list[float],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Flat train pairs in sample-major order (all pairs for sample 0, then sample 1, ...)."""
    train_data: list[dict[str, Any]] = []
    scheduler_meta = scheduler_meta_from_samples(samples)

    for sample, rew, raw_r in zip(samples, rewards, raw_rewards, strict=True):
        traj, denoising_env, rollout_log_probs = _sample_required_inputs(sample)
        per_sample_features = _build_per_sample_features(
            sample,
            reward=rew,
            raw_reward=raw_r,
            denoising_env=denoising_env,
        )
        per_timestep_features, idx = _build_per_timestep_features(
            sample,
            traj=traj,
            rollout_log_probs=rollout_log_probs,
        )
        pair_debug_steps = None
        if sample.rollout_debug_tensors is not None:
            pair_debug_steps = _slice_rollout_debug_for_train_pairs(sample.rollout_debug_tensors, sde_idx=idx)
        sample_t_steps = int(per_timestep_features["latent"].shape[0])
        if pair_debug_steps is not None and len(pair_debug_steps) != sample_t_steps:
            raise ValueError(
                f"rollout_debug_tensors step count {len(pair_debug_steps)} != train pairs {sample_t_steps} "
                f"(sample_index={sample.index})"
            )

        for t in range(sample_t_steps):
            pair: dict[str, Any] = {
                **per_sample_features,
                **{key: tensor[t].detach().cpu() for key, tensor in per_timestep_features.items()},
            }
            if pair_debug_steps is not None:
                pair["rollout_debug_tensors"] = pair_debug_steps[t]
            train_data.append(pair)

    if not train_data:
        raise ValueError("No train pairs were produced from rollout samples")

    return train_data, scheduler_meta


def _sample_required_inputs(sample: Sample):
    traj = sample.dit_trajectory
    denoising_env = sample.denoising_env
    rollout_log_probs = sample.rollout_log_probs
    if traj is None or traj.timesteps is None or denoising_env is None or rollout_log_probs is None:
        raise ValueError("Sample missing dit_trajectory, denoising_env, or rollout_log_probs")
    return traj, denoising_env, rollout_log_probs


def _build_per_sample_features(
    sample: Sample,
    *,
    reward: float,
    raw_reward: float,
    denoising_env,
) -> dict[str, Any]:
    """Fields shared by every train pair produced from one sample."""
    return {
        "advantage": float(reward),
        "denoising_env": denoising_env,
        "sample_index": sample.index,
        "prompt": sample.prompt,
        "raw_reward": float(raw_reward),
    }


def _build_per_timestep_features(
    sample: Sample,
    *,
    traj,
    rollout_log_probs: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Fields with one row per selected denoising step.

    ``traj.latents`` is either the full 0..T trajectory or the filtered window
    the rollout requested; ``latent_step_indices`` maps original step numbers
    to array positions, so pairing never assumes the array is contiguous.
    Timesteps are the full [T+1] schedule (terminal included) and log_probs
    the full [T]; both are indexed by original step number.
    """
    sde_idx = (sample.train_metadata or {}).get("sde_step_indices")
    assert sde_idx is not None, "SDE step indices are required for training"
    # Keep producer dtype: the train actor casts on consumption (loss_hub _stack_pair_field).
    all_latents = traj.latents
    timesteps = traj.timesteps

    if traj.latent_step_indices is None:
        position = {step: step for step in range(int(all_latents.shape[0]))}
    else:
        position = {int(step): pos for pos, step in enumerate(traj.latent_step_indices.tolist())}
    needed = sorted({int(s) for s in sde_idx} | {int(s) + 1 for s in sde_idx})
    missing = [step for step in needed if step not in position]
    if missing:
        provenance = "echoed" if traj.latent_step_indices is not None else "absent (engine must echo it)"
        raise ValueError(
            f"trajectory lacks latents for steps {missing} (have {sorted(position)}); "
            f"latent_step_indices {provenance}"
        )

    idx = torch.as_tensor(sde_idx, dtype=torch.long)
    latent_pos = torch.as_tensor([position[int(s)] for s in sde_idx], dtype=torch.long)
    next_pos = torch.as_tensor([position[int(s) + 1] for s in sde_idx], dtype=torch.long)
    return {
        "latent": all_latents[latent_pos],
        "next_latent": all_latents[next_pos],
        "timestep": timesteps[idx],
        "next_timestep": timesteps[idx + 1],
        "log_prob_old": rollout_log_probs[idx],
    }, idx


def _slice_rollout_debug_for_train_pairs(
    dbg: RolloutDebugTensors,
    *,
    sde_idx: torch.Tensor | None = None,
) -> list[dict[str, torch.Tensor]] | None:
    """Slice per-sample rollout debug tensors into one debug payload per train pair."""
    rollout_to_train_pair_fields = {
        "rollout_variance_noises": "rollout_step_variance_noise",
        "rollout_prev_sample_means": "rollout_step_prev_sample_mean",
        "rollout_noise_std_devs": "rollout_step_noise_std_dev",
        "rollout_model_outputs": "rollout_step_model_output",
    }
    indexed: dict[str, torch.Tensor] = {}
    for rollout_key, train_pair_key in rollout_to_train_pair_fields.items():
        tensor = getattr(dbg, rollout_key, None)
        if tensor is None:
            continue
        tensor = tensor.detach().cpu()
        if sde_idx is not None:
            tensor = tensor[sde_idx]
        indexed[train_pair_key] = tensor

    if not indexed:
        return None

    num_steps = int(next(iter(indexed.values())).shape[0])
    for key, tensor in indexed.items():
        if int(tensor.shape[0]) != num_steps:
            raise ValueError(f"Rollout debug field {key!r} has {tensor.shape[0]} steps, expected {num_steps}")

    return [{key: tensor[step] for key, tensor in indexed.items()} for step in range(num_steps)]
