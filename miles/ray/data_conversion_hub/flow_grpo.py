"""Convert Flow-GRPO rollout samples into flat train-pair payloads."""

from typing import Any

import torch

from miles.utils.types import RolloutDebugTensors, RolloutStreamData, Sample


def expand_samples_to_train_pairs(
    args,
    samples: list[Sample],
    rewards: list[float],
    raw_rewards: list[float],
) -> dict[str, Any]:
    """Flat train pairs in sample-major order (all pairs for sample 0, then sample 1, ...).

    A pair's per-step fields are keyed by latent stream name, like an SFT pair's ``latent``.
    """
    train_data: list[dict[str, Any]] = []
    for sample, rew, raw_r in zip(samples, rewards, raw_rewards, strict=True):
        if sample.denoising_env is None:
            raise ValueError("Sample missing denoising_env")
        per_sample_features = _build_per_sample_features(
            sample,
            reward=rew,
            raw_reward=raw_r,
            denoising_env=sample.denoising_env,
        )
        stream_steps = {
            stream_name: _build_stream_steps(sample, stream_rollout)
            for stream_name, stream_rollout in _stream_rollouts(args, sample).items()
        }
        # The engine rolls every stream out over the same SDE steps, so the streams' step lists line up.
        for step_by_stream in zip(*stream_steps.values(), strict=True):
            pair: dict[str, Any] = dict(per_sample_features)
            for stream_name, per_step in zip(stream_steps, step_by_stream, strict=True):
                for key, value in per_step.items():
                    pair.setdefault(key, {})[stream_name] = value
            train_data.append(pair)

    if not train_data:
        raise ValueError("No train pairs were produced from rollout samples")

    return {"train_data": train_data}


def _to_cpu(value):
    return value.detach().cpu() if isinstance(value, torch.Tensor) else value


def _own_row(row):
    # A view would pickle the whole trajectory into every pair of the shard
    return row.detach().cpu().clone() if isinstance(row, torch.Tensor) else row


def _stream_rollouts(args, sample: Sample) -> dict[str, RolloutStreamData]:
    """The rollout data of every stream the run trains on; the primary stream's is the sample's top-level fields."""
    missing = [stream_name for stream_name in args.rollout_extra_streams if stream_name not in sample.extra_streams]
    if missing:
        raise ValueError(
            f"the rollout response carries no rollout data for --rollout-extra-streams {missing} "
            f"(sample_index={sample.index}); the response has no rolled-out stream of that name"
        )
    return {
        args.primary_stream: RolloutStreamData(
            rollout_log_probs=sample.rollout_log_probs,
            rollout_debug_tensors=sample.rollout_debug_tensors,
            dit_trajectory=sample.dit_trajectory,
        ),
        **{stream_name: sample.extra_streams[stream_name] for stream_name in args.rollout_extra_streams},
    }


def _build_stream_steps(sample: Sample, stream_rollout: RolloutStreamData) -> list[dict[str, Any]]:
    """One stream's per-step pair fields, one dict per selected SDE step."""
    traj = stream_rollout.dit_trajectory
    rollout_log_probs = stream_rollout.rollout_log_probs
    if traj is None or traj.timesteps is None or traj.sigmas is None or rollout_log_probs is None:
        raise ValueError("Sample missing dit_trajectory timesteps/sigmas or rollout_log_probs")
    per_timestep_features, idx = _build_per_timestep_features(
        sample,
        traj=traj,
        rollout_log_probs=rollout_log_probs,
    )
    steps = [{key: _to_cpu(tensor[t]) for key, tensor in per_timestep_features.items()} for t in range(len(idx))]
    pair_debug_steps = None
    if stream_rollout.rollout_debug_tensors is not None:
        pair_debug_steps = _slice_rollout_debug_for_train_pairs(stream_rollout.rollout_debug_tensors, sde_idx=idx)
    if pair_debug_steps is not None:
        if len(pair_debug_steps) != len(steps):
            raise ValueError(
                f"rollout_debug_tensors step count {len(pair_debug_steps)} != train pairs {len(steps)} "
                f"(sample_index={sample.index})"
            )
        for per_step, debug_step in zip(steps, pair_debug_steps, strict=True):
            per_step["rollout_debug_tensors"] = debug_step
    return steps


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
) -> tuple[dict[str, Any], torch.Tensor]:
    """Fields with one row per selected denoising step.

    ``traj.latents`` is either the full 0..T trajectory or the filtered window
    the rollout requested; ``latent_step_indices`` maps original step numbers
    to array positions, so pairing never assumes the array is contiguous.
    Timesteps and sigmas are the full [T+1] schedule (terminal included) and
    log_probs the full [T]; all are indexed by original step number.
    """
    sde_idx = (sample.train_metadata or {}).get("sde_step_indices")
    assert sde_idx is not None, "SDE step indices are required for training"
    # Keep producer dtype: the train actor casts on consumption (loss_hub _stack_pair_field).
    all_latents = traj.latents
    timesteps = traj.timesteps
    sigmas = traj.sigmas

    if traj.latent_step_indices is None:
        position = {step: step for step in range(len(all_latents))}
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
    return {
        "latent": [_own_row(all_latents[position[int(s)]]) for s in sde_idx],
        "next_latent": [_own_row(all_latents[position[int(s) + 1]]) for s in sde_idx],
        "timestep": timesteps[idx],
        "sigma": sigmas[idx],
        "next_sigma": sigmas[idx + 1],
        # The first sigma below 1 on this sample's grid; the SDE kernel substitutes it at sigma == 1.
        "sigma_max": sigmas[1].expand(len(idx)),
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
    indexed: dict[str, Any] = {}
    for rollout_key, train_pair_key in rollout_to_train_pair_fields.items():
        tensor = getattr(dbg, rollout_key, None)
        if tensor is None:
            continue
        tensor = _to_cpu(tensor)
        if sde_idx is not None:
            tensor = [_own_row(tensor[int(s)]) for s in sde_idx]
        indexed[train_pair_key] = tensor

    if not indexed:
        return None

    num_steps = len(next(iter(indexed.values())))
    for key, tensor in indexed.items():
        if len(tensor) != num_steps:
            raise ValueError(f"Rollout debug field {key!r} has {len(tensor)} steps, expected {num_steps}")

    return [{key: tensor[step] for key, tensor in indexed.items()} for step in range(num_steps)]
