from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DiffusionRolloutSpec:
    # Required rollout keys to reconstruct per-step log_prob_new and PPO ratio in training.
    # latents and next latents for log_prob_new in training, log_prob_old used with log_prob_new (in training) to get ratio.
    required_keys: tuple[str, ...] = ("timesteps", "sigmas", "latents", "next_latents", "log_prob_old")
    # Optional rollout keys for KL regularization or debugging.
    # mean of distribution p(x_{t+1} | x_t), for KL
    optional_keys: tuple[str, ...] = ("prev_latents_mean",)


@dataclass(frozen=True)
class DiffusionTrainSpec:
    required_keys: tuple[str, ...] = ("log_prob_old", "log_prob_new", "advantage")


def _as_tuple(value: Iterable[int] | torch.Size) -> tuple[int, ...]:
    return tuple(int(v) for v in value)


def _normalize_time_major(tensor: torch.Tensor) -> tuple[int, int]:
    """
    Return (batch, steps) for 1D/2D tensors, treating 1D as batch=1.
    """
    if tensor.ndim == 1:
        return 1, tensor.shape[0]
    if tensor.ndim == 2:
        return tensor.shape[0], tensor.shape[1]
    raise ValueError(f"Expected 1D or 2D tensor, got shape {_as_tuple(tensor.shape)}")


def _normalize_latents(tensor: torch.Tensor) -> tuple[int, int]:
    """
    Return (batch, steps) for 4D/5D tensors, treating 4D as batch=1.
    """
    if tensor.ndim == 4:
        return 1, tensor.shape[0]
    if tensor.ndim == 5:
        return tensor.shape[0], tensor.shape[1]
    raise ValueError(f"Expected 4D or 5D tensor, got shape {_as_tuple(tensor.shape)}")


def validate_rollout_metadata(metadata: dict) -> list[str]:
    # Validate that rollout metadata contains required tensors and aligned shapes.
    errors: list[str] = []
    spec = DiffusionRolloutSpec()

    for key in spec.required_keys:
        if key not in metadata:
            errors.append(f"missing metadata key: {key}")
    if errors:
        return errors

    timesteps = metadata["timesteps"]
    sigmas = metadata["sigmas"]
    latents = metadata["latents"]
    next_latents = metadata["next_latents"]
    log_prob_old = metadata["log_prob_old"]

    if not isinstance(timesteps, torch.Tensor):
        errors.append("timesteps must be a torch.Tensor")
    if not isinstance(sigmas, torch.Tensor):
        errors.append("sigmas must be a torch.Tensor")
    if not isinstance(latents, torch.Tensor):
        errors.append("latents must be a torch.Tensor")
    if not isinstance(next_latents, torch.Tensor):
        errors.append("next_latents must be a torch.Tensor")
    if not isinstance(log_prob_old, torch.Tensor):
        errors.append("log_prob_old must be a torch.Tensor")
    if errors:
        return errors

    try:
        b_t, t_t = _normalize_time_major(timesteps)
    except ValueError as exc:
        errors.append(str(exc))
        b_t, t_t = 0, 0

    try:
        b_s, t_s = _normalize_time_major(sigmas)
    except ValueError as exc:
        errors.append(str(exc))
        b_s, t_s = 0, 0

    try:
        b_l, t_l = _normalize_latents(latents)
    except ValueError as exc:
        errors.append(str(exc))
        b_l, t_l = 0, 0

    try:
        b_n, t_n = _normalize_latents(next_latents)
    except ValueError as exc:
        errors.append(str(exc))
        b_n, t_n = 0, 0

    try:
        b_p, t_p = _normalize_time_major(log_prob_old)
    except ValueError as exc:
        errors.append(str(exc))
        b_p, t_p = 0, 0

    if b_t and b_l and b_t != b_l:
        errors.append(f"batch mismatch: timesteps batch {b_t} != latents batch {b_l}")
    if b_s and b_t and b_s != b_t:
        errors.append(f"batch mismatch: sigmas batch {b_s} != timesteps batch {b_t}")
    if b_l and b_n and b_l != b_n:
        errors.append(f"batch mismatch: latents batch {b_l} != next_latents batch {b_n}")
    if b_l and b_p and b_l != b_p:
        errors.append(f"batch mismatch: latents batch {b_l} != log_prob_old batch {b_p}")

    if t_t and t_l and t_t != t_l:
        errors.append(f"timestep mismatch: timesteps steps {t_t} != latents steps {t_l}")
    if t_s and t_t and t_s not in (t_t, t_t + 1):
        errors.append(f"timestep mismatch: sigmas steps {t_s} not in (timesteps {t_t}, timesteps+1 {t_t + 1})")
    if t_l and t_n and t_l != t_n:
        errors.append(f"timestep mismatch: latents steps {t_l} != next_latents steps {t_n}")
    if t_l and t_p and t_l != t_p:
        errors.append(f"timestep mismatch: latents steps {t_l} != log_prob_old steps {t_p}")

    if "prev_latents_mean" in metadata:
        prev_latents_mean = metadata["prev_latents_mean"]
        if not isinstance(prev_latents_mean, torch.Tensor):
            errors.append("prev_latents_mean must be a torch.Tensor")
        else:
            if _as_tuple(prev_latents_mean.shape) != _as_tuple(latents.shape):
                errors.append(
                    "prev_latents_mean must match latents shape "
                    f"{_as_tuple(latents.shape)}, got {_as_tuple(prev_latents_mean.shape)}"
                )

    return errors


def validate_train_inputs(train_data: dict) -> list[str]:
    errors: list[str] = []
    spec = DiffusionTrainSpec()

    for key in spec.required_keys:
        if key not in train_data:
            errors.append(f"missing train_data key: {key}")
    if errors:
        return errors

    log_prob_old = train_data["log_prob_old"]
    log_prob_new = train_data["log_prob_new"]
    advantage = train_data["advantage"]

    if not isinstance(log_prob_old, torch.Tensor):
        errors.append("log_prob_old must be a torch.Tensor")
    if not isinstance(log_prob_new, torch.Tensor):
        errors.append("log_prob_new must be a torch.Tensor")
    if not isinstance(advantage, torch.Tensor):
        errors.append("advantage must be a torch.Tensor")
    if errors:
        return errors

    try:
        b_old, t_old = _normalize_time_major(log_prob_old)
    except ValueError as exc:
        errors.append(str(exc))
        b_old, t_old = 0, 0

    try:
        b_new, t_new = _normalize_time_major(log_prob_new)
    except ValueError as exc:
        errors.append(str(exc))
        b_new, t_new = 0, 0

    try:
        b_adv, t_adv = _normalize_time_major(advantage)
    except ValueError as exc:
        errors.append(str(exc))
        b_adv, t_adv = 0, 0

    if b_old and b_new and b_old != b_new:
        errors.append(f"batch mismatch: log_prob_old batch {b_old} != log_prob_new batch {b_new}")
    if b_old and b_adv and b_old != b_adv:
        errors.append(f"batch mismatch: log_prob_old batch {b_old} != advantage batch {b_adv}")

    if t_old and t_new and t_old != t_new:
        errors.append(f"timestep mismatch: log_prob_old steps {t_old} != log_prob_new steps {t_new}")
    if t_old and t_adv and t_old != t_adv:
        errors.append(f"timestep mismatch: log_prob_old steps {t_old} != advantage steps {t_adv}")

    return errors


def broadcast_advantage(reward: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """
    Broadcast (B,) or (B, 1) rewards into (B, T) advantage aligned to timesteps.
    """
    if reward.ndim == 1:
        reward = reward.unsqueeze(1)
    if reward.ndim != 2 or reward.shape[1] != 1:
        raise ValueError(f"reward must be (B,) or (B, 1), got shape {_as_tuple(reward.shape)}")

    _, steps = _normalize_time_major(timesteps)
    return reward.repeat(1, steps)
