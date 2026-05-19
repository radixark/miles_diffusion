import logging
from typing import Any

import torch

from miles.utils.types import RolloutDebugTensors, Sample

logger = logging.getLogger(__name__)


def stack_train_pair_rollout_debug(
    batch: list[dict],
    key: str,
) -> torch.Tensor | None:
    """Stack one rollout debug field across a train micro-batch."""
    if not batch:
        return None
    for item in batch:
        rollout_debug_tensors = item.get("rollout_debug_tensors")
        if not isinstance(rollout_debug_tensors, dict) or rollout_debug_tensors.get(key) is None:
            return None
    return torch.stack([item["rollout_debug_tensors"][key] for item in batch], dim=0)


def scheduler_meta_from_rollout(
    rollout_data: dict,
    *,
    device: torch.device,
    num_train_timesteps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use rollout-side scheduler metadata for train/rollout alignment."""
    if "scheduler_timesteps" not in rollout_data:
        raise ValueError("rollout_data missing scheduler_timesteps")
    timesteps = rollout_data["scheduler_timesteps"].to(device=device, dtype=torch.float32)
    if "scheduler_sigmas" in rollout_data:
        sigmas = rollout_data["scheduler_sigmas"].to(device=device, dtype=torch.float32)
    else:
        sigmas = torch.cat([timesteps / float(num_train_timesteps), timesteps.new_zeros(1)])
    return timesteps, sigmas


class RolloutTrainDataConverter:
    """Convert rollout samples into the flat train-pair payload."""

    def convert_samples(
        self,
        samples: list[Sample],
        rewards: list[float],
        raw_rewards: list[float],
    ) -> dict[str, Any]:
        train_data, scheduler_meta = self._expand_samples_to_train_pairs(samples, rewards, raw_rewards)
        return {"train_data": train_data, **scheduler_meta}

    def _expand_samples_to_train_pairs(
        self,
        samples: list[Sample],
        rewards: list[float],
        raw_rewards: list[float],
    ) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
        """Flat train pairs in sample-major order (all pairs for sample 0, then sample 1, ...)."""
        device = torch.device("cpu")
        train_data: list[dict[str, Any]] = []
        first_traj = samples[0].dit_trajectory
        # assume all samples have the same scheduler meta
        scheduler_meta: dict[str, torch.Tensor] = {
            "scheduler_timesteps": first_traj.timesteps.detach().cpu().float()
        }

        if first_traj.sigmas is not None:
            scheduler_meta["scheduler_sigmas"] = first_traj.sigmas.detach().cpu().float()

        for sample, rew, raw_r in zip(samples, rewards, raw_rewards, strict=True):
            traj, denoising_env, rollout_log_probs = self._sample_required_inputs(sample)
            # build per-sample features for train pairs
            per_sample_features = self._build_per_sample_features(
                sample,
                reward=rew,
                raw_reward=raw_r,
                denoising_env=denoising_env,
            )
            # build per-timestep features for train pairs
            per_timestep_features, idx = self._build_per_timestep_features(
                sample,
                traj=traj,
                rollout_log_probs=rollout_log_probs,
                device=device,
            )
            # build debug tensors for train pairs
            pair_debug_steps = None
            if sample.rollout_debug_tensors is not None:
                pair_debug_steps = self._slice_rollout_debug_for_train_pairs(sample.rollout_debug_tensors, sde_idx=idx)
            # validate debug tensors
            sample_t_steps = int(per_timestep_features["latent"].shape[0])
            if pair_debug_steps is not None and len(pair_debug_steps) != sample_t_steps:
                raise ValueError(
                    f"rollout_debug_tensors step count {len(pair_debug_steps)} != train pairs {sample_t_steps} "
                    f"(sample_index={sample.index})"
                )

            for t in range(sample_t_steps):
                pair: dict[str, Any] = {
                    **per_sample_features,
                    **{
                        key: tensor[t].detach().cpu()
                        for key, tensor in per_timestep_features.items()
                    },
                }
                # attach debug tensors to train pair
                if pair_debug_steps is not None:
                    pair["rollout_debug_tensors"] = pair_debug_steps[t]
                train_data.append(pair)

        if not train_data:
            raise ValueError("No train pairs were produced from rollout samples")

        return train_data, scheduler_meta

    @staticmethod
    def _sample_required_inputs(sample: Sample):
        traj = sample.dit_trajectory
        denoising_env = sample.denoising_env
        rollout_log_probs = sample.rollout_log_probs
        if traj is None or traj.timesteps is None or denoising_env is None or rollout_log_probs is None:
            raise ValueError("Sample missing dit_trajectory, denoising_env, or rollout_log_probs")
        return traj, denoising_env, rollout_log_probs

    @staticmethod
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

    @staticmethod
    def _build_per_timestep_features(
        sample: Sample,
        *,
        traj,
        rollout_log_probs: torch.Tensor,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Fields with one row per selected denoising step."""
        all_latents = traj.latents.to(device, dtype=torch.float32)
        latents = all_latents[:-1]
        next_latents = all_latents[1:]
        timesteps = traj.timesteps.to(device, dtype=torch.float32)

        sde_idx = (sample.train_metadata or {}).get("sde_step_indices")
        assert sde_idx is not None, "SDE step indices are required for training"
        idx = torch.as_tensor(sde_idx, dtype=torch.long)
        return {
            "latent": latents[idx],
            "next_latent": next_latents[idx],
            "timestep": timesteps[idx],
            "log_prob_old": rollout_log_probs[idx],
        }, idx

    @staticmethod
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
                raise ValueError(
                    f"Rollout debug field {key!r} has {tensor.shape[0]} steps, expected {num_steps}"
                )

        return [{key: tensor[step] for key, tensor in indexed.items()} for step in range(num_steps)]


class TrainDataDPSplitter:
    """Split flat train-pair payloads across DP ranks."""

    def split_by_dp(self, data: dict[str, Any], dp_size: int) -> list[dict[str, list[dict[str, Any]]]]:
        """Split train data across DP ranks using equal contiguous pair ranges."""
        if dp_size <= 0:
            raise ValueError(f"dp_size must be positive, got {dp_size}")
        train_data = data["train_data"]
        scheduler_timesteps = data.get("scheduler_timesteps")
        scheduler_sigmas = data.get("scheduler_sigmas")
        num_pairs = len(train_data)
        if num_pairs < dp_size:
            raise ValueError(
                f"num_pairs={num_pairs} is smaller than dp_size={dp_size}; "
                "would drop all pairs when enforcing equal DP shards"
            )
        dropped_pairs = num_pairs % dp_size
        if dropped_pairs:
            logger.warning(
                "Drop last %s train pairs after DP split so every DP rank has the same number of "
                "pairs (num_pairs=%s, dp_size=%s)",
                dropped_pairs,
                num_pairs,
                dp_size,
            )

        pairs_per_rank = num_pairs // dp_size
        shards: list[dict[str, list[dict[str, Any]]]] = []
        for rank in range(dp_size):
            pair_lo = rank * pairs_per_rank
            pair_hi = pair_lo + pairs_per_rank
            shard_pairs = train_data[pair_lo:pair_hi]
            shard: dict[str, Any] = {"train_data": shard_pairs}
            if scheduler_timesteps is not None:
                shard["scheduler_timesteps"] = scheduler_timesteps
            if scheduler_sigmas is not None:
                shard["scheduler_sigmas"] = scheduler_sigmas
            shards.append(shard)
        return shards
