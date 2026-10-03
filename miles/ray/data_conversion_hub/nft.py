"""Convert DiffusionNFT rollout samples into timestep-expanded train pairs."""

from argparse import Namespace
from typing import Any

import torch

from miles.utils.hash_utils import stable_hash
from miles.utils.misc import load_function
from miles.utils.train_data_utils import scheduler_meta_from_samples
from miles.utils.types import Sample


def _clean_x0_from_sample(sample: Sample) -> torch.Tensor:
    traj = sample.dit_trajectory
    if traj is None or traj.latents is None or traj.latents.shape[0] < 1:
        raise ValueError(
            f"sample {sample.index} missing dit_trajectory.latents; "
            "NFT needs the final clean latent x0 from rollout"
        )
    if traj.latent_step_indices is not None:
        # NFT asks the rollout for the final step alone, so the last latent must be x0.
        final_step = int(traj.latent_step_indices[-1])
        num_steps = int(traj.sigmas.shape[0]) - 1 if traj.sigmas is not None else final_step
        if final_step != num_steps:
            raise ValueError(
                f"sample {sample.index} trajectory ends at step {final_step}, not the final "
                f"step {num_steps}; NFT needs x0"
            )
    return traj.latents[-1].detach().cpu().float()


# ---- Timestep strategies: (args, num_steps, generator) -> the sampling-step indices to train on, in training
# order. ``generator`` is seeded per sample. Point --diffusion-nft-timestep-strategy-path at any such function.


def drop_random_steps(args: Namespace, num_steps: int, generator: torch.Generator) -> list[int]:
    """Every sampling step but ``--diffusion-nft-num-dropped-timesteps`` ones drawn per sample, in a random
    order, so across samples every step is trained."""
    kept_steps = torch.randperm(num_steps, generator=generator)[: num_steps - args.diffusion_nft_num_dropped_timesteps]
    return kept_steps.tolist()


def drop_final_steps(args: Namespace, num_steps: int, generator: torch.Generator) -> list[int]:
    """Every sampling step but the final ``--diffusion-nft-num-dropped-timesteps`` (lowest-noise) ones,
    in schedule order."""
    return list(range(num_steps - args.diffusion_nft_num_dropped_timesteps))


# ---- Conversion


def expand_samples_to_train_pairs(
    args: Namespace,
    samples: list[Sample],
    rewards: list[float],
    raw_rewards: list[float],
) -> dict[str, Any]:
    """Expand NFT rollout samples into K ``(x0, t)`` train pairs."""
    if not samples:
        raise ValueError("NFT convert received empty samples")
    if len(samples) != len(rewards) or len(samples) != len(raw_rewards):
        raise ValueError(
            f"NFT convert length mismatch: samples={len(samples)} "
            f"rewards={len(rewards)} raw_rewards={len(raw_rewards)}"
        )
    scheduler_meta = scheduler_meta_from_samples(samples)
    select_train_steps = load_function(args.diffusion_nft_timestep_strategy_path)
    train_data: list[dict[str, Any]] = []
    for position, (sample, adv, raw) in enumerate(zip(samples, rewards, raw_rewards, strict=True)):
        if sample.denoising_env is None:
            raise ValueError(f"sample {sample.index} missing denoising_env")
        x0 = _clean_x0_from_sample(sample)
        # The trajectory grid ends with the terminal sigma 0; the sigmas before it are the sampling steps.
        step_sigmas = scheduler_meta["scheduler_sigmas"][:-1]
        # Keyed on the sample's global index, which the data source advances across rollouts,
        # so each sample draws its own steps and order and the run reproduces.
        stream = sample.index if sample.index is not None else position
        generator = torch.Generator().manual_seed(stable_hash("nft_sigma_shuffle", int(args.seed), int(stream)))
        train_steps = torch.tensor(select_train_steps(args, len(step_sigmas), generator), dtype=torch.long)
        sample_sigmas = step_sigmas[train_steps]
        num_timesteps = len(sample_sigmas)
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
