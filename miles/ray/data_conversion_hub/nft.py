"""Convert DiffusionNFT rollout samples into timestep-expanded train pairs."""

from argparse import Namespace
from typing import Any

import torch

from miles.utils.types import Sample


def _clean_x0_from_sample(sample: Sample) -> torch.Tensor:
    traj = sample.dit_trajectory
    if traj is None or traj.latents is None or traj.latents.shape[0] < 1:
        raise ValueError(
            f"sample {sample.index} missing dit_trajectory.latents; "
            "NFT needs the final clean latent x0 from rollout"
        )
    return traj.latents[-1].detach().cpu().float()


def resolve_nft_sigmas(
    sigmas: torch.Tensor,
    *,
    training_timestep_fraction: float = 0.99,
) -> torch.Tensor:
    ts = sigmas.detach().float().flatten()
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
    first_traj = samples[0].dit_trajectory
    if first_traj is None:
        raise ValueError("sample 0 missing dit_trajectory")
    if first_traj.timesteps is None:
        raise ValueError("NFT needs dit_trajectory.timesteps from rollout")
    if first_traj.sigmas is None:
        raise ValueError("NFT needs dit_trajectory.sigmas from rollout; no timesteps-derived fallback")
    scheduler_meta = {
        "scheduler_timesteps": first_traj.timesteps.detach().cpu().float(),
        "scheduler_sigmas": first_traj.sigmas.detach().cpu().float(),
    }
    sigmas = resolve_nft_sigmas(
        scheduler_meta["scheduler_sigmas"],
        training_timestep_fraction=args.diffusion_nft_timestep_fraction,
    )
    num_timesteps = int(sigmas.numel())

    train_data: list[dict[str, Any]] = []
    for sample, adv, raw in zip(samples, rewards, raw_rewards, strict=True):
        if sample.denoising_env is None:
            raise ValueError(f"sample {sample.index} missing denoising_env")
        traj = sample.dit_trajectory
        if (
            traj is None
            or traj.timesteps is None
            or not torch.equal(traj.timesteps.detach().cpu().float(), scheduler_meta["scheduler_timesteps"])
        ):
            raise ValueError(
                f"sample {sample.index} has different scheduler_timesteps than sample 0; "
                "the converter assumes one shared schedule across the batch"
            )
        if traj.sigmas is None or not torch.equal(
            traj.sigmas.detach().cpu().float(), scheduler_meta["scheduler_sigmas"]
        ):
            raise ValueError(
                f"sample {sample.index} has different scheduler_sigmas than sample 0; "
                "the converter assumes one shared schedule across the batch"
            )
        x0 = _clean_x0_from_sample(sample)
        sample_sigmas = sigmas[torch.randperm(num_timesteps)] if args.diffusion_nft_shuffle_timesteps else sigmas
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
