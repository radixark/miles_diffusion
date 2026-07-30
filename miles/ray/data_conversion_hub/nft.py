"""Convert DiffusionNFT rollout samples into timestep-expanded train pairs."""

from argparse import Namespace
from typing import Any

import torch

from miles.utils.types import Sample

# ---------------------------------------------------------------------------
# Converter (K-expanded pairs; rewards already post-processed by rollout)
# ---------------------------------------------------------------------------


def _clean_x0_from_sample(sample: Sample) -> torch.Tensor:
    traj = sample.dit_trajectory
    if traj is None or traj.latents is None or traj.latents.shape[0] < 1:
        raise ValueError(
            f"sample {sample.index} missing dit_trajectory.latents; "
            "NFT needs the final clean latent x0 from rollout"
        )
    return traj.latents[-1].detach().cpu().float()


# TODO: remove and replace sigmas by rollout results
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
    for sample, adv, raw in zip(samples, rewards, raw_rewards, strict=True):
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
