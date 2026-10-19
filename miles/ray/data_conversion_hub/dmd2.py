"""Convert DMD2 rollout samples, each stopped at its step i, into one train pair per sample."""

from argparse import Namespace
from typing import Any

from miles.utils.types import Sample


def expand_samples_to_train_pairs(
    args: Namespace,
    samples: list[Sample],
    rewards: list[float],
    raw_rewards: list[float],
) -> dict[str, Any]:
    """One pair per sample: the student's input ``x_i`` at its stop step ``i`` and its x0 prediction there."""
    train_data: list[dict[str, Any]] = []
    for sample in samples:
        traj = sample.dit_trajectory
        if traj is None or traj.latent_step_indices is None or sample.denoising_env is None:
            raise ValueError(
                f"sample {sample.index} missing dit_trajectory.latent_step_indices or denoising_env; "
                "DMD2 rollouts need --diffusion-step-strategy-path miles.rollout.step_strategy_hub.stop_at_random_step"
            )
        step_index = int(traj.latent_step_indices[0])
        # The engine sends step i straight to sigma 0, so the grid ends right after sigma_i.
        if traj.latent_step_indices.tolist() != [step_index, step_index + 1] or len(traj.sigmas) != step_index + 2:
            raise ValueError(
                f"sample {sample.index} returned steps {traj.latent_step_indices.tolist()} on a "
                f"{len(traj.sigmas)}-point sigma grid; DMD2 needs a rollout stopped at its first returned step"
            )
        train_data.append(
            {
                "latent": traj.latents[0],
                "x0": traj.latents[1],
                "timestep": float(traj.timesteps[step_index]),
                "sigma": float(traj.sigmas[step_index]),
                "step_index": step_index,
                "denoising_env": sample.denoising_env,
                "sample_index": sample.index,
                "prompt": sample.prompt,
            }
        )
    return {"train_data": train_data}
