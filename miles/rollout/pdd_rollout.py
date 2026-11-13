"""PDD rollout hooks: validate SGLang trajectories for the block-ordered updates.

SGLang returns the start state of all 32 intervals plus the endpoint. The actor
expands each trajectory into its eight blocks after DP splitting; block j trains
from the rollout state at its first interval, and the teacher target starts from the
rollout state at the sampled interval. Uses the shared rollout engine, response
parser, dataset cursor and weight sync.
"""

import torch

from miles.distillation.integration import pdd_grid


async def no_reward(args, samples, **kwargs):
    # PDD has a teacher loss, no reward model. The shared rollout logger expects
    # a numeric reward; this placeholder must not be interpreted as image quality.
    return [0.0] * len(samples)


def convert_samples_to_train_data(args, samples):
    grid = pdd_grid(args)
    states_count = args.pdd_num_intervals + 1
    pairs = []
    for sample in samples:
        traj, env = sample.dit_trajectory, sample.denoising_env
        if traj is None or env is None or env.pos_cond_kwargs is None:
            raise ValueError("PDD requires SGLang trajectory and positive conditioning")
        if traj.sigmas is None or not torch.equal(traj.sigmas.cpu().float(), grid):
            raise ValueError("PDD rollout sigma grid differs from training")
        # Require every interval start + the endpoint, never silently index a
        # filtered trajectory as if it were a complete one.
        if traj.latents is None or traj.latents.shape[0] != states_count:
            raise ValueError(f"PDD requires the full {states_count}-state interval trajectory")
        if traj.latent_step_indices is not None and traj.latent_step_indices.tolist() != list(range(states_count)):
            raise ValueError(f"PDD trajectory indices must be 0..{states_count - 1}")
        states = traj.latents.detach().cpu().float()
        if states.ndim != 3:
            raise ValueError(f"PDD expects per-sample packed latent tokens [{states_count}, S, C]")
        if not torch.isfinite(states).all():
            raise ValueError("PDD rollout contains nonfinite states")
        pairs.append(dict(sample_index=sample.index, pdd_trajectory=states, denoising_env=env))
    if not pairs:
        raise ValueError("PDD rollout produced no training samples")
    return {"train_data": pairs}
