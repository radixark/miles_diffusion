"""DMD2 rollout samples -> train pairs.

    stop_at_random_step request (stop = i, return [i, i+1]) ──► engine runs steps 0..i, step i goes to sigma 0
      dit_trajectory.latents             [x_i, x0_i]   (x0_i = the student's x0 prediction at step i)
      dit_trajectory.latent_step_indices [i, i+1]
      dit_trajectory.sigmas              [sigma_0 .. sigma_i, 0]
                │
    expand_samples_to_train_pairs ──► one pair per sample:
      latent = x_i, x0 = x0_i, timestep = timesteps[i], sigma = sigma_i, step_index = i,
      denoising_env, sample_index, prompt; rewards are ignored
    A trajectory that did not stop at its first returned step is rejected.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.ray.data_conversion_hub.dmd2 import expand_samples_to_train_pairs
from miles.utils.types import DenoisingEnv, DiTTrajectory, Sample

FULL_SIGMAS = [1.0, 0.75, 0.5, 0.25, 0.0]
FULL_TIMESTEPS = [1000.0, 750.0, 500.0, 250.0]


def _stopped_sample(index, step):
    return Sample(
        index=index,
        prompt=f"prompt {index}",
        denoising_env=DenoisingEnv(),
        dit_trajectory=DiTTrajectory(
            latents=torch.stack([torch.full((2, 3), float(index)), torch.full((2, 3), -float(index))]),
            timesteps=torch.tensor(FULL_TIMESTEPS[: step + 1]),
            sigmas=torch.tensor(FULL_SIGMAS[: step + 1] + [0.0]),
            latent_step_indices=torch.tensor([step, step + 1]),
        ),
    )


def test_each_stopped_sample_becomes_one_pair():
    samples = [_stopped_sample(10, 0), _stopped_sample(11, 2)]

    pairs = expand_samples_to_train_pairs(None, samples, [0.0, 0.0], [0.0, 0.0])["train_data"]

    assert [pair["sample_index"] for pair in pairs] == [10, 11]
    assert [pair["step_index"] for pair in pairs] == [0, 2]
    assert [pair["sigma"] for pair in pairs] == [1.0, 0.5]
    assert [pair["timestep"] for pair in pairs] == [1000.0, 500.0]
    assert torch.equal(pairs[1]["latent"], torch.full((2, 3), 11.0))
    assert torch.equal(pairs[1]["x0"], torch.full((2, 3), -11.0))
    assert pairs[1]["denoising_env"] is samples[1].denoising_env
    assert pairs[1]["prompt"] == "prompt 11"


def test_full_trajectory_is_rejected():
    sample = _stopped_sample(10, 1)
    sample.dit_trajectory.latent_step_indices = None
    with pytest.raises(ValueError, match="stop_at_random_step"):
        expand_samples_to_train_pairs(None, [sample], [0.0], [0.0])


def test_trajectory_that_runs_past_its_returned_step_is_rejected():
    sample = _stopped_sample(10, 1)
    sample.dit_trajectory.sigmas = torch.tensor(FULL_SIGMAS)
    with pytest.raises(ValueError, match="stopped at its first returned step"):
        expand_samples_to_train_pairs(None, [sample], [0.0], [0.0])
