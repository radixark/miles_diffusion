"""Windowed trajectory -> train pairs: provenance mapping in the converter.

Mental model (10-step rollout, SDE window S=[2,3]):

    engine ships latents for L = S U (S+1) = [2,3,4]     (was: all 11)
           echoes latent_step_indices=[2,3,4]
           timesteps / log_probs stay full-length
    converter pairs latent[s] with latent[s+1] BY PROVENANCE, never by
           array adjacency

Covered: the windowed trajectory yields bitwise the same train-pair tensors as
the full trajectory (1), full trajectories without provenance keep the legacy
behaviour (2), a missing window latent raises instead of mispairing (3), a
non-contiguous kept set still pairs correctly (4), and NFT reads x0 off a
final-step-only trajectory but rejects one ending anywhere else (5).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.ray.data_conversion_hub.flow_grpo import _build_per_timestep_features
from miles.utils.types import DiTTrajectory, Sample

T = 10
SDE = [2, 3]


def _full_traj() -> DiTTrajectory:
    g = torch.Generator().manual_seed(7)
    return DiTTrajectory(
        latents=torch.randn(T + 1, 4, 6, generator=g),
        timesteps=torch.linspace(1000.0, 0.0, T + 1),
        sigmas=torch.linspace(1.0, 0.0, T + 1),
    )


def _windowed(traj: DiTTrajectory, kept: list[int]) -> DiTTrajectory:
    return DiTTrajectory(
        latents=traj.latents[kept],
        timesteps=traj.timesteps,
        sigmas=traj.sigmas,
        latent_step_indices=torch.tensor(kept, dtype=torch.long),
    )


def _sample(sde=SDE) -> Sample:
    return Sample(index=0, prompt="p", train_metadata={"sde_step_indices": list(sde)})


def _build(traj, sde=SDE):
    log_probs = torch.linspace(-1.0, -2.0, T)
    return _build_per_timestep_features(_sample(sde), traj=traj, rollout_log_probs=log_probs)


def test_windowed_matches_full_bitwise():
    full = _full_traj()
    feats_full, idx_full = _build(full)
    feats_win, idx_win = _build(_windowed(full, [2, 3, 4]))
    assert torch.equal(idx_full, idx_win)
    assert feats_full.keys() == feats_win.keys()
    for key in feats_full:
        assert torch.equal(feats_full[key], feats_win[key]), key


def test_full_trajectory_without_provenance_is_legacy():
    feats, idx = _build(_full_traj())
    assert idx.tolist() == SDE
    full = _full_traj()
    assert torch.equal(feats["latent"], full.latents[SDE])
    assert torch.equal(feats["next_latent"], full.latents[[s + 1 for s in SDE]])
    assert torch.equal(feats["log_prob_old"], torch.linspace(-1.0, -2.0, T)[SDE])


def test_terminal_step_next_timestep_is_zero():
    feats, _ = _build(_full_traj(), sde=[T - 1])
    assert feats["next_timestep"].item() == 0.0


def test_missing_window_latent_raises():
    with pytest.raises(ValueError, match="lacks latents"):
        _build(_windowed(_full_traj(), [2, 3]))  # boundary latent 4 missing


def test_non_contiguous_kept_set_pairs_by_provenance():
    full = _full_traj()
    win = _windowed(full, [2, 3, 4, 7, 8])
    feats, _ = _build(win, sde=[2, 3, 7])
    assert torch.equal(feats["latent"], full.latents[[2, 3, 7]])
    assert torch.equal(feats["next_latent"], full.latents[[3, 4, 8]])


def test_nft_x0_comes_from_the_final_step():
    """NFT requests the final step alone; a tail from any other step is not x0."""
    from miles.ray.data_conversion_hub.nft import _clean_x0_from_sample

    full = _full_traj()
    sample = _sample()
    sample.dit_trajectory = full
    assert torch.equal(_clean_x0_from_sample(sample), full.latents[-1].float())

    sample.dit_trajectory = _windowed(full, [2, 3, 4])
    with pytest.raises(ValueError, match="needs x0"):
        _clean_x0_from_sample(sample)
