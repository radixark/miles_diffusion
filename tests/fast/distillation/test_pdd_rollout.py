"""Rollout side: engine heads, trajectory conversion, timestep matching and resume signature."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import copy
from types import SimpleNamespace

import pytest
import torch

from miles.backends.sglang_diffusion_utils.monkey_patches.patch_pdd_qwen_image21 import PDDProjection
from miles.distillation import integration
from miles.distillation.integration import embedded_sigmas, engine_timesteps, expand_trajectories
from miles.distillation.pdd import block_update
from miles.rollout import pdd_rollout


def test_engine_interval_steps_match_pdd_and_see_live_weight_updates():
    """The engine's 32 Euler steps on one cached forward per block reproduce every interval start."""
    torch.manual_seed(8)
    projection = PDDProjection(torch.nn.Linear(3, 2))
    grid = torch.linspace(1, 0, 33).square()
    hidden, state = torch.randn(1, 5, 3), torch.randn(1, 5, 2)
    x = state
    for interval in range(32):
        start = interval - interval % 4
        if interval == start:
            head = projection(hidden)
            torch.testing.assert_close(head, projection.parallel(hidden)[0])
            block_state = x
        x = x + projection.block[interval] * (grid[interval + 1] - grid[interval])
        expected = block_update(block_state, projection.parallel(hidden), grid, start, interval + 1)
        torch.testing.assert_close(x, expected, rtol=1e-6, atol=1e-6)
    before = projection(hidden).clone()
    with torch.no_grad():
        projection.parallel.weight.add_(0.5)
    assert not torch.equal(before, projection(hidden))
    assert set(projection.state_dict()) == {"parallel.weight", "parallel.bias"}


def test_converter_validates_grid_and_complete_trajectory(monkeypatch):
    grid = torch.linspace(1, 0, 33)
    monkeypatch.setattr(integration, "make_grid", lambda *args: grid)
    sample = SimpleNamespace(
        index=17,
        denoising_env=SimpleNamespace(pos_cond_kwargs={}),
        dit_trajectory=SimpleNamespace(
            sigmas=grid,
            latents=torch.randn(33, 4, 2),
            latent_step_indices=None,
        ),
    )
    args = SimpleNamespace(
        diffusion_model_family="qwen_image21", hf_checkpoint="unused", diffusion_height=32, pdd_num_intervals=32
    )
    pair = pdd_rollout.convert_samples_to_train_data(args, [sample])["train_data"][0]
    torch.testing.assert_close(pair["pdd_trajectory"], sample.dit_trajectory.latents)
    blocks = expand_trajectories([pair], intervals=32, block_size=4)
    assert [block["pdd_start"] for block in blocks] == list(range(0, 32, 4))
    for field, value in [
        ("sigmas", grid[::4]),
        ("sigmas", grid + 0.01),
        ("latents", torch.randn(9, 4, 2)),
        ("latents", torch.full((33, 4, 2), float("nan"))),
        ("latent_step_indices", torch.arange(1, 34)),
    ]:
        bad = copy.deepcopy(sample)
        setattr(bad.dit_trajectory, field, value)
        with pytest.raises(ValueError):
            pdd_rollout.convert_samples_to_train_data(args, [bad])


def test_engine_timesteps_reproduce_trainer_bf16_sigma():
    # Native DiT embeds bf16(bf16(t) / 1000); the Diffusers trainer embeds bf16(sigma).
    sigmas = torch.linspace(1, 0.01, 2000)
    naive = embedded_sigmas(sigmas * 1000) != sigmas.bfloat16().float()
    embedded = embedded_sigmas(engine_timesteps(sigmas))
    fixed = embedded != sigmas.bfloat16().float()
    assert naive.float().mean() > 0.2 and fixed.float().mean() < 0.05 and not (fixed & ~naive).any()
    # Unreachable sigmas fall back to the nearest embeddable value, which the trainer adopts.
    assert (embedded - sigmas).abs().max() < 8e-3
