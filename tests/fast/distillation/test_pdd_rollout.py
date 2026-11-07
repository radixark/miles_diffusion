"""Rollout side: engine heads, trajectory conversion, timestep matching and resume signature."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import copy
from types import SimpleNamespace

import pytest
import torch
from tests.fast.distillation.test_miles_pdd_integration import valid_args

from miles.backends.sglang_diffusion_utils.monkey_patches.patch_pdd_qwen_image21 import PDDProjection
from miles.distillation.integration import embedded_sigmas, engine_timesteps, expand_trajectories
from miles.distillation.pdd import block_update
from miles.rollout import pdd_rollout


def test_native_ode_projection_matches_pdd_and_sees_live_weight_updates():
    torch.manual_seed(8)
    projection = PDDProjection(torch.nn.Linear(3, 2))
    projection.grid = torch.linspace(1, 0, 33).square()
    hidden, state = torch.randn(1, 5, 3), torch.randn(1, 5, 2)
    for start in range(0, 32, 4):
        projection.start = start
        expected = block_update(state, projection.parallel(hidden), projection.grid, start, start + 4)
        actual = state + projection(hidden) * (projection.grid[start + 4] - projection.grid[start])
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    before = projection(hidden).clone()
    with torch.no_grad():
        projection.parallel.weight.add_(0.5)
    assert not torch.equal(before, projection(hidden))
    assert set(projection.state_dict()) == {"parallel.weight", "parallel.bias"}


def test_converter_validates_grid_and_complete_trajectory(monkeypatch):
    grid = torch.linspace(1, 0, 33)
    monkeypatch.setattr(pdd_rollout, "make_grid", lambda *args: grid)
    sample = SimpleNamespace(
        index=17,
        denoising_env=SimpleNamespace(pos_cond_kwargs={}),
        dit_trajectory=SimpleNamespace(
            sigmas=grid[::4],
            latents=torch.randn(9, 4, 2),
            latent_step_indices=None,
        ),
    )
    args = SimpleNamespace(hf_checkpoint="unused", diffusion_height=32)
    pair = pdd_rollout.convert_samples_to_train_data(args, [sample])["train_data"][0]
    torch.testing.assert_close(pair["pdd_trajectory"], sample.dit_trajectory.latents)
    blocks = expand_trajectories([pair], intervals=32, block_size=4)
    assert [block["pdd_start"] for block in blocks] == list(range(0, 32, 4))
    for field, value in [
        ("sigmas", grid[::4] + 0.01),
        ("latents", torch.randn(8, 4, 2)),
        ("latents", torch.full((9, 4, 2), float("nan"))),
        ("latent_step_indices", torch.arange(1, 10)),
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


def test_resume_signature_tracks_rollout_and_training_settings(monkeypatch):
    from miles.distillation import integration

    args = valid_args()
    for field in (
        *("seed", "fsdp_reduce_dtype", "train_dp_split_mode", "actor_num_nodes", "actor_num_gpus_per_node"),
        *("lr", "weight_decay", "adam_beta1", "adam_beta2", "adam_eps", "clip_grad"),
        *("rollout_seed", "diffusion_generator_device", "diffusion_quality"),
    ):
        vars(args).setdefault(field, 0)
    monkeypatch.setattr(integration, "file_sha256", lambda path: str(path))
    monkeypatch.setattr(integration, "make_grid", lambda *args: torch.linspace(1, 0, 33))
    signature = integration.resume_signature(args)
    assert signature["version"] == 3 and signature["prompt_sha256"] == args.prompt_data
    args.rollout_seed = 1
    assert integration.resume_signature(args) != signature
