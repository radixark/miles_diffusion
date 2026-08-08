"""Smoke tests for diffusion SFT hooks (prepare + loss formula; actor owns DiT)."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import torch
import torch.nn as nn

from miles.backends.fsdp_utils.loss_hub.sft import prepare_sft_batch, sample_grid_indices, sft_loss_formula
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext

NUM_TRAIN_TIMESTEPS = 1000
NUM_GRID = 8


class _Config:
    needs_timestep_scaling = False

    def collate_cond_for_sample_batch(self, per_sample_cond_kwargs, device, pad_to_len=None):
        return {"encoder_hidden_states": torch.cat([kw["encoder_hidden_states"] for kw in per_sample_cond_kwargs])}

    def component_for_timestep(self, timestep, num_train_timesteps):
        return "transformer" if timestep >= 0.875 * num_train_timesteps else "transformer_2"


class _SingleConfig(_Config):
    def component_for_timestep(self, timestep, num_train_timesteps):
        return "transformer"


def _scheduler():
    sigmas = torch.linspace(1.0, 1.0 / NUM_GRID, NUM_GRID)
    return Namespace(
        timesteps=sigmas * NUM_TRAIN_TIMESTEPS,
        sigmas=torch.cat([sigmas, torch.zeros(1)]),
        config=Namespace(num_train_timesteps=NUM_TRAIN_TIMESTEPS),
    )


def _ctx(models, rollout_id=3, microbatch_id=0, dp_rank=0, config=None):
    return DiffusionLossContext(
        models=models,
        train_pipeline_config=config if config is not None else _Config(),
        sde_backend=None,
        scheduler=_scheduler(),
        args=Namespace(seed=42),
        forward_dtype=torch.float32,
        device=torch.device("cpu"),
        rollout_id=rollout_id,
        microbatch_id=microbatch_id,
        dp_rank=dp_rank,
    )


def _batch(bsz=4):
    return [
        {"latent": torch.randn(16, 2, 4, 4), "cond_kwargs": {"encoder_hidden_states": torch.randn(1, 6, 8)}}
        for _ in range(bsz)
    ]


class _Metrics:
    def __init__(self):
        self.seen = {}

    def emit_mean(self, key, *, total, count):
        self.seen[key] = (float(total), count)


class TestPrepareSftBatch:
    def test_corruption_and_target_identity(self):
        torch.manual_seed(0)
        ctx = _ctx({"transformer": nn.Identity()})
        batch = _batch()
        prepared = prepare_sft_batch(ctx, batch)

        x0 = torch.stack([pair["latent"] for pair in batch]).float()
        sigma = (prepared.timesteps / NUM_TRAIN_TIMESTEPS).view(-1, 1, 1, 1, 1)
        assert torch.allclose(prepared.latents, x0 + sigma * prepared.extras["target"], atol=1e-5)
        assert not prepared.use_cfg
        assert prepared.pos_cond["encoder_hidden_states"].shape == (4, 6, 8)
        assert torch.equal(prepared.timesteps_for_model, prepared.timesteps)

    def test_timestep_scaling(self):
        torch.manual_seed(0)
        ctx = _ctx({"transformer": nn.Identity()})
        ctx.train_pipeline_config.needs_timestep_scaling = True
        prepared = prepare_sft_batch(ctx, _batch())
        assert torch.allclose(prepared.timesteps_for_model, prepared.timesteps / NUM_TRAIN_TIMESTEPS)

    def test_single_model_indices_cover_grid_uniformly(self):
        ctx = _ctx({"transformer": nn.Identity()}, config=_SingleConfig())
        _, _, idx = sample_grid_indices(
            ctx,
            bsz=20000,
            generator=torch.Generator().manual_seed(1),
        )
        counts = torch.bincount(idx, minlength=NUM_GRID).float()
        assert counts.min() > 0
        assert ((counts / 20000) - 1 / NUM_GRID).abs().max() < 0.02

    def test_single_wan_expert_only_samples_its_timesteps(self):
        config = _Config()
        timesteps = _scheduler().timesteps
        for component_name in ("transformer", "transformer_2"):
            ctx = _ctx({component_name: nn.Identity()}, config=config)
            name, _, idx = sample_grid_indices(
                ctx,
                bsz=1000,
                generator=torch.Generator().manual_seed(1),
            )
            expected = {
                i
                for i, timestep in enumerate(timesteps)
                if config.component_for_timestep(float(timestep), NUM_TRAIN_TIMESTEPS) == component_name
            }
            assert name == component_name
            assert set(idx.tolist()) == expected

    def test_dual_expert_micro_batch_is_phase_pure(self):
        models = {"transformer": nn.Identity(), "transformer_2": nn.Identity()}
        ctx = _ctx(models)
        config = ctx.train_pipeline_config
        picked = set()
        for call in range(20):
            ctx.microbatch_id = call
            name, model, idx = sample_grid_indices(
                ctx,
                bsz=4,
                generator=torch.Generator().manual_seed(call + 100),
            )
            picked.add(name)
            assert model is models[name]
            for i in idx.tolist():
                t = float(ctx.scheduler.timesteps[i])
                assert config.component_for_timestep(t, NUM_TRAIN_TIMESTEPS) == name
        assert picked == {"transformer", "transformer_2"}

    def test_prepare_is_independent_of_global_rng_state(self):
        batch = _batch()
        torch.manual_seed(0)
        global_state = torch.get_rng_state()

        first = prepare_sft_batch(_ctx({"transformer": nn.Identity()}), batch)
        assert torch.equal(torch.get_rng_state(), global_state)

        torch.rand(1000)
        second = prepare_sft_batch(_ctx({"transformer": nn.Identity()}), batch)
        assert torch.equal(first.timesteps, second.timesteps)
        assert torch.equal(first.latents, second.latents)
        assert torch.equal(first.extras["target"], second.extras["target"])

    def test_samples_within_microbatch_use_different_noise(self):
        batch = _batch(bsz=4)
        prepared = prepare_sft_batch(_ctx({"transformer": nn.Identity()}), batch)
        x0 = torch.stack([pair["latent"] for pair in batch]).float()
        noise = prepared.extras["target"] + x0
        assert all(not torch.equal(noise[0], noise[i]) for i in range(1, len(batch)))

    def test_identity_changes_draws(self):
        batch = _batch()
        base = prepare_sft_batch(_ctx({"transformer": nn.Identity()}), batch)
        other_rollout = prepare_sft_batch(_ctx({"transformer": nn.Identity()}, rollout_id=4), batch)
        other_slot = prepare_sft_batch(_ctx({"transformer": nn.Identity()}, microbatch_id=1), batch)
        other_dp_rank = prepare_sft_batch(_ctx({"transformer": nn.Identity()}, dp_rank=1), batch)
        assert not torch.equal(base.extras["target"], other_rollout.extras["target"])
        assert not torch.equal(base.extras["target"], other_slot.extras["target"])
        assert not torch.equal(base.extras["target"], other_dp_rank.extras["target"])

    def test_dual_expert_choice_is_rank_aligned(self):
        models = {"transformer": nn.Identity(), "transformer_2": nn.Identity()}
        for slot in range(10):
            # Different DP ranks choose the same expert but use independent sample RNG.
            a = prepare_sft_batch(_ctx(models, microbatch_id=slot, dp_rank=0), _batch())
            b = prepare_sft_batch(_ctx(models, microbatch_id=slot, dp_rank=1), _batch())
            assert a.component_name == b.component_name
            assert not torch.equal(a.extras["target"], b.extras["target"])


class TestSftLossFormula:
    def test_zero_loss_on_exact_velocity(self):
        torch.manual_seed(0)
        ctx = _ctx({"transformer": nn.Identity()})
        batch = _batch()
        prepared = prepare_sft_batch(ctx, batch)
        metrics = _Metrics()
        loss = sft_loss_formula(
            ctx, batch, prepared, new_pred=prepared.extras["target"], ref_pred=None, metrics=metrics
        )
        assert torch.allclose(loss, torch.zeros(()))

    def test_unit_offset_loss(self):
        torch.manual_seed(0)
        ctx = _ctx({"transformer": nn.Identity()})
        batch = _batch()
        prepared = prepare_sft_batch(ctx, batch)
        metrics = _Metrics()
        loss = sft_loss_formula(
            ctx,
            batch,
            prepared,
            new_pred=prepared.extras["target"] + 1.0,
            ref_pred=None,
            metrics=metrics,
        )
        assert torch.allclose(loss, torch.tensor(float(len(batch))))
        assert metrics.seen["loss"] == (float(len(batch)), len(batch))
