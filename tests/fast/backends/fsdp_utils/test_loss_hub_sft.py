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


def _scheduler():
    sigmas = torch.linspace(1.0, 1.0 / NUM_GRID, NUM_GRID)
    return Namespace(
        timesteps=sigmas * NUM_TRAIN_TIMESTEPS,
        sigmas=torch.cat([sigmas, torch.zeros(1)]),
        config=Namespace(num_train_timesteps=NUM_TRAIN_TIMESTEPS),
    )


def _ctx(models):
    return DiffusionLossContext(
        models=models,
        train_pipeline_config=_Config(),
        sde_backend=None,
        scheduler=_scheduler(),
        args=Namespace(),
        forward_dtype=torch.float32,
        device=torch.device("cpu"),
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
        torch.manual_seed(0)
        ctx = _ctx({"transformer": nn.Identity()})
        _, _, idx = sample_grid_indices(ctx, bsz=20000)
        counts = torch.bincount(idx, minlength=NUM_GRID).float()
        assert counts.min() > 0
        assert ((counts / 20000) - 1 / NUM_GRID).abs().max() < 0.02

    def test_dual_expert_micro_batch_is_phase_pure(self):
        torch.manual_seed(0)
        models = {"transformer": nn.Identity(), "transformer_2": nn.Identity()}
        ctx = _ctx(models)
        config = ctx.train_pipeline_config
        picked = set()
        for _ in range(20):
            name, model, idx = sample_grid_indices(ctx, bsz=4)
            picked.add(name)
            assert model is models[name]
            for i in idx.tolist():
                t = float(ctx.scheduler.timesteps[i])
                assert config.component_for_timestep(t, NUM_TRAIN_TIMESTEPS) == name
        assert picked == {"transformer", "transformer_2"}


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
