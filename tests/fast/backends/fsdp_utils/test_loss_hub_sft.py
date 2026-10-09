"""Diffusion SFT hooks: prepare + loss formula over named latent streams (the actor owns the DiT).

    cached pair["latent"]: {"visual": T}  (single-stream families)  or  {"visual": T, "audio": T}  (joint)
                │
    prepare_sft_batch:
      grid       ──► FlowNoiseSchedule.shifted(--fsdp-flow-shift[name])
      select_component(visual grid), once, rank-aligned ──► DiT component + the visual grid points it serves
      for each stream, its own seeded generator draws
                       (visual: the single-stream seed; other streams add their name to it)
      grid index ──► uniform over its pool (visual: the component's points; others: the whole grid)
                 ──► timesteps[name], extras["sigmas"][name]
      noise      ──► latents[name] = (1 - sigma) x0 + sigma noise,  extras["target"][name] = noise - x0
                │
    sft_loss_formula(new_pred keyed like latents): per pair, sum_streams mean((pred - target)^2)
                                                   + sigma-bucket metrics of the visual stream;
                                                   a stream absent from new_pred carries no loss

What each test pins:
  TestPrepareSftBatch   corruption identity, select_component's expert routing and served grid points,
                        determinism, known flow-shift grid values,
                        the visual stream draws exactly what the single-stream seed draws
  TestSftLossFormula    exact-velocity zero loss, unit offset, sigma buckets partition the loss,
                        --log-loss-sigma-bucket 0 emits only the loss
  TestJointStreams      each stream on its own --fsdp-flow-shift grid, draws independent across streams,
                        per-stream MSE normalization, an unpredicted stream adds no loss,
                        rank-aligned expert choice, shape/stream-set rejection
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest
import torch
import torch.nn as nn

from miles.backends.fsdp_utils.loss_hub.sft import prepare_sft_batch, select_component, sft_loss_formula
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, FlowNoiseSchedule
from miles.utils.hash_utils import stable_hash

NUM_TRAIN_TIMESTEPS = 1000
NUM_GRID = 8


class _Config:
    # The SD3 default: the DiT takes the trajectory timestep unchanged.
    process_timestep_as_input = staticmethod(lambda timesteps: timesteps)
    num_train_timesteps = NUM_TRAIN_TIMESTEPS

    def collate_cond_for_sample_batch(self, per_sample_cond_kwargs, device, pad_to_len=None):
        return {"encoder_hidden_states": torch.cat([kw["encoder_hidden_states"] for kw in per_sample_cond_kwargs])}

    def component_for_timestep(self, timestep):
        return "transformer" if timestep >= 0.875 * self.num_train_timesteps else "transformer_2"


class _SingleConfig(_Config):
    def component_for_timestep(self, timestep):
        return "transformer"


FLOW_SHIFT = 3.0
# --fsdp-flow-shift for every stream the tests build; each stream gets a distinct grid.
FLOW_SHIFTS = {"visual": FLOW_SHIFT, "audio": 1.0, "action": 6.0}


def _noise_schedule():
    sigmas = torch.linspace(1.0, 1.0 / NUM_GRID, NUM_GRID)
    return FlowNoiseSchedule(sigmas * NUM_TRAIN_TIMESTEPS, torch.cat([sigmas, torch.zeros(1)]))


def _ctx(models, rollout_id=3, microbatch_id=0, dp_rank=0, config=None):
    return DiffusionLossContext(
        models=models,
        train_pipeline_config=config if config is not None else _Config(),
        sde_backend=None,
        args=Namespace(seed=42, log_loss_sigma_bucket=5, fsdp_flow_shift=FLOW_SHIFTS),
        forward_dtype=torch.float32,
        device=torch.device("cpu"),
        rollout_id=rollout_id,
        microbatch_id=microbatch_id,
        dp_rank=dp_rank,
    )


def _batch(bsz=4):
    return [
        {
            "latent": {"visual": torch.randn(16, 2, 4, 4)},
            "cond_kwargs": {"encoder_hidden_states": torch.randn(1, 6, 8)},
        }
        for _ in range(bsz)
    ]


def _visual_target(prepared):
    return prepared.extras["target"]["visual"]


class _Metrics:
    def __init__(self):
        self.seen = {}

    def emit_mean(self, key, *, total, count):
        prev_total, prev_count = self.seen.get(key, (0.0, 0))
        self.seen[key] = (prev_total + float(total), prev_count + count)


class TestPrepareSftBatch:
    def test_corruption_and_target_identity(self):
        torch.manual_seed(0)
        ctx = _ctx({"transformer": nn.Identity()})
        batch = _batch()
        prepared = prepare_sft_batch(ctx, batch)

        x0 = torch.stack([pair["latent"]["visual"] for pair in batch]).float()
        sigma = (prepared.timesteps["visual"] / NUM_TRAIN_TIMESTEPS).view(-1, 1, 1, 1, 1)
        assert torch.allclose(prepared.latents["visual"], x0 + sigma * _visual_target(prepared), atol=1e-5)
        assert not prepared.use_cfg
        assert prepared.pos_cond["encoder_hidden_states"].shape == (4, 6, 8)
        # SD3's DiT takes the raw trajectory timestep, so the hook is the identity here.
        assert torch.equal(prepared.timesteps_for_model["visual"], prepared.timesteps["visual"])

    def test_single_model_serves_the_whole_grid(self):
        ctx = _ctx({"transformer": nn.Identity()}, config=_SingleConfig())
        _, _, pool = select_component(ctx, _noise_schedule(), torch.device("cpu"))
        assert torch.equal(pool, torch.arange(NUM_GRID))

    def test_single_wan_expert_serves_only_its_timesteps(self):
        config = _Config()
        timesteps = _noise_schedule().timesteps
        for component_name in ("transformer", "transformer_2"):
            ctx = _ctx({component_name: nn.Identity()}, config=config)
            name, _, pool = select_component(ctx, _noise_schedule(), torch.device("cpu"))
            expected = [
                i
                for i, timestep in enumerate(timesteps)
                if config.component_for_timestep(float(timestep)) == component_name
            ]
            assert name == component_name
            assert pool.tolist() == expected

    def test_dual_expert_micro_batch_is_phase_pure(self):
        models = {"transformer": nn.Identity(), "transformer_2": nn.Identity()}
        ctx = _ctx(models)
        config = ctx.train_pipeline_config
        noise_schedule = _noise_schedule()
        picked = set()
        for call in range(20):
            ctx.microbatch_id = call
            name, model, pool = select_component(ctx, noise_schedule, torch.device("cpu"))
            picked.add(name)
            assert model is models[name]
            for i in pool.tolist():
                t = float(noise_schedule.timesteps[i])
                assert config.component_for_timestep(t) == name
        assert picked == {"transformer", "transformer_2"}

    def test_prepare_is_independent_of_global_rng_state(self):
        batch = _batch()
        torch.manual_seed(0)
        global_state = torch.get_rng_state()

        first = prepare_sft_batch(_ctx({"transformer": nn.Identity()}), batch)
        assert torch.equal(torch.get_rng_state(), global_state)

        torch.rand(1000)
        second = prepare_sft_batch(_ctx({"transformer": nn.Identity()}), batch)
        assert torch.equal(first.timesteps["visual"], second.timesteps["visual"])
        assert torch.equal(first.latents["visual"], second.latents["visual"])
        assert torch.equal(_visual_target(first), _visual_target(second))

    def test_visual_stream_draws_what_the_single_stream_seed_draws(self):
        # seed (no stream name) --randint over the high-noise expert's grid points--> grid index --randn--> noise:
        # the draws single-stream SFT made before latents were keyed by stream, rebuilt by hand.
        ctx = _ctx({"transformer": nn.Identity()})
        batch = _batch()
        prepared = prepare_sft_batch(ctx, batch)

        generator = torch.Generator().manual_seed(stable_hash("sample", 42, 3, 0, 0))
        noise_schedule = FlowNoiseSchedule.shifted(FLOW_SHIFT, NUM_TRAIN_TIMESTEPS)
        high_noise = [i for i, t in enumerate(noise_schedule.timesteps.tolist()) if t >= 0.875 * NUM_TRAIN_TIMESTEPS]
        idx = torch.tensor(high_noise)[torch.randint(len(high_noise), (len(batch),), generator=generator)]
        x0 = torch.stack([pair["latent"]["visual"] for pair in batch]).float()
        noise = torch.randn(x0.shape, generator=generator)
        assert torch.equal(prepared.timesteps["visual"], noise_schedule.timesteps.float()[idx])
        assert torch.equal(_visual_target(prepared), noise - x0)

    def test_samples_within_microbatch_use_different_noise(self):
        batch = _batch(bsz=4)
        prepared = prepare_sft_batch(_ctx({"transformer": nn.Identity()}), batch)
        x0 = torch.stack([pair["latent"]["visual"] for pair in batch]).float()
        noise = _visual_target(prepared) + x0
        assert all(not torch.equal(noise[0], noise[i]) for i in range(1, len(batch)))

    def test_identity_changes_draws(self):
        batch = _batch()
        base = prepare_sft_batch(_ctx({"transformer": nn.Identity()}), batch)
        other_rollout = prepare_sft_batch(_ctx({"transformer": nn.Identity()}, rollout_id=4), batch)
        other_slot = prepare_sft_batch(_ctx({"transformer": nn.Identity()}, microbatch_id=1), batch)
        other_dp_rank = prepare_sft_batch(_ctx({"transformer": nn.Identity()}, dp_rank=1), batch)
        for other in (other_rollout, other_slot, other_dp_rank):
            assert not torch.equal(_visual_target(base), _visual_target(other))

    def test_dual_expert_choice_is_rank_aligned(self):
        models = {"transformer": nn.Identity(), "transformer_2": nn.Identity()}
        for slot in range(10):
            # Different DP ranks choose the same expert but use independent sample RNG.
            a = prepare_sft_batch(_ctx(models, microbatch_id=slot, dp_rank=0), _batch())
            b = prepare_sft_batch(_ctx(models, microbatch_id=slot, dp_rank=1), _batch())
            assert a.component_name == b.component_name
            assert not torch.equal(_visual_target(a), _visual_target(b))

    def test_prepare_draws_from_the_flow_shift_grid(self):
        # shift 3 over 4 points: s = 1, .75, .5, .25 -> 3s / (1 + 2s) = 1, .9, .75, .5, then the terminal 0
        noise_schedule = FlowNoiseSchedule.shifted(3.0, 4)
        torch.testing.assert_close(noise_schedule.sigmas, torch.tensor([1.0, 0.9, 0.75, 0.5, 0.0]))
        torch.testing.assert_close(noise_schedule.timesteps, torch.tensor([4.0, 3.6, 3.0, 2.0]))
        prepared = prepare_sft_batch(_ctx({"transformer": nn.Identity()}), _batch(bsz=16))
        grid = FlowNoiseSchedule.shifted(FLOW_SHIFT, NUM_TRAIN_TIMESTEPS)
        assert set(prepared.extras["sigmas"]["visual"].tolist()) <= set(grid.sigmas[:-1].tolist())
        assert set(prepared.timesteps["visual"].tolist()) <= set(grid.timesteps.tolist())


class TestSftLossFormula:
    def _loss(self, prediction_offset, metrics, log_loss_sigma_bucket=5):
        ctx = _ctx({"transformer": nn.Identity()})
        ctx.args.log_loss_sigma_bucket = log_loss_sigma_bucket
        batch = _batch()
        prepared = prepare_sft_batch(ctx, batch)
        new_pred = {"visual": _visual_target(prepared) + prediction_offset}
        return prepared, sft_loss_formula(
            ctx, batch, prepared, new_pred=new_pred, old_pred=None, ref_pred=None, metrics=metrics
        )

    def test_zero_loss_on_exact_velocity(self):
        _, loss = self._loss(0.0, _Metrics())
        assert torch.allclose(loss, torch.zeros(()))

    def test_unit_offset_loss(self):
        metrics = _Metrics()
        _, loss = self._loss(1.0, metrics)
        assert torch.allclose(loss, torch.tensor(4.0))
        assert metrics.seen["loss"] == (4.0, 4)

    def test_sigma_buckets_partition_the_loss(self):
        from miles.backends.fsdp_utils.metrics import new_metric_buffer, sigma_bucket_key

        metrics = _Metrics()
        prepared, loss = self._loss(1.0, metrics)
        bucket_keys = [key for key in metrics.seen if key.startswith("loss_sigma_")]
        declared = new_metric_buffer(None, torch.device("cpu"), ["transformer"], sigma_buckets=5)._schema
        assert set(bucket_keys) <= set(declared), "emitted buckets must be pre-declared for the DP reduce layout"
        sigmas = prepared.extras["sigmas"]["visual"]
        assert set(bucket_keys) == {sigma_bucket_key(min(int(float(s) * 5), 4), 5) for s in sigmas}
        assert torch.allclose(torch.tensor(sum(metrics.seen[key][0] for key in bucket_keys)), loss)

    def test_disabled_sigma_buckets(self):
        metrics = _Metrics()
        self._loss(0.0, metrics, log_loss_sigma_bucket=0)
        assert metrics.seen == {"loss": (0.0, 4)}


def test_sigma_bucket_key_edges_are_floats():
    from miles.backends.fsdp_utils.metrics import sigma_bucket_key

    assert sigma_bucket_key(0, 10) == "loss_sigma_0.0_0.1"
    assert sigma_bucket_key(9, 10) == "loss_sigma_0.9_1.0"


def _joint_batch(bsz=4):
    return [
        {
            "latent": {"visual": torch.randn(8, 12), "audio": torch.randn(3, 4)},
            "cond_kwargs": {"encoder_hidden_states": torch.randn(1, 6, 8)},
        }
        for _ in range(bsz)
    ]


class TestJointStreams:
    def test_each_stream_draws_from_its_own_flow_shift_grid(self):
        # --fsdp-flow-shift visual=3,audio=1: each stream's sigmas lie on its own grid, not on the other's.
        batch = _joint_batch(16)
        prepared = prepare_sft_batch(_ctx({"transformer": nn.Identity()}, config=_SingleConfig()), batch)
        assert set(prepared.latents) == {"visual", "audio"}
        grids = {
            name: set(FlowNoiseSchedule.shifted(FLOW_SHIFTS[name], NUM_TRAIN_TIMESTEPS).sigmas[:-1].tolist())
            for name in ("visual", "audio")
        }
        for name in ("visual", "audio"):
            x0 = torch.stack([pair["latent"][name] for pair in batch])
            sigmas = prepared.extras["sigmas"][name]
            assert set(sigmas.tolist()) <= grids[name]
            assert torch.allclose(
                prepared.latents[name], x0 + sigmas.view(-1, 1, 1) * prepared.extras["target"][name], atol=1e-6
            )
        assert not set(prepared.extras["sigmas"]["audio"].tolist()) <= grids["visual"]

    def test_stream_draws_are_independent_of_other_streams(self):
        # Same clean latent in both streams, yet different draws; adding a third stream changes neither.
        ctx = _ctx({"transformer": nn.Identity()}, config=_SingleConfig())
        batch = _joint_batch(32)
        for pair in batch:
            pair["latent"]["audio"] = pair["latent"]["visual"].clone()
        first = prepare_sft_batch(ctx, batch)
        assert not torch.equal(first.timesteps["visual"], first.timesteps["audio"])
        assert not torch.equal(first.extras["target"]["visual"], first.extras["target"]["audio"])
        for pair in batch:
            pair["latent"]["action"] = torch.ones(10, 7)
        second = prepare_sft_batch(ctx, batch)
        for name in ("visual", "audio"):
            assert torch.equal(first.extras["target"][name], second.extras["target"][name])
            assert torch.equal(first.timesteps[name], second.timesteps[name])

    def test_loss_averages_each_stream_over_its_own_size(self):
        # visual (96 elements) off by 1, audio (12 elements) off by 2: per pair 1 + 4, not a size-weighted mix.
        ctx = _ctx({"transformer": nn.Identity()}, config=_SingleConfig())
        batch = _joint_batch(2)
        prepared = prepare_sft_batch(ctx, batch)
        predictions = {
            "visual": (prepared.extras["target"]["visual"] + 1.0).requires_grad_(),
            "audio": (prepared.extras["target"]["audio"] + 2.0).requires_grad_(),
        }
        metrics = _Metrics()
        loss = sft_loss_formula(
            ctx, batch, prepared, new_pred=predictions, old_pred=None, ref_pred=None, metrics=metrics
        )
        assert loss.item() == pytest.approx(10.0)
        # Sigma buckets follow the visual stream only: 2 pairs, each off by 1.
        assert sum(total for key, (total, _) in metrics.seen.items() if key.startswith("loss_sigma_")) == 2.0
        loss.backward()
        assert all(prediction.grad.abs().sum() > 0 for prediction in predictions.values())

    def test_a_stream_the_config_does_not_predict_adds_no_loss(self):
        # H3 Ref2VA on a silent target: the audio stream is noised and fed to the DiT, but only visual is predicted.
        ctx = _ctx({"transformer": nn.Identity()}, config=_SingleConfig())
        batch = _joint_batch(2)
        prepared = prepare_sft_batch(ctx, batch)
        loss = sft_loss_formula(
            ctx,
            batch,
            prepared,
            new_pred={"visual": prepared.extras["target"]["visual"] + 1.0},
            old_pred=None,
            ref_pred=None,
            metrics=_Metrics(),
        )
        assert loss.item() == pytest.approx(2.0)

    def test_visual_stream_selects_a_rank_aligned_expert(self):
        models = {"transformer": nn.Identity(), "transformer_2": nn.Identity()}
        batch = _joint_batch(4)
        for slot in range(5):
            contexts = [_ctx(models, microbatch_id=slot, dp_rank=rank) for rank in (0, 1)]
            first, second = [prepare_sft_batch(context, batch) for context in contexts]
            assert first.component_name == second.component_name
            for timestep in first.timesteps["visual"]:
                component = contexts[0].train_pipeline_config.component_for_timestep(float(timestep))
                assert component == first.component_name

    def test_rejects_misshaped_predictions(self):
        ctx = _ctx({"transformer": nn.Identity()}, config=_SingleConfig())
        batch = _joint_batch(1)
        prepared = prepare_sft_batch(ctx, batch)
        predictions = dict(prepared.extras["target"])
        predictions["audio"] = predictions["audio"][0]
        with pytest.raises(ValueError, match="prediction shape"):
            sft_loss_formula(
                ctx, batch, prepared, new_pred=predictions, old_pred=None, ref_pred=None, metrics=_Metrics()
            )

    def test_rejects_different_stream_sets_in_one_microbatch(self):
        batch = _joint_batch(2)
        del batch[1]["latent"]["audio"]
        with pytest.raises(ValueError, match="latent streams"):
            prepare_sft_batch(_ctx({"transformer": nn.Identity()}), batch)
