"""Smoke tests for DiffusionNFT hooks (prepare + loss formula; actor owns DiT)."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import torch

from miles.backends.fsdp_utils.configs.qwen_image import QwenImageTrainPipelineConfig
from miles.backends.fsdp_utils.configs.train_pipeline_config import TrainPipelineConfig
from miles.backends.fsdp_utils.ema import EmaShadow
from miles.backends.fsdp_utils.loss_hub.nft import corrupt, nft_r_from_advantages, prepare_nft_batch
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext
from miles.ray.data_conversion_hub.nft import expand_samples_to_train_pairs, resolve_nft_sigmas
from miles.utils.types import Sample


def _args(**overrides):
    base = dict(
        n_samples_per_prompt=2,
        globalize_reward_mean=False,
        globalize_reward_std=False,
        grpo_std_normalization=True,
        reward_key=None,
        diffusion_nft_timestep_fraction=1.0,
        diffusion_nft_shuffle_timesteps=False,
        seed=42,
        custom_prepare_train_batch_path=None,
        custom_loss_function_path=None,
    )
    base.update(overrides)
    return Namespace(**base)


class TestNftMath:
    def test_nft_r_remap(self):
        r = nft_r_from_advantages(torch.tensor([-5.0, 0.0, 5.0]), adv_clip_max=5.0)
        assert torch.allclose(r, torch.tensor([0.0, 0.5, 1.0]))

    def test_corrupt_linear_flow(self):
        x0 = torch.ones(2, 4)
        eps = torch.zeros(2, 4)
        t = torch.tensor([0.25, 0.75])
        xt = corrupt(x0, t, eps)
        assert torch.allclose(xt[0], torch.full((4,), 0.75))
        assert torch.allclose(xt[1], torch.full((4,), 0.25))

    def test_resolve_sigmas_drops_zero_and_fraction(self):
        sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        ts = resolve_nft_sigmas(sigmas, training_timestep_fraction=0.99)
        assert torch.allclose(ts, torch.tensor([1.0, 0.8, 0.6, 0.4]))


class TestNftHooks:
    def test_convert_expands_k_timestep_pairs(self):
        class _Traj:
            def __init__(self):
                self.timesteps = torch.tensor([999.0, 500.0, 0.0])
                self.sigmas = torch.tensor([1.0, 0.5, 0.0])
                self.latents = torch.zeros(3, 2, 2)
                self.latent_step_indices = None

        class _Env:
            pos_cond_kwargs = {}
            neg_cond_kwargs = None

        samples = [
            Sample(index=0, prompt="a", reward=1.0, dit_trajectory=_Traj(), denoising_env=_Env()),
            Sample(index=1, prompt="b", reward=3.0, dit_trajectory=_Traj(), denoising_env=_Env()),
        ]
        args = _args()
        raw_rewards = [1.0, 3.0]
        rewards = [-1.0, 1.0]
        out = expand_samples_to_train_pairs(args, samples, rewards, raw_rewards)
        assert len(out["train_data"]) == 4
        assert {p["timestep"] for p in out["train_data"]} == {1.0, 0.5}
        assert out["train_data"][0]["x0"] is out["train_data"][1]["x0"]
        assert out["train_data"][0]["advantage"] == rewards[0]
        assert out["train_data"][2]["advantage"] == rewards[1]

    def test_convert_rejects_mismatched_scheduler_meta(self):
        # One shared schedule per batch, same contract as the flow_grpo converter.
        class _Traj:
            def __init__(self):
                self.timesteps = torch.tensor([999.0, 500.0, 0.0])
                self.sigmas = torch.tensor([1.0, 0.5, 0.0])
                self.latents = torch.zeros(3, 2, 2)
                self.latent_step_indices = None

        class _Env:
            pos_cond_kwargs = {}
            neg_cond_kwargs = None

        samples = [
            Sample(index=0, prompt="a", reward=1.0, dit_trajectory=_Traj(), denoising_env=_Env()),
            Sample(index=1, prompt="b", reward=3.0, dit_trajectory=_Traj(), denoising_env=_Env()),
        ]
        samples[1].dit_trajectory.sigmas = samples[1].dit_trajectory.sigmas + 1.0  # tamper
        try:
            expand_samples_to_train_pairs(_args(), samples, [1.0, 2.0], [1.0, 2.0])
        except ValueError as e:
            assert "scheduler_sigmas" in str(e)
        else:
            raise AssertionError("expected ValueError for mismatched scheduler_sigmas")

    def test_convert_requires_rollout_sigmas(self):
        # Sigmas come from the rollout scheduler snapshot; no timesteps/1000 fallback.
        class _Traj:
            def __init__(self):
                self.timesteps = torch.tensor([999.0, 500.0, 0.0])
                self.sigmas = None
                self.latents = torch.zeros(3, 2, 2)
                self.latent_step_indices = None

        class _Env:
            pos_cond_kwargs = {}
            neg_cond_kwargs = None

        samples = [Sample(index=0, prompt="a", reward=1.0, dit_trajectory=_Traj(), denoising_env=_Env())]
        try:
            expand_samples_to_train_pairs(_args(), samples, [1.0], [1.0])
        except ValueError as e:
            assert "sigmas" in str(e)
        else:
            raise AssertionError("expected ValueError for missing dit_trajectory.sigmas")


class _StubConfig:
    """Cond plumbing stubbed out. Binds both hooks so a prepare wired to the wrong one fails on
    the numbers, not on a missing attribute."""

    process_timestep_as_input = TrainPipelineConfig.process_timestep_as_input
    process_sigma_as_timesteps_input = TrainPipelineConfig.process_sigma_as_timesteps_input

    def prepare_cond_kwargs(self, cond, device):
        return {}

    def collate_cond_for_sample_batch(self, per_sample_cond_kwargs, device, pad_to_len=None):
        return {}


class _Sd3StyleConfig(_StubConfig):
    pass


class _QwenStyleConfig(_StubConfig):
    process_timestep_as_input = QwenImageTrainPipelineConfig.process_timestep_as_input
    process_sigma_as_timesteps_input = QwenImageTrainPipelineConfig.process_sigma_as_timesteps_input


class TestPrepareNftBatch:
    NUM_TRAIN_TIMESTEPS = 1000
    # 0.8474... does not survive a multiply then divide by 1000 in fp32.
    SIGMAS = [0.8474337458610535, 0.5]

    class _Env:
        pos_cond_kwargs = None
        neg_cond_kwargs = None

    def _ctx(self, config, microbatch_id=0):
        return DiffusionLossContext(
            models={"transformer": torch.nn.Identity()},
            train_pipeline_config=config,
            sde_backend=None,
            scheduler=Namespace(config=Namespace(num_train_timesteps=self.NUM_TRAIN_TIMESTEPS)),
            args=Namespace(seed=42),
            forward_dtype=torch.float32,
            device=torch.device("cpu"),
            microbatch_id=microbatch_id,
        )

    def _batch(self):
        return [
            {
                "x0": torch.zeros(2, 2),
                "timestep": sigma,
                "denoising_env": self._Env(),
                "advantage": 1.0,
                "nft_num_timesteps": len(self.SIGMAS),
            }
            for sigma in self.SIGMAS
        ]

    def test_raw_sigma_reaches_the_family_hook(self):
        seen = {}

        class _Recording(_StubConfig):
            def process_sigma_as_timesteps_input(self, sigmas, *, num_train_timesteps):
                seen["sigmas"] = sigmas.clone()
                seen["num_train_timesteps"] = num_train_timesteps
                return sigmas

        prepare_nft_batch(self._ctx(_Recording()), self._batch())
        assert torch.equal(seen["sigmas"], torch.tensor(self.SIGMAS))
        assert seen["num_train_timesteps"] == self.NUM_TRAIN_TIMESTEPS

    def test_sd3_style_family_gets_the_scheduler_range(self):
        prepared = prepare_nft_batch(self._ctx(_Sd3StyleConfig()), self._batch())
        assert torch.equal(prepared.timesteps, torch.tensor(self.SIGMAS))
        assert torch.equal(prepared.timesteps_for_model, torch.tensor(self.SIGMAS) * float(self.NUM_TRAIN_TIMESTEPS))

    def test_qwen_style_family_gets_the_sigma_bit_exactly(self):
        prepared = prepare_nft_batch(self._ctx(_QwenStyleConfig()), self._batch())
        # equal, not allclose: the wrong hook would still pass allclose.
        assert torch.equal(prepared.timesteps_for_model, torch.tensor(self.SIGMAS))


class TestNftDeterminism:
    # NFT draws two random streams -- the sigma permutation in the converter and the
    # corruption noise in prepare. Both used the global RNG, so two runs of the same
    # configuration trained on different data and no metric reproduced.
    def _traj(self):
        class _Traj:
            def __init__(self):
                self.timesteps = torch.tensor([999.0, 750.0, 500.0, 250.0, 0.0])
                self.sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
                self.latents = torch.zeros(5, 2, 2)
                self.latent_step_indices = None

        return _Traj()

    def _samples(self):
        class _Env:
            pos_cond_kwargs = None
            neg_cond_kwargs = None

        return [
            Sample(index=i, prompt=p, reward=r, dit_trajectory=self._traj(), denoising_env=_Env())
            for i, (p, r) in enumerate([("a", 1.0), ("b", 3.0)])
        ]

    def test_sigma_shuffle_reproduces(self):
        args = _args(diffusion_nft_shuffle_timesteps=True)
        first = expand_samples_to_train_pairs(args, self._samples(), [-1.0, 1.0], [1.0, 3.0])
        second = expand_samples_to_train_pairs(args, self._samples(), [-1.0, 1.0], [1.0, 3.0])
        got = [p["timestep"] for p in first["train_data"]]
        assert got == [p["timestep"] for p in second["train_data"]]
        # Shuffled, not just handed back in scheduler order.
        assert got[: len(got) // 2] != sorted(got[: len(got) // 2], reverse=True)

    def test_each_sample_draws_its_own_permutation(self):
        args = _args(diffusion_nft_shuffle_timesteps=True)
        out = expand_samples_to_train_pairs(args, self._samples(), [-1.0, 1.0], [1.0, 3.0])
        per_sample = {}
        for pair in out["train_data"]:
            per_sample.setdefault(pair["sample_index"], []).append(pair["timestep"])
        assert len(per_sample) == 2
        assert list(per_sample.values())[0] != list(per_sample.values())[1]

    def test_corruption_noise_reproduces(self):
        ctx = TestPrepareNftBatch()._ctx(_Sd3StyleConfig())
        batch = TestPrepareNftBatch()._batch()
        first = prepare_nft_batch(ctx, batch)
        second = prepare_nft_batch(ctx, batch)
        assert torch.equal(first.latents, second.latents)

    def test_a_different_microbatch_draws_different_noise(self):
        harness = TestPrepareNftBatch()
        first = prepare_nft_batch(harness._ctx(_Sd3StyleConfig()), harness._batch())
        second = prepare_nft_batch(harness._ctx(_Sd3StyleConfig(), microbatch_id=1), harness._batch())
        assert not torch.equal(first.latents, second.latents)


class TestEmaShadow:
    def _model(self):
        return torch.nn.Linear(4, 4, bias=False)

    def test_snapshot_and_update(self):
        m = self._model()
        ema = EmaShadow(m.parameters(), decay=0.5, uprate=0.001, uphold=0.5, flat_steps=10)
        init = m.weight.detach().clone()
        with torch.no_grad():
            m.weight.add_(1.0)
        delta = ema.update()
        assert delta == 0.5
        assert torch.allclose(ema.shadow[0], init + 0.5)

    def test_swap_in_restores_exactly(self):
        m = self._model()
        ema = EmaShadow(m.parameters(), decay=0.1)
        live = m.weight.detach().clone()
        with torch.no_grad():
            m.weight.add_(2.0)
        with ema.swap_in():
            assert torch.equal(m.weight.detach(), live)
        assert torch.equal(m.weight.detach(), live + 2.0)
