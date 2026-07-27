"""Smoke tests for DiffusionNFT hooks (prepare + loss formula; actor owns DiT)."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest
import torch

from miles.backends.fsdp_utils.loss_hub.losses import resolve_loss_formula_fn, resolve_prepare_fn
from miles.backends.fsdp_utils.loss_hub.nft import (
    convert_samples_to_nft_train_data,
    corrupt,
    nft_loss_formula,
    nft_r_from_advantages,
    prepare_nft_batch,
    resolve_nft_sigmas,
)
from miles.backends.fsdp_utils.lora_ema import (
    LoraEmaShadow,
    lora_ema_rollout_policy,
    lora_ema_shadow_enabled,
    resolve_lora_ema_kwargs,
)
from miles.backends.fsdp_utils.metrics import new_metric_buffer
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
        class _Sched:
            sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

        ts = resolve_nft_sigmas(_Sched(), training_timestep_fraction=0.99)
        assert torch.allclose(ts, torch.tensor([1.0, 0.8, 0.6, 0.4]))


class TestNftHooks:
    def test_convert_expands_k_timestep_pairs(self):
        class _Traj:
            def __init__(self):
                self.timesteps = torch.tensor([999.0, 500.0, 0.0])
                self.sigmas = torch.tensor([1.0, 0.5, 0.0])
                self.latents = torch.zeros(3, 2, 2)

        class _Env:
            pos_cond_kwargs = {}
            neg_cond_kwargs = None

        samples = [
            Sample(index=0, prompt="a", reward=1.0, dit_trajectory=_Traj(), denoising_env=_Env()),
            Sample(index=1, prompt="b", reward=3.0, dit_trajectory=_Traj(), denoising_env=_Env()),
        ]
        out = convert_samples_to_nft_train_data(_args(), samples)
        assert len(out["train_data"]) == 4
        assert {p["timestep"] for p in out["train_data"]} == {1.0, 0.5}
        assert out["train_data"][0]["x0"] is out["train_data"][1]["x0"]

    def test_formula_write_old_log_prob_is_noop(self):
        metrics = new_metric_buffer(None, torch.device("cpu"), ())
        loss = nft_loss_formula(
            None,  # type: ignore[arg-type]
            [],
            None,  # type: ignore[arg-type]
            new_pred=None,  # type: ignore[arg-type]
            ref_pred=None,
            metrics=metrics,
            write_old_log_prob=True,
        )
        assert loss is None

    def test_formula_declares_ref_and_window_attrs(self):
        assert getattr(nft_loss_formula, "ref_mode", None) == "ema"
        assert getattr(nft_loss_formula, "requires_sample_aligned_windows", False) is True

    def test_resolve_defaults_are_flow_grpo(self):
        from miles.backends.fsdp_utils.loss_hub.losses import flow_grpo_loss_formula, prepare_flow_grpo_batch

        assert resolve_prepare_fn(_args()) is prepare_flow_grpo_batch
        assert resolve_loss_formula_fn(_args()) is flow_grpo_loss_formula

    def test_resolve_custom_paths(self):
        args = _args(
            custom_prepare_train_batch_path="miles.backends.fsdp_utils.loss_hub.nft.prepare_nft_batch",
            custom_loss_function_path="miles.backends.fsdp_utils.loss_hub.nft.nft_loss_formula",
        )
        assert resolve_prepare_fn(args) is prepare_nft_batch
        assert resolve_loss_formula_fn(args) is nft_loss_formula


class TestLoraEmaShadow:
    def _model(self):
        return torch.nn.Linear(4, 4, bias=False)

    def test_snapshot_and_update(self):
        m = self._model()
        ema = LoraEmaShadow(m.parameters(), decay=0.5, uprate=0.001, uphold=0.5, flat_steps=10)
        init = m.weight.detach().clone()
        with torch.no_grad():
            m.weight.add_(1.0)
        delta = ema.update()
        assert delta == 0.5
        assert torch.allclose(ema.shadow[0], init + 0.5)

    def test_swap_in_restores_exactly(self):
        m = self._model()
        ema = LoraEmaShadow(m.parameters(), decay=0.1)
        live = m.weight.detach().clone()
        with torch.no_grad():
            m.weight.add_(2.0)
        with ema.swap_in():
            assert torch.equal(m.weight.detach(), live)
        assert torch.equal(m.weight.detach(), live + 2.0)


class TestLoraEmaArgs:
    def test_shadow_enabled(self):
        assert lora_ema_shadow_enabled(Namespace(lora_ema_shadow=True))
        assert not lora_ema_shadow_enabled(Namespace())

    def test_rollout_policy(self):
        assert lora_ema_rollout_policy(Namespace(lora_ema_rollout_policy="ema")) == "ema"
        assert lora_ema_rollout_policy(Namespace()) == "live"

    def test_resolve_kwargs(self):
        kwargs = resolve_lora_ema_kwargs(
            Namespace(lora_ema_decay=0.01, lora_ema_uprate=0.02, lora_ema_uphold=0.3, lora_ema_flat_steps=5)
        )
        assert kwargs == {"decay": 0.01, "uprate": 0.02, "uphold": 0.3, "flat_steps": 5}
