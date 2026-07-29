"""CPU tests for --ref-mode resolution and train-data convert layering."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from miles.backends.fsdp_utils.loss_hub.nft import NftTrainDataConverter
from miles.utils.arguments import resolve_and_validate_ref_mode
from miles.utils.train_data_utils import RolloutTrainDataConverter, resolve_train_data_converter


def _ref_args(**overrides):
    base = dict(
        ref_mode=None,
        use_lora=True,
        lora_ema_shadow=False,
        diffusion_kl_beta=0.0,
        diffusion_nft_ref_mode="ema",
        loss_type="policy_loss",
    )
    base.update(overrides)
    return Namespace(**base)


class TestResolveAndValidateRefMode:
    def test_default_none(self):
        args = _ref_args()
        resolve_and_validate_ref_mode(args, is_nft=False, ema_enabled=False)
        assert args.ref_mode == "none"

    def test_kl_beta_auto_lora_base(self):
        args = _ref_args(diffusion_kl_beta=0.01)
        resolve_and_validate_ref_mode(args, is_nft=False, ema_enabled=False)
        assert args.ref_mode == "lora_base"

    def test_nft_auto_ema_when_shadow_on(self):
        args = _ref_args(loss_type="nft", lora_ema_shadow=True)
        resolve_and_validate_ref_mode(args, is_nft=True, ema_enabled=True)
        assert args.ref_mode == "ema"

    def test_nft_auto_fallback_without_shadow(self):
        args = _ref_args(loss_type="nft", lora_ema_shadow=False)
        resolve_and_validate_ref_mode(args, is_nft=True, ema_enabled=False)
        assert args.ref_mode == "lora_base"

    def test_nft_rejects_explicit_none(self):
        args = _ref_args(ref_mode="none", loss_type="nft")
        with pytest.raises(ValueError, match="nft requires a reference model"):
            resolve_and_validate_ref_mode(args, is_nft=True, ema_enabled=False)

    def test_explicit_ema_requires_shadow(self):
        args = _ref_args(ref_mode="ema")
        with pytest.raises(ValueError, match="requires --lora-ema-shadow"):
            resolve_and_validate_ref_mode(args, is_nft=False, ema_enabled=False)

    def test_kl_beta_rejects_none(self):
        args = _ref_args(ref_mode="none", diffusion_kl_beta=0.1)
        with pytest.raises(ValueError, match="diffusion-kl-beta"):
            resolve_and_validate_ref_mode(args, is_nft=False, ema_enabled=False)

    def test_nft_alias_base_maps_to_lora_base(self):
        args = _ref_args(diffusion_nft_ref_mode="base")
        resolve_and_validate_ref_mode(args, is_nft=True, ema_enabled=False)
        assert args.ref_mode == "lora_base"


class TestResolveTrainDataConverter:
    def test_default_is_rollout_converter(self):
        conv = resolve_train_data_converter(Namespace(loss_type="policy_loss"))
        assert isinstance(conv, RolloutTrainDataConverter)

    def test_nft_selects_nft_converter(self):
        conv = resolve_train_data_converter(
            Namespace(
                loss_type="nft",
                diffusion_nft_timestep_fraction=1.0,
                diffusion_nft_shuffle_timesteps=False,
            )
        )
        assert isinstance(conv, NftTrainDataConverter)


class TestConvertSamplesLayering:
    def test_custom_convert_short_circuits_before_post_process(self):
        from miles.ray.rollout import RolloutManager

        cls = RolloutManager.__ray_actor_class__
        called = {"post": False, "custom": False}

        def custom_convert(args, samples):
            called["custom"] = True
            return {"train_data": []}

        fake = SimpleNamespace(
            custom_convert_samples_to_train_data_func=custom_convert,
            args=Namespace(loss_type="policy_loss"),
            rollout_id=0,
            train_data_converter=MagicMock(),
        )

        def boom_post_process(self, samples):
            called["post"] = True
            raise AssertionError("post_process should be skipped for full custom_convert")

        fake._post_process_rewards = boom_post_process.__get__(fake, type(fake))
        out = cls._convert_samples_to_train_data(fake, [])
        assert out == {"train_data": []}
        assert called["custom"] is True
        assert called["post"] is False
        fake.train_data_converter.convert_samples.assert_not_called()
