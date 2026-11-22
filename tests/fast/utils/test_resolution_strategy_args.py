"""A mixed-resolution rollout is only accepted where the training side can replay it.

    family declares supports_mixed_resolution     ─► allowed
    family does not                               ─► rejected (its RoPE reads the global size)
    --micro-batch-size-sample > 1 (legacy tiles)  ─► rejected (flat micro-batching only)
    weighted_buckets without buckets / bad buckets ─► rejected at startup, not in the rollout actor
    no strategy at all                            ─► nothing is checked

Validation also derives args.diffusion_resolution_bucket_list, which is what the train side
declares its per-resolution metrics from.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import sys
from argparse import Namespace

import pytest

from miles.utils.arguments import validate_resolution_strategy

WEIGHTED_BUCKETS = "miles.rollout.resolution_strategy_hub.weighted_buckets"


class _MixedResolutionConfig:
    supports_mixed_resolution = True


class _FixedResolutionConfig:
    supports_mixed_resolution = False


def _args(**overrides):
    values = dict(
        custom_expand_samples_to_train_pairs_path=None,
        custom_convert_samples_to_train_data_path=None,
        diffusion_resolution_strategy_path=WEIGHTED_BUCKETS,
        diffusion_resolution_buckets="512x512,672x384,384x672",
        diffusion_resolution_weights=None,
        micro_batch_size=1,
        micro_batch_size_sample=None,
        micro_batch_size_tstep=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_accepts_a_family_that_declares_mixed_resolution():
    validate_resolution_strategy(_args(), _MixedResolutionConfig)


def test_no_strategy_checks_nothing():
    # a fixed-resolution family with a wide micro-batch stays valid
    validate_resolution_strategy(
        _args(diffusion_resolution_strategy_path=None, micro_batch_size=8), _FixedResolutionConfig
    )


@pytest.mark.parametrize(
    "overrides, cfg_cls, match",
    [
        (dict(), _FixedResolutionConfig, "supports_mixed_resolution=False"),
        (
            dict(micro_batch_size=8, micro_batch_size_sample=8, micro_batch_size_tstep=1),
            _MixedResolutionConfig,
            "flat micro-batching only",
        ),
        (dict(diffusion_resolution_buckets=None), _MixedResolutionConfig, "needs --diffusion-resolution-buckets"),
        (dict(diffusion_resolution_buckets="512x500"), _MixedResolutionConfig, "multiple of 16"),
        (dict(diffusion_resolution_weights="1,1"), _MixedResolutionConfig, "one weight per bucket"),
    ],
)
def test_rejects_rollouts_training_cannot_replay(overrides, cfg_cls, match):
    with pytest.raises(ValueError, match=match):
        validate_resolution_strategy(_args(**overrides), cfg_cls)


def test_a_flat_micro_batch_is_accepted():
    # the train actor checks each micro-batch holds one resolution (check_single_resolution)
    validate_resolution_strategy(_args(micro_batch_size=8), _MixedResolutionConfig)


def test_a_custom_strategy_needs_no_buckets():
    # buckets belong to weighted_buckets; another strategy may read the resolution off the sample
    validate_resolution_strategy(
        _args(diffusion_resolution_strategy_path="miles.utils.misc.load_function", diffusion_resolution_buckets=None),
        _MixedResolutionConfig,
    )  # the conversion stays the declaring default


def test_weights_without_buckets_are_rejected():
    with pytest.raises(ValueError, match="which is unset"):
        validate_resolution_strategy(
            _args(
                diffusion_resolution_strategy_path="miles.utils.misc.load_function",
                diffusion_resolution_buckets=None,
                diffusion_resolution_weights="2,1,1",
            ),
            _MixedResolutionConfig,
        )


def test_validation_derives_the_declared_buckets():
    args = _args()

    validate_resolution_strategy(args, _MixedResolutionConfig)

    assert args.diffusion_resolution_bucket_list == ((512, 512), (672, 384), (384, 672))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(diffusion_resolution_strategy_path=None),  # fixed resolution
        # a custom strategy free to pick any size declares nothing
        dict(
            diffusion_resolution_strategy_path="miles.utils.misc.load_function",
            diffusion_resolution_buckets=None,
        ),
    ],
)
def test_nothing_is_declared_without_a_bucket_list(overrides):
    args = _args(**overrides)

    validate_resolution_strategy(args, _MixedResolutionConfig)

    assert args.diffusion_resolution_bucket_list == ()


def test_the_default_conversion_labels_its_pairs():
    # unset means Flow-GRPO, which declares it
    validate_resolution_strategy(_args(), _MixedResolutionConfig)


@pytest.mark.parametrize("conversion", ["nft", "dmd2"])
def test_a_conversion_that_does_not_declare_labelling_is_rejected(conversion):
    # --loss-type nft / dmd2 select these; their pairs carry no resolution, which would leave
    # the train-side micro-batch check vacuous
    path = f"miles.ray.data_conversion_hub.{conversion}.expand_samples_to_train_pairs"
    with pytest.raises(ValueError, match="does not label train pairs"):
        validate_resolution_strategy(
            _args(custom_expand_samples_to_train_pairs_path=path), _MixedResolutionConfig
        )


def test_a_conversion_is_unsupported_until_it_declares_itself():
    # the safe default: a conversion that has not thought about resolution is rejected, rather
    # than silently making the train-side check vacuous
    with pytest.raises(ValueError, match="does not label train pairs"):
        validate_resolution_strategy(
            _args(custom_expand_samples_to_train_pairs_path="miles.utils.misc.load_function"),
            _MixedResolutionConfig,
        )


def test_a_declaring_conversion_is_accepted():
    from miles.ray.data_conversion_hub import labels_rollout_resolution

    @labels_rollout_resolution
    def convert(args, samples, rewards, raw_rewards):
        raise NotImplementedError

    sys.modules[__name__].convert = convert
    validate_resolution_strategy(
        _args(custom_expand_samples_to_train_pairs_path=f"{__name__}.convert"), _MixedResolutionConfig
    )


# ---------------------------------------------------------------------------
# Validation must resolve the conversion the way the run will
# ---------------------------------------------------------------------------

FULL_OVERRIDE = "custom_convert_samples_to_train_data_path"
EXPAND_HOOK = "custom_expand_samples_to_train_pairs_path"
NFT = "miles.ray.data_conversion_hub.nft.expand_samples_to_train_pairs"


def test_the_full_override_is_what_gets_checked():
    """RolloutManager returns from --custom-convert-samples-to-train-data-path and never reaches
    the expand hook, so an undeclared full override must be rejected even next to a declared
    one -- and a declared full override must not be rejected for an expand hook that is dead."""
    from miles.ray.data_conversion_hub import labels_rollout_resolution

    @labels_rollout_resolution
    def declared(args, samples):
        raise NotImplementedError

    def undeclared(args, samples):
        raise NotImplementedError

    sys.modules[__name__].declared = declared
    sys.modules[__name__].undeclared = undeclared

    # the dead expand hook does not save an undeclared override, nor condemn a declared one
    with pytest.raises(ValueError, match="does not label train pairs"):
        validate_resolution_strategy(
            _args(**{FULL_OVERRIDE: f"{__name__}.undeclared"}), _MixedResolutionConfig
        )
    validate_resolution_strategy(
        _args(**{FULL_OVERRIDE: f"{__name__}.declared", EXPAND_HOOK: NFT}), _MixedResolutionConfig
    )


def test_the_expand_hook_is_checked_when_no_full_override_is_set():
    with pytest.raises(ValueError, match="does not label train pairs"):
        validate_resolution_strategy(_args(**{EXPAND_HOOK: NFT}), _MixedResolutionConfig)
