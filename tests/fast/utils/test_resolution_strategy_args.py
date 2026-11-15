"""A mixed-resolution rollout is only accepted where the training side can replay it.

    family declares supports_mixed_resolution     ─► allowed
    family does not                               ─► rejected (its RoPE reads the global size)
    --micro-batch-size-sample > 1 (legacy tiles)  ─► rejected (a tile spans prompt groups)
    weighted_buckets without buckets / bad buckets ─► rejected at startup, not in the rollout actor
    no strategy at all                            ─► nothing is checked
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

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
            "span prompt groups",
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
    )


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
