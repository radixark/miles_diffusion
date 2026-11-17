"""A resolution strategy draws one resolution per prompt group, and the whole group
-- every microgroup of it -- renders at that resolution.

    --diffusion-resolution-buckets 512x512,672x384  ─► parsed once, height first
    --diffusion-resolution-weights 2,1              ─► normalized draw weights
    weighted_buckets(args, group[0], seed_base)     ─► reproducible from the group's seed
    apply_resolution_strategy(...)                  ─► request body + every sample in the group
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.rollout.resolution_strategy_hub import parse_resolution_buckets, parse_resolution_weights, weighted_buckets
from miles.rollout.sglang_diffusion_rollout import apply_resolution_strategy
from miles.rollout.step_strategy_hub import sde_window
from miles.utils.hash_utils import stable_hash
from miles.utils.types import Sample


def _args(**overrides):
    values = dict(diffusion_resolution_buckets="512x512,672x384,384x672", diffusion_resolution_weights=None)
    values.update(overrides)
    return Namespace(**values)


# ---------------------------------------------------------------------------
# Bucket / weight parsing
# ---------------------------------------------------------------------------


def test_buckets_parse_height_first():
    assert parse_resolution_buckets(" 512x512, 672x384 ") == [(512, 512), (672, 384)]


@pytest.mark.parametrize(
    "raw, match",
    [
        (None, "is required"),
        ("512", "must be 'HxW'"),
        ("512xfoo", "integers"),
        ("512x500", "multiple of 16"),
        ("512x0", "multiple of 16"),
        ("512x512,512x512", "repeats a bucket"),
    ],
)
def test_malformed_buckets_are_rejected(raw, match):
    with pytest.raises(ValueError, match=match):
        parse_resolution_buckets(raw)


def test_unset_weights_mean_uniform():
    assert parse_resolution_weights(None, 3) is None


def test_weights_are_normalized_to_probabilities():
    assert parse_resolution_weights("2,1,1", 3) == [0.5, 0.25, 0.25]


@pytest.mark.parametrize(
    "raw, match",
    [
        ("1,1", "one weight per bucket"),
        ("1,-1,1", "non-negative"),
        ("0,0,0", "must not sum to 0"),
        ("1,x,1", "comma-separated numbers"),
    ],
)
def test_malformed_weights_are_rejected(raw, match):
    with pytest.raises(ValueError, match=match):
        parse_resolution_weights(raw, 3)


# ---------------------------------------------------------------------------
# weighted_buckets
# ---------------------------------------------------------------------------


def test_draw_is_a_bucket_and_reproducible_from_the_seed():
    args = _args()
    drawn = weighted_buckets(args, Sample(), seed=7)

    assert drawn in parse_resolution_buckets(args.diffusion_resolution_buckets)
    assert weighted_buckets(args, Sample(), seed=7) == drawn


def test_a_zero_weight_bucket_is_never_drawn():
    args = _args(diffusion_resolution_weights="1,0,0")

    assert {weighted_buckets(args, Sample(), seed=seed) for seed in range(32)} == {(512, 512)}


# ---------------------------------------------------------------------------
# One resolution for the whole group
# ---------------------------------------------------------------------------


def test_group_shares_one_resolution_on_request_and_samples():
    group = [Sample(prompt="p") for _ in range(4)]
    sampling_params = {"height": 512, "width": 512}

    height, width = apply_resolution_strategy(
        _args(),
        group,
        sampling_params,
        resolution_strategy_fn=lambda args, sample, seed: (384, 672),
        seed=11,
    )

    assert (height, width) == (384, 672)
    # every microgroup copies this body, so one draw covers the group
    assert (sampling_params["height"], sampling_params["width"]) == (384, 672)
    assert [(s.height, s.width) for s in group] == [(384, 672)] * 4


def test_an_off_grid_custom_strategy_is_rejected_before_the_request():
    # sgl-d would floor 1000 to 992 silently; catch it before a full generation runs
    group = [Sample(prompt="p") for _ in range(2)]
    sampling_params = {"height": 512, "width": 512}

    with pytest.raises(ValueError, match="multiple of 16"):
        apply_resolution_strategy(
            _args(),
            group,
            sampling_params,
            resolution_strategy_fn=lambda args, sample, seed: (1000, 512),
            seed=0,
        )
    assert (sampling_params["height"], sampling_params["width"]) == (512, 512)
    assert all(s.height is None for s in group)


# ---------------------------------------------------------------------------
# The resolution draw gets its own stream
# ---------------------------------------------------------------------------


def test_the_draw_does_not_track_the_sde_window():
    """seed_base is also microgroup 0's request seed, which the step strategy draws its SDE
    window from. Seeding both with it pairs each bucket with one window start forever, so a
    group of one microgroup would only ever train that bucket at that step."""
    step_args = Namespace(diffusion_num_sde_steps=1, diffusion_sde_window_range="3,5")
    res_args = _args(diffusion_resolution_buckets="512x512,672x384")

    pairs = set()
    for group_index in range(200):
        seed_base = (7 + group_index * 16) % (2**31)
        bucket = weighted_buckets(res_args, Sample(), stable_hash("resolution", seed_base))
        window, _, _ = sde_window(step_args, Sample(), 10, seed_base)
        pairs.add((bucket, window[0]))

    # 2 buckets x 2 window starts, and the draw must reach all four
    assert len(pairs) == 4
