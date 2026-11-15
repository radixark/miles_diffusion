"""A train micro-batch holds pairs of one rollout resolution.

    pairs of one resolution / fixed-resolution pairs  ─► accepted
    672x384 next to 384x672 (same token count)        ─► rejected: qwen_image's RoPE would use
                                                         the first pair's grid for both
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import pytest

from miles.utils.train_data_utils import check_single_resolution


def test_one_resolution_is_accepted():
    check_single_resolution([{"resolution": (672, 384)}] * 4)


def test_fixed_resolution_pairs_are_accepted():
    # without a resolution strategy pairs carry None, or no key at all (custom conversions)
    check_single_resolution([{"resolution": None}, {}])


def test_a_transposed_resolution_is_rejected():
    with pytest.raises(ValueError, match="mixes rollout resolutions"):
        check_single_resolution([{"resolution": (672, 384)}, {"resolution": (384, 672)}])
