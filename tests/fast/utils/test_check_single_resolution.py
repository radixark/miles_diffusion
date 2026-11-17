"""A train micro-batch holds pairs of one rollout resolution.

    pairs of one resolution / fixed-resolution pairs  ─► accepted
    672x384 next to 384x672 (same token count)        ─► rejected: qwen_image's RoPE would use
                                                         the first pair's grid for both
    unlabeled pairs while a strategy draws            ─► rejected: the check would be vacuous
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


def test_unlabeled_pairs_are_rejected_while_a_strategy_draws():
    # a conversion that does not copy the sample's size onto its pairs (nft / dmd2 today) would
    # make every micro-batch look single-resolution
    with pytest.raises(ValueError, match="without one"):
        check_single_resolution([{"latent": None}, {"latent": None}], labels_required=True)


def test_unlabeled_pairs_are_fine_at_a_fixed_resolution():
    check_single_resolution([{"latent": None}, {"latent": None}], labels_required=False)
