from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import pytest

from miles.rollout.rm_hub.core import ColocatedRewardSlots, _dispersal_order


def test_sd3_dispersal_order():
    assert _dispersal_order(
        list(range(8)),
        [0, 1, 2, 3, 0, 1, 2, 3],
        num_gpus_per_node=4,
        num_gpus_per_engine=1,
    ) == (0, 4, 1, 5, 2, 6, 3, 7)


def test_wan_dispersal_order():
    assert _dispersal_order(
        list(range(16)),
        list(range(8)) * 2,
        num_gpus_per_node=8,
        num_gpus_per_engine=4,
    ) == (1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15, 0, 8, 4, 12)


def test_pools_receive_disjoint_contiguous_slices():
    slots = ColocatedRewardSlots((0, 4, 1, 5, 2, 6, 3, 7))

    hps = slots.allocate("HPS", 3)
    pickscore = slots.allocate("PickScore", 3)

    assert hps == [0, 4, 1]
    assert pickscore == [5, 2, 6]
    assert set(hps).isdisjoint(pickscore)


@pytest.mark.parametrize(
    ("bundle_indices", "gpu_ids", "num_gpus_per_node"),
    [
        ([0, 1, 2, 3], [0, 1, 2, 3], 5),
        ([7, 9], [0, 2], 3),
    ],
)
def test_partial_placement_group_retains_every_bundle(bundle_indices, gpu_ids, num_gpus_per_node):
    order = _dispersal_order(
        bundle_indices,
        gpu_ids,
        num_gpus_per_node=num_gpus_per_node,
        num_gpus_per_engine=2,
    )

    assert len(order) == len(bundle_indices)
    assert set(order) == set(bundle_indices)


def test_incomplete_engine_span_is_preferred_to_base_bundles():
    order = _dispersal_order(
        list(range(5)),
        list(range(5)),
        num_gpus_per_node=5,
        num_gpus_per_engine=2,
    )

    assert order == (4, 1, 3, 0, 2)


@pytest.mark.parametrize(
    "gpu_ids",
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ],
)
def test_information_poor_placements_retain_every_bundle(gpu_ids):
    bundle_indices = list(range(len(gpu_ids)))

    order = _dispersal_order(
        bundle_indices,
        gpu_ids,
        num_gpus_per_node=8 if len(gpu_ids) > 8 else 4,
        num_gpus_per_engine=4,
    )

    assert len(order) == len(bundle_indices)
    assert set(order) == set(bundle_indices)


def test_exhaustion_is_atomic_and_reports_ownership():
    slots = ColocatedRewardSlots((0, 1, 2, 3))
    slots.allocate("HPS", 2)
    slots.allocate("PickScore", 1)

    with pytest.raises(RuntimeError, match=r"only 1/4 remain .*HPS×2, PickScore×1"):
        slots.allocate("Other", 2)

    assert slots.allocate("Last", 1) == [3]


def test_duplicate_pool_name_does_not_consume_a_slice():
    slots = ColocatedRewardSlots((0, 1, 2))
    slots.allocate("HPS", 1)

    with pytest.raises(RuntimeError, match="HPS already owns reward slots"):
        slots.allocate("HPS", 1)

    assert slots.allocate("PickScore", 1) == [1]
