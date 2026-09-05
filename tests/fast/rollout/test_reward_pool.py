"""The colocated reward slot ledger: which bundle each reward actor gets, and that pools never overlap.

Mental model (8 bundles = 2 nodes x 4 GPUs, 4-GPU engines; `x` = an engine's own claim):

    node 0   bundle 0[x] 1 2 3        bundle_deal_order deals non-`x` bundles first, alternating
    node 1   bundle 4[x] 5 6 7        nodes, and the `x` bundles last -> [1, 5, 2, 6, 3, 7, 0, 4]
    ColocatedRewardSlots(order).allocate("hps", 3) -> [1, 5, 2]; allocate("pickscore", 3) -> [6, 3, 7]

Covered: the deal order for single-GPU (SD3) and 4-GPU (Wan) engines (1-2); pools get disjoint
contiguous slices (3); every bundle survives partial or information-poor placements (4-6); an
over-subscribing pool is rejected with the per-bundle owner map and nothing is consumed (7); a
pool cannot allocate twice (8).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import pytest

from miles.rollout.rm_hub.core import ColocatedRewardSlots, bundle_deal_order


def test_sd3_bundle_deal_order():
    assert bundle_deal_order(
        list(range(8)),
        [0, 1, 2, 3, 0, 1, 2, 3],
        num_gpus_per_node=4,
        num_gpus_per_engine=1,
    ) == [0, 4, 1, 5, 2, 6, 3, 7]


def test_wan_bundle_deal_order():
    assert bundle_deal_order(
        list(range(16)),
        list(range(8)) * 2,
        num_gpus_per_node=8,
        num_gpus_per_engine=4,
    ) == [1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15, 0, 8, 4, 12]


def test_pools_receive_disjoint_contiguous_slices():
    slots = ColocatedRewardSlots([0, 4, 1, 5, 2, 6, 3, 7])

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
    order = bundle_deal_order(
        bundle_indices,
        gpu_ids,
        num_gpus_per_node=num_gpus_per_node,
        num_gpus_per_engine=2,
    )

    assert len(order) == len(bundle_indices)
    assert set(order) == set(bundle_indices)


def test_incomplete_engine_span_is_preferred_to_base_bundles():
    order = bundle_deal_order(
        list(range(5)),
        list(range(5)),
        num_gpus_per_node=5,
        num_gpus_per_engine=2,
    )

    assert order == [4, 1, 3, 0, 2]


@pytest.mark.parametrize(
    "gpu_ids",
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ],
)
def test_information_poor_placements_retain_every_bundle(gpu_ids):
    bundle_indices = list(range(len(gpu_ids)))

    order = bundle_deal_order(
        bundle_indices,
        gpu_ids,
        num_gpus_per_node=8 if len(gpu_ids) > 8 else 4,
        num_gpus_per_engine=4,
    )

    assert len(order) == len(bundle_indices)
    assert set(order) == set(bundle_indices)


def test_exhaustion_is_atomic_and_reports_ownership():
    slots = ColocatedRewardSlots([0, 1, 2, 3])
    slots.allocate("HPS", 2)
    slots.allocate("PickScore", 1)

    with pytest.raises(
        RuntimeError, match=r"only 1/4 remain \(bundle 0: HPS, bundle 1: HPS, bundle 2: PickScore, bundle 3: None\)"
    ):
        slots.allocate("Other", 2)

    assert slots.allocate("Last", 1) == [3]


def test_duplicate_pool_name_does_not_consume_a_slice():
    slots = ColocatedRewardSlots([0, 1, 2])
    slots.allocate("HPS", 1)

    with pytest.raises(RuntimeError, match="HPS already owns reward slots"):
        slots.allocate("HPS", 1)

    assert slots.allocate("PickScore", 1) == [1]
