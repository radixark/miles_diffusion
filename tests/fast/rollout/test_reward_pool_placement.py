"""A colocated reward pool takes one slot per rollout bundle and only RolloutManager may seat it.

Mental model (2 rollout bundles, --hps-reward-colocate --hps-num-workers 2):

    bundle 0  [ train | engine | reward slot ]   <- worker 0, PlacementGroupSchedulingStrategy
    bundle 1  [ train | engine | reward slot ]   <- worker 1

Covered: each worker lands on its own bundle at the colocated GPU share (1); a colocated
pool built outside RolloutManager, i.e. without seats, is rejected before any actor is
created (2). Over-subscription is covered by test_reward_pool.py.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import pytest

from miles.ray.utils import COLOCATED_REWARD_GPU
from miles.rollout.rm_hub.core import AsyncRewardActorPool, ColocatedRewardSlots


class _FakeActorCls:
    """Records the options each worker was created with instead of starting Ray actors."""

    def __init__(self):
        self.created = []

    def options(self, **options):
        self.created.append(options)
        return self

    def remote(self, **kwargs):
        return object()


def _pool(actor_cls, *, num_workers, colocate, placement_group=None, slots=None):
    return AsyncRewardActorPool(
        actor_cls=actor_cls,
        actor_kwargs={},
        num_workers=num_workers,
        batch_size=8,
        num_gpus_per_worker=1.0,
        colocate=colocate,
        name="hps",
        placement_group=placement_group,
        slots=slots,
    )


def test_colocated_workers_take_one_slot_each_at_the_colocated_share():
    """Two workers on one bundle, or a full-GPU claim on a bundle, would pend in Ray forever."""
    actor_cls = _FakeActorCls()
    slots = ColocatedRewardSlots([0, 1])

    _pool(actor_cls, num_workers=2, colocate=True, placement_group=("pg", [0, 1], [0, 1]), slots=slots)

    assert [o["scheduling_strategy"].placement_group_bundle_index for o in actor_cls.created] == [0, 1]
    assert [o["num_gpus"] for o in actor_cls.created] == [COLOCATED_REWARD_GPU] * 2
    assert slots.remaining == 0


def test_colocated_pool_built_without_seats_is_rejected():
    """Only RolloutManager seats colocated pools; a lazy build means the flag was set for an unseated type."""
    actor_cls = _FakeActorCls()
    with pytest.raises(RuntimeError, match="not seated by RolloutManager"):
        _pool(actor_cls, num_workers=1, colocate=True)
    assert actor_cls.created == []
