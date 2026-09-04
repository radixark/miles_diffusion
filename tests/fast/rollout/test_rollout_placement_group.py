"""The rollout placement group reaches actor pools as an argument, never as module state."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import pytest

from miles.rollout.base_types import call_rollout_fn
from miles.rollout.rm_hub.core import AsyncRewardActorPool


def test_call_rollout_fn_passes_the_placement_group_only_to_signatures_that_take_it():
    """A custom rollout function written before the kwarg existed must keep working."""
    seen = {}

    def legacy(args, rollout_id, data_source, evaluation=False):
        seen["legacy"] = "called"
        return []

    def seated(args, rollout_id, data_source, evaluation=False, placement_group=None):
        seen["seated"] = placement_group
        return []

    pg = object()
    call_rollout_fn(legacy, None, 0, None, evaluation=False, placement_group=pg)
    call_rollout_fn(seated, None, 0, None, evaluation=False, placement_group=pg)

    assert seen == {"legacy": "called", "seated": pg}


def test_colocated_pool_without_seats_fails_before_creating_actors():
    """An unplaced 0.05-GPU actor would pend in Ray forever instead of erroring."""
    with pytest.raises(RuntimeError, match="rollout placement group"):
        AsyncRewardActorPool(
            actor_cls=None,
            actor_kwargs={},
            num_workers=1,
            batch_size=1,
            num_gpus_per_worker=1.0,
            colocate=True,
            name="PickScore",
        )
