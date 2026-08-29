from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

import miles.backends.fsdp_utils.actor as actor_module
import miles.ray.rollout as rollout_module


def _manager(trace):
    """A RolloutManager with only the boot-offload state, wired to a trace list.

    RolloutManager is a Ray actor class; exercise the underlying class so the
    ordering invariant can be tested without a Ray runtime.
    """
    cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(cls)
    manager._boot_offload_done = False
    manager._ensure_engines_ready = lambda: trace.append("engines_ready")
    manager.offload = lambda: trace.append("offload")
    return manager


def test_boot_offload_releases_weights_when_the_train_gate_asks_first():
    # Ray orders actor tasks per caller, so the train actor's gate can reach the
    # manager before the driver's own boot_offload submission. The gate must
    # still return with the engine weights released, or a colocated train actor
    # would materialize its model while they are resident.
    trace = []
    manager = _manager(trace)

    assert manager.boot_offload() is True
    assert trace == ["engines_ready", "offload"]


def test_boot_offload_runs_once_across_callers():
    trace = []
    manager = _manager(trace)

    manager.boot_offload()
    manager.boot_offload()

    assert trace.count("offload") == 1


def _train_actor(monkeypatch, calls, offload_rollout):
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor._engines_evicted = False
    actor.args = Namespace(offload_rollout=offload_rollout)

    class _FakeManager:
        class boot_offload:  # noqa: N801 — mimics a Ray method handle
            @staticmethod
            def remote():
                calls.append("submitted")
                return "ref"

    actor.rollout_manager = _FakeManager()
    monkeypatch.setattr(actor_module.ray, "get", lambda ref: calls.append(f"awaited:{ref}"))
    return actor


def test_train_gate_awaits_the_boot_offload(monkeypatch):
    calls = []
    actor = _train_actor(monkeypatch, calls, offload_rollout=True)

    actor._wait_rollout_engines_evicted()

    assert calls == ["submitted", "awaited:ref"]


def test_train_gate_runs_only_before_the_first_gpu_allocation(monkeypatch):
    # wan2.2-style recipes wrap two components; only the first one pays the gate.
    calls = []
    actor = _train_actor(monkeypatch, calls, offload_rollout=True)

    actor._wait_rollout_engines_evicted()
    actor._wait_rollout_engines_evicted()

    assert calls.count("submitted") == 1


@pytest.mark.parametrize("rollout_manager_missing", [True, False])
def test_train_gate_is_skipped_when_engines_keep_their_weights(monkeypatch, rollout_manager_missing):
    # Disaggregated placement gives rollout its own GPUs: nothing to wait for.
    calls = []
    actor = _train_actor(monkeypatch, calls, offload_rollout=False)
    if rollout_manager_missing:
        actor.rollout_manager = None

    actor._wait_rollout_engines_evicted()

    assert calls == []
