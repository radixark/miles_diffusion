from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import miles.ray.placement_group as placement_module


def _args(**overrides):
    values = dict(
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        rollout_num_gpus=None,
        colocate=False,
        train_only=True,
        debug_rollout_only=False,
    )
    values.update(overrides)
    return Namespace(**values)


def _record_created_groups(monkeypatch):
    calls = []

    def fake_create(num_gpus):
        calls.append(num_gpus)
        return object(), list(range(num_gpus)), list(range(num_gpus))

    monkeypatch.setattr(placement_module, "_create_placement_group", fake_create)
    return calls


def test_train_only_reuses_actor_group_by_default(monkeypatch):
    calls = _record_created_groups(monkeypatch)
    groups = placement_module.create_placement_groups(_args())

    assert calls == [4]
    assert groups["actor"][0] is groups["rollout"][0]


def test_train_only_reserves_explicit_rollout_group(monkeypatch):
    calls = _record_created_groups(monkeypatch)
    groups = placement_module.create_placement_groups(_args(rollout_num_gpus=1))

    assert calls == [4, 1]
    assert groups["actor"][0] is not groups["rollout"][0]
    assert len(groups["rollout"][1]) == 1


def test_regular_disaggregated_training_is_unchanged(monkeypatch):
    calls = _record_created_groups(monkeypatch)
    placement_module.create_placement_groups(_args(train_only=False, rollout_num_gpus=2))

    assert calls == [4, 2]
