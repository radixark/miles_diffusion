from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from types import SimpleNamespace

import pytest

import miles.ray.placement_group as placement_module
from miles.ray.actor_group import RayTrainGroup
from miles.ray.utils import TRAIN_ACTOR_GPU
from miles.utils.trainer_config import TrainerConfig


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


class _FakeTrainGroup:
    def __init__(self, role, start_rollout_id, events):
        self._role = role
        self._start_rollout_id = start_rollout_id
        self._events = events

    def async_init(self, args, role, with_ref):
        self._events.append(("init", role, args.name))
        return [self._start_rollout_id] * 2

    def set_rollout_manager(self, rollout_manager):
        self._events.append(("set_rollout_manager", self._role))


def _create_training_models(monkeypatch, start_rollout_ids):
    events, allocations = [], []

    def fake_allocate(**kwargs):
        allocations.append(kwargs)
        return _FakeTrainGroup(kwargs["role"], start_rollout_ids[kwargs["role"]], events)

    trainer_configs = [TrainerConfig(role=role) for role in start_rollout_ids]
    monkeypatch.setattr(placement_module, "compute_trainer_configs", lambda args: trainer_configs)
    monkeypatch.setattr(
        placement_module, "compute_trainer_args", lambda args, config: Namespace(name=f"{config.role}_args")
    )
    monkeypatch.setattr(placement_module, "allocate_train_group", fake_allocate)
    monkeypatch.setattr(placement_module.ray, "get", lambda refs: refs)
    args = _args(start_rollout_id=None, rollout_global_dataset=False)
    groups = placement_module.create_training_models(args, {"actor": "actor_pg"}, rollout_manager=None)
    return args, groups, events, allocations


@pytest.mark.parametrize("roles", [["actor"], ["critic", "actor"], ["critic", "reward", "actor"]])
def test_trainer_groups_split_the_train_gpu_share(monkeypatch, roles):
    args, groups, events, allocations = _create_training_models(monkeypatch, dict.fromkeys(roles, 3))

    assert list(groups) == roles
    assert args.start_rollout_id == 3
    assert [a["num_gpus_per_actor"] for a in allocations] == [TRAIN_ACTOR_GPU / len(roles)] * len(roles)
    assert sum(int(a["num_gpus_per_actor"] * 10_000) for a in allocations) <= int(TRAIN_ACTOR_GPU * 10_000)
    assert all(a["pg"] == "actor_pg" and a["args"].name == f"{a['role']}_args" for a in allocations)
    assert events == [("init", role, f"{role}_args") for role in roles] + [("set_rollout_manager", "actor")]


def test_trainers_restored_to_different_rollouts_are_rejected(monkeypatch):
    with pytest.raises(AssertionError, match="trainers restored different rollouts"):
        _create_training_models(monkeypatch, {"critic": 2, "actor": 3})


def test_each_rank_receives_the_same_rank_output_of_earlier_groups():
    class _Handle:
        def __init__(self, rank):
            self.train = SimpleNamespace(remote=lambda *args: (rank, *args))

    group = object.__new__(RayTrainGroup)
    group._actor_handlers = [_Handle(rank) for rank in range(2)]
    external_data = {"critic": ["critic_0", "critic_1"], "reward": ["reward_0", "reward_1"]}

    assert group.async_train(7, "data", external_data) == [
        (0, 7, "data", {"critic": "critic_0", "reward": "reward_0"}),
        (1, 7, "data", {"critic": "critic_1", "reward": "reward_1"}),
    ]
    assert group.async_train(7, "data") == [(0, 7, "data", {}), (1, 7, "data", {})]
