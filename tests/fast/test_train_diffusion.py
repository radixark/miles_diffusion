"""train_diffusion.py trains every trainer group on each rollout in declared order, handing each the earlier outputs."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import train_diffusion


class _Ref:
    def __init__(self, role, rollout_id):
        self.role = role
        self.rollout_id = rollout_id


class _Remote:
    def __init__(self, fn):
        self.remote = fn


class _FakeRolloutManager:
    def __init__(self):
        self.generate = _Remote(lambda rollout_id: ("rollout_data", rollout_id))
        self.dispose = _Remote(lambda: None)


class _FakeTrainGroup:
    def __init__(self, role, events):
        self._role = role
        self._events = events

    def async_train(self, rollout_id, rollout_data_ref, external_data):
        received = {role: [(ref.role, ref.rollout_id) for ref in refs] for role, refs in external_data.items()}
        self._events.append(("train_start", self._role, rollout_id, rollout_data_ref, received))
        return [_Ref(self._role, rollout_id)]

    def save_model(self, rollout_id, force_sync=False):
        self._events.append(("save_model", self._role, rollout_id))

    def offload(self):
        self._events.append(("offload", self._role))

    def update_weights(self):
        self._events.append(("update_weights", self._role))


def _fake_ray_get(events):
    def get(ref):
        if isinstance(ref, list):
            return [get(r) for r in ref]
        if isinstance(ref, _Ref):
            events.append(("train_done", ref.role, ref.rollout_id))
        return ref

    return get


def test_every_trainer_group_trains_each_rollout_in_order(monkeypatch):
    roles = ["critic", "actor"]
    events = []
    monkeypatch.setattr(train_diffusion, "configure_logger", lambda: None)
    monkeypatch.setattr(train_diffusion, "init_tracking", lambda args: None)
    monkeypatch.setattr(train_diffusion, "create_placement_groups", lambda args: {"rollout": None})
    monkeypatch.setattr(train_diffusion, "create_rollout_manager", lambda args, pg: (_FakeRolloutManager(), None))
    monkeypatch.setattr(
        train_diffusion,
        "create_training_models",
        lambda args, pgs, rollout_manager: {role: _FakeTrainGroup(role, events) for role in roles},
    )
    monkeypatch.setattr(train_diffusion.ray, "get", _fake_ray_get(events))
    args = Namespace(
        start_rollout_id=0,
        num_rollout=2,
        offload_rollout=False,
        offload_train=True,
        eval_interval=None,
        save_interval=1,
        rollout_global_dataset=False,
    )

    train_diffusion.train(args)

    expected = [("update_weights", "actor")]
    for rollout_id in range(2):
        for i, role in enumerate(roles):
            received = {earlier: [(earlier, rollout_id)] for earlier in roles[:i]}
            expected += [
                ("train_start", role, rollout_id, ("rollout_data", rollout_id), received),
                ("train_done", role, rollout_id),
            ]
        expected += [("save_model", role, rollout_id) for role in roles]
        expected += [("offload", role) for role in roles]
        expected += [("update_weights", "actor")]
    assert events == expected
