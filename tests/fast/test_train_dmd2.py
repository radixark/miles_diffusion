"""train_dmd2.py orders the critic and the actor within a rollout as --dmd2-critic-scoring asks.

    reuse-critic-step   critic trains (all K steps) ──► actor trains on its output
    original            critic trains steps 0..K-2 (none when K = 1) ──► critic scores, forward only
                        ──► actor trains on the scores ──► critic trains step K-1

The critic trains exactly K steps per rollout and the actor once; the scoring comes before the critic's last update.
A group saves after its last train of the rollout and, under --offload-train, sleeps before another group wakes;
the rollout is released after the last group's train.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

import train_dmd2


class _Ref:
    def __init__(self, kind, role, rollout_id):
        self.kind = kind
        self.role = role
        self.rollout_id = rollout_id


class _Remote:
    def __init__(self, fn):
        self.remote = fn


class _FakeRolloutManager:
    def __init__(self, events):
        self.generate = _Remote(lambda rollout_id: ("rollout_data", rollout_id))
        self.dispose = _Remote(lambda: None)
        self.release = _Remote(lambda rollout_id: events.append(("release", rollout_id)))


class _FakeTrainGroup:
    def __init__(self, role, events):
        self._role = role
        self._events = events

    def async_train(self, rollout_id, rollout_data_ref, external_data=None, optim_steps=None):
        received = {
            role: [(ref.kind, ref.role, ref.rollout_id) for ref in refs]
            for role, refs in (external_data or {}).items()
        }
        self._events.append(("train_start", self._role, rollout_id, rollout_data_ref, received, optim_steps))
        return [_Ref("train", self._role, rollout_id)]

    def async_compute(self, rollout_id, rollout_data_ref, external_data, hooks_key):
        self._events.append(("compute_start", self._role, rollout_id, rollout_data_ref, external_data, hooks_key))
        return [_Ref("compute", self._role, rollout_id)]

    def save_model(self, rollout_id, force_sync=False):
        self._events.append(("save_model", self._role, rollout_id, force_sync))

    def offload(self):
        self._events.append(("offload", self._role))

    def clear_memory(self):
        self._events.append(("clear_memory", self._role))

    def update_weights(self):
        self._events.append(("update_weights", self._role))


def _fake_ray_get(events):
    def get(ref):
        if isinstance(ref, list):
            return [get(r) for r in ref]
        if isinstance(ref, _Ref):
            events.append((f"{ref.kind}_done", ref.role, ref.rollout_id))
        return ref

    return get


def _run(monkeypatch, scoring, critic_steps, offload_train, num_rollout=2):
    events = []
    monkeypatch.setattr(train_dmd2, "configure_logger", lambda: None)
    monkeypatch.setattr(train_dmd2, "init_tracking", lambda args: None)
    monkeypatch.setattr(train_dmd2, "create_placement_groups", lambda args: {"rollout": None})
    monkeypatch.setattr(train_dmd2, "create_rollout_manager", lambda args, pg: (_FakeRolloutManager(events), None))
    monkeypatch.setattr(
        train_dmd2,
        "create_training_models",
        lambda args, pgs, rollout_manager: {role: _FakeTrainGroup(role, events) for role in ("critic", "actor")},
    )
    monkeypatch.setattr(train_dmd2.ray, "get", _fake_ray_get(events))
    args = Namespace(
        loss_type="dmd2",
        dmd2_critic_scoring=scoring,
        dmd2_critic_steps_per_rollout=critic_steps,
        start_rollout_id=0,
        num_rollout=num_rollout,
        offload_rollout=False,
        offload_train=offload_train,
        eval_interval=None,
        save_interval=1,
        rollout_global_dataset=False,
    )
    train_dmd2.train(args)
    return events


def _rollout_events(scoring, critic_steps, offload_train, rollout_id, last):
    data = ("rollout_data", rollout_id)

    def train(role, received, optim_steps):
        return [("train_start", role, rollout_id, data, received, optim_steps), ("train_done", role, rollout_id)]

    def done(role):
        return [("save_model", role, rollout_id, last)] + ([("offload", role)] if offload_train else [])

    if scoring == "reuse-critic-step":
        expected = train("critic", {}, None) + done("critic")
        expected += train("actor", {"critic": [("train", "critic", rollout_id)]}, None) + done("actor")
    else:
        expected = train("critic", {}, (0, critic_steps - 1)) if critic_steps > 1 else []
        expected += [
            ("compute_start", "critic", rollout_id, data, {}, "critic-scoring"),
            ("compute_done", "critic", rollout_id),
        ] + ([("offload", "critic")] if offload_train else [])
        expected += train("actor", {"critic": [("compute", "critic", rollout_id)]}, None) + done("actor")
        expected += train("critic", {}, (critic_steps - 1, critic_steps)) + done("critic")
    expected += [("release", rollout_id)]
    if not offload_train:
        expected += [("clear_memory", "critic"), ("clear_memory", "actor")]
    return expected + [("update_weights", "actor")]


@pytest.mark.parametrize("offload_train", [False, True])
@pytest.mark.parametrize(("scoring", "critic_steps"), [("reuse-critic-step", 3), ("original", 3), ("original", 1)])
def test_each_rollout_runs_the_order_of_the_scoring_mode(monkeypatch, scoring, critic_steps, offload_train):
    events = _run(monkeypatch, scoring, critic_steps, offload_train)

    expected = [("update_weights", "actor")]
    for rollout_id in range(2):
        expected += _rollout_events(scoring, critic_steps, offload_train, rollout_id, last=rollout_id == 1)
    assert events == expected


def test_the_original_order_is_the_reference_s():
    """The explicit trace of one rollout with K = 3 under --offload-train."""
    expected = [
        ("train_start", "critic", 0, ("rollout_data", 0), {}, (0, 2)),
        ("train_done", "critic", 0),
        ("compute_start", "critic", 0, ("rollout_data", 0), {}, "critic-scoring"),
        ("compute_done", "critic", 0),
        ("offload", "critic"),
        ("train_start", "actor", 0, ("rollout_data", 0), {"critic": [("compute", "critic", 0)]}, None),
        ("train_done", "actor", 0),
        ("save_model", "actor", 0, True),
        ("offload", "actor"),
        ("train_start", "critic", 0, ("rollout_data", 0), {}, (2, 3)),
        ("train_done", "critic", 0),
        ("save_model", "critic", 0, True),
        ("offload", "critic"),
        ("release", 0),
        ("update_weights", "actor"),
    ]
    assert _rollout_events("original", 3, True, 0, last=True) == expected


@pytest.mark.parametrize("scoring", ["reuse-critic-step", "original"])
@pytest.mark.parametrize("critic_steps", [1, 4])
def test_the_critic_trains_k_steps_and_the_actor_once_after_the_scoring(monkeypatch, scoring, critic_steps):
    events = [
        event
        for event in _run(monkeypatch, scoring, critic_steps, False, num_rollout=1)
        if event[0] != "update_weights"
    ]

    critic_trains = [event for event in events if event[0] == "train_start" and event[1] == "critic"]
    actor_trains = [event for event in events if event[0] == "train_start" and event[1] == "actor"]
    steps_trained = sum(critic_steps if event[5] is None else event[5][1] - event[5][0] for event in critic_trains)
    assert steps_trained == critic_steps
    assert len(actor_trains) == 1
    # The actor's scores come from the critic before its last update.
    actor_at = events.index(actor_trains[0])
    if scoring == "original":
        assert events.index(("compute_done", "critic", 0)) < actor_at < events.index(critic_trains[-1])
    else:
        assert events.index(("train_done", "critic", 0)) < actor_at


def test_other_loss_types_are_sent_to_train_diffusion(monkeypatch):
    monkeypatch.setattr(train_dmd2, "configure_logger", lambda: None)
    with pytest.raises(ValueError, match="run through train_diffusion.py"):
        train_dmd2.train(Namespace(loss_type="nft"))
