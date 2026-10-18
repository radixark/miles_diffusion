"""train_sft.py generates rollout i+1 while rollout i trains, without breaking resume.

The fakes record every remote submission and every ray.get as one event; the loop must produce
this stream (3 rollouts, save_interval=1):

    generate 0 | save 0 | train_start 0 | generate 1 | train_done 0 | save_model 0 |
                 ^^^^^^                    ^^^^^^^^^^   ^^^^^^^^^^^^
                 data-source cursor saved   next rollout submitted before train 0 is waited for
                 before generate 1 moves it
    save 1 | train_start 1 | generate 2 | train_done 1 | save_model 1 |
    save 2 | train_start 2 |              train_done 2 | save_model 2 | dispose
                                          no generate 3: num_rollout reached

    generate i+1 carries train_in_flight = train i's refs when the rollout and actor placement
    groups share GPUs (colocated encoders wait for that step), and None when they do not.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

import train_sft


class _Ref:
    def __init__(self, kind, rollout_id):
        self.kind = kind
        self.rollout_id = rollout_id


class _Submit:
    def __init__(self, events, name):
        self._events = events
        self._name = name

    def remote(self, *args, **kwargs):
        self._events.append((self._name, *args, *kwargs.values()))
        return _Ref(self._name, args[0] if args else None)


class _FakeRolloutManager:
    def __init__(self, events):
        for name in ("generate", "save", "dispose"):
            setattr(self, name, _Submit(events, name))


class _FakeActorModel:
    def __init__(self, events):
        self._events = events

    def async_train(self, rollout_id, rollout_data_ref):
        assert rollout_data_ref == ("rollout_data", rollout_id), "trained on another rollout's data"
        self._events.append(("train_start", rollout_id))
        return [_Ref("train", rollout_id)]

    def save_model(self, rollout_id, force_sync=False):
        self._events.append(("save_model", rollout_id))

    def clear_memory(self):
        pass

    def offload(self):
        pass


def _fake_ray_get(events):
    def get(ref):
        if isinstance(ref, list):
            return [get(r) for r in ref]
        if ref.kind == "train":
            events.append(("train_done", ref.rollout_id))
        if ref.kind == "generate":
            return ("rollout_data", ref.rollout_id)
        return None

    return get


def _run(monkeypatch, *, shared_gpus=True, num_rollout=3, save_interval=None):
    events = []
    actor_pg, rollout_pg = object(), object()
    pgs = {
        "actor": (actor_pg, [0], [0]),
        "rollout": (actor_pg if shared_gpus else rollout_pg, [0], [0]),
    }
    monkeypatch.setattr(train_sft, "configure_logger", lambda: None)
    monkeypatch.setattr(train_sft, "init_tracking", lambda args: None)
    monkeypatch.setattr(train_sft, "create_placement_groups", lambda args: pgs)
    monkeypatch.setattr(train_sft, "create_rollout_manager", lambda args, pg: (_FakeRolloutManager(events), None))
    monkeypatch.setattr(
        train_sft, "create_training_models", lambda args, pgs, rollout_manager: _FakeActorModel(events)
    )
    monkeypatch.setattr(train_sft.ray, "get", _fake_ray_get(events))
    args = Namespace(
        train_only=True,
        start_rollout_id=0,
        num_rollout=num_rollout,
        offload_train=False,
        save_interval=save_interval,
        rollout_global_dataset=True,
    )
    train_sft.train(args)
    return events


def _index(events, name, rollout_id):
    return events.index(next(e for e in events if e[0] == name and e[1] == rollout_id))


def test_the_next_rollout_is_generated_before_training_finishes(monkeypatch):
    """Waiting for train i before submitting generate i+1 would serialize rollout and training again."""
    events = _run(monkeypatch)

    assert [e[1] for e in events if e[0] == "generate"] == [0, 1, 2]
    for rollout_id in (0, 1):
        assert _index(events, "generate", rollout_id + 1) < _index(events, "train_done", rollout_id)


def test_data_source_cursor_is_saved_before_the_prefetch_moves_it(monkeypatch):
    """A cursor saved after the prefetch would skip one rollout's samples on resume."""
    events = _run(monkeypatch, save_interval=1)

    for rollout_id in (0, 1):
        assert _index(events, "save", rollout_id) < _index(events, "generate", rollout_id + 1)
        assert _index(events, "save_model", rollout_id) > _index(events, "train_done", rollout_id)


@pytest.mark.parametrize("shared_gpus", [True, False], ids=["colocated", "dedicated"])
def test_only_colocated_encoders_are_handed_the_in_flight_train_step(monkeypatch, shared_gpus):
    """Colocated encoders without the refs would load onto a GPU mid train step; dedicated ones must not wait."""
    events = _run(monkeypatch, shared_gpus=shared_gpus)

    first, *prefetched = [e for e in events if e[0] == "generate"]
    assert first == ("generate", 0)
    for _, rollout_id, train_in_flight in prefetched:
        if shared_gpus:
            assert [ref.rollout_id for ref in train_in_flight] == [rollout_id - 1]
        else:
            assert train_in_flight is None
