"""train_diffusion.py runs each rollout's train schedule in order, handing every phase the earlier phases' outputs.

A phase trains a group's optimizer steps or runs a compute hook on it. A group saves after its last phase; with
--offload-train it sleeps whenever the next phase belongs to another group, before that group wakes.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

import train_diffusion
from miles.utils.trainer_config import TrainPhase

SCORE = "loss_hub.dmd2.score_held_out_samples"
ACTOR_ONLY = [TrainPhase(role="actor", key="actor")]
CRITIC_THEN_ACTOR = [TrainPhase(role="critic", key="critic"), TrainPhase(role="actor", key="actor")]
ORIGINAL_DMD2 = [
    TrainPhase(role="critic", key="critic_earlier_steps", optim_steps=(0, 4)),
    TrainPhase(role="critic", key="critic", compute_path=SCORE),
    TrainPhase(role="actor", key="actor"),
    TrainPhase(role="critic", key="critic_last_step", optim_steps=(4, 5)),
]


class _Ref:
    def __init__(self, key, rollout_id):
        self.key = key
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

    def _received(self, external_data):
        return {key: [(ref.key, ref.rollout_id) for ref in refs] for key, refs in external_data.items()}

    def async_train(self, rollout_id, rollout_data_ref, external_data, optim_steps):
        self._events.append(
            ("train", self._role, rollout_id, rollout_data_ref, self._received(external_data), optim_steps)
        )
        return [_Ref(f"{self._role}:{optim_steps}", rollout_id)]

    def async_compute(self, rollout_id, rollout_data_ref, external_data, func_path):
        self._events.append(
            ("compute", self._role, rollout_id, rollout_data_ref, self._received(external_data), func_path)
        )
        return [_Ref(f"{self._role}:{func_path}", rollout_id)]

    def save_model(self, rollout_id, force_sync=False):
        self._events.append(("save_model", self._role, rollout_id))

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
            events.append(("done", ref.key, ref.rollout_id))
        return ref

    return get


def _run(monkeypatch, schedule, offload_train, num_rollout):
    events = []
    roles = list(dict.fromkeys(phase.role for phase in schedule))
    monkeypatch.setattr(train_diffusion, "configure_logger", lambda: None)
    monkeypatch.setattr(train_diffusion, "init_tracking", lambda args: None)
    monkeypatch.setattr(train_diffusion, "create_placement_groups", lambda args: {"rollout": None})
    monkeypatch.setattr(train_diffusion, "create_rollout_manager", lambda args, pg: (_FakeRolloutManager(), None))
    monkeypatch.setattr(
        train_diffusion,
        "create_training_models",
        lambda args, pgs, rollout_manager: {role: _FakeTrainGroup(role, events) for role in roles},
    )
    monkeypatch.setattr(train_diffusion, "compute_train_schedule", lambda args: schedule)
    monkeypatch.setattr(train_diffusion.ray, "get", _fake_ray_get(events))
    args = Namespace(
        start_rollout_id=0,
        num_rollout=num_rollout,
        offload_rollout=False,
        offload_train=offload_train,
        eval_interval=None,
        save_interval=1,
        rollout_global_dataset=False,
    )

    train_diffusion.train(args)
    return events


def _phase_ref(phase):
    return f"{phase.role}:{phase.compute_path if phase.compute_path is not None else phase.optim_steps}"


@pytest.mark.parametrize("schedule", [ACTOR_ONLY, CRITIC_THEN_ACTOR, ORIGINAL_DMD2])
@pytest.mark.parametrize("offload_train", [True, False])
def test_every_phase_runs_each_rollout_in_order(monkeypatch, schedule, offload_train):
    events = _run(monkeypatch, schedule, offload_train, num_rollout=2)

    roles = list(dict.fromkeys(phase.role for phase in schedule))
    last_phase_index = {phase.role: index for index, phase in enumerate(schedule)}
    expected = [("update_weights", "actor")]
    for rollout_id in range(2):
        for index, phase in enumerate(schedule):
            received = {earlier.key: [(_phase_ref(earlier), rollout_id)] for earlier in schedule[:index]}
            kind, detail = ("compute", phase.compute_path) if phase.compute_path else ("train", phase.optim_steps)
            expected += [
                (kind, phase.role, rollout_id, ("rollout_data", rollout_id), received, detail),
                ("done", _phase_ref(phase), rollout_id),
            ]
            if index == last_phase_index[phase.role]:
                expected += [("save_model", phase.role, rollout_id)]
            if offload_train and (index + 1 == len(schedule) or schedule[index + 1].role != phase.role):
                expected += [("offload", phase.role)]
        if not offload_train:
            expected += [("clear_memory", role) for role in roles]
        expected += [("update_weights", "actor")]
    assert events == expected


def test_original_dmd2_scores_between_the_critic_steps_and_sleeps_the_critic_around_the_actor(monkeypatch):
    events = _run(monkeypatch, ORIGINAL_DMD2, offload_train=True, num_rollout=1)

    data = ("rollout_data", 0)
    earlier = {"critic_earlier_steps": [("critic:(0, 4)", 0)]}
    scored = {**earlier, "critic": [(f"critic:{SCORE}", 0)]}
    assert events == [
        ("update_weights", "actor"),
        ("train", "critic", 0, data, {}, (0, 4)),
        ("done", "critic:(0, 4)", 0),
        ("compute", "critic", 0, data, earlier, SCORE),
        ("done", f"critic:{SCORE}", 0),
        ("offload", "critic"),
        ("train", "actor", 0, data, scored, None),
        ("done", "actor:None", 0),
        ("save_model", "actor", 0),
        ("offload", "actor"),
        ("train", "critic", 0, data, {**scored, "actor": [("actor:None", 0)]}, (4, 5)),
        ("done", "critic:(4, 5)", 0),
        ("save_model", "critic", 0),
        ("offload", "critic"),
        ("update_weights", "actor"),
    ]
