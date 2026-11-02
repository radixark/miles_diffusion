"""A train call may cover a range of the rollout's optimizer steps, keeping the micro-batch ids of the full run."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=["fsdp"])

import pytest
import torch

from tests.fast.backends.fsdp_utils.test_train_input_hook import _actor, _load_train_core


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


def test_a_step_range_trains_those_steps_only():
    pairs = [{"sample_index": index} for index in range(6)]
    forwarded = []
    actor = _actor(None, None, forwarded)
    actor.args.num_steps_per_rollout = 3

    _load_train_core()(actor, rollout_id=0, rollout_data={"train_data": pairs}, external_data={}, optim_steps=(1, 3))

    assert forwarded == [(index, [pair]) for index, pair in enumerate(pairs)][2:]
    assert actor.global_step == 2


@pytest.mark.parametrize("optim_steps", [(1, 3), (1, 1), (-1, 1)])
def test_a_step_range_outside_the_schedule_is_rejected(optim_steps):
    pairs = [{"sample_index": index} for index in range(4)]
    with pytest.raises(ValueError, match="not a range within the 2 steps"):
        _load_train_core()(
            _actor(None, None, []),
            rollout_id=0,
            rollout_data={"train_data": pairs},
            external_data={},
            optim_steps=optim_steps,
        )
