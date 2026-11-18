"""A train call may cover a range of the rollout's optimizer steps; the rollout's EMA step follows its last step.

    train(optim_steps=(lo, hi)) ──► _train_core trains windows lo..hi-1 with the micro-batch ids of the full run
                                     (the earlier windows' micro-batches are counted, not run)
    a range outside the schedule, or one under --diffusion-recompute-old-log-prob ──► ValueError
    ema_optimizer.step ──► once per rollout, by the call whose range ends at the last step (or trains them all)
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=["fsdp"])

from types import SimpleNamespace

import pytest
from tests.fast.backends.fsdp_utils.test_train_input_hook import _actor, _load_train_core
from tests.fast.backends.fsdp_utils.test_train_output_handoff import _load_methods

from miles.utils.ray_utils import Box


def test_a_step_range_trains_those_steps_only():
    pairs = [{"sample_index": index} for index in range(6)]
    forwarded = []
    actor = _actor(None, None, forwarded)
    actor.args.num_steps_per_rollout = 3

    _load_train_core()(actor, rollout_id=0, rollout_data={"train_data": pairs}, external_data={}, optim_steps=(1, 3))

    # Windows 1 and 2: micro-batches 2..5, numbered as in the full run.
    assert forwarded == [(index, index // 2, False, [pair]) for index, pair in enumerate(pairs)][2:]
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


def test_a_step_range_cannot_recompute_old_log_probs():
    pairs = [{"sample_index": index} for index in range(4)]
    actor = _actor(None, None, [], diffusion_recompute_old_log_prob=True)
    with pytest.raises(ValueError, match="no step range"):
        _load_train_core()(
            actor, rollout_id=0, rollout_data={"train_data": pairs}, external_data={}, optim_steps=(0, 1)
        )


@pytest.mark.parametrize(
    ("optim_steps", "ema_steps"),
    [(None, 1), ((0, 2), 0), ((2, 3), 1), ((0, 3), 1)],
)
def test_the_ema_steps_once_per_rollout_after_its_last_step(optim_steps, ema_steps):
    namespace, store, put = _load_methods("train")
    ema_calls = []
    actor = SimpleNamespace(
        args=SimpleNamespace(
            offload_train=False, debug_rollout_only=False, use_output_store=False, num_steps_per_rollout=3
        ),
        ema_optimizer=SimpleNamespace(step=ema_calls.append),
        global_step=7,
        _train_core=lambda rollout_id, rollout_data, external_data, optim_steps: None,
        parallel_state=SimpleNamespace(get_mesh=lambda name: SimpleNamespace(get_local_rank=lambda: 0)),
    )

    namespace["train"](actor, 0, [Box(put({"train_data": []}))], {}, optim_steps)

    assert ema_calls == [7] * ema_steps
