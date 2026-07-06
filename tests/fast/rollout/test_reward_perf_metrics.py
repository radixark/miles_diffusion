"""CPU unit tests for _compute_reward_perf_metrics (reward-bottleneck perf).

Pure timestamp arithmetic over the generate/reward completion stamps collected
during streaming rollout. No ray / engine / model needed — the state is faked
with a SimpleNamespace exposing only the three fields the function reads.
"""

from types import SimpleNamespace

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from miles.rollout.sglang_diffusion_rollout import _compute_reward_perf_metrics


def _state(generate_done_ts, reward_done_ts, reward_max_inflight):
    return SimpleNamespace(
        generate_done_ts=generate_done_ts,
        reward_done_ts=reward_done_ts,
        reward_max_inflight=reward_max_inflight,
    )


def test_no_stamps_returns_empty():
    # e.g. group_rm with no reward recorded, or an aborted rollout.
    assert _compute_reward_perf_metrics(_state([], [], 0), gen_start=0.0) == {}
    assert _compute_reward_perf_metrics(_state([1.0], [], 0), gen_start=0.0) == {}


def test_reward_hidden_under_generation_is_not_bottleneck():
    # last generate at t=10, last reward at t=10.1 -> tiny tail, ratio ~0.
    m = _compute_reward_perf_metrics(_state([2.0, 5.0, 10.0], [3.0, 6.0, 10.1], 3), gen_start=0.0)
    assert m["perf/reward/tail_time"] == pytest.approx(0.1)
    assert m["perf/reward/tail_ratio"] == pytest.approx(0.1 / 10.1)
    assert m["perf/reward/max_inflight"] == 3.0


def test_reward_tail_is_the_bottleneck():
    # generation drained at t=10, reward keeps going to t=18 -> tail=8, ratio=8/18.
    m = _compute_reward_perf_metrics(_state([4.0, 8.0, 10.0], [9.0, 14.0, 18.0], 12), gen_start=0.0)
    assert m["perf/reward/tail_time"] == pytest.approx(8.0)
    assert m["perf/reward/tail_ratio"] == pytest.approx(8.0 / 18.0)
    assert m["perf/reward/max_inflight"] == 12.0


def test_tail_clamped_non_negative():
    # A late generation stamp (e.g. an oversampled task) must not yield a
    # negative tail — reward finished before the last generate.
    m = _compute_reward_perf_metrics(_state([5.0, 20.0], [6.0, 10.0], 2), gen_start=0.0)
    assert m["perf/reward/tail_time"] == 0.0
    assert m["perf/reward/tail_ratio"] == 0.0


def test_ratio_uses_gen_start_as_wall_clock_origin():
    # wall = last_reward - gen_start = 18 - 2 = 16; tail = 18 - 12 = 6.
    m = _compute_reward_perf_metrics(_state([8.0, 12.0], [15.0, 18.0], 4), gen_start=2.0)
    assert m["perf/reward/tail_time"] == pytest.approx(6.0)
    assert m["perf/reward/tail_ratio"] == pytest.approx(6.0 / 16.0)
