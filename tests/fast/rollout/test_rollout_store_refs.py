"""Every group a rollout generates but does not train on is recorded for release: filtered, surplus, or in flight."""

import asyncio
from argparse import Namespace

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from radix_transfer_plugins import StoreRef

from miles.rollout import sglang_diffusion_rollout as rollout
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.output_store import take_dropped_refs
from miles.utils.types import Sample


def _group(key: str, reward: float) -> list[Sample]:
    ref = StoreRef(backend="mooncake", key=key, fields=())
    return [Sample(index=int(key), prompt="p", reward=reward, rollout_store_ref=ref)]


class FakeState:
    """The loop's view of GenerateState: each submitted prompt resolves to the next scripted group."""

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.remaining_batch_size = 0
        self.pendings = set()

    def submit_generate_tasks(self, samples):
        for _ in samples:
            delay, group = next(self.outcomes)
            self.pendings.add(asyncio.create_task(self._finish(delay, group)))
        self.remaining_batch_size += len(samples)

    async def _finish(self, delay, group):
        await asyncio.sleep(delay)
        return group

    def reset(self):
        self.remaining_batch_size = 0
        self.pendings = set()


def drop_zero_reward(args, samples, **kwargs):
    return DynamicFilterOutput(keep=samples[0].reward != 0.0)


def _args() -> Namespace:
    return Namespace(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        over_sampling_batch_size=2,
        n_samples_per_prompt=1,
        dynamic_sampling_filter_path=f"{__name__}.drop_zero_reward",
        rollout_sample_filter_path=None,
    )


@pytest.mark.parametrize("fourth_delay", [0.02, 0.3], ids=["surplus_in_the_same_batch", "still_in_flight"])
async def test_filtered_surplus_and_in_flight_groups_are_recorded(monkeypatch, fourth_delay):
    outcomes = [
        (0.0, _group("1", 0.0)),
        (0.01, _group("2", 1.0)),
        (0.02, _group("3", 1.0)),
        (fourth_delay, _group("4", 1.0)),
    ]
    state = FakeState(outcomes)
    monkeypatch.setattr(rollout, "GenerateState", lambda args: state)
    take_dropped_refs()

    output = await rollout.generate_rollout_async(_args(), 0, lambda n: [None] * n)

    kept = {group[0].index for group in output.samples}
    dropped = set(take_dropped_refs())
    assert len(kept) == 2 and 1 not in kept
    assert "1" in dropped and len(dropped) == 2
    assert dropped | {str(index) for index in kept} == {"1", "2", "3", "4"}
    assert state.pendings == set()
