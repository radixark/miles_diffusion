"""--rm-type none: a constant 0 reward for losses that read none, with no scorer behind it."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace

from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.types import Sample


def _args(rm_type):
    return Namespace(rm_type=rm_type, custom_rm_path=None)


def test_none_rewards_every_sample_with_zero():
    samples = [Sample(index=index) for index in range(3)]

    assert asyncio.run(batched_async_rm(_args("none"), samples)) == [0.0, 0.0, 0.0]


def test_per_sample_none_overrides_the_global_reward():
    sample = Sample(index=0, metadata={"rm_type": "none"})

    assert asyncio.run(async_rm(_args("pickscore"), sample)) == 0.0
