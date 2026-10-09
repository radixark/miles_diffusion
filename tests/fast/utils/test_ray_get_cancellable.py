"""Cancelling a rollout task must stop the Ray work it is waiting on, not just the asyncio task.

Covered: a plain call returns its result (1); with distributed POST enabled, cancelling post() cancels the
in-flight async actor call (2); a parser call waiting behind a busy parser never runs once cancelled (3),
since Ray itself can't cancel a call already queued on a sync actor (4).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import asyncio
import time

import pytest
import ray

from miles.rollout.sglang_diffusion_rollout import GenerateState
from miles.utils import http_utils
from miles.utils.async_utils import ray_get_cancellable


@pytest.fixture(scope="module", autouse=True)
def local_ray():
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


@ray.remote
class AsyncPoster:
    """Stands in for http_utils' POST actor: an async actor whose call can sit on a slow request."""

    def __init__(self):
        self.started = 0
        self.cancelled = 0
        self.finished = 0

    async def do_post(self, url, payload, max_retries=60, raw=False):
        self.started += 1
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        self.finished += 1
        return "done"

    async def echo(self, x):
        return x

    async def counts(self):
        return self.started, self.cancelled, self.finished


@ray.remote
class SyncParser:
    """Stands in for RolloutImageResponseParserActor: sync, one call at a time, later calls queue."""

    def __init__(self):
        self.ran = []

    def work(self, tag, seconds):
        time.sleep(seconds)
        self.ran.append(tag)
        return tag

    def ran_tags(self):
        return self.ran


async def _until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not await predicate():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.05)


def _parser_state(parser):
    """GenerateState holding just one response parser; the rest needs a router and GPUs."""
    state = object.__new__(GenerateState)
    state.response_parsers = [parser]
    state._parser_rr = 0
    state.parser_inflight = [0]
    state.parser_slots = [asyncio.Semaphore(1)]
    return state


async def _cancel(task):
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_returns_the_result():
    actor = AsyncPoster.remote()
    assert asyncio.run(ray_get_cancellable(lambda: actor.echo.remote(3))) == 3


def test_cancelling_distributed_post_cancels_the_actor_call(monkeypatch):
    actor = AsyncPoster.remote()
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(http_utils, "_post_actors", [actor])

    async def counts():
        return await asyncio.to_thread(ray.get, actor.counts.remote())

    async def scenario():
        task = asyncio.create_task(http_utils.post("http://unused/rollout/generate", {}))
        await _until(lambda: _equals(counts, (1, 0, 0)))
        await _cancel(task)
        await _until(lambda: _equals(counts, (1, 1, 0)))

    asyncio.run(scenario())


async def _equals(get, expected):
    return await get() == expected


def test_cancelled_parser_call_waiting_on_a_busy_parser_never_runs():
    parser = SyncParser.remote()
    state = _parser_state(parser)

    async def scenario():
        busy = asyncio.create_task(state.call_parser("work", "busy", 1.5))
        waiting = asyncio.create_task(state.call_parser("work", "cancelled", 0))
        await asyncio.sleep(0.3)
        await _cancel(waiting)
        assert await busy == ("busy", 0)
        await asyncio.sleep(1.0)  # give a wrongly-surviving call time to run
        return await asyncio.to_thread(ray.get, parser.ran_tags.remote())

    assert asyncio.run(scenario()) == ["busy"]


def test_ray_cannot_cancel_a_call_queued_on_a_sync_actor():
    """Why call_parser holds its own slot: if this starts failing, Ray cancels queued sync-actor calls itself."""
    parser = SyncParser.remote()
    busy = parser.work.remote("busy", 1.5)
    queued = parser.work.remote("queued", 0)
    time.sleep(0.3)
    ray.cancel(queued)
    ray.get(busy)
    time.sleep(1.0)

    assert ray.get(parser.ran_tags.remote()) == ["busy", "queued"]
