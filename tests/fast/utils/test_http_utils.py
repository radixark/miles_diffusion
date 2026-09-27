from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest
import ray

from miles.utils import http_utils


class _StubActorHandle:
    pass


class _StubActorClass:
    """Stands in for the class ``ray.remote`` returns, recording ``.options`` kwargs."""

    def __init__(self, recorded_options: list[dict]):
        self._recorded_options = recorded_options

    def options(self, **kwargs):
        self._recorded_options.append(kwargs)
        return self

    def remote(self, *args, **kwargs):
        return _StubActorHandle()


# NodeAffinitySchedulingStrategy validates this for real: a node id is 28 bytes of hex.
_FAKE_NODE_ID = "ab" * 28


@pytest.fixture
def recorded_options(monkeypatch):
    """Run _init_ray_distributed_post against stubbed Ray, returning the actor options."""
    options: list[dict] = []
    monkeypatch.setattr(ray, "remote", lambda cls: _StubActorClass(options))
    monkeypatch.setattr(ray, "nodes", lambda: [{"NodeID": _FAKE_NODE_ID, "Alive": True}])
    monkeypatch.setattr(http_utils, "_post_actors", [])
    monkeypatch.setattr(http_utils, "_client_concurrency", 8)

    http_utils._init_ray_distributed_post(Namespace(num_gpus_per_node=2))
    return options


class TestDistributedPostActorLifetime:
    # These actors only serve the creating process's rollout POSTs. Creating them
    # detached leaked one per node per GPU on every run: a detached actor outlives
    # the job by design, and an unnamed one leaves no handle to kill it afterwards.
    def test_actors_are_not_detached(self, recorded_options):
        assert recorded_options, "expected the POST actors to be created"
        for options in recorded_options:
            assert options.get("lifetime") is None, f"POST actor must not be detached, got {options!r}"

    def test_one_actor_per_gpu_per_node(self, recorded_options):
        assert len(recorded_options) == 2
