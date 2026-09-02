"""Router pick/ack endpoints: the in-flight ledger behind direct engine fetch.

Mental model (one request, direct mode):

    actor -- GET /pick_worker_for_request --> router   count[worker] += 1
    actor -- POST {worker}/rollout/generate            (bytes skip the router)
    actor -- POST /worker_finish_request?url=worker    count[worker] -= 1

Covered: the pick returns the least-loaded worker and counts it in-flight (1),
the ack returns the count so load balancing converges (2), an unknown ack is a
400 and mutates nothing (3), and an empty pool is a 503, not a crash (4).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from argparse import Namespace

from starlette.testclient import TestClient

from miles.router.router import MilesRouter


def _router():
    args = Namespace(
        miles_router_max_connections=4,
        miles_router_timeout=None,
        miles_router_health_check_failure_threshold=3,
        sglang_server_concurrency=2,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1,
    )
    return MilesRouter(args)


def test_pick_counts_in_flight_and_ack_returns_it():
    router = _router()
    router.worker_request_counts = {"http://a:1": 1, "http://b:2": 0}
    with TestClient(router.app) as client:
        picked = client.get("/pick_worker_for_request").json()["url"]
        assert picked == "http://b:2"  # least loaded
        assert router.worker_request_counts["http://b:2"] == 1

        assert client.post("/worker_finish_request", params={"url": picked}).status_code == 200
        assert router.worker_request_counts["http://b:2"] == 0


def test_unknown_ack_is_rejected_without_mutation():
    router = _router()
    router.worker_request_counts = {"http://a:1": 1}
    with TestClient(router.app) as client:
        assert client.post("/worker_finish_request", params={"url": "http://ghost:9"}).status_code == 400
        assert router.worker_request_counts == {"http://a:1": 1}


def test_empty_pool_returns_503():
    router = _router()
    with TestClient(router.app) as client:
        assert client.get("/pick_worker_for_request").status_code == 503
