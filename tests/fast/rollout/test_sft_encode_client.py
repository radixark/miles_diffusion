"""SftEncodeClient against a mocked encode-only server: what is retried and how failures surface.

What each test pins:
  transient errors   transport errors, 429 and 5xx are retried, and the sample is cached once one succeeds
  client errors      a 4xx other than 429 fails at once, with the server's detail in the error
  exhausted retries  the error names the attempt count and the last server detail
  partial failure    one failed sample does not stop the others, which stay cached; the batch still raises
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import functools
from argparse import Namespace
from types import SimpleNamespace

import httpx
import msgpack
import pytest
import torch

from miles.rollout import encoder_hub, sft_rollout
from miles.rollout.sft_rollout import REMOTE_ENCODE_ATTEMPTS, SftEncodeClient
from miles.utils.types import Sample

_OK = msgpack.packb({"request_id": "r", "encoded": {"prompt_len": 3}})


@pytest.fixture
def serve(monkeypatch):
    """Route the client to ``handler(request, call_index)`` for /encode; /model_info and /server_info always answer."""
    SftEncodeClient.clear_instances()
    monkeypatch.setattr(sft_rollout.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        encoder_hub,
        "get_encoder",
        lambda family, task: SimpleNamespace(pair_from_encoded=lambda encoded, sample: encoded),
    )

    def install(handler):
        calls = []

        def route(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/model_info":
                return httpx.Response(200, json={"model_path": "m"})
            if request.url.path == "/server_info":
                return httpx.Response(200, json={"version": "1"})
            calls.append(request)
            return handler(request, len(calls) - 1)

        monkeypatch.setattr(
            sft_rollout.httpx, "Client", functools.partial(httpx.Client, transport=httpx.MockTransport(route))
        )
        args = Namespace(
            sft_encode_url="http://encoder",
            diffusion_model_family="f",
            diffusion_task="t2i",
            diffusion_width=64,
            diffusion_height=64,
        )
        return SftEncodeClient(args), calls

    yield install
    SftEncodeClient.clear_instances()


def _sample(tmp_path, name):
    target = tmp_path / f"{name}.png"
    target.write_bytes(name.encode())
    return Sample(prompt=name, target={"visual": str(target)}, metadata={"sft_cache_name": f"{name}.pt"})


def test_transient_errors_are_retried(serve, tmp_path):
    def handler(request, index):
        if index == 0:
            raise httpx.ReadError("connection reset", request=request)
        if index == 1:
            return httpx.Response(503, json={"detail": "busy"})
        return httpx.Response(200, content=_OK)

    client, calls = serve(handler)
    client.encode([_sample(tmp_path, "a")], tmp_path)
    assert len(calls) == REMOTE_ENCODE_ATTEMPTS
    assert torch.load(tmp_path / "a.pt") == {"prompt_len": 3}


def test_client_error_is_not_retried(serve, tmp_path):
    client, calls = serve(
        lambda request, index: httpx.Response(400, json={"detail": "Invalid sampling params: width"})
    )
    with pytest.raises(RuntimeError, match="Invalid sampling params: width"):
        client.encode([_sample(tmp_path, "a")], tmp_path)
    assert len(calls) == 1


def test_exhausted_retries_report_the_server_detail(serve, tmp_path):
    client, calls = serve(lambda request, index: httpx.Response(500, json={"detail": "Encode failed: bad image"}))
    with pytest.raises(RuntimeError, match=f"after {REMOTE_ENCODE_ATTEMPTS} attempt.*bad image"):
        client.encode([_sample(tmp_path, "a")], tmp_path)
    assert len(calls) == REMOTE_ENCODE_ATTEMPTS


def test_one_failure_keeps_the_rest_cached(serve, tmp_path):
    def handler(request, index):
        if b'"prompt":"bad"' in request.content:
            return httpx.Response(400, json={"detail": "rejected"})
        return httpx.Response(200, content=_OK)

    client, _ = serve(handler)
    samples = [_sample(tmp_path, name) for name in ("a", "bad", "c")]
    with pytest.raises(RuntimeError, match="1/3 SFT encodes failed"):
        client.encode(samples, tmp_path)
    assert (tmp_path / "a.pt").exists() and (tmp_path / "c.pt").exists()
    assert not (tmp_path / "bad.pt").exists()
