"""API rewards pair each generated image with its prompt and return one numeric score.

Mental model:

    generated_output -> actor -> prompt + PNG request -> validated numeric score
    api_rm -> singleton pool -> raw tensors in, scores and queue depth out

Covered: image/prompt pairing and client configuration; fatal HTTP errors; score validation;
image-only input; reward dispatch; YAML rubric loading and credential safety.
Ray worker configuration is covered by test_api_reward_pool.py.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import base64
import io
import json
import pickle
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
import torch
import yaml
from PIL import Image

import miles.rollout.rm_hub.api as api_module
from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.rollout.rm_hub.api import (
    ApiRewardActor,
    _parse_score,
    api_rm,
)
from miles.utils.api_rm_config import ApiRewardConfig
from miles.utils.types import Sample


def _config(**overrides):
    return ApiRewardConfig(model="judge-v1", api_key_env="TEST_RM_KEY", **overrides)


def _args():
    return Namespace(rm_type="api", custom_rm_path=None, _api_rm_config=_config())


def _sample(index):
    return Sample(index=index, prompt=str(index), generated_output=torch.full((3, 1, 8, 8), index / 4))


def _response(score):
    return httpx.Response(
        200,
        json={
            "id": "test-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "judge-v1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"score": score}),
                        "refusal": None,
                    },
                }
            ],
        },
    )


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("TEST_RM_KEY", "test-only-secret")


@pytest.fixture
def sdk_transport(monkeypatch):
    original = openai.OpenAI
    created, clients = [], []

    def install(handler):
        def factory(**kwargs):
            created.append(kwargs)
            client = original(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
            clients.append(client)
            return client

        monkeypatch.setattr(openai, "OpenAI", factory)
        return created

    yield install
    for client in clients:
        client.close()


def test_actor_preserves_image_prompt_pairing(sdk_transport):
    requests_started = Barrier(2)

    def handler(request):
        payload = json.loads(request.content)
        content = payload["messages"][1]["content"]
        index = int(content[0]["text"])
        data_url = content[1]["image_url"]["url"]
        assert data_url.startswith("data:image/png;base64,")
        image = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
        assert image.getpixel((0, 0)) == (round(index / 4 * 255),) * 3
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["model"] == "judge-v1"
        assert request.headers["authorization"] == "Bearer test-only-secret"
        requests_started.wait(timeout=5)
        return _response(index)

    clients = sdk_transport(handler)
    actor = ApiRewardActor(config=_config())
    samples = [_sample(2), _sample(1)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(actor.score_batch, [s.generated_output], [s.prompt]) for s in samples]
        assert [future.result()[0] for future in futures] == [2.0, 1.0]
    assert clients[0]["max_retries"] == 2


async def test_rm_passes_raw_tensor_and_preserves_pool_results_and_errors(monkeypatch):
    pool = AsyncMock()
    pool.score.return_value = ([1.0], 3)
    monkeypatch.setattr(api_module, "AsyncApiRewardPool", lambda args: pool)
    args = _args()
    sample = _sample(1)
    assert await api_rm(args, [sample]) == [1.0]
    (output,), prompts = pool.score.await_args.args
    assert output is sample.generated_output
    assert prompts == [sample.prompt]
    assert sample.reward_max_queue_depth == {"api": 3.0}

    failure = ValueError("Invalid API score")
    pool.score.side_effect = failure
    with pytest.raises(ValueError) as exc:
        await api_rm(args, [sample])
    assert exc.value is failure


@pytest.mark.parametrize("status, error", [(429, openai.RateLimitError), (503, openai.InternalServerError)])
def test_http_error_propagates_after_sdk_retries(sdk_transport, status, error):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"message": "test failure"}})

    sdk_transport(handler)
    actor = ApiRewardActor(config=_config())
    with pytest.raises(error):
        actor.score_batch([_sample(1).generated_output], ["1"])
    assert calls == 3


@pytest.mark.parametrize(
    "content",
    [
        "Score: 2",
        '{"score": "2"}',
        '{"score": true}',
        '{"score": NaN}',
        '{"score": -1}',
        '{"score": 5}',
    ],
)
def test_invalid_response_never_becomes_a_reward(content):
    with pytest.raises(ValueError):
        _parse_score(content, _config())


def test_video_is_rejected_before_http(sdk_transport):
    def handler(request):
        pytest.fail("Unsupported media must not be sent to the API")

    sdk_transport(handler)
    actor = ApiRewardActor(config=_config())
    with pytest.raises(ValueError):
        actor.score_batch([torch.zeros(3, 2, 8, 8)], ["1"])


async def test_builtin_dispatch_and_per_sample_override(monkeypatch):
    pool = AsyncMock()
    pool.score.side_effect = [([1.0, 2.0], 0), ([3.0], 0)]
    monkeypatch.setattr(api_module, "AsyncApiRewardPool", lambda args: pool)
    args = _args()
    assert await batched_async_rm(args, [_sample(1), _sample(2)]) == [1.0, 2.0]
    args.rm_type = "unused"
    sample = _sample(3)
    sample.metadata = {"rm_type": "api"}
    assert await async_rm(args, sample) == 3.0


def test_config_prompt_resolution_and_no_credentials_in_serialized_args(tmp_path):
    from miles.utils.arguments import load_api_rm_config

    (tmp_path / "rubric.txt").write_text("Evaluate prompt adherence from 0 to 10. Return JSON with score.")
    config_path = tmp_path / "rm.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": "judge-version-123",
                "api_key_env": "TEST_RM_KEY",
                "prompt_path": "rubric.txt",
                "score_max": 10,
            }
        )
    )
    args = Namespace(api_rm_config=str(config_path), _api_rm_config=load_api_rm_config(str(config_path)))
    args = pickle.loads(pickle.dumps(args))
    assert args._api_rm_config.model == "judge-version-123"
    (tmp_path / "rubric.txt").unlink()
    assert "0 to 10" in args._api_rm_config.prompt
    assert b"test-only-secret" not in pickle.dumps(args)


def test_inline_api_key_is_rejected(tmp_path):
    from miles.utils.arguments import load_api_rm_config

    path = tmp_path / "rm.yaml"
    path.write_text("model: judge\napi_key_env: TEST_RM_KEY\napi_key: not-allowed\n")
    with pytest.raises(TypeError, match="api_key"):
        load_api_rm_config(str(path))


def test_multiple_api_configs_are_rejected(tmp_path):
    from miles.utils.arguments import load_api_rm_config

    path = tmp_path / "rm.yaml"
    path.write_text(yaml.safe_dump({"alignment": {"model": "judge"}, "aesthetic": {"model": "judge"}}))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        load_api_rm_config(str(path))


def test_zero_api_concurrency_is_rejected_at_startup(tmp_path):
    from miles.utils.arguments import load_api_rm_config

    path = tmp_path / "rm.yaml"
    path.write_text("model: judge\napi_key_env: TEST_RM_KEY\nmax_concurrency: 0\n")
    with pytest.raises(ValueError, match="max_concurrency must be a positive integer"):
        load_api_rm_config(str(path))
