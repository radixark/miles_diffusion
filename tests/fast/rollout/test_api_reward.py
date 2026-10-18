"""API actors return one numeric score per input; the OpenAI actor handles images.

Mental model:

    generated_output -> actor -> prompt + PNG request -> validated numeric score
    custom_api_rm -> singleton pool -> raw tensors in, scores and queue depth out

Covered: the shared scoring contract; OpenAI image/prompt pairing and client configuration;
fatal HTTP errors; image-only input; reward dispatch; YAML rubric loading and credential safety.
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
from dataclasses import asdict
from threading import Barrier
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
import torch
import yaml
from PIL import Image

import miles.rollout.rm_hub.api as api_module
import miles.rollout.rm_hub.openai_api as openai_api_module
from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.rollout.rm_hub.api import ApiRewardActor, custom_api_rm
from miles.rollout.rm_hub.openai_api import OpenAIImageRewardActor, openai_api_rm
from miles.utils.api_rm_config import (
    ApiRewardConfig,
    OpenAIImageRewardConfig,
    load_api_rm_config,
    resolve_api_rm_configs,
)
from miles.utils.types import Sample


def _config(**overrides):
    return OpenAIImageRewardConfig(model="judge-v1", api_key_env="TEST_RM_KEY", **overrides)


def _args():
    return Namespace(
        rm_type="custom_api",
        custom_rm_path=None,
        _custom_api_rm_config=ApiRewardConfig(actor_kwargs=asdict(_config())),
        _openai_api_rm_config=ApiRewardConfig(actor_kwargs=asdict(_config())),
    )


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
        assert str(request.url) == "http://judge.test/v1/chat/completions"
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
    actor = OpenAIImageRewardActor(**asdict(_config(base_url="http://judge.test/v1", timeout_s=17)))
    samples = [_sample(2), _sample(1)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(actor.score_batch, [s.generated_output], [s.prompt]) for s in samples]
        assert [future.result()[0] for future in futures] == [2.0, 1.0]
    assert clients[0]["max_retries"] == 2
    assert clients[0]["timeout"] == 17


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module, pool_name, rm_function, reward_name",
    [
        (api_module, "AsyncApiRewardPool", custom_api_rm, "custom_api"),
        (openai_api_module, "AsyncOpenAIPool", openai_api_rm, "openai_api"),
    ],
)
async def test_rm_passes_raw_tensor_and_preserves_pool_results_and_errors(
    monkeypatch, module, pool_name, rm_function, reward_name
):
    pool = AsyncMock()
    pool.score.return_value = ([1.0], 3)
    monkeypatch.setattr(module, pool_name, lambda args: pool)
    args = _args()
    sample = _sample(1)
    assert await rm_function(args, [sample]) == [1.0]
    (output,), prompts = pool.score.await_args.args
    assert output is sample.generated_output
    assert prompts == [sample.prompt]
    assert sample.reward_max_queue_depth == {reward_name: 3.0}

    failure = ValueError("Invalid API score")
    pool.score.side_effect = failure
    with pytest.raises(ValueError) as exc:
        await rm_function(args, [sample])
    assert exc.value is failure


@pytest.mark.parametrize("status, error", [(429, openai.RateLimitError), (503, openai.InternalServerError)])
def test_http_error_propagates_after_sdk_retries(sdk_transport, status, error):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"message": "test failure"}})

    sdk_transport(handler)
    actor = OpenAIImageRewardActor(**asdict(_config()))
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
def test_invalid_response_never_becomes_a_reward(content, sdk_transport):
    requests = []

    def handler(request):
        requests.append(request)
        data = _response(0).json()
        data["choices"][0]["message"]["content"] = content
        return httpx.Response(200, json=data)

    sdk_transport(handler)
    actor = OpenAIImageRewardActor(**asdict(_config()))
    with pytest.raises(ValueError):
        actor.score_batch([_sample(1).generated_output], ["1"])
    assert len(requests) == 1


def test_video_is_rejected_before_http(sdk_transport):
    def handler(request):
        pytest.fail("Unsupported media must not be sent to the API")

    sdk_transport(handler)
    actor = OpenAIImageRewardActor(**asdict(_config()))
    with pytest.raises(ValueError):
        actor.score_batch([torch.zeros(3, 2, 8, 8)], ["1"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module, pool_name, rm_type",
    [(api_module, "AsyncApiRewardPool", "custom_api"), (openai_api_module, "AsyncOpenAIPool", "openai_api")],
)
async def test_builtin_dispatch_and_per_sample_override(monkeypatch, module, pool_name, rm_type):
    pool = AsyncMock()
    pool.score.side_effect = [([1.0, 2.0], 0), ([3.0], 0)]
    monkeypatch.setattr(module, pool_name, lambda args: pool)
    args = _args()
    args.rm_type = rm_type
    assert await batched_async_rm(args, [_sample(1), _sample(2)]) == [1.0, 2.0]
    args.rm_type = "unused"
    sample = _sample(3)
    sample.metadata = {"rm_type": rm_type}
    assert await async_rm(args, sample) == 3.0


@pytest.mark.parametrize("flat_config", [True, False])
def test_config_prompt_resolution_and_no_credentials_in_serialized_args(tmp_path, flat_config):
    (tmp_path / "rubric.txt").write_text("Evaluate prompt adherence from 0 to 10. Return JSON with score.")
    config_path = tmp_path / "rm.yaml"
    kwargs = {
        "model": "judge-version-123",
        "api_key_env": "TEST_RM_KEY",
        "prompt_path": "rubric.txt",
        "score_max": 10,
    }
    data = kwargs if flat_config else {"actor_kwargs": kwargs}
    config_path.write_text(yaml.safe_dump({**data, "max_concurrency": 3}))
    args = Namespace(custom_api_rm_config=None, openai_api_rm_config=str(config_path))
    resolve_api_rm_configs(args)
    args = pickle.loads(pickle.dumps(args))
    assert args._openai_api_rm_config.actor_kwargs["model"] == "judge-version-123"
    assert args._openai_api_rm_config.max_concurrency == 3
    (tmp_path / "rubric.txt").unlink()
    assert "0 to 10" in args._openai_api_rm_config.actor_kwargs["prompt"]
    assert b"test-only-secret" not in pickle.dumps(args)


def test_inline_api_key_is_rejected(tmp_path):
    path = tmp_path / "rm.yaml"
    path.write_text("model: judge\napi_key_env: TEST_RM_KEY\napi_key: not-allowed\n")
    with pytest.raises(TypeError, match="api_key"):
        load_api_rm_config(str(path))


def test_multiple_api_configs_are_rejected(tmp_path):
    path = tmp_path / "rm.yaml"
    path.write_text(yaml.safe_dump({"alignment": {"model": "judge"}, "aesthetic": {"model": "judge"}}))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        load_api_rm_config(str(path))


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5])
def test_invalid_api_concurrency_is_rejected_at_startup(tmp_path, concurrency):
    path = tmp_path / "rm.yaml"
    path.write_text(yaml.safe_dump({"model": "judge", "api_key_env": "TEST_RM_KEY", "max_concurrency": concurrency}))
    with pytest.raises(ValueError, match="max_concurrency must be a positive integer"):
        load_api_rm_config(str(path))


class StubApiRewardActor(ApiRewardActor):
    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = []

    def build_request(self, output, prompt):
        self.calls.append((output, prompt))
        return prompt

    def send_request(self, request):
        return next(self.scores)

    def parse_response(self, response):
        return response


def test_base_actor_preserves_raw_inputs_and_backend_score_range():
    outputs = [torch.zeros(3, 2, 8, 8), _sample(1).generated_output]
    prompts = ["video", "image"]
    actor = StubApiRewardActor([-7, 12.5])
    assert actor.score_batch(outputs, prompts) == [-7.0, 12.5]
    assert len(actor.calls) == len(outputs)
    for (output, prompt), expected_output, expected_prompt in zip(actor.calls, outputs, prompts, strict=True):
        assert output is expected_output
        assert prompt == expected_prompt


def test_invalid_input_pairing_and_empty_batches_do_not_call_backend():
    actor = StubApiRewardActor([])
    with pytest.raises(ValueError, match="one prompt per output"):
        actor.score_batch([_sample(1).generated_output], [])
    assert actor.score_batch([], []) == []
    assert actor.calls == []


@pytest.mark.parametrize("score", [[], [1, 2], True, "2", None, float("nan"), float("inf"), -float("inf")])
def test_base_actor_rejects_invalid_backend_scores(score):
    with pytest.raises(ValueError, match="finite numbers"):
        StubApiRewardActor([score]).score_batch([_sample(1).generated_output], ["prompt"])


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"actor_class": ""}, "actor_class"),
        ({"actor_class": None}, "actor_class"),
        ({"actor_kwargs": []}, "actor_kwargs"),
    ],
)
def test_invalid_actor_settings_are_rejected_at_startup(tmp_path, settings, message):
    path = tmp_path / "rm.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match=message):
        load_api_rm_config(str(path))
