"""API actor selection and pool configuration, without starting Ray or an HTTP server.

Mental model: max_concurrency=2 -> one zero-GPU actor handling up to two requests concurrently.
The shared pool's placement rules are covered by test_reward_pool_placement.py.
Custom APIs supply their own actor and constructor kwargs through YAML.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import base64
import json
from argparse import Namespace
from unittest.mock import Mock, call

import httpx
import openai
import pytest
import torch
import yaml

import miles.rollout.rm_hub.core as core_module
from miles.rollout.rm_hub.api import ApiRewardActor, AsyncApiRewardPool
from miles.rollout.rm_hub.openai_api import AsyncOpenAIPool, OpenAIImageRewardActor
from miles.utils.api_rm_config import ApiRewardConfig, load_api_rm_config


@pytest.fixture
def ray_worker(monkeypatch):
    monkeypatch.setattr(AsyncApiRewardPool, "_instances", {})
    actor_cls = Mock()
    actor_cls.options.return_value = actor_cls
    actor_cls.remote.side_effect = lambda **kwargs: Mock()
    remote = Mock(return_value=actor_cls)
    monkeypatch.setattr(core_module.ray, "remote", remote)
    return remote, actor_cls


@pytest.fixture
def api_transport(monkeypatch):
    original = httpx.Client
    clients = []

    def install(handler):
        def factory(**kwargs):
            client = original(**kwargs, transport=httpx.MockTransport(handler))
            clients.append(client)
            return client

        monkeypatch.setattr(httpx, "Client", factory)

    yield install
    for client in clients:
        client.close()


@pytest.mark.parametrize("pool_cls", [AsyncApiRewardPool, AsyncOpenAIPool])
def test_pool_reuses_one_concurrent_zero_gpu_worker(ray_worker, pool_cls):
    remote, actor_cls = ray_worker
    config = ApiRewardConfig(actor_kwargs={"model": "judge", "api_key_env": "TEST_RM_KEY"}, max_concurrency=2)
    args = Namespace(**{f"_{pool_cls.name}_rm_config": config})
    pool = pool_cls(args)
    assert pool_cls(args) is pool

    remote.assert_called_once_with(OpenAIImageRewardActor)
    actor_cls.options.assert_called_once_with(num_cpus=0, num_gpus=0, scheduling_strategy="DEFAULT", max_concurrency=2)
    actor_cls.remote.assert_called_once_with(**config.actor_kwargs)
    assert pool._batch_size == 1


class CustomApiRewardActor(ApiRewardActor):
    """A user-owned service with its own payload, no model/key, and no OpenAI schema."""

    def __init__(self, *, endpoint, timeout_s):
        self.client = httpx.Client(base_url=endpoint, timeout=timeout_s)

    def build_request(self, output, prompt):
        return {"prompt": prompt, "num_frames": output.shape[1]}

    def send_request(self, request):
        response = self.client.post("/score", json=request)
        response.raise_for_status()
        return response

    def parse_response(self, response):
        return response.json()["reward"]


@pytest.mark.parametrize("inline", [False, True], ids=["file", "base64"])
def test_custom_api_loads_config_and_owns_request_and_response_formats(
    tmp_path, monkeypatch, ray_worker, api_transport, inline
):
    remote, actor_cls = ray_worker
    path = tmp_path / "rm.yaml"
    kwargs = {"endpoint": "http://custom-reward.test", "timeout_s": 17}
    path.write_text(
        yaml.safe_dump(
            {"actor_class": f"{__name__}.CustomApiRewardActor", "actor_kwargs": kwargs, "max_concurrency": 3}
        )
    )
    value = "base64:" + base64.b64encode(path.read_bytes()).decode() if inline else str(path)
    config = load_api_rm_config(value)

    def create_worker(cls):
        actor_cls.remote.side_effect = cls
        return actor_cls

    remote.side_effect = create_worker
    monkeypatch.setattr(
        openai, "OpenAI", Mock(side_effect=AssertionError("Custom APIs must not create an OpenAI client"))
    )
    payloads = []

    def handler(request):
        assert str(request.url) == "http://custom-reward.test/score"
        assert "authorization" not in request.headers
        payload = json.loads(request.content)
        payloads.append(payload)
        return httpx.Response(200, json={"reward": float(payload["prompt"])})

    api_transport(handler)
    pool = AsyncApiRewardPool(Namespace(_custom_api_rm_config=config))
    remote.assert_called_once_with(CustomApiRewardActor)
    actor_cls.remote.assert_called_once_with(**kwargs)
    actor_cls.options.assert_called_once_with(num_cpus=0, num_gpus=0, scheduling_strategy="DEFAULT", max_concurrency=3)
    actor = pool._actors[0]
    outputs = [torch.zeros(3, 2, 8, 8), torch.zeros(3, 1, 8, 8)]
    assert actor.score_batch(outputs, ["-7", "12.5"]) == [-7.0, 12.5]
    assert payloads == [{"prompt": "-7", "num_frames": 2}, {"prompt": "12.5", "num_frames": 1}]


@pytest.mark.parametrize("actor_class", ["builtins.dict", "miles.rollout.rm_hub.api.custom_api_rm"])
def test_invalid_actor_class_is_rejected_before_creating_ray_worker(ray_worker, actor_class):
    remote, _ = ray_worker
    config = ApiRewardConfig(actor_class=actor_class)
    with pytest.raises(TypeError, match="ApiRewardActor subclass"):
        AsyncApiRewardPool(Namespace(_custom_api_rm_config=config))
    remote.assert_not_called()


def test_openai_pool_rejects_other_api_implementations(ray_worker):
    remote, _ = ray_worker
    config = ApiRewardConfig(actor_class=f"{__name__}.CustomApiRewardActor")
    with pytest.raises(TypeError, match="OpenAIImageRewardActor subclass"):
        AsyncOpenAIPool(Namespace(_openai_api_rm_config=config))
    remote.assert_not_called()


def test_custom_and_openai_pools_use_independent_configs_and_workers(ray_worker):
    remote, actor_cls = ray_worker
    custom_config = ApiRewardConfig(
        actor_class=f"{__name__}.CustomApiRewardActor",
        actor_kwargs={"endpoint": "http://custom-reward.test", "timeout_s": 17},
        max_concurrency=3,
    )
    openai_config = ApiRewardConfig(actor_kwargs={"model": "judge", "api_key_env": "TEST_RM_KEY"}, max_concurrency=7)
    args = Namespace(_custom_api_rm_config=custom_config, _openai_api_rm_config=openai_config)
    custom_pool = AsyncApiRewardPool(args)
    openai_pool = AsyncOpenAIPool(args)
    assert custom_pool is not openai_pool
    assert custom_pool.name == "custom_api"
    assert openai_pool.name == "openai_api"
    assert remote.call_args_list == [call(CustomApiRewardActor), call(OpenAIImageRewardActor)]
    assert actor_cls.remote.call_args_list == [call(**custom_config.actor_kwargs), call(**openai_config.actor_kwargs)]
    assert [c.kwargs["max_concurrency"] for c in actor_cls.options.call_args_list] == [3, 7]


@pytest.mark.parametrize("pool_cls", [AsyncApiRewardPool, AsyncOpenAIPool])
def test_pool_requires_its_own_config_before_creating_worker(ray_worker, pool_cls):
    remote, _ = ray_worker
    configs = {
        "_custom_api_rm_config": ApiRewardConfig(),
        "_openai_api_rm_config": ApiRewardConfig(),
    }
    configs[f"_{pool_cls.name}_rm_config"] = None
    with pytest.raises(ValueError, match=f"--{pool_cls.name.replace('_', '-')}-rm-config"):
        pool_cls(Namespace(**configs))
    remote.assert_not_called()
