"""API pool worker configuration, without starting Ray or an HTTP server.

Mental model: max_concurrency=2 -> one zero-GPU actor handling up to two requests concurrently.
The shared pool's placement rules are covered by test_reward_pool_placement.py.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from unittest.mock import Mock

import miles.rollout.rm_hub.core as core_module
from miles.rollout.rm_hub.api import ApiRewardActor, AsyncApiRewardPool
from miles.utils.api_rm_config import ApiRewardConfig


def test_pool_reuses_one_concurrent_zero_gpu_worker(monkeypatch):
    monkeypatch.setattr(AsyncApiRewardPool, "_instances", {})
    actor_cls = Mock()
    actor_cls.options.return_value = actor_cls
    actor_cls.remote.side_effect = lambda **kwargs: Mock()
    remote = Mock(return_value=actor_cls)
    monkeypatch.setattr(core_module.ray, "remote", remote)
    config = ApiRewardConfig(model="judge", api_key_env="TEST_RM_KEY", max_concurrency=2)
    args = Namespace(_api_rm_config=config)
    pool = AsyncApiRewardPool(args)
    assert AsyncApiRewardPool(args) is pool

    remote.assert_called_once_with(ApiRewardActor)
    actor_cls.options.assert_called_once_with(num_cpus=0, num_gpus=0, scheduling_strategy="DEFAULT", max_concurrency=2)
    actor_cls.remote.assert_called_once_with(config=config)
    assert pool._batch_size == 1
