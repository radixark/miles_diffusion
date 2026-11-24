"""VideoAlign reward: dict-per-sample contract and the rm_hub dispatch fast path.

Mental model (one video scored by VideoAlign):

    sample.generated_output (float CFHW, T frames)
        -> VideoAlignRewardActor.score_batch : truncate -> uint8 frames -> VideoAlignScorer
        -> videoalign_rm ships only --videoalign-num-frames frames across the object store
        -> reward is a dict {"vq", "mq", "ta", "overall"} per sample (--reward-key picks the scalar)

Covered: the actor decodes every frame and returns one dict per sample (1); videoalign_rm hands the
raw tensor to its pool and records its own queue depth (2); videoalign_rm selects frames before
shipping (3); batched_async_rm takes the videoalign fast path (4) and preserves sample order for
mixed rm_types (5).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

import miles.rollout.rm_hub.videoalign as videoalign_module
from miles.rollout.rm_hub import batched_async_rm
from miles.rollout.rm_hub.videoalign import VideoAlignRewardActor
from miles.utils.types import Sample

# one pixel row [0, 0.5, 1] per channel; truncation gives 127 in the middle (the actor truncates)
_ROW = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 1, 1, 3).repeat(3, 1, 1, 1)


def _sample_reward() -> dict[str, float]:
    return {"vq": 1.0, "mq": 2.0, "ta": 3.0, "overall": 6.0}


def test_videoalign_actor_decodes_every_frame_and_returns_one_dict_per_sample():
    actor = VideoAlignRewardActor.__new__(VideoAlignRewardActor)
    actor.num_frames = None
    decoded = {}

    def scorer(prompts, videos):
        for prompt, frames in zip(prompts, videos, strict=True):
            decoded[prompt] = [np.asarray(frame).shape for frame in frames]
        return [_sample_reward() for _ in prompts]

    actor.scorer = scorer
    two_frames = torch.cat([_ROW, _ROW], dim=1)

    assert actor.score_batch([two_frames], ["prompt"]) == [_sample_reward()]
    assert len(decoded["prompt"]) == 2  # both frames decoded and handed to the scorer


@pytest.mark.asyncio
async def test_videoalign_rm_hands_the_raw_tensor_to_its_pool(monkeypatch):
    pool = AsyncMock()
    pool.score.return_value = ([_sample_reward()], 2)
    monkeypatch.setattr(videoalign_module, "AsyncVideoAlignPool", lambda args: pool)
    sample = Sample(prompt="prompt", generated_output=_ROW)

    assert await videoalign_module.videoalign_rm(Namespace(videoalign_num_frames=None), [sample]) == [_sample_reward()]

    (shipped,), prompts = pool.score.await_args.args
    assert torch.equal(shipped, _ROW)
    assert prompts == ["prompt"]
    assert sample.reward_max_queue_depth == {"videoalign": 2.0}


@pytest.mark.asyncio
async def test_videoalign_rm_selects_frames_before_shipping(monkeypatch):
    """Only the --videoalign-num-frames frames should cross the object store, not the whole video."""
    pool = AsyncMock()
    pool.score.return_value = ([_sample_reward()], 0)
    monkeypatch.setattr(videoalign_module, "AsyncVideoAlignPool", lambda args: pool)
    video = torch.cat([_ROW * 0.0, _ROW, _ROW * 0.0], dim=1)

    await videoalign_module.videoalign_rm(
        Namespace(videoalign_num_frames=1), [Sample(prompt="p", generated_output=video)]
    )

    (shipped,), _ = pool.score.await_args.args
    assert shipped.shape[1] == 1 and torch.equal(shipped[:, 0], _ROW[:, 0])


@pytest.mark.asyncio
async def test_batched_async_rm_uses_videoalign_batch_fast_path(monkeypatch):
    videoalign_rm = AsyncMock(return_value=[_sample_reward(), _sample_reward()])
    monkeypatch.setattr(videoalign_module, "videoalign_rm", videoalign_rm)
    args = Namespace(custom_rm_path=None, rm_type="videoalign")
    samples = [Sample(index=3), Sample(index=1)]

    assert await batched_async_rm(args, samples) == [_sample_reward(), _sample_reward()]
    videoalign_rm.assert_awaited_once_with(args, samples)


@pytest.mark.asyncio
async def test_batched_async_rm_preserves_order_for_mixed_reward_types(monkeypatch):
    monkeypatch.setattr(videoalign_module, "videoalign_rm", AsyncMock(return_value=[_sample_reward()]))
    from miles.rollout.rm_hub import hps as hps_module

    monkeypatch.setattr(hps_module, "hps_rm", AsyncMock(return_value=[10.0]))
    args = Namespace(custom_rm_path=None, rm_type=None)
    samples = [
        Sample(metadata={"rm_type": "videoalign"}),
        Sample(metadata={"rm_type": "hps"}),
    ]

    assert await batched_async_rm(args, samples) == [_sample_reward(), 10.0]


@pytest.mark.asyncio
async def test_shared_pool_preserves_dict_rewards_and_scalar_conversion(monkeypatch):
    from miles.rollout.rm_hub import core

    pool = core.AsyncRewardActorPool.__new__(core.AsyncRewardActorPool)
    rewards = [_sample_reward(), np.float32(2.5), {**_sample_reward(), "mq": 4.0}]
    pool._actors = [Namespace(score_batch=Namespace(remote=lambda outputs, prompts: [rewards[i] for i in outputs]))]
    pool._batch_size = 2
    pool._round_robin_index = 0
    pool._inflight = [0]
    monkeypatch.setattr(core.ray, "get", lambda refs: refs)

    scores, depth = await pool.score([0, 1, 2], ["a", "b", "c"])
    assert scores == rewards
    assert isinstance(scores[1], float)
    assert depth == 1
    assert pool._inflight == [0]


def test_videoalign_pool_uses_default_model_when_cli_path_is_none(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        videoalign_module.AsyncRewardActorPool, "__init__", lambda self, **kwargs: captured.update(kwargs)
    )
    pool = videoalign_module.AsyncVideoAlignPool.__new__(videoalign_module.AsyncVideoAlignPool)
    pool.__init__(
        Namespace(
            videoalign_num_frames=8,
            videoalign_model_path=None,
            videoalign_checkpoint_path=None,
            videoalign_max_pixels=200704,
            videoalign_no_normalize=False,
            videoalign_num_workers=1,
            videoalign_batch_size=1,
            videoalign_num_gpus_per_worker=1,
            videoalign_reward_colocate=False,
        )
    )
    assert captured["actor_kwargs"]["model_path"] == videoalign_module._BASE_MODEL
