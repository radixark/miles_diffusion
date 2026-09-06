"""Rewards receive ``generated_output`` itself and quantise to uint8 inside each actor.

Mental model (one sample scored by two rewards):

    sample.generated_output (float CFHW)
        -> HPSRewardActor.score_batch       : round    -> uint8 -> HPSv2
        -> PickScoreRewardActor.score_batch : truncate -> uint8 -> mean over frames -> PickScore

Covered: HPS rounds when quantising (1); PickScore truncates and returns one mean per sample (2);
hps_rm and pickscore_rm hand the raw tensor to their pool and record their own queue depth (3);
pickscore_rm keeps only --pickscore-num-frames frames before shipping (4); batched_async_rm takes
the HPS fast path (5) and keeps sample order for mixed rm_types (6).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

import miles.rollout.rm_hub.hps as hps_module
import miles.rollout.rm_hub.pickscore as pickscore_module
from miles.rollout.rm_hub import batched_async_rm
from miles.rollout.rm_hub.hps import HPSRewardActor
from miles.rollout.rm_hub.pickscore import PickScoreRewardActor
from miles.utils.types import Sample

# one pixel row [0, 0.5, 1] per channel; rounding gives 128 in the middle, truncation 127
_ROW = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 1, 1, 3).repeat(3, 1, 1, 1)


def _middle_pixel(images) -> list[float]:
    return [float(np.asarray(image)[0, 1, 0]) for image in images]


def test_hps_actor_rounds_when_quantising():
    """Truncating would shift pixels one level below what the HPSv2 reference scores."""
    actor = HPSRewardActor.__new__(HPSRewardActor)
    actor.scorer = lambda prompts, images: _middle_pixel(images)

    assert actor.score_batch([_ROW], ["prompt"]) == [128.0]


def test_pickscore_actor_truncates_and_averages_frames_per_sample():
    """PickScore keeps flow_grpo's truncation, a two-frame sample yields one score, and each forward sees
    --pickscore-batch-size frames (the chunking the e2e standards were recorded with), not one sample."""
    actor = PickScoreRewardActor.__new__(PickScoreRewardActor)
    actor.frames_per_forward = 1
    forward_sizes = []

    def scorer(prompts, images):
        forward_sizes.append(len(images))
        return [p + i for p, i in zip(prompts, _middle_pixel(images), strict=True)]

    actor.scorer = scorer
    two_frames = torch.cat([_ROW, _ROW], dim=1)

    assert actor.score_batch([two_frames], [1000.0]) == [1127.0]
    assert forward_sizes == [1, 1]


@pytest.mark.asyncio
async def test_rm_functions_hand_the_raw_tensor_to_their_pool(monkeypatch):
    """Decoding belongs to the actor, and each pool records its own backlog instead of overwriting a shared one."""
    pool = AsyncMock()
    pool.score.side_effect = [([1.0], 2), ([1.0], 1)]
    monkeypatch.setattr(hps_module, "AsyncHPSPool", lambda args: pool)
    monkeypatch.setattr(pickscore_module, "AsyncPickScorePool", lambda args: pool)
    sample = Sample(prompt="prompt", generated_output=_ROW)

    assert await hps_module.hps_rm(Namespace(), [sample]) == [1.0]
    assert await pickscore_module.pickscore_rm(Namespace(pickscore_num_frames=None), [sample]) == [1.0]

    calls = pool.score.await_args_list
    assert [call.args[1] for call in calls] == [["prompt"]] * 2
    assert all(torch.equal(call.args[0][0], _ROW) for call in calls)
    assert sample.reward_max_queue_depth == {"hps": 2.0, "pickscore": 1.0}


@pytest.mark.asyncio
async def test_pickscore_rm_selects_frames_before_shipping(monkeypatch):
    """Only the --pickscore-num-frames frames should cross the object store, not the whole video."""
    pool = AsyncMock()
    pool.score.return_value = ([1.0], 0)
    monkeypatch.setattr(pickscore_module, "AsyncPickScorePool", lambda args: pool)
    video = torch.cat([_ROW * 0.0, _ROW, _ROW * 0.0], dim=1)

    await pickscore_module.pickscore_rm(
        Namespace(pickscore_num_frames=1), [Sample(prompt="p", generated_output=video)]
    )

    (shipped,), _ = pool.score.await_args.args
    assert shipped.shape[1] == 1 and torch.equal(shipped[:, 0], _ROW[:, 0])


@pytest.mark.asyncio
async def test_batched_async_rm_uses_hps_batch_fast_path(monkeypatch):
    hps_rm = AsyncMock(return_value=[3.0, 1.0])
    monkeypatch.setattr(hps_module, "hps_rm", hps_rm)
    args = Namespace(custom_rm_path=None, rm_type="hps")
    samples = [Sample(index=3), Sample(index=1)]

    assert await batched_async_rm(args, samples) == [3.0, 1.0]
    hps_rm.assert_awaited_once_with(args, samples)


@pytest.mark.asyncio
async def test_batched_async_rm_preserves_order_for_mixed_reward_types(monkeypatch):
    monkeypatch.setattr(hps_module, "hps_rm", AsyncMock(return_value=[10.0]))
    monkeypatch.setattr(pickscore_module, "pickscore_rm", AsyncMock(return_value=[20.0]))
    args = Namespace(custom_rm_path=None, rm_type=None)
    samples = [
        Sample(metadata={"rm_type": "hps"}),
        Sample(metadata={"rm_type": "pickscore"}),
    ]

    assert await batched_async_rm(args, samples) == [10.0, 20.0]
