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
from miles.rollout.rm_hub.hps import _sample_to_rgb_hwc_uint8
from miles.utils.types import Sample


def test_sample_to_rgb_hwc_uint8_matches_hps_rounding():
    channel = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 1, 1, 3)
    sample = Sample(generated_output=channel.repeat(3, 1, 1, 1))

    actual = _sample_to_rgb_hwc_uint8(sample)

    expected = np.array([[[0, 0, 0], [128, 128, 128], [255, 255, 255]]], dtype=np.uint8)
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous


def test_sample_to_rgb_hwc_uint8_rejects_video_outputs():
    with pytest.raises(ValueError, match="supports image outputs only"):
        _sample_to_rgb_hwc_uint8(Sample(generated_output=torch.zeros(3, 2, 4, 4)))


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
