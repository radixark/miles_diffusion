from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace

import numpy as np
import pytest
import torch
from PIL import Image

from miles.rollout.rm_hub import batched_async_rm
from miles.rollout.rm_hub.hps import HPSScorer, _HPSImageTransform, _sample_to_rgb_hwc_uint8
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


def test_hps_scorer_returns_aligned_diagonal_scores():
    class FakeModel(torch.nn.Module):
        def forward(self, image_batch, text_batch):
            assert image_batch.shape[0] == text_batch.shape[0] == 2
            return {
                "image_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "text_features": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
            }

    scorer = HPSScorer.__new__(HPSScorer)
    torch.nn.Module.__init__(scorer)
    scorer.device = torch.device("cpu")
    scorer.model = FakeModel()
    scorer.preprocess = lambda _: torch.zeros(3, 2, 2)
    scorer.tokenizer = lambda prompts: torch.zeros(len(prompts), 1, dtype=torch.long)

    images = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]

    assert scorer(["first", "second"], images) == [2.0, 5.0]


def test_hps_image_transform_fits_longest_side_and_pads():
    transform = _HPSImageTransform(image_size=(4, 4), mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    image = Image.fromarray(np.full((2, 4, 3), 255, dtype=np.uint8))

    actual = transform(image)

    assert actual.shape == (3, 4, 4)
    torch.testing.assert_close(actual[:, 1:3], torch.ones(3, 2, 4))
    torch.testing.assert_close(actual[:, [0, 3]], torch.zeros(3, 2, 4))


def test_batched_async_rm_uses_hps_batch_fast_path(monkeypatch):
    import miles.rollout.rm_hub.hps as hps_module

    calls = []

    async def fake_hps_rm(args, samples):
        calls.append(samples)
        return [float(sample.index) for sample in samples]

    monkeypatch.setattr(hps_module, "hps_rm", fake_hps_rm)
    args = Namespace(custom_rm_path=None, rm_type="hps")
    samples = [Sample(index=3), Sample(index=1)]

    rewards = asyncio.run(batched_async_rm(args, samples))

    assert rewards == [3.0, 1.0]
    assert calls == [samples]


def test_batched_async_rm_preserves_order_for_mixed_reward_types(monkeypatch):
    import miles.rollout.rm_hub.hps as hps_module
    import miles.rollout.rm_hub.pickscore as pickscore_module

    async def fake_hps_rm(args, samples):
        await asyncio.sleep(0.01)
        return [10.0]

    async def fake_pickscore_rm(args, samples):
        return [20.0]

    monkeypatch.setattr(hps_module, "hps_rm", fake_hps_rm)
    monkeypatch.setattr(pickscore_module, "pickscore_rm", fake_pickscore_rm)
    args = Namespace(custom_rm_path=None, rm_type=None)
    samples = [
        Sample(metadata={"rm_type": "hps"}),
        Sample(metadata={"rm_type": "pickscore"}),
    ]

    rewards = asyncio.run(batched_async_rm(args, samples))

    assert rewards == [10.0, 20.0]
