"""DOVER keeps video boundaries, calibrates named branches, and uses the shared reward plumbing."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from unittest.mock import AsyncMock

import pytest
import torch

import miles.rollout.rm_hub.dover as dover_module
import miles.rollout.rm_hub.pickscore as pickscore_module
from miles.rollout.rm_hub import batched_async_rm, create_colocated_reward_pools
from miles.rollout.rm_hub.dover import DOVERScorer, _rescale_scores
from miles.rollout.rm_hub.weighted_mixture_rm import weighted_mixture_rm
from miles.utils.types import Sample


def test_calibration_preserves_branch_identity_and_fuses_before_sigmoid():
    scores = _rescale_scores(
        {"technical": torch.tensor([0.1107, 0.18425]), "aesthetic": torch.tensor([-0.08285, -0.15833])}
    )

    assert scores["technical"].tolist() == pytest.approx([0.5, 0.73105858])
    assert scores["aesthetic"].tolist() == pytest.approx([0.5, 0.11920292])
    assert scores["overall"].tolist() == pytest.approx([0.5, 0.45789991])


def test_batch_reduction_averages_clips_within_each_video():
    scorer = DOVERScorer.__new__(DOVERScorer)
    torch.nn.Module.__init__(scorer)
    scorer.device = torch.device("cpu")
    scorer.score_type = "technical"
    scorer.preprocess = lambda value: {"technical": value.repeat(3), "aesthetic": value.repeat(1)}

    def model(views, **kwargs):
        assert views["technical"].tolist() == [1, 1, 1, 2, 2, 2]
        assert views["aesthetic"].tolist() == [1, 2]
        return [torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), torch.tensor([0.0, 0.0])]

    scorer.model = model
    expected = torch.sigmoid((torch.tensor([0.2, 0.5]) - 0.1107) / 0.07355)
    assert scorer([torch.tensor(1), torch.tensor(2)]) == pytest.approx(expected.tolist())


@pytest.mark.asyncio
async def test_dover_passes_video_tensors_and_records_its_queue(monkeypatch):
    pool = AsyncMock()
    pool.score.return_value = ([0.7], 2)
    monkeypatch.setattr(dover_module, "AsyncDOVERPool", lambda args: pool)
    sample = Sample(prompt="p", generated_output=torch.zeros(3, 4, 8, 8), reward_max_queue_depth={"pickscore": 1})

    assert await dover_module.dover_rm(Namespace(), [sample]) == [0.7]
    pool.score.assert_awaited_once_with([sample.generated_output], ["p"])
    assert sample.reward_max_queue_depth == {"pickscore": 1, "dover": 2.0}


@pytest.mark.asyncio
async def test_batch_dispatch_and_metadata_override_preserve_order(monkeypatch):
    dover_rm = AsyncMock(return_value=[0.2, 0.8])
    monkeypatch.setattr(dover_module, "dover_rm", dover_rm)
    args = Namespace(custom_rm_path=None, rm_type="dover")
    samples = [Sample(), Sample()]
    assert await batched_async_rm(args, samples) == [0.2, 0.8]
    dover_rm.assert_awaited_once_with(args, samples)

    dover_rm.return_value = [0.2]
    monkeypatch.setattr(pickscore_module, "pickscore_rm", AsyncMock(return_value=[0.9]))
    samples[1].metadata = {"rm_type": "pickscore"}
    assert await batched_async_rm(args, samples) == [0.2, 0.9]


def test_colocated_dover_pool_uses_the_managers_shared_slots(monkeypatch):
    calls = []
    pool = object()
    monkeypatch.setattr(dover_module, "AsyncDOVERPool", lambda args, **kwargs: calls.append(kwargs) or pool)
    args = Namespace(pickscore_reward_colocate=False, hps_reward_colocate=False, dover_reward_colocate=True)
    placement_group, slots = object(), object()

    assert create_colocated_reward_pools(args, placement_group, slots) == [pool]
    assert calls == [{"placement_group": placement_group, "slots": slots}]


@pytest.mark.asyncio
async def test_dover_is_available_in_the_shipped_mixture(monkeypatch):
    pool = AsyncMock()
    pool.score.return_value = ([0.2, 0.8], 0)
    monkeypatch.setattr(dover_module, "AsyncDOVERPool", lambda args: pool)
    args = Namespace(custom_rm_args="dover=0.5", reward_key="weighted")
    samples = [Sample(), Sample()]

    assert await weighted_mixture_rm(args, samples) == [
        {"dover": 0.2, "weighted": 0.1},
        {"dover": 0.8, "weighted": 0.4},
    ]
