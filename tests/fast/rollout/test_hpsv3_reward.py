"""HPSv3 reward plumbing that does not need the 7B model; parity with the official scorer is
tests/fast-gpu/test_hpsv3_alignment.py.

Mental model (one image sample):

    sample.generated_output (float CFHW, F == 1)
        -> HPSv3RewardActor.score_batch : round -> uint8 -> PIL
        -> HPSv3Scorer                  : smart_resize to ~256 visual tokens -> Qwen2-VL -> rm_head at <|Reward|> -> mu

Covered: the actor rounds when quantising (1) and refuses videos (2); the official checkpoint keys
split onto Qwen2VLModel and rm_head, dropping only lm_head (3); smart_resize keeps the official
token budget (4); hpsv3_rm hands the raw tensor to its pool and records its backlog (5);
batched_async_rm takes the hpsv3 fast path (6); weighted_mixture_rm accepts hpsv3 (7).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

import miles.rollout.rm_hub.hpsv3 as hpsv3_module
from miles.rollout.rm_hub import batched_async_rm
from miles.rollout.rm_hub.hpsv3 import HPSv3RewardActor, split_official_state_dict
from miles.rollout.rm_hub.weighted_mixture_rm import parse_weights
from miles.utils.types import Sample

# one pixel row [0, 0.5, 1] per channel; rounding gives 128 in the middle, truncation 127
_ROW = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 1, 1, 3).repeat(3, 1, 1, 1)


def _actor_with(scorer) -> HPSv3RewardActor:
    actor = HPSv3RewardActor.__new__(HPSv3RewardActor)
    actor.scorer = scorer
    return actor


def test_actor_rounds_when_quantising():
    """The official scorer reads PNGs, which diffusers writes with rounding."""
    actor = _actor_with(lambda prompts, images: [float(np.asarray(image)[0, 1, 0]) for image in images])

    assert actor.score_batch([_ROW], ["prompt"]) == [128.0]


def test_actor_rejects_videos():
    actor = _actor_with(lambda prompts, images: [0.0] * len(images))

    with pytest.raises(ValueError, match="images only, got 2 frames"):
        actor.score_batch([torch.cat([_ROW, _ROW], dim=1)], ["prompt"])


def test_official_keys_split_onto_model_and_head():
    model_state, head_state = split_official_state_dict(
        {
            "model.layers.0.mlp.up_proj.weight": 1,
            "model.embed_tokens.weight": 2,
            "visual.blocks.0.attn.qkv.weight": 3,
            "rm_head.5.bias": 4,
            "lm_head.weight": 5,
        }
    )

    assert model_state == {
        "language_model.layers.0.mlp.up_proj.weight": 1,
        "language_model.embed_tokens.weight": 2,
        "visual.blocks.0.attn.qkv.weight": 3,
    }
    assert head_state == {"5.bias": 4}
    with pytest.raises(KeyError, match="score_head"):
        split_official_state_dict({"score_head.weight": 0})


@pytest.mark.parametrize(
    "height, width, expected",
    [(512, 512, (448, 448)), (1024, 1024, (448, 448)), (768, 1024, (364, 504)), (1344, 768, (588, 336))],
)
def test_smart_resize_keeps_the_official_token_budget(height, width, expected):
    assert hpsv3_module._smart_resize(height, width) == expected


@pytest.mark.asyncio
async def test_hpsv3_rm_hands_the_raw_tensor_to_its_pool(monkeypatch):
    pool = AsyncMock()
    pool.score.return_value = ([7.5], 3)
    monkeypatch.setattr(hpsv3_module, "AsyncHPSv3Pool", lambda args: pool)
    sample = Sample(prompt="prompt", generated_output=_ROW)

    assert await hpsv3_module.hpsv3_rm(Namespace(), [sample]) == [7.5]
    (outputs, prompts), _ = pool.score.await_args
    assert torch.equal(outputs[0], _ROW) and prompts == ["prompt"]
    assert sample.reward_max_queue_depth == {"hpsv3": 3.0}


@pytest.mark.asyncio
async def test_batched_async_rm_uses_hpsv3_batch_fast_path(monkeypatch):
    hpsv3_rm = AsyncMock(return_value=[3.0, 1.0])
    monkeypatch.setattr(hpsv3_module, "hpsv3_rm", hpsv3_rm)
    args = Namespace(custom_rm_path=None, rm_type="hpsv3")
    samples = [Sample(index=3), Sample(index=1)]

    assert await batched_async_rm(args, samples) == [3.0, 1.0]
    hpsv3_rm.assert_awaited_once_with(args, samples)


def test_weighted_mixture_accepts_hpsv3():
    assert parse_weights("hpsv3=0.5,hps=0.5") == [("hpsv3", 0.5), ("hps", 0.5)]
