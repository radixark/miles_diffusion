"""OpenAI-compatible image API rewards."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from miles.utils.api_rm_config import OpenAIImageRewardConfig
from miles.utils.processing_utils import encode_image_as_png_data_url, generated_output_to_rgb_hwc_uint8_frames
from miles.utils.types import Sample

from .api import ApiRewardActor, AsyncApiRewardPool
from .core import record_reward_queue_depth

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "image_reward",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
            "additionalProperties": False,
        },
    },
}


class OpenAIImageRewardActor(ApiRewardActor):
    """Score prompt/image pairs using the OpenAI-compatible Chat Completions API."""

    def __init__(self, **kwargs) -> None:
        from openai import OpenAI

        self.config = OpenAIImageRewardConfig(**kwargs)
        self.client = OpenAI(
            api_key=os.environ[self.config.api_key_env],
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
            max_retries=2,
        )

    def build_request(self, output: torch.Tensor, prompt: str) -> dict[str, Any]:
        (frame,) = generated_output_to_rgb_hwc_uint8_frames(output, None, round_normalized=True)
        image = Image.fromarray(frame)
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.config.prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": encode_image_as_png_data_url(image)}},
                    ],
                },
            ],
            "response_format": _RESPONSE_FORMAT,
        }

    def send_request(self, request: dict[str, Any]) -> ChatCompletion:
        return self.client.chat.completions.create(**request)

    def parse_response(self, response: ChatCompletion) -> float:
        score = json.loads(response.choices[0].message.content)["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Reward score must be a finite number")
        if not self.config.score_min <= score <= self.config.score_max:
            raise ValueError(f"Reward score must be in [{self.config.score_min}, {self.config.score_max}]")
        return float(score)


class AsyncOpenAIPool(AsyncApiRewardPool):
    """Ray pool for OpenAI-compatible image rewards."""

    name = "openai_api"
    actor_base_cls = OpenAIImageRewardActor


async def openai_api_rm(args, samples: Sequence[Sample], **kwargs) -> list[float]:
    pool = AsyncOpenAIPool(args)
    scores, max_queue_depth = await pool.score([s.generated_output for s in samples], [s.prompt for s in samples])
    record_reward_queue_depth(samples, "openai_api", max_queue_depth)
    return scores
