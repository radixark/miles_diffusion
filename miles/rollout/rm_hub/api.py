"""OpenAI-compatible image rewards."""

from __future__ import annotations

import base64
import io
import json
import math
import os
from collections.abc import Sequence

import torch
from PIL import Image

from miles.utils.api_rm_config import ApiRewardConfig
from miles.utils.misc import SingletonMeta
from miles.utils.processing_utils import generated_output_to_rgb_hwc_uint8_frames
from miles.utils.types import Sample

from .core import AsyncRewardActorPool, record_reward_queue_depth

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


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse_score(content: str, config: ApiRewardConfig) -> float:
    score = json.loads(content)["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("Reward score must be a finite number")
    if not config.score_min <= score <= config.score_max:
        raise ValueError(f"Reward score must be in [{config.score_min}, {config.score_max}]")
    return float(score)


class OpenAIImageScorer:
    """Score prompt/image pairs using the OpenAI-compatible Chat Completions API."""

    def __init__(self, config: ApiRewardConfig):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            api_key=os.environ[config.api_key_env],
            base_url=config.base_url,
            timeout=config.timeout_s,
            max_retries=2,
        )

    def __call__(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> list[float]:
        scores = []
        for prompt, image in zip(prompts, images, strict=True):
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.config.prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": _encode_image(image)}},
                        ],
                    },
                ],
                response_format=_RESPONSE_FORMAT,
            )
            scores.append(_parse_score(response.choices[0].message.content, self.config))
        return scores


class ApiRewardActor:
    def __init__(self, *, config: ApiRewardConfig) -> None:
        self.scorer = OpenAIImageScorer(config)

    def score_batch(self, outputs: list[torch.Tensor], prompts: list[str]) -> list[float]:
        images = []
        for output in outputs:
            (frame,) = generated_output_to_rgb_hwc_uint8_frames(output, None, round_normalized=True)
            images.append(Image.fromarray(frame))
        return self.scorer(prompts, images)


class AsyncApiRewardPool(AsyncRewardActorPool, metaclass=SingletonMeta):
    """API reward pool with one zero-GPU actor handling concurrent HTTP requests."""

    def __init__(self, args) -> None:
        config = args._api_rm_config
        if config is None:
            raise ValueError("API reward requires --api-rm-config.")
        super().__init__(
            actor_cls=ApiRewardActor,
            actor_kwargs={"config": config},
            num_workers=1,
            batch_size=1,
            num_gpus_per_worker=0,
            colocate=False,
            name="api",
            actor_max_concurrency=config.max_concurrency,
        )


async def api_rm(args, samples: Sequence[Sample], **kwargs) -> list[float]:
    pool = AsyncApiRewardPool(args)
    scores, max_queue_depth = await pool.score([s.generated_output for s in samples], [s.prompt for s in samples])
    record_reward_queue_depth(samples, "api", max_queue_depth)
    return scores
