import asyncio
import io
import json
import logging
import threading
from collections.abc import Iterable

import numpy as np
import pybase64
import ray
from PIL import Image

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_POOL_LOCK = threading.Lock()
_OCR_POOL = None

_IMAGE_CANDIDATE_KEYS = (
    "image",
    "image_base64",
    "image_b64",
    "generated_image",
    "generated_image_base64",
    "output_image",
    "output_image_base64",
    "diffusion_image",
    "diffusion_image_base64",
)


def _decode_image_base64(value: str) -> np.ndarray:
    text = value.strip().strip('"').strip("'")
    if "base64," in text:
        text = text.split("base64,", 1)[1]
    text = "".join(text.split())
    if not text:
        raise ValueError("empty base64 image payload")
    pad = len(text) % 4
    if pad:
        text += "=" * (4 - pad)
    data = pybase64.b64decode(text.encode("ascii"), validate=False)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(image)


def _extract_from_json_payload(text: str) -> Iterable[str]:
    try:
        payload = json.loads(text)
    except Exception:
        return []

    if isinstance(payload, dict):
        values = []
        for key in _IMAGE_CANDIDATE_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                values.append(value)
        return values
    return []


def _iter_image_candidates(sample: Sample) -> Iterable[str]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    for key in _IMAGE_CANDIDATE_KEYS:
        value = metadata.get(key)
        if isinstance(value, str):
            yield value

    for key, value in metadata.items():
        if isinstance(key, str) and "image" in key.lower() and isinstance(value, str):
            yield value

    if isinstance(sample.response, str) and sample.response:
        for candidate in _extract_from_json_payload(sample.response):
            yield candidate
        yield sample.response


def _extract_image(sample: Sample) -> np.ndarray:
    for candidate in _iter_image_candidates(sample):
        try:
            return _decode_image_base64(candidate)
        except Exception:
            continue
    raise ValueError("Could not extract a base64 image from sample.response or sample.metadata.")


def _prompt_to_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt, ensure_ascii=False)


@ray.remote
class OcrRewardActor:
    def __init__(self, use_gpu: bool = False):
        from flow_grpo.ocr import OcrScorer

        self.scorer = OcrScorer(use_gpu=use_gpu)

    def score_single(self, image: np.ndarray, prompt: str) -> float:
        rewards = self.scorer([image], [prompt])
        if not rewards:
            return 0.0
        return float(rewards[0])


class AsyncOcrPool:
    def __init__(self, num_workers: int, use_gpu: bool = False):
        if num_workers <= 0:
            raise ValueError(f"ocr_num_workers must be > 0, got {num_workers}")
        self.actors = [OcrRewardActor.options(num_cpus=1).remote(use_gpu=use_gpu) for _ in range(num_workers)]
        self._counter = 0

    def _next_actor(self):
        actor = self.actors[self._counter % len(self.actors)]
        self._counter += 1
        return actor

    async def score(self, image: np.ndarray, prompt: str) -> float:
        actor = self._next_actor()
        ref = actor.score_single.remote(image, prompt)
        loop = asyncio.get_running_loop()
        return float(await loop.run_in_executor(None, ray.get, ref))


def _resolve_num_workers(args) -> int:
    num_workers = int(getattr(args, "ocr_num_workers", 4) or 4)
    if num_workers <= 0:
        raise ValueError(f"ocr_num_workers must be > 0, got {num_workers}")
    return num_workers


def init_ocr_pool(args):
    global _OCR_POOL
    if _OCR_POOL is not None:
        return _OCR_POOL

    with _POOL_LOCK:
        if _OCR_POOL is not None:
            return _OCR_POOL

        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. OCR RM requires Ray runtime for OcrRewardActor.")
        num_workers = _resolve_num_workers(args)
        _OCR_POOL = AsyncOcrPool(num_workers=num_workers, use_gpu=False)
        logger.info("Initialized OCR reward actor pool with %d workers.", num_workers)
        return _OCR_POOL


def _to_reward_dict(score: float) -> dict[str, float]:
    value = float(score)
    return {"ocr": value, "avg": value}


async def ocr_rm(args, sample: Sample):
    pool = init_ocr_pool(args)
    image = _extract_image(sample)
    prompt = _prompt_to_text(sample.prompt)
    score = await pool.score(image, prompt)
    return _to_reward_dict(score)


async def batched_ocr_rm(args, samples: list[Sample]):
    if len(samples) == 0:
        return []

    pool = init_ocr_pool(args)
    coros = []
    for sample in samples:
        image = _extract_image(sample)
        prompt = _prompt_to_text(sample.prompt)
        coros.append(pool.score(image, prompt))

    scores = await asyncio.gather(*coros)
    return [_to_reward_dict(score) for score in scores]
