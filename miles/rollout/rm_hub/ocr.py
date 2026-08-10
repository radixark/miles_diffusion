import asyncio
import logging

import numpy as np
import ray
import torch
from Levenshtein import distance
from paddleocr import PaddleOCR
from PIL import Image

from miles.utils.misc import SingletonMeta
from miles.utils.processing_utils import cfhw_to_fhwc, image_or_video_to_uint8
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _init_paddleocr(use_gpu: bool) -> PaddleOCR:
    return PaddleOCR(
        use_angle_cls=False,
        lang="en",
        use_gpu=use_gpu,
        show_log=False,
    )


class OcrScorer:
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        self.ocr = _init_paddleocr(use_gpu)

    @torch.no_grad()
    def __call__(self, images: list[Image.Image] | list[np.ndarray], prompts: list[str]) -> list[float]:
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward tensor (CPU)
        """
        prompts = [prompt.split('"')[1] for prompt in prompts]
        rewards = []
        # Ensure input lengths are consistent
        assert len(images) == len(
            prompts
        ), f"Images({len(images)}) and prompts({len(prompts)}) must have the same length"
        for img, prompt in zip(images, prompts, strict=False):
            # Convert image format
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                # OCR recognition
                result = self.ocr.ocr(img, cls=False)
                # Extract recognized text (handle possible multi-line results)
                recognized_text = (
                    "".join([res[1][0] if res[1][1] > 0 else "" for res in result[0]]) if result[0] else ""
                )

                recognized_text = recognized_text.replace(" ", "").lower()
                prompt = prompt.replace(" ", "").lower()
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt)
                # Recognized many unrelated characters, only add one character penalty
                if dist > len(prompt):
                    dist = len(prompt)

            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                logger.warning(f"OCR processing failed: {e}")
                dist = len(prompt)  # Maximum penalty
            reward = 1 - dist / (len(prompt))
            rewards.append(reward)

        return rewards


@ray.remote
class OcrRewardActor:
    def __init__(self, use_gpu: bool = False):
        self.scorer = OcrScorer(use_gpu=use_gpu)

    def score_single(self, image: np.ndarray, prompt: str) -> float:
        return self.scorer([image], [prompt])[0]


class AsyncOcrPool(metaclass=SingletonMeta):
    """Ray-backed round-robin pool of :class:`OcrRewardActor` (same lifetime pattern as ``GenerateState``)."""

    def __init__(self, args) -> None:
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. OCR RM requires Ray for OcrRewardActor.")
        num_workers = args.ocr_num_workers
        if num_workers <= 0:
            raise ValueError(f"ocr_num_workers must be > 0, got {num_workers}")
        self._actors = [OcrRewardActor.options(num_cpus=1).remote(use_gpu=False) for _ in range(num_workers)]
        self._round_robin_index = 0
        logger.info("Initialized OCR reward actor pool with %d workers.", num_workers)

    def _next_actor(self):
        i = self._round_robin_index % len(self._actors)
        self._round_robin_index += 1
        return self._actors[i]

    async def score(self, image: np.ndarray, prompt: str) -> float:
        ref = self._next_actor().score_single.remote(image, prompt)
        loop = asyncio.get_running_loop()
        return float(await loop.run_in_executor(None, ray.get, ref))


def _rgb_hwc_from_generated(sample: Sample) -> np.ndarray:
    """``generated_output``: ``[C, F, H, W]`` or ``[C, H, W]``; use frame index 0.

    Accepts both the local-rollout format ``[C, F, H, W]`` (video frames) and
    the sglang-diffusion SD3 format ``[C, H, W]`` (static image, no frame dim).

    Feeds PaddleOCR the exact same ``(RGB, uint8 HWC)`` array that flow_grpo's
    ``ocr_score`` wrapper does — `(images * 255).round().clamp(0,255).to(uint8)`
    then ``transpose(0, 2, 3, 1)``, no channel swap. PaddleOCR's OpenCV stack
    would prefer BGR, but flow_grpo trains against the (slightly-off) RGB
    convention, so we match that to keep the reward signal bit-identical.
    """
    t = sample.generated_output
    if t is None:
        raise ValueError("generated_output is None")
    t = t.detach().cpu().float()
    if t.ndim == 3:
        t = t.unsqueeze(1)
    if t.ndim != 4:
        raise ValueError(f"generated_output must be 3D [C, H, W] or 4D [C, F, H, W], got {tuple(t.shape)}")
    fhwc = cfhw_to_fhwc(t)
    if fhwc.shape[0] != 1:
        raise ValueError(f"generated_output frame dim F must be 1 for image models, got F={fhwc.shape[0]}")
    return image_or_video_to_uint8(fhwc[0], round_normalized=True).numpy()


async def ocr_rm(args, sample: Sample):
    pool = AsyncOcrPool(args)
    image = _rgb_hwc_from_generated(sample)
    score = await pool.score(image, sample.prompt)
    return score
