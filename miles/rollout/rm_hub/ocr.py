import logging
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image

from miles.utils.misc import SingletonMeta
from miles.utils.processing_utils import generated_output_to_rgb_hwc_uint8_frames
from miles.utils.types import Sample

from .core import AsyncRewardActorPool, record_reward_queue_depth

logger = logging.getLogger(__name__)


def _init_paddleocr(use_gpu: bool):
    # actor-only dependency: the manager imports this module just to dispatch
    from paddleocr import PaddleOCR

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
        from Levenshtein import distance

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


class OcrRewardActor:
    def __init__(self, use_gpu: bool = False):
        self.scorer = OcrScorer(use_gpu=use_gpu)

    def score_batch(self, outputs: list[torch.Tensor], prompts: list[str]) -> list[float]:
        assert len(outputs) == 1, f"OCR scores one image per call, got {len(outputs)}"
        # flow_grpo feeds PaddleOCR rounded RGB (not BGR) uint8; matching it keeps the reward bit-identical
        (image,) = generated_output_to_rgb_hwc_uint8_frames(outputs[0], None, round_normalized=True)
        return self.scorer([image], prompts)


class AsyncOcrPool(AsyncRewardActorPool, metaclass=SingletonMeta):
    """Ray actor pool for CPU PaddleOCR reward inference."""

    def __init__(self, args) -> None:
        super().__init__(
            actor_cls=OcrRewardActor,
            actor_kwargs={"use_gpu": False},
            num_workers=args.ocr_num_workers,
            batch_size=1,
            num_gpus_per_worker=0,
            colocate=False,
            name="ocr",
        )


async def ocr_rm(args, samples: Sequence[Sample]) -> list[float]:
    pool = AsyncOcrPool(args)
    scores, max_queue_depth = await pool.score([s.generated_output for s in samples], [s.prompt for s in samples])
    record_reward_queue_depth(samples, "ocr", max_queue_depth)
    return scores
