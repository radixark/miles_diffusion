import asyncio
import logging
import numpy as np
import ray
import torch
import argparse
from paddleocr import PaddleOCR
from Levenshtein import distance
from PIL import Image
from typing import List, Union
from miles.utils.misc import SingletonMeta
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

def _init_paddleocr(use_gpu: bool) -> PaddleOCR:
    def make_ocr() -> PaddleOCR:
        return PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=use_gpu,
            show_log=False,
        )
    return make_ocr()

class OcrScorer:
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        self.ocr = _init_paddleocr(use_gpu)

    @torch.no_grad()
    def __call__(self, 
                images: Union[List[Image.Image], List[np.ndarray]], 
                prompts: List[str]) -> List[float]:
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward tensor (CPU)
        """
        prompts = [prompt.split('"')[1] for prompt in prompts]
        rewards = []
        # Ensure input lengths are consistent
        assert len(images) == len(prompts), f"Images({len(images)}) and prompts({len(prompts)}) must have the same length"
        for img, prompt in zip(images, prompts):
            # Convert image format
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            try:
                # OCR recognition
                result = self.ocr.ocr(img, cls=False)
                # Extract recognized text (handle possible multi-line results)
                recognized_text = ''.join([res[1][0] if res[1][1] > 0 else '' for res in result[0]]) if result[0] else ''
                
                recognized_text = recognized_text.replace(' ', '').lower()
                prompt = prompt.replace(' ', '').lower()
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt)
                # Recognized many unrelated characters, only add one character penalty
                if dist > len(prompt):
                    dist = len(prompt)
                
            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f"OCR processing failed: {str(e)}")
                dist = len(prompt)  # Maximum penalty
            reward = 1-dist/(len(prompt))
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
        num_workers = int(getattr(args, "ocr_num_workers", 4) or 4)
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
    """``generated_output``: ``[C, F, H, W]``; use time index 0 (``F==1`` for still images)."""
    t = sample.generated_output
    if t is None:
        raise ValueError("generated_output is None")
    if t.ndim != 4:
        raise ValueError(f"generated_output must be 4D [C, F, H, W], got {tuple(t.shape)}")
    frame_chw = t[:, 0, :, :]
    hwc = frame_chw.numpy().transpose(1, 2, 0)
    if float(hwc.max()) <= 1.0 + 1e-3:
        out = (hwc * 255.0).clip(0, 255).astype(np.uint8)
    else:
        out = hwc.clip(0, 255).astype(np.uint8)
    # VAE outputs RGB; PaddleOCR (OpenCV-based) expects BGR.
    out = out[:, :, ::-1]
    return out

async def ocr_rm(args, sample: Sample):
    pool = AsyncOcrPool(args)
    image = _rgb_hwc_from_generated(sample)
    score = await pool.score(image, sample.prompt)
    return score

if __name__ == "__main__":
    args = argparse.Namespace(ocr_num_workers=4)
    pil_image = Image.open("imgs/miles_logo.png").convert("RGB")
    image_tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).unsqueeze(1).float()
    sample = Sample(
        prompt="A logo of Miles saying \"Miles\"",
        generated_output=image_tensor,
    )
    img = np.array(pil_image)
    print(OcrScorer()([img], [sample.prompt])[0])