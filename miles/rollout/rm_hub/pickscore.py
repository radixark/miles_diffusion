from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import numpy as np
import ray
import torch
from PIL import Image

from miles.utils.misc import SingletonMeta
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def sample_frame_indices(num_total_frames: int, num_frames: int) -> list[int]:
    if num_total_frames <= 0:
        raise ValueError(f"video has no frames: {num_total_frames}")
    if num_total_frames <= num_frames:
        return list(range(num_total_frames))
    if num_frames == 1:
        return [num_total_frames // 2]
    step = (num_total_frames - 1) / (num_frames - 1)
    return [int(round(i * step)) for i in range(num_frames)]


def generated_output_to_fchw(t: torch.Tensor) -> torch.Tensor:
    """Return ``[F, C, H, W]`` float tensor in ``[0, 1]``."""
    t = t.detach().cpu().float()
    if t.ndim == 3:
        if t.shape[0] not in (1, 3):
            raise ValueError(f"expected [C, H, W] with C in {{1, 3}}, got {tuple(t.shape)}")
        t = t.unsqueeze(0)
    elif t.ndim == 4:
        if t.shape[-1] in (1, 3):
            t = t.permute(0, 3, 1, 2)
        elif t.shape[0] in (1, 3):
            t = t.permute(1, 0, 2, 3)
        elif t.shape[1] not in (1, 3):
            raise ValueError(f"unrecognized 4D video layout: {tuple(t.shape)}")
    elif t.ndim == 5:
        if t.shape[0] == 1 and t.shape[-1] in (1, 3):
            t = t[0].permute(0, 3, 1, 2)
        else:
            raise ValueError(f"unrecognized 5D video layout: {tuple(t.shape)}")
    else:
        raise ValueError(f"generated_output must be 3D–5D, got {tuple(t.shape)}")

    if float(t.max()) > 1.0 + 1e-3:
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def fchw_frame_to_hwc_uint8(frame_chw: torch.Tensor) -> np.ndarray:
    hwc = frame_chw.numpy().transpose(1, 2, 0)
    if float(hwc.max()) <= 1.0 + 1e-3:
        hwc = hwc * 255.0
    return np.ascontiguousarray(hwc.clip(0, 255).astype(np.uint8))


def first_frame_for_wandb(t: torch.Tensor) -> np.ndarray | None:
    """First frame as HWC uint8 for wandb logging; None if layout is unsupported."""
    try:
        return fchw_frame_to_hwc_uint8(generated_output_to_fchw(t)[0])
    except (ValueError, TypeError):
        if t.ndim != 4:
            return None
        frame = t[:, 0, :, :].float().cpu().numpy().transpose(1, 2, 0)
        if float(frame.max()) <= 1.0 + 1e-3:
            frame = frame * 255.0
        return np.clip(frame, 0, 255).astype(np.uint8)


def fchw_to_pil_frames(video_fchw: torch.Tensor, frame_indices: Sequence[int]) -> list[Image.Image]:
    return [Image.fromarray(fchw_frame_to_hwc_uint8(video_fchw[idx])) for idx in frame_indices]


def is_video_generated_output(t: torch.Tensor) -> bool:
    """True when output carries multiple temporal frames (LTX / sglang video)."""
    fchw = generated_output_to_fchw(t)
    return fchw.shape[0] > 1


def _feature_tensor(features):
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output"):
        pooled = features.pooler_output
        if isinstance(pooled, torch.Tensor):
            return pooled
    for attr in ("image_embeds", "text_embeds"):
        value = getattr(features, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(features, tuple):
        for item in reversed(features):
            if isinstance(item, torch.Tensor) and item.ndim == 2:
                return item
        raise TypeError(f"No 2-D tensor in model output tuple (len={len(features)})")
    raise TypeError(f"Cannot extract embedding tensor from {type(features)!r}")


def _sample_to_rgb_hwc_uint8(sample: Sample) -> np.ndarray:
    fchw = generated_output_to_fchw(sample.generated_output)
    return fchw_frame_to_hwc_uint8(fchw[fchw.shape[0] // 2])


class PickScoreScorer(torch.nn.Module):
    """PickScore for static images (SD3 / single-frame outputs)."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        processor_path: str,
        model_path: str,
    ) -> None:
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> list[float]:
        image_inputs = self.processor(
            images=list(images),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = self.processor(
            text=list(prompts),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {key: value.to(device=self.device) for key, value in image_inputs.items()}
        text_inputs = {key: value.to(device=self.device) for key, value in text_inputs.items()}

        image_embs = _feature_tensor(self.model.get_image_features(**image_inputs))
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        text_embs = _feature_tensor(self.model.get_text_features(**text_inputs))
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)
        scores = scores.diag() / 26.0
        return [float(score) for score in scores.detach().cpu()]


class VideoPickScoreScorer(torch.nn.Module):
    """Multi-frame PickScore for LTX video (matches trainer-rollout / verl-omni)."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        processor_path: str,
        model_path: str,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.device = torch.device(device)
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = AutoModel.from_pretrained(model_path).eval().to(device=self.device, dtype=dtype)

    @torch.no_grad()
    def score_videos(
        self,
        videos_fchw: Sequence[torch.Tensor],
        prompts: Sequence[str],
        *,
        num_frames: int,
        batch_size: int,
    ) -> list[float]:
        if len(videos_fchw) != len(prompts):
            raise ValueError(f"#videos ({len(videos_fchw)}) != #prompts ({len(prompts)})")

        flat_images: list[Image.Image] = []
        flat_prompts: list[str] = []
        per_sample_counts: list[int] = []

        for video_fchw, prompt in zip(videos_fchw, prompts, strict=True):
            frame_indices = sample_frame_indices(video_fchw.shape[0], num_frames)
            per_sample_counts.append(len(frame_indices))
            flat_images.extend(fchw_to_pil_frames(video_fchw, frame_indices))
            flat_prompts.extend([prompt] * len(frame_indices))

        logit_scale = self.model.logit_scale.exp()
        flat_scores: list[torch.Tensor] = []
        for start in range(0, len(flat_images), batch_size):
            image_chunk = flat_images[start : start + batch_size]
            prompt_chunk = flat_prompts[start : start + batch_size]

            image_inputs = self.processor(images=image_chunk, return_tensors="pt", padding=True)
            image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
            if "pixel_values" in image_inputs:
                image_inputs["pixel_values"] = image_inputs["pixel_values"].to(self.dtype)

            text_inputs = self.processor(
                text=prompt_chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

            image_embs = _feature_tensor(self.model.get_image_features(**image_inputs))
            image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

            text_embs = _feature_tensor(self.model.get_text_features(**text_inputs))
            text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

            chunk_scores = logit_scale * (text_embs * image_embs).sum(dim=-1)
            flat_scores.append(chunk_scores.float())

        all_scores = torch.cat(flat_scores, dim=0)
        rewards: list[float] = []
        offset = 0
        for count in per_sample_counts:
            rewards.append(float(all_scores[offset : offset + count].mean()))
            offset += count
        return rewards


@ray.remote
class PickScoreRewardActor:
    def __init__(
        self,
        *,
        processor_path: str,
        model_path: str,
    ) -> None:
        gpu_ids = ray.get_gpu_ids()
        use_cuda = bool(gpu_ids) and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        device = "cuda" if use_cuda else "cpu"
        self.device = device
        self.image_scorer = PickScoreScorer(
            device=device,
            processor_path=processor_path,
            model_path=model_path,
        )
        self.video_scorer = VideoPickScoreScorer(
            device=device,
            processor_path=processor_path,
            model_path=model_path,
            dtype=torch.float16 if use_cuda else torch.float32,
        )

    def score_batch(self, images: list[np.ndarray], prompts: list[str]) -> list[float]:
        pil_images = [Image.fromarray(image) for image in images]
        return self.image_scorer(prompts, pil_images)

    def score_videos_batch(
        self,
        videos_fchw: list[torch.Tensor],
        prompts: list[str],
        *,
        num_frames: int,
        batch_size: int,
    ) -> list[float]:
        return self.video_scorer.score_videos(
            videos_fchw,
            prompts,
            num_frames=num_frames,
            batch_size=batch_size,
        )


class AsyncPickScorePool(metaclass=SingletonMeta):
    """Ray actor pool for GPU PickScore reward inference."""

    def __init__(self, args) -> None:
        num_workers = args.pickscore_num_workers
        num_gpus_per_worker = args.pickscore_num_gpus_per_worker
        self._batch_size = args.pickscore_batch_size
        self._num_frames = int(getattr(args, "pickscore_num_frames", 3) or 3)
        self._actors = [
            PickScoreRewardActor.options(
                num_cpus=1,
                num_gpus=num_gpus_per_worker,
                scheduling_strategy="DEFAULT",
            ).remote(
                processor_path=args.pickscore_processor_path,
                model_path=args.pickscore_model_path,
            )
            for _ in range(num_workers)
        ]
        self._round_robin_index = 0
        logger.info(
            "Initialized PickScore actor pool with %d workers, %.3f GPUs/worker, batch_size=%d.",
            num_workers,
            num_gpus_per_worker,
            self._batch_size,
        )

    def _next_actor(self):
        i = self._round_robin_index % len(self._actors)
        self._round_robin_index += 1
        return self._actors[i]

    async def score(self, images: list[np.ndarray], prompts: list[str]) -> list[float]:
        refs = []
        for start in range(0, len(images), self._batch_size):
            end = start + self._batch_size
            refs.append(self._next_actor().score_batch.remote(images[start:end], prompts[start:end]))

        loop = asyncio.get_running_loop()
        chunked_scores = await loop.run_in_executor(None, ray.get, refs)
        return [float(score) for chunk in chunked_scores for score in chunk]

    async def score_videos(
        self,
        videos_fchw: list[torch.Tensor],
        prompts: list[str],
    ) -> list[float]:
        actor = self._next_actor()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            ray.get,
            actor.score_videos_batch.remote(
                videos_fchw,
                prompts,
                num_frames=self._num_frames,
                batch_size=self._batch_size,
            ),
        )


async def pickscore_rm(args, samples: Sequence[Sample]) -> list[float]:
    pool = AsyncPickScorePool(args)
    if any(is_video_generated_output(sample.generated_output) for sample in samples):
        videos = [generated_output_to_fchw(sample.generated_output) for sample in samples]
        prompts = [sample.prompt for sample in samples]
        return await pool.score_videos(videos, prompts)

    images = [_sample_to_rgb_hwc_uint8(sample) for sample in samples]
    prompts = [sample.prompt for sample in samples]
    return await pool.score(images, prompts)
