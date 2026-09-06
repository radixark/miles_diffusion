from __future__ import annotations

from collections.abc import Sequence

import ray
import torch
from PIL import Image

from miles.utils.misc import SingletonMeta
from miles.utils.processing_utils import generated_output_to_rgb_hwc_uint8_frames, sample_frame_indices
from miles.utils.types import Sample

from .core import AsyncRewardActorPool


def _feature_tensor(features):
    # transformers <5.0 returns a plain tensor; >=5.0 returns BaseModelOutputWithPooling.
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output") and isinstance(features.pooler_output, torch.Tensor):
        return features.pooler_output
    raise TypeError(f"Cannot extract embedding tensor from {type(features)!r}")


class PickScoreScorer(torch.nn.Module):
    """CLIP PickScore for (prompt, image) pairs; raw logits scaled to ~0-1."""

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
        self.model = CLIPModel.from_pretrained(model_path).eval().to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> list[float]:
        image_inputs = self.processor(images=list(images), return_tensors="pt", padding=True)
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        if "pixel_values" in image_inputs:
            image_inputs["pixel_values"] = image_inputs["pixel_values"].float()

        text_inputs = self.processor(
            text=list(prompts),
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

        scores = self.model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)
        # Flow-Factory convention: scale raw PickScore logits (~0-26) to ~0-1.
        scores = scores.float() / 26.0
        return [float(score) for score in scores.detach().cpu()]


class PickScoreRewardActor:
    def __init__(
        self,
        *,
        processor_path: str,
        model_path: str,
        frames_per_forward: int,
    ) -> None:
        self.frames_per_forward = frames_per_forward
        gpu_ids = ray.get_gpu_ids()
        use_cuda = bool(gpu_ids) and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        device = "cuda" if use_cuda else "cpu"
        self.scorer = PickScoreScorer(
            device=device,
            processor_path=processor_path,
            model_path=model_path,
        )

    def score_batch(self, outputs: list[torch.Tensor], prompts: list[str]) -> list[float]:
        # a video sample is scored on every frame it arrives with and gets their mean
        images, frame_counts = [], []
        for output in outputs:
            frames = generated_output_to_rgb_hwc_uint8_frames(output, None)
            images.extend(Image.fromarray(frame) for frame in frames)
            frame_counts.append(len(frames))
        flat_prompts = [p for p, n in zip(prompts, frame_counts, strict=True) for _ in range(n)]
        # forwards see --pickscore-batch-size frames, the chunking the e2e standards were recorded with
        flat_scores = []
        for start in range(0, len(images), self.frames_per_forward):
            end = start + self.frames_per_forward
            flat_scores.extend(self.scorer(flat_prompts[start:end], images[start:end]))
        scores, offset = [], 0
        for count in frame_counts:
            scores.append(float(sum(flat_scores[offset : offset + count]) / count))
            offset += count
        return scores


class AsyncPickScorePool(AsyncRewardActorPool, metaclass=SingletonMeta):
    """Ray actor pool for GPU PickScore reward inference."""

    def __init__(self, args, placement_group=None, slots=None) -> None:
        super().__init__(
            actor_cls=PickScoreRewardActor,
            actor_kwargs={
                "processor_path": args.pickscore_processor_path,
                "model_path": args.pickscore_model_path,
                "frames_per_forward": args.pickscore_batch_size,
            },
            num_workers=args.pickscore_num_workers,
            batch_size=args.pickscore_batch_size,
            num_gpus_per_worker=args.pickscore_num_gpus_per_worker,
            colocate=args.pickscore_reward_colocate,
            name="pickscore",
            placement_group=placement_group,
            slots=slots,
        )


async def pickscore_rm(args, samples: Sequence[Sample]) -> list[float]:
    pool = AsyncPickScorePool(args)
    # pick --pickscore-num-frames here so only the scored frames cross the object store
    outputs = [
        s.generated_output[:, sample_frame_indices(s.generated_output.shape[1], args.pickscore_num_frames)]
        for s in samples
    ]
    scores, max_queue_depth = await pool.score(outputs, [s.prompt for s in samples])
    for sample in samples:
        sample.reward_max_queue_depth = float(max_queue_depth)
    return scores
