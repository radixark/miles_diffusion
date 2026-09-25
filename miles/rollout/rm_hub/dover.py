from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import ray
import torch

from miles.utils.misc import SingletonMeta
from miles.utils.processing_utils import image_or_video_to_uint8
from miles.utils.types import Sample

from .core import AsyncRewardActorPool, record_reward_queue_depth

_VIEW_OPTIONS = {
    "technical": {
        "fragments_h": 7,
        "fragments_w": 7,
        "fsize_h": 32,
        "fsize_w": 32,
        "aligned": 32,
    },
    "aesthetic": {"size_h": 224, "size_w": 224},
}


class _DOVERVideoTransform:
    """Adapt DOVER's file-based preprocessing to in-memory CFHW videos."""

    def __init__(self, seed: int = 0) -> None:
        from dover.datasets import UnifiedFrameSampler, get_single_view

        self.seed = seed
        self.get_single_view = get_single_view
        # dover.yml: three technical clips and one temporally fragmented aesthetic clip.
        self.samplers = {
            "technical": UnifiedFrameSampler(32, 3, 2),
            "aesthetic": UnifiedFrameSampler(1, 32, 2),
        }
        self.mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1, 1)
        self.std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1, 1)

    def __call__(self, output: torch.Tensor) -> dict[str, torch.Tensor]:
        if output.ndim != 4 or output.shape[0] != 3 or output.shape[1] < 1:
            raise ValueError(f"DOVER expects a non-empty RGB CFHW video, got {tuple(output.shape)}")

        # Fix crops per video so actor assignment and batch order do not change the reward.
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.seed)
                np.random.seed(self.seed)
                indices = {name: sampler(output.shape[1]) for name, sampler in self.samplers.items()}
                views = {}
                for name, opt in _VIEW_OPTIONS.items():
                    video = image_or_video_to_uint8(output[:, indices[name]].detach().cpu(), round_normalized=True)
                    view = self.get_single_view(video, name, **opt)
                    view = (view - self.mean) / self.std
                    views[name] = view.reshape(3, -1, 32, 224, 224).transpose(0, 1)
                return views
        finally:
            np.random.set_state(numpy_state)


def _rescale_scores(scores: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Official evaluate_a_set_of_videos.py calibration, not EvalCrafter's generated-video calibration.
    technical = (scores["technical"] - 0.1107) / 0.07355
    aesthetic = (scores["aesthetic"] + 0.08285) / 0.03774
    return {
        "technical": technical.sigmoid(),
        "aesthetic": aesthetic.sigmoid(),
        "overall": (0.6104 * technical + 0.3896 * aesthetic).sigmoid(),
    }


class DOVERScorer(torch.nn.Module):
    """DOVER video quality in [0, 1], independent of the generation prompt."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        checkpoint_path: str | None = None,
        score_type: str = "overall",
        seed: int = 0,
    ) -> None:
        super().__init__()
        from dover import DOVER
        from huggingface_hub import hf_hub_download

        self.device = torch.device(device)
        self.score_type = score_type
        self.preprocess = _DOVERVideoTransform(seed)
        self.model = DOVER(
            backbone={"technical": {"type": "swin_tiny_grpb"}, "aesthetic": {"type": "conv_tiny"}},
            backbone_preserve_keys="technical,aesthetic",
            divide_head=True,
            vqa_head={"in_channels": 768, "hidden_channels": 64},
        )
        if checkpoint_path is None:
            checkpoint_path = hf_hub_download("teowu/DOVER", "DOVER.pth")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def forward(self, outputs: Sequence[torch.Tensor]) -> list[float]:
        views = [self.preprocess(output) for output in outputs]
        batch = {name: torch.cat([view[name] for view in views]).to(self.device) for name in _VIEW_OPTIONS}
        predictions = self.model(batch, reduce_scores=False)
        scores = {
            name: prediction.reshape(len(outputs), -1).mean(dim=1)
            for name, prediction in zip(batch, predictions, strict=True)
        }
        return _rescale_scores(scores)[self.score_type].float().cpu().tolist()


class DOVERRewardActor:
    def __init__(self, *, checkpoint_path: str | None, score_type: str, seed: int) -> None:
        use_cuda = bool(ray.get_gpu_ids()) and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        self.scorer = DOVERScorer(
            device="cuda" if use_cuda else "cpu", checkpoint_path=checkpoint_path, score_type=score_type, seed=seed
        )

    def score_batch(self, outputs: list[torch.Tensor], prompts: list[str]) -> list[float]:
        return self.scorer(outputs)


class AsyncDOVERPool(AsyncRewardActorPool, metaclass=SingletonMeta):
    def __init__(self, args, placement_group=None, slots=None) -> None:
        super().__init__(
            actor_cls=DOVERRewardActor,
            actor_kwargs={
                "checkpoint_path": args.dover_checkpoint_path,
                "score_type": args.dover_score_type,
                "seed": args.seed,
            },
            num_workers=args.dover_num_workers,
            batch_size=args.dover_batch_size,
            num_gpus_per_worker=args.dover_num_gpus_per_worker,
            colocate=args.dover_reward_colocate,
            name="dover",
            placement_group=placement_group,
            slots=slots,
        )


async def dover_rm(args, samples: Sequence[Sample]) -> list[float]:
    pool = AsyncDOVERPool(args)
    scores, max_queue_depth = await pool.score([s.generated_output for s in samples], [s.prompt for s in samples])
    record_reward_queue_depth(samples, "dover", max_queue_depth)
    return scores
