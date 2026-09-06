from __future__ import annotations

from collections.abc import Sequence

import ray
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from miles.utils.misc import SingletonMeta
from miles.utils.processing_utils import generated_output_to_rgb_hwc_uint8_frames
from miles.utils.types import Sample

from .core import AsyncRewardActorPool

_HPS_VERSION_TO_FILENAME = {
    "v2.0": "HPS_v2_compressed.pt",
    "v2.1": "HPS_v2.1_compressed.pt",
}


class _HPSImageTransform:
    """HPSv2's inference resize: fit the longest side, then zero-pad."""

    def __init__(self, image_size: tuple[int, int], mean: Sequence[float], std: Sequence[float]) -> None:
        self.image_size = image_size[0]
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        tensor = vision_functional.to_tensor(image)
        height, width = tensor.shape[-2:]
        scale = self.image_size / float(max(height, width))
        new_height, new_width = (round(height * scale), round(width * scale))
        if (new_height, new_width) != (height, width):
            tensor = vision_functional.resize(
                tensor,
                [new_height, new_width],
                interpolation=InterpolationMode.BICUBIC,
            )

        pad_height = self.image_size - new_height
        pad_width = self.image_size - new_width
        tensor = vision_functional.pad(
            tensor,
            [
                pad_width // 2,
                pad_height // 2,
                pad_width - pad_width // 2,
                pad_height - pad_height // 2,
            ],
            fill=0,
        )
        return vision_functional.normalize(tensor, mean=self.mean, std=self.std)


class HPSScorer(torch.nn.Module):
    """HPSv2 scorer for aligned prompt/image batches."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        hps_version: str = "v2.1",
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        import huggingface_hub
        from open_clip import create_model, get_tokenizer
        from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

        self.device = torch.device(device)
        model = create_model(
            "ViT-H-14",
            pretrained=None,
            precision="amp",
            device=str(self.device),
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            output_dict=True,
        )

        if checkpoint_path is None:
            checkpoint_path = huggingface_hub.hf_hub_download("xswu/HPSv2", _HPS_VERSION_TO_FILENAME[hps_version])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        self.model = model.to(self.device).eval()
        self.preprocess = _HPSImageTransform(model.visual.image_size, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        self.tokenizer = get_tokenizer("ViT-H-14")

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> list[float]:
        image_batch = torch.stack([self.preprocess(image) for image in images]).to(self.device, non_blocking=True)
        text_batch = self.tokenizer(list(prompts)).to(self.device, non_blocking=True)
        with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            outputs = self.model(image_batch, text_batch)
            scores = torch.diagonal(outputs["image_features"] @ outputs["text_features"].T)
        return [float(score) for score in scores.detach().float().cpu()]


class HPSRewardActor:
    def __init__(self, *, hps_version: str, checkpoint_path: str | None = None) -> None:
        use_cuda = bool(ray.get_gpu_ids()) and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        self.scorer = HPSScorer(
            device="cuda" if use_cuda else "cpu",
            hps_version=hps_version,
            checkpoint_path=checkpoint_path,
        )

    def score_batch(self, outputs: list[torch.Tensor], prompts: list[str]) -> list[float]:
        # HPSv2 rounds when quantising to uint8; matching it keeps scores comparable with the reference
        images = []
        for output in outputs:
            (image,) = generated_output_to_rgb_hwc_uint8_frames(output, None, round_normalized=True)
            images.append(Image.fromarray(image))
        return self.scorer(prompts, images)


class AsyncHPSPool(AsyncRewardActorPool, metaclass=SingletonMeta):
    """Ray actor pool for HPS reward inference."""

    def __init__(self, args, placement_group=None, slots=None) -> None:
        super().__init__(
            actor_cls=HPSRewardActor,
            actor_kwargs={
                "hps_version": args.hps_version,
                "checkpoint_path": args.hps_checkpoint_path,
            },
            num_workers=args.hps_num_workers,
            batch_size=args.hps_batch_size,
            num_gpus_per_worker=args.hps_num_gpus_per_worker,
            colocate=args.hps_reward_colocate,
            name="hps",
            placement_group=placement_group,
            slots=slots,
        )


async def hps_rm(args, samples: Sequence[Sample]) -> list[float]:
    pool = AsyncHPSPool(args)
    scores, max_queue_depth = await pool.score([s.generated_output for s in samples], [s.prompt for s in samples])
    for sample in samples:
        sample.reward_max_queue_depth = float(max_queue_depth)
    return scores
