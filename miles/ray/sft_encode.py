"""Family-generic SFT cache building: encode actors driven by TrainPipelineConfig hooks."""

import logging
import os
from pathlib import Path

import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils.misc import load_function

logger = logging.getLogger(__name__)


def read_video_clip(path: str, *, height: int, width: int, num_frames: int, frame_stride: int) -> torch.Tensor:
    import torchvision

    frames, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")
    span = (num_frames - 1) * frame_stride + 1
    if frames.shape[0] < span:
        raise ValueError(f"{path} has {frames.shape[0]} frames, need {span}")
    start = (frames.shape[0] - span) // 2
    frames = frames[start : start + span : frame_stride].float() / 127.5 - 1.0

    scale = max(height / frames.shape[2], width / frames.shape[3])
    new_h = max(height, round(frames.shape[2] * scale))
    new_w = max(width, round(frames.shape[3] * scale))
    frames = torch.nn.functional.interpolate(frames, size=(new_h, new_w), mode="bilinear", antialias=True)
    top = (new_h - height) // 2
    left = (new_w - width) // 2
    return frames[:, :, top : top + height, left : left + width].permute(1, 0, 2, 3)


@ray.remote
class SftEncodeActor:
    def __init__(self, args):
        self.args = args
        self.config = load_function(args.train_pipeline_config_path)()
        self.encoder = self.config.load_sft_encoder(args, torch.device("cuda"))

    def encode(self, items: list[dict], cache_dir: str) -> int:
        args = self.args
        for item in items:
            pixels = read_video_clip(
                item["video"],
                height=args.sft_height,
                width=args.sft_width,
                num_frames=args.sft_num_frames,
                frame_stride=args.sft_frame_stride,
            )
            generator = torch.Generator().manual_seed(item["latent_seed"])
            pair = self.config.encode_sft_sample(self.encoder, pixels, item["prompt"], generator)
            out_path = Path(cache_dir) / item["cache_name"]
            # Temp-then-rename so an interrupted write never leaves a loadable-looking cache entry.
            tmp_path = out_path.with_name(out_path.name + ".tmp")
            torch.save(pair, tmp_path)
            os.replace(tmp_path, out_path)
        return len(items)


def build_sft_cache(args, items: list[dict], cache_dir: Path, pg) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    num_workers = min(len(pg.bundle_specs), len(items))
    actors = [
        SftEncodeActor.options(
            num_cpus=1,
            num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            ),
        ).remote(args)
        for i in range(num_workers)
    ]
    done = ray.get([actor.encode.remote(items[i::num_workers], str(cache_dir)) for i, actor in enumerate(actors)])
    for actor in actors:
        ray.kill(actor)
    logger.info("SFT cache: encoded %d new samples into %s (%d workers)", sum(done), cache_dir, num_workers)
