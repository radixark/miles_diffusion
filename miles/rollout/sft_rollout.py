"""SFT rollout plugin: lazy content-addressed encode cache behind the standard RolloutManager flow.

Plugged via --loss-type sft_loss (see set_default_diffusion_args): generate_rollout replaces the
sglang rollout, convert_samples_to_train_data replaces the reward/advantage conversion, and
log_rollout_data replaces the reward logging. The encoder actor pool plays the architectural role
sglang engines play in RL: the GPU data producer behind the rollout function, colocated on the
manager placement group.
"""

import hashlib
import logging
import os
import time
from pathlib import Path

import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.rm_hub.core import get_manager_placement_group
from miles.utils import tracking_utils
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

ENCODE_GPU_FRACTION = 0.3


def sft_sample_key(args, item: dict) -> tuple[str, int]:
    """Content-addressed cache filename and latent-sampling seed for one (media, prompt) item."""
    stat = Path(item["media"]).stat()
    digest = hashlib.sha256(
        f"{args.sft_encoder_checkpoint}|{args.diffusion_height}x{args.diffusion_width}"
        f"|{args.diffusion_output_num_frames}s{args.sft_frame_stride}"
        f"|{item['media']}|{stat.st_size}|{stat.st_mtime_ns}|{item['prompt']}".encode()
    ).digest()
    return digest.hex()[:16] + ".pt", int.from_bytes(digest[8:16], "big") % 2**63


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def read_media_clip(path: str, *, height: int, width: int, num_frames: int, frame_stride: int) -> torch.Tensor:
    if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
        if num_frames != 1:
            raise ValueError(f"{path} is an image, which requires --diffusion-output-num-frames 1")
        import numpy as np
        from PIL import Image

        frames = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).permute(2, 0, 1)[None].float()
    else:
        import torchvision

        video, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")
        span = (num_frames - 1) * frame_stride + 1
        if video.shape[0] < span:
            raise ValueError(f"{path} has {video.shape[0]} frames, need {span}")
        start = (video.shape[0] - span) // 2
        frames = video[start : start + span : frame_stride].float()
    frames = frames / 127.5 - 1.0

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
        from miles.rollout.encoder_hub import get_encoder

        self.args = args
        self.encoder_module = get_encoder(args.diffusion_model_family)
        self.encoder = self.encoder_module.load_encoder(args, torch.device("cuda"))

    def encode(self, items: list[dict], cache_dir: str) -> int:
        args = self.args
        for item in items:
            pixels = read_media_clip(
                item["media"],
                height=args.diffusion_height,
                width=args.diffusion_width,
                num_frames=args.diffusion_output_num_frames,
                frame_stride=args.sft_frame_stride,
            )
            generator = torch.Generator().manual_seed(item["latent_seed"])
            pair = self.encoder_module.encode_sample(self.encoder, pixels, item["prompt"], generator)
            out_path = Path(cache_dir) / item["cache_name"]
            # Temp-then-rename so an interrupted write never leaves a loadable-looking cache entry.
            tmp_path = out_path.with_name(out_path.name + ".tmp")
            torch.save(pair, tmp_path)
            os.replace(tmp_path, out_path)
        return len(items)


_encode_actors: list | None = None
_scheduler_grid: tuple[torch.Tensor, torch.Tensor] | None = None


def _encode_pool(args) -> list:
    global _encode_actors
    if _encode_actors is None:
        # Encode is SFT's rollout: the pool takes the rollout placement seats sglang engines use in RL.
        pg, bundle_indices, _ = get_manager_placement_group()
        _encode_actors = [
            SftEncodeActor.options(
                num_cpus=ENCODE_GPU_FRACTION,
                num_gpus=ENCODE_GPU_FRACTION,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
            ).remote(args)
            for i in bundle_indices
        ]
        logger.info("SFT encode pool: %d workers at %.2f GPU each", len(_encode_actors), ENCODE_GPU_FRACTION)
    return _encode_actors


def _get_scheduler_grid(args) -> tuple[torch.Tensor, torch.Tensor]:
    global _scheduler_grid
    if _scheduler_grid is None:
        config = load_function(args.train_pipeline_config_path)()
        scheduler = load_function(args.model_backend_path)(config).load_scheduler(args)
        num_train_timesteps = int(scheduler.config.num_train_timesteps)
        shift = args.fsdp_flow_shift
        sigmas = torch.linspace(1.0, 1.0 / num_train_timesteps, num_train_timesteps, dtype=torch.float64)
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        _scheduler_grid = (
            (sigmas * num_train_timesteps).to(torch.float32),
            torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)]).to(torch.float32),
        )
    return _scheduler_grid


def generate_rollout(args, rollout_id, data_source, evaluation: bool = False) -> RolloutFnTrainOutput:
    assert not evaluation, "sft_loss does not support eval rollouts"
    # Deterministic per epoch and idempotent; non-divisible datasets may repeat a few
    # boundary samples across an epoch wrap.
    data_source.dataset.shuffle(data_source.epoch_id)
    groups = data_source.get_samples(args.rollout_batch_size)
    samples = [sample for group in groups for sample in group]

    cache_dir = Path(args.prompt_data).parent / ".sft_cache"
    items = []
    for sample in samples:
        media = sample.metadata.get("video") or sample.metadata.get("image")
        if media is None:
            raise ValueError(f"sample {sample.index} metadata has neither 'video' nor 'image': {sample.metadata}")
        item = {"media": media, "prompt": sample.prompt}
        item["cache_name"], item["latent_seed"] = sft_sample_key(args, item)
        items.append(item)

    missing = {item["cache_name"]: item for item in items if not (cache_dir / item["cache_name"]).exists()}
    encode_seconds = 0.0
    if missing:
        cache_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        actors = _encode_pool(args)
        miss_items = list(missing.values())
        shards = [miss_items[i :: len(actors)] for i in range(len(actors))]
        ray.get(
            [actor.encode.remote(shard, str(cache_dir)) for actor, shard in zip(actors, shards, strict=True) if shard]
        )
        encode_seconds = time.time() - start

    for sample, item in zip(samples, items, strict=True):
        sample.train_metadata = {"sft_pair": torch.load(cache_dir / item["cache_name"], map_location="cpu")}
        sample.status = Sample.Status.COMPLETED

    metrics = {
        "sft_cache_miss": len(missing),
        "sft_encode_seconds": round(encode_seconds, 3),
        "sft_epoch": data_source.epoch_id,
    }
    return RolloutFnTrainOutput(samples=groups, metrics=metrics)


def convert_samples_to_train_data(args, samples: list[Sample]) -> dict:
    scheduler_timesteps, scheduler_sigmas = _get_scheduler_grid(args)
    return {
        "train_data": [sample.train_metadata["sft_pair"] for sample in samples],
        "scheduler_timesteps": scheduler_timesteps,
        "scheduler_sigmas": scheduler_sigmas,
    }


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    log_dict = {f"rollout/{key}": value for key, value in (rollout_extra_metrics or {}).items()}
    log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
    tracking_utils.log(args, log_dict, step_key="rollout/step")
    logger.info("sft rollout %d: %s (%.1fs)", rollout_id, log_dict, rollout_time)
    return True
