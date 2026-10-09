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
from miles.utils import tracking_utils
from miles.utils.media import ImageSource, visual_source
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.misc import SingletonMeta
from miles.utils.timer import timer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

ENCODE_GPU_FRACTION = 0.3


def resolve_media_path(media: str, prompt_data: str) -> str:
    """Relative media paths are anchored at the dataset jsonl's directory (portable datasets)."""
    if os.path.isabs(media):
        return media
    return str(Path(prompt_data).parent / media)


def sft_sample_key(args, item: dict) -> tuple[str, int]:
    """Content-addressed cache filename and latent-sampling seed for one (media, prompt) item."""
    stat = Path(item["media"]).stat()
    # Bump the version whenever the cached pair changes.
    digest = hashlib.sha256(
        f"v3|{args.diffusion_model_family}|{args.sft_encoder_checkpoint}"
        f"|{args.diffusion_height}x{args.diffusion_width}"
        f"|{args.diffusion_output_num_frames}s{args.sft_frame_stride}"
        f"|{item['media']}|{stat.st_size}|{stat.st_mtime_ns}|{item['prompt']}".encode()
    ).digest()
    return digest.hex()[:16] + ".pt", int.from_bytes(digest[8:16], "big") % 2**63


def read_media_clip(path: str, *, height: int, width: int, num_frames: int, frame_stride: int) -> dict:
    """Every ``frame_stride``-th frame of a file prepared offline as exactly ``(num_frames - 1) * frame_stride + 1``
    frames of ``width x height`` square pixels; any other file is an error.

    Returns ``{"video": uint8 [C, T, H, W], "fps", "frame_times_seconds"}``, the times on the file timeline so a
    family can cut the target audio over the same interval.
    """
    source = visual_source(path)
    num_source_frames = (num_frames - 1) * frame_stride + 1
    if source.info.num_frames != num_source_frames:
        raise ValueError(
            f"{path} has {source.info.num_frames} frames; --diffusion-output-num-frames {num_frames} at "
            f"--sft-frame-stride {frame_stride} needs exactly {num_source_frames}; cut the file to that length offline"
        )
    if (source.info.width, source.info.height) != (width, height):
        raise ValueError(
            f"{path} is {source.info.width}x{source.info.height}, not the {width}x{height} canvas; resize it offline"
        )
    # Scaling then cropping leaves aspect residues such as 4096:4095, which display within half a pixel.
    if round(width * source.info.sample_aspect_ratio) != width:
        raise ValueError(
            f"{path} has non-square pixels (sample aspect ratio {source.info.sample_aspect_ratio}); "
            "resample it to square pixels offline"
        )
    frames = source.decode(0, num_source_frames)[::frame_stride]
    return {
        "video": frames.permute(1, 0, 2, 3),
        "fps": source.info.fps,
        "frame_times_seconds": (
            None if isinstance(source, ImageSource) else source.info.frame_times_seconds[::frame_stride]
        ),
    }


def _relocate(obj, device: torch.device):
    """Move every module/tensor found in a family's encoder structure."""
    if isinstance(obj, (torch.nn.Module, torch.Tensor)):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _relocate(value, device) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_relocate(value, device) for value in obj)
    return obj


@ray.remote
class SftEncodeActor:
    """Loads the family encoder once; with --sft-offload-encoder it sleeps in
    host RAM between encode bursts instead of holding the GPU for the whole run."""

    def __init__(self, args):
        from miles.rollout.encoder_hub import get_encoder

        self.args = args
        self.encoder_module = get_encoder(args.diffusion_model_family)
        self.encoder = self.encoder_module.load_encoder(args, torch.device("cuda"))
        if args.sft_offload_encoder:
            self.encoder = _relocate(self.encoder, torch.device("cpu"))
            torch.cuda.empty_cache()

    def encode(self, items: list[dict], cache_dir: str) -> int:
        args = self.args
        encoder = self.encoder
        if args.sft_offload_encoder:
            encoder = _relocate(encoder, torch.device("cuda"))
        for item in items:
            media_clip = read_media_clip(
                item["media"],
                height=args.diffusion_height,
                width=args.diffusion_width,
                num_frames=args.diffusion_output_num_frames,
                frame_stride=args.sft_frame_stride,
            )
            generator = torch.Generator().manual_seed(item["latent_seed"])
            pair = self.encoder_module.encode_sample(encoder, media_clip, item["prompt"], generator)
            out_path = Path(cache_dir) / item["cache_name"]
            # Temp-then-rename so an interrupted write never leaves a loadable-looking cache entry.
            tmp_path = out_path.with_name(out_path.name + ".tmp")
            torch.save(pair, tmp_path)
            os.replace(tmp_path, out_path)
        if args.sft_offload_encoder:
            self.encoder = _relocate(encoder, torch.device("cpu"))
            torch.cuda.empty_cache()
        return len(items)


class SftEncodePool(metaclass=SingletonMeta):
    """One encode actor per rollout placement-group bundle: encode is SFT's rollout, seated by RolloutManager."""

    def __init__(self, args, placement_group=None) -> None:
        if placement_group is None:
            raise RuntimeError("SftEncodePool is seated by RolloutManager; --loss-type sft_loss was not set.")
        self._args = args
        self._placement_group = placement_group
        self._actors: list | None = None
        # set per generate by RolloutManager; None when the encoders have their own GPUs
        self.train_in_flight: list | None = None

    @property
    def actors(self) -> list:
        # a fully cached dataset never encodes, so the encoders (tens of GB each) load on the first miss only
        if self._actors is not None:
            return self._actors
        pg, bundle_indices, _ = self._placement_group
        self._actors = [
            SftEncodeActor.options(
                num_cpus=ENCODE_GPU_FRACTION,
                num_gpus=ENCODE_GPU_FRACTION,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
            ).remote(self._args)
            for i in bundle_indices
        ]
        logger.info("SFT encode pool: %d workers at %.2f GPU each", len(self._actors), ENCODE_GPU_FRACTION)
        return self._actors

    def wait_for_train_step(self) -> None:
        """A prefetched rollout's encode burst must not share a GPU with the train step it overlaps."""
        if not self.train_in_flight:
            return
        with timer("sft_encode_wait"):
            ray.get(self.train_in_flight)


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
        item = {"media": resolve_media_path(media, args.prompt_data), "prompt": sample.prompt}
        item["cache_name"], item["latent_seed"] = sft_sample_key(args, item)
        items.append(item)

    missing = {item["cache_name"]: item for item in items if not (cache_dir / item["cache_name"]).exists()}
    encode_seconds = 0.0
    if missing:
        cache_dir.mkdir(parents=True, exist_ok=True)
        pool = SftEncodePool(args)
        pool.wait_for_train_step()
        start = time.time()
        actors = pool.actors
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
    return {"train_data": [sample.train_metadata["sft_pair"] for sample in samples]}


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    log_dict = {f"rollout/{key}": value for key, value in (rollout_extra_metrics or {}).items()}
    log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
    tracking_utils.log(args, log_dict, step_key="rollout/step")
    logger.info("sft rollout %d: %s (%.1fs)", rollout_id, log_dict, rollout_time)
    return True
