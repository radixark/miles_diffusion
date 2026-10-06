"""SFT rollout plugin: lazy content-addressed encode cache behind the standard RolloutManager flow.

Plugged via --loss-type sft_loss (see set_default_diffusion_args): generate_rollout replaces the
sglang rollout, convert_samples_to_train_data replaces the reward/advantage conversion, and
log_rollout_data replaces the reward logging. The encoder actor pool plays the architectural role
sglang engines play in RL: the GPU data producer behind the rollout function, colocated on the
manager placement group.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.utils import tracking_utils
from miles.utils.media import ImageSource, maybe_download_media, media_fingerprint, visual_source
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.misc import SingletonMeta
from miles.utils.timer import timer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

ENCODE_GPU_FRACTION = 0.3

# Every args field an encoder reads; the cache key and latent seed depend on exactly these.
SFT_CACHE_KEY_ARGS = (
    "diffusion_model_family",
    "sft_encoder_checkpoint",
    "diffusion_height",
    "diffusion_width",
    "diffusion_output_num_frames",
    "sft_frame_stride",
    "diffusion_task",
)


def localize_sft_media(sample: Sample, media_cache_dir: Path) -> None:
    """Point the sample's target and conditions at local files, downloading URLs once into media_cache_dir."""
    sample.target = {kind: maybe_download_media(uri, media_cache_dir) for kind, uri in sample.target.items()}
    sample.conditions = [
        {**condition, "uri": maybe_download_media(condition["uri"], media_cache_dir)}
        for condition in sample.conditions
    ]


def sft_sample_key(args, sample: Sample) -> tuple[str, int]:
    """Cache identity includes the prompt, ordered conditions, target, all media files, and the args encoders read."""
    paths = {*sample.target.values(), *(condition["uri"] for condition in sample.conditions)}
    payload = {
        # Bump the version whenever the cached pair changes.
        "version": 3,
        "config": {key: vars(args)[key] for key in SFT_CACHE_KEY_ARGS},
        "sample": {"prompt": sample.prompt, "conditions": sample.conditions, "target": sample.target},
        "files": [media_fingerprint(path) for path in sorted(paths)],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).digest()
    return digest.hex() + ".pt", int.from_bytes(digest[8:16], "big") % 2**63


def read_media_clip(path: str, *, height: int, width: int, num_frames: int, frame_stride: int) -> dict:
    """Every ``frame_stride``-th frame of a file prepared offline as exactly ``(num_frames - 1) * frame_stride + 1``
    frames of ``width x height`` square pixels; any other file is an error.

    Returns ``{"video": uint8 [C, T, H, W], "fps", "frame_times_seconds", "audio_sample_rate"}``, the times on the
    file timeline so a family can cut the soundtrack over the same interval.
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
        "audio_sample_rate": source.info.audio_sample_rate,
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
        self.encoder_module = get_encoder(args.diffusion_model_family, args.diffusion_task)
        self.encoder = self.encoder_module.load_encoder(args, torch.device("cuda"))
        if args.sft_offload_encoder:
            self.encoder = _relocate(self.encoder, torch.device("cpu"))
            torch.cuda.empty_cache()

    def encode(self, samples: list[Sample], cache_dir: str) -> int:
        args = self.args
        encoder = self.encoder
        if args.sft_offload_encoder:
            encoder = _relocate(encoder, torch.device("cuda"))
        for sample in samples:
            media_clip = read_media_clip(
                sample.target["visual"],
                height=args.diffusion_height,
                width=args.diffusion_width,
                num_frames=args.diffusion_output_num_frames,
                frame_stride=args.sft_frame_stride,
            )
            generator = torch.Generator().manual_seed(sample.seed)
            pair = self.encoder_module.encode_sample(encoder, sample, media_clip, generator, args)
            out_path = Path(cache_dir) / sample.metadata["sft_cache_name"]
            # Temp-then-rename so an interrupted write never leaves a loadable-looking cache entry.
            tmp_path = out_path.with_name(out_path.name + ".tmp")
            torch.save(pair, tmp_path)
            os.replace(tmp_path, out_path)
        if args.sft_offload_encoder:
            self.encoder = _relocate(encoder, torch.device("cpu"))
            torch.cuda.empty_cache()
        return len(samples)


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
    for sample in samples:
        localize_sft_media(sample, cache_dir / "media")
        sample.metadata["sft_cache_name"], sample.seed = sft_sample_key(args, sample)

    # Keyed by cache name so a sample repeated across an epoch wrap is encoded once.
    missing = {
        sample.metadata["sft_cache_name"]: sample
        for sample in samples
        if not (cache_dir / sample.metadata["sft_cache_name"]).exists()
    }
    encode_seconds = 0.0
    if missing:
        cache_dir.mkdir(parents=True, exist_ok=True)
        pool = SftEncodePool(args)
        pool.wait_for_train_step()
        start = time.time()
        actors = pool.actors
        misses = list(missing.values())
        shards = [misses[i :: len(actors)] for i in range(len(actors))]
        ray.get(
            [actor.encode.remote(shard, str(cache_dir)) for actor, shard in zip(actors, shards, strict=True) if shard]
        )
        encode_seconds = time.time() - start

    for sample in samples:
        cache_name = sample.metadata.pop("sft_cache_name")
        sample.train_metadata = {"sft_pair": torch.load(cache_dir / cache_name, map_location="cpu")}
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
