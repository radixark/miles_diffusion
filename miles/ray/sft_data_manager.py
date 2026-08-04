"""Serves auto-cached SFT samples through the RolloutManager driver interface."""

import hashlib
import json
import logging
from pathlib import Path

import ray
import torch

from miles.ray.sft_encode import build_sft_cache
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import load_function
from miles.utils.ray_utils import Box
from miles.utils.train_data_utils import TrainDataDPSplitter

logger = logging.getLogger(__name__)


def sft_sample_key(args, item: dict) -> tuple[str, int]:
    """Content-addressed cache filename and latent-sampling seed for one (video, prompt) item."""
    stat = Path(item["video"]).stat()
    digest = hashlib.sha256(
        f"{args.hf_checkpoint}|{args.sft_height}x{args.sft_width}|{args.sft_num_frames}s{args.sft_frame_stride}"
        f"|{item['video']}|{stat.st_size}|{stat.st_mtime_ns}|{item['prompt']}".encode()
    ).digest()
    return digest.hex()[:16] + ".pt", int.from_bytes(digest[8:16], "big") % 2**63


@ray.remote
class SftDataManager:
    """Encodes --sft-data-path (jsonl) into a content-addressed cache on first run,
    then serves per-epoch shuffled pairs; generate() is stateless in rollout_id."""

    def __init__(self, args, pg):
        configure_logger()
        self.args = args
        data_path = Path(args.sft_data_path)
        items = [
            {"video": row[args.sft_video_key], "prompt": row[args.sft_prompt_key]}
            for row in (json.loads(line) for line in data_path.read_text().splitlines() if line.strip())
        ]
        if len(items) < args.rollout_batch_size:
            raise ValueError(
                f"--sft-data-path holds {len(items)} samples, fewer than rollout_batch_size={args.rollout_batch_size}"
            )
        dropped = len(items) % args.rollout_batch_size
        if dropped:
            logger.warning(
                "SFT drop-last: %d of %d samples unused per epoch (rollout_batch_size=%d); "
                "the per-epoch reshuffle rotates which samples are dropped",
                dropped,
                len(items),
                args.rollout_batch_size,
            )

        cache_dir = data_path.parent / ".sft_cache"
        for item in items:
            item["cache_name"], item["latent_seed"] = sft_sample_key(args, item)
        self.files = [cache_dir / item["cache_name"] for item in items]
        missing = [item for item in items if not (cache_dir / item["cache_name"]).exists()]
        if missing:
            logger.info("SftDataManager: encoding %d of %d samples into %s", len(missing), len(items), cache_dir)
            build_sft_cache(args, missing, cache_dir, pg[0])
        assert all(f.exists() for f in self.files)
        self.train_data_dp_splitter = TrainDataDPSplitter()

        train_pipeline_config = load_function(args.train_pipeline_config_path)()
        scheduler = load_function(args.model_backend_path)(train_pipeline_config).load_scheduler(args)
        num_train_timesteps = int(scheduler.config.num_train_timesteps)
        shift = args.diffusion_flow_shift
        sigmas = torch.linspace(1.0, 1.0 / num_train_timesteps, num_train_timesteps, dtype=torch.float64)
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        self.scheduler_timesteps = (sigmas * num_train_timesteps).to(torch.float32)
        self.scheduler_sigmas = torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)]).to(torch.float32)
        logger.info(
            "SftDataManager: %d samples, cache=%s, flow_shift=%s, num_train_timesteps=%d",
            len(self.files),
            cache_dir,
            shift,
            num_train_timesteps,
        )

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def get_num_rollout_per_epoch(self):
        return len(self.files) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        batch_size = self.args.rollout_batch_size
        epoch, slot = divmod(rollout_id, len(self.files) // batch_size)
        generator = torch.Generator().manual_seed(self.args.seed + epoch)
        perm = torch.randperm(len(self.files), generator=generator)
        indices = perm[slot * batch_size : (slot + 1) * batch_size].tolist()
        pairs = [torch.load(self.files[i], map_location="cpu") for i in indices]
        data = {
            "train_data": pairs,
            "scheduler_timesteps": self.scheduler_timesteps,
            "scheduler_sigmas": self.scheduler_sigmas,
        }
        shards = self.train_data_dp_splitter.split_by_dp(data, self.train_parallel_config["dp_size"])
        return [Box(ray.put(shard)) for shard in shards]

    def save(self, rollout_id):
        pass

    def load(self, rollout_id=None):
        pass

    def offload(self):
        pass

    def onload_weights(self):
        pass

    def dispose(self):
        pass
