"""Serves pre-encoded SFT samples through the RolloutManager driver interface."""

import logging
from pathlib import Path

import ray
import torch

from miles.utils.logging_utils import configure_logger
from miles.utils.misc import load_function
from miles.utils.ray_utils import Box
from miles.utils.train_data_utils import TrainDataDPSplitter

logger = logging.getLogger(__name__)


@ray.remote
class SftDataManager:
    """Deterministically shuffles --sft-data-path pairs per epoch; generate() is stateless in rollout_id."""

    def __init__(self, args, pg):
        configure_logger()
        self.args = args
        self.files = sorted(Path(args.sft_data_path).glob("*.pt"))
        if len(self.files) < args.rollout_batch_size:
            raise ValueError(
                f"--sft-data-path holds {len(self.files)} samples, "
                f"fewer than rollout_batch_size={args.rollout_batch_size}"
            )
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
            "SftDataManager: %d samples, flow_shift=%s, num_train_timesteps=%d",
            len(self.files),
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
