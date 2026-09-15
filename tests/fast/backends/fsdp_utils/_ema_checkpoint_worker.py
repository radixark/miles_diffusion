"""Write distinct, uneven EMA shards for the CPU resharding test."""

import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from miles.backends.fsdp_utils.ema import EmaShadow

if __name__ == "__main__":
    dist.init_process_group("gloo")
    mesh = init_device_mesh("cpu", (dist.get_world_size(),))
    full = torch.arange(15).reshape(5, 3).float()
    param = torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]))
    ema = EmaShadow([param], decay=0.5, flat_steps=10)
    for _ in range(2):
        with torch.no_grad():
            param.add_(1)
        ema.update()
    dcp.save({"ema": ema}, checkpoint_id=str(Path(sys.argv[1]) / "ema"))
    dist.destroy_process_group()
