import logging
from argparse import Namespace

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from miles.utils.distributed_utils import get_gloo_group

from ..training_utils.parallel import ParallelState

logger = logging.getLogger(__name__)


def build_fsdp_meshes(
    device_type: str,
    world_size: int,
    dp_replicate: int,
    sp_size: int,
) -> dict[str, DeviceMesh]:
    """Build the FSDP hybrid-shard and DP/SP views."""
    world_mesh = init_device_mesh(device_type, (world_size,), mesh_dim_names=("world",))

    shard_view = world_mesh._unflatten(0, (dp_replicate, world_size // dp_replicate), ("dp_replicate", "fsdp"))
    # A degree-1 replicate axis would all-reduce over a single rank every bucket.
    fsdp_mesh = shard_view if dp_replicate > 1 else shard_view["fsdp"]
    meshes = {"world": world_mesh, "fsdp": fsdp_mesh, "dp": world_mesh}

    if sp_size > 1:
        dp_sp_view = world_mesh._unflatten(0, (world_size // sp_size, sp_size), ("dp", "sp"))
        meshes["dp"] = dp_sp_view["dp"]
        meshes["sp"] = dp_sp_view["sp"]
    return meshes


def build_sp_groups(
    sp_mesh: DeviceMesh | None,
    ring_degree: int,
    ulysses_degree: int,
) -> tuple[dist.ProcessGroup | None, dist.ProcessGroup | None]:
    if sp_mesh is None:
        return None, None

    sp_group = sp_mesh.get_group()
    if ring_degree == 1:
        return sp_group, None
    if ulysses_degree == 1:
        return None, sp_group
    usp_mesh = sp_mesh._unflatten(0, (ring_degree, ulysses_degree), ("ring", "ulysses"))
    return usp_mesh["ulysses"].get_group(), usp_mesh["ring"].get_group()


def create_fsdp_parallel_state(args: Namespace) -> ParallelState:
    """ParallelState for FSDP with optional hybrid sharding and SP."""
    world_size = dist.get_world_size()
    sp_size = args.sequence_parallel_size
    ulysses_degree = args.ulysses_degree or sp_size
    ring_degree = sp_size // ulysses_degree
    meshes = build_fsdp_meshes("cuda", world_size, args.dp_replicate_size, sp_size)
    ulysses_group, ring_group = build_sp_groups(meshes.get("sp"), ring_degree, ulysses_degree)
    dp_mesh = meshes["dp"]
    sp_mesh = meshes.get("sp")

    logger.info(
        f"[Rank {meshes['world'].get_local_rank()}] mesh dp={dp_mesh.size()} "
        f"(replicate={args.dp_replicate_size}) "
        f"sp={sp_size} (ulysses={ulysses_degree} ring={ring_degree}), "
        f"dp_rank={dp_mesh.get_local_rank()}"
    )

    return ParallelState(
        dp_rank=dp_mesh.get_local_rank(),
        dp_src_rank=0,
        dp_size=dp_mesh.size(),
        dp_group=dp_mesh.get_group(),
        sp_rank=sp_mesh.get_local_rank() if sp_mesh is not None else 0,
        sp_size=sp_mesh.size() if sp_mesh is not None else 1,
        sp_group=sp_mesh.get_group() if sp_mesh is not None else None,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        ulysses_group=ulysses_group,
        ring_group=ring_group,
        dp_sp_rank=meshes["world"].get_local_rank(),
        dp_sp_size=meshes["world"].size(),
        dp_sp_group_gloo=get_gloo_group(),
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        meshes=meshes,
    )
