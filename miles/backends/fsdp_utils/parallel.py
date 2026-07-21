import logging
from argparse import Namespace

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from miles.utils.distributed_utils import get_gloo_group

from ..training_utils.parallel import ParallelState
from .sequence_parallel.topology import locate_rank, sp_subgroups, validate_sp_config

logger = logging.getLogger(__name__)


def create_fsdp_parallel_state(args: Namespace) -> ParallelState:
    """ParallelState for FSDP + optional sequence parallelism.

    SP gets its own process groups. FSDP shards parameters over every mesh
    axis (dp x sp flattened) — should a replicate axis (HSDP) ever be added,
    flatten only the shard axes, not the whole world. Data dispatch is by
    dp_rank; sp peers share samples.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    sp_size, ulysses_degree, ring_degree = validate_sp_config(
        world_size, args.sequence_parallel_size, args.ulysses_degree
    )
    dp_rank, sp_rank, _, _ = locate_rank(rank, sp_size, ulysses_degree)
    dp_size = world_size // sp_size

    mesh = init_device_mesh("cuda", mesh_shape=(dp_size, sp_size), mesh_dim_names=("dp", "sp"))
    dp_group = mesh.get_group("dp")
    sp_group = mesh.get_group("sp")
    logger.info(
        f"[Rank {rank}] mesh dp={dp_size} sp={sp_size} (ulysses={ulysses_degree} ring={ring_degree}), "
        f"dp_rank={dp_rank} sp_rank={sp_rank}"
    )

    # dist.new_group is collective: every rank must create every group.
    # Degree-1 dimensions stay None (usp_attention treats None as local).
    ulysses_group = ring_group = None
    if sp_size > 1:
        _, _, _, ulysses_groups, ring_groups = sp_subgroups(world_size, sp_size, ulysses_degree)
        if ulysses_degree > 1:
            for ranks in ulysses_groups:
                group = dist.new_group(ranks)
                if rank in ranks:
                    ulysses_group = group
        if ring_degree > 1:
            for ranks in ring_groups:
                group = dist.new_group(ranks)
                if rank in ranks:
                    ring_group = group

    parallel_state = ParallelState(
        dp_rank=dp_rank,
        dp_src_rank=0,
        dp_size=dp_size,
        dp_group=dp_group,
        sp_rank=sp_rank,
        sp_size=sp_size,
        sp_group=sp_group,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        ulysses_group=ulysses_group,
        ring_group=ring_group,
        dp_sp_rank=rank,
        dp_sp_size=world_size,
        dp_sp_group_gloo=get_gloo_group(),
        tp_size=1,
        tp_rank=0,
        tp_group=None,
    )
    parallel_state.fsdp_mesh = mesh[("dp", "sp")]._flatten("dp_sp") if sp_size > 1 else mesh["dp"]
    return parallel_state
