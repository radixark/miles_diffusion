import logging
from argparse import Namespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from miles.utils.distributed_utils import get_gloo_group

from ..training_utils.parallel import ParallelState
from .sp_mesh import locate_rank, sp_subgroups, validate_sp_config

logger = logging.getLogger(__name__)


def create_fsdp_parallel_state(args: Namespace) -> ParallelState:
    """ParallelState for FSDP + optional sequence parallelism.

    FSDP shards parameters on the dp mesh dim only; SP gets its own process
    groups and parameters are replicated across sp ranks.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    sp_size, ulysses_degree, ring_degree = validate_sp_config(
        world_size,
        args.sequence_parallel_size,
        args.ulysses_degree,
        args.ring_degree,
        getattr(args, "num_attention_heads", None),
    )
    dp_rank, sp_rank, _, _ = locate_rank(rank, sp_size, ulysses_degree, ring_degree)
    dp_size = world_size // sp_size

    mesh = init_device_mesh("cuda", mesh_shape=(dp_size, sp_size), mesh_dim_names=("dp", "sp"))
    dp_group = mesh.get_group("dp")
    sp_group = mesh.get_group("sp")
    logger.info(
        f"[Rank {rank}] mesh dp={dp_size} sp={sp_size} (ulysses={ulysses_degree} ring={ring_degree}), "
        f"dp_rank={dp_rank} sp_rank={sp_rank}"
    )

    # USPAttention reads sglang's module-global _SP coordinator;
    # set_seq_parallel_pg_by_sp_groups only sets ULYSSES_PG/RING_PG.
    ulysses_group = ring_group = None
    if sp_size > 1:
        from sglang.multimodal_gen.runtime.distributed import parallel_state as sgl_sp
        from sglang.multimodal_gen.runtime.distributed.parallel_groups import (
            PROCESS_GROUP,
            set_seq_parallel_pg_by_sp_groups,
        )

        _, _, sp_groups, _, _ = sp_subgroups(world_size, sp_size, ulysses_degree, ring_degree)
        set_seq_parallel_pg_by_sp_groups(ulysses_degree, ring_degree, rank, sp_groups)
        ulysses_group = PROCESS_GROUP.ULYSSES_PG
        ring_group = PROCESS_GROUP.RING_PG
        sgl_sp._SP = sgl_sp.init_parallel_group_coordinator(
            group_ranks=sp_groups,
            local_rank=torch.cuda.current_device(),
            backend=dist.get_backend(),
            parallel_mode="sequence",
            ulysses_group=ulysses_group,
            ring_group=ring_group,
        )

    parallel_state = ParallelState(
        dp_rank=dp_rank,
        dp_src_rank=dp_rank // world_size,
        dp_size=dp_size,
        cp_rank=sp_rank,
        cp_size=sp_size,
        dp_cp_rank=rank,
        dp_cp_size=world_size,
        dp_group=dp_group,
        dp_cp_group=dist.group.WORLD,
        dp_cp_group_gloo=get_gloo_group(),
        cp_group=sp_group,
        tp_size=1,
        tp_rank=0,
        tp_group=dist.new_group([rank]),
        sp_rank=sp_rank,
        sp_size=sp_size,
        sp_group=sp_group,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        ulysses_group=ulysses_group,
        ring_group=ring_group,
    )
    parallel_state.dp_mesh = mesh["dp"]
    return parallel_state
