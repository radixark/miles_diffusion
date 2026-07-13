"""Multi-GPU SP init smoke test: NCCL group members must match the sp_mesh layout.

Usage: torchrun --standalone --nproc_per_node=N tests/sp/sp_init_smoke.py --sp S [--ulysses U --ring R]
"""

import argparse

import torch
import torch.distributed as dist

from miles.backends.fsdp_utils.parallel import create_fsdp_parallel_state
from miles.backends.fsdp_utils.sp_mesh import sp_subgroups
from miles.utils.distributed_utils import init_gloo_group


def _members(group):
    n = dist.get_world_size(group)
    t = torch.tensor([dist.get_rank()], device="cuda")
    out = [torch.zeros_like(t) for _ in range(n)]
    dist.all_gather(out, t, group=group)
    return sorted(int(x.item()) for x in out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sp", type=int, default=2)
    p.add_argument("--ulysses", type=int, default=0)
    p.add_argument("--ring", type=int, default=0)
    cli = p.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    init_gloo_group()

    args = argparse.Namespace(
        sequence_parallel_size=cli.sp,
        ulysses_degree=cli.ulysses,
        ring_degree=cli.ring,
    )
    ps = create_fsdp_parallel_state(args)

    dp_size, sp_size, sp_groups, ulysses_groups, ring_groups = sp_subgroups(world, cli.sp, cli.ulysses, cli.ring)
    assert ps.sp_size == sp_size and ps.dp_size == dp_size
    assert dist.get_world_size(ps.sp_group) == sp_size
    assert dist.get_world_size(ps.dp_group) == dp_size
    assert ps.fsdp_mesh.size() == world, f"fsdp mesh {ps.fsdp_mesh.size()} != world {world}"
    assert ps.dp_mesh.size() == dp_size, f"dp mesh {ps.dp_mesh.size()} != dp {dp_size}"

    my_sp = next(g for g in sp_groups if rank in g)
    assert _members(ps.sp_group) == my_sp, f"rank{rank} sp_group {_members(ps.sp_group)} != {my_sp}"
    if ps.ulysses_degree > 1:
        my_u = next(g for g in ulysses_groups if rank in g)
        assert _members(ps.ulysses_group) == my_u, f"rank{rank} ulysses {_members(ps.ulysses_group)} != {my_u}"
    else:
        assert ps.ulysses_group is None
    if ps.ring_degree > 1:
        my_r = sorted(next(g for g in ring_groups if rank in g))
        assert _members(ps.ring_group) == my_r, f"rank{rank} ring {_members(ps.ring_group)} != {my_r}"
    else:
        assert ps.ring_group is None

    dist.barrier()
    if rank == 0:
        print(
            f"[SP-INIT-SMOKE OK] world={world} dp={dp_size} sp={sp_size} "
            f"ulysses={ps.ulysses_degree} ring={ps.ring_degree} fsdp_mesh={ps.fsdp_mesh.size()}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
