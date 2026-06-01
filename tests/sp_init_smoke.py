"""阶段1 多卡 SP 初始化 smoke test（torchrun，需 NCCL）。AC-2。

验证 create_fsdp_parallel_state 在真实 NCCL 下建出的 dp/sp/ulysses/ring 组成员，
与 sp_mesh 纯函数预期一致。
用法: torchrun --standalone --nproc_per_node=N tests/sp_init_smoke.py --sp S [--ulysses U --ring R]
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
    init_gloo_group()  # create_fsdp_parallel_state 依赖（正常训练在 train_actor 初始化）

    args = argparse.Namespace(
        sequence_parallel_size=cli.sp,
        ulysses_degree=cli.ulysses,
        ring_degree=cli.ring,
        context_parallel_size=1,
    )
    ps = create_fsdp_parallel_state(args)

    dp_size, sp_size, sp_groups, ulysses_groups, ring_groups = sp_subgroups(
        world, cli.sp, cli.ulysses, cli.ring
    )
    assert ps.sp_size == sp_size and ps.dp_size == dp_size
    assert dist.get_world_size(ps.sp_group) == sp_size
    assert dist.get_world_size(ps.dp_group) == dp_size

    my_sp = next(g for g in sp_groups if rank in g)
    assert _members(ps.sp_group) == my_sp, f"rank{rank} sp_group {_members(ps.sp_group)} != {my_sp}"
    if sp_size > 1:
        my_u = next(g for g in ulysses_groups if rank in g)
        my_r = sorted(next(g for g in ring_groups if rank in g))
        assert _members(ps.ulysses_group) == my_u, f"rank{rank} ulysses {_members(ps.ulysses_group)} != {my_u}"
        assert _members(ps.ring_group) == my_r, f"rank{rank} ring {_members(ps.ring_group)} != {my_r}"

        # USPAttention 经这些 getter 读 _SP coordinator —— 必须由 create_fsdp_parallel_state 注册好。
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_ring_parallel_world_size,
            get_sequence_parallel_world_size,
            get_sp_parallel_rank,
            get_sp_world_size,
            get_ulysses_parallel_world_size,
        )

        assert get_sp_world_size() == sp_size
        assert get_sequence_parallel_world_size() == sp_size
        assert get_sp_parallel_rank() == ps.sp_rank
        assert get_ulysses_parallel_world_size() == ps.ulysses_degree
        assert get_ring_parallel_world_size() == ps.ring_degree

    dist.barrier()
    if rank == 0:
        print(f"[SP-INIT-SMOKE OK] world={world} dp={dp_size} sp={sp_size} "
              f"ulysses={ps.ulysses_degree} ring={ps.ring_degree}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
