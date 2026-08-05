"""Gloo worker for test_hybrid_shard_mesh; run under torch.distributed.run."""

import os
import sys

import torch.distributed as dist

from miles.backends.fsdp_utils.parallel import build_fsdp_meshes, build_sp_groups

# (dp_replicate, dp_shard, ring, ulysses) over 4 ranks
CONFIGS = [
    (1, 4, 1, 1),  # flat FSDP, no SP
    (2, 2, 1, 1),  # hybrid sharding, no SP
    (2, 1, 1, 2),  # hybrid sharding x Ulysses
    (2, 1, 2, 1),  # hybrid sharding x Ring
    (1, 1, 2, 2),  # full USP, no replicate
]


def mesh_group_ranks(mesh):
    return dist.get_process_group_ranks(mesh.get_group())


def check(rank, world_size, dp_replicate, dp_shard, ring_degree, ulysses_degree):
    sp_size = ring_degree * ulysses_degree
    meshes = build_fsdp_meshes("cpu", world_size, dp_replicate, sp_size)
    sp_mesh = meshes.get("sp")
    ulysses_group, ring_group = build_sp_groups(sp_mesh, ring_degree, ulysses_degree)
    dp_mesh = meshes["dp"]
    dp_rank, sp_rank = divmod(rank, sp_size)

    assert dp_mesh.get_local_rank() == dp_rank
    assert dp_mesh.size() == dp_replicate * dp_shard

    if sp_size == 1:
        assert sp_mesh is None
    else:
        sp_start = dp_rank * sp_size
        assert mesh_group_ranks(sp_mesh) == list(range(sp_start, sp_start + sp_size))
        assert sp_mesh.get_local_rank() == sp_rank
    if ulysses_degree > 1:
        ulysses_start = sp_start + sp_rank // ulysses_degree * ulysses_degree
        assert dist.get_process_group_ranks(ulysses_group) == list(
            range(ulysses_start, ulysses_start + ulysses_degree)
        )
        assert dist.get_rank(ulysses_group) == sp_rank % ulysses_degree
        if ring_degree == 1:
            assert ulysses_group is sp_mesh.get_group()
    else:
        assert ulysses_group is None
    if ring_degree > 1:
        ring_start = sp_start + sp_rank % ulysses_degree
        assert dist.get_process_group_ranks(ring_group) == list(range(ring_start, sp_start + sp_size, ulysses_degree))
        assert dist.get_rank(ring_group) == sp_rank // ulysses_degree
        if ulysses_degree == 1:
            assert ring_group is sp_mesh.get_group()
    else:
        assert ring_group is None

    # fully_shard reads dim 0 as replicate, dim 1 as shard; a degree-1 replicate is dropped.
    fsdp = meshes["fsdp"]
    if dp_replicate > 1:
        assert fsdp.ndim == 2, fsdp
        assert fsdp.mesh_dim_names == ("dp_replicate", "dp_shard")
        assert fsdp["dp_replicate"].size() == dp_replicate
        assert fsdp["dp_shard"].size() == dp_shard * sp_size
        # dp_replicate outermost makes a shard group a contiguous rank run, i.e. one node.
        shard_ranks = mesh_group_ranks(fsdp["dp_shard"])
        assert shard_ranks == list(range(shard_ranks[0], shard_ranks[0] + len(shard_ranks)))
    else:
        assert fsdp.ndim == 1, fsdp
        assert fsdp.size() == dp_shard * sp_size


def main():
    dist.init_process_group("gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    for config in CONFIGS:
        assert config[0] * config[1] * config[2] * config[3] == world_size, config
        check(rank, world_size, *config)
    dist.destroy_process_group()
    if rank == 0:
        print("OK", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    sys.exit(main())
