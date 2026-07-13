"""SP rank layout / subgroup unit tests. Pure functions, no GPU."""

import pytest

from miles.backends.fsdp_utils.sp_mesh import locate_rank, resolve_sp_degrees, sp_subgroups, validate_sp_config


def test_resolve_auto_degrees():
    assert resolve_sp_degrees(4) == (4, 4, 1)
    assert resolve_sp_degrees(4, ulysses_degree=2) == (4, 2, 2)
    assert resolve_sp_degrees(8, 2) == (8, 2, 4)
    assert resolve_sp_degrees(1) == (1, 1, 1)


def test_resolve_illegal():
    with pytest.raises(ValueError):
        resolve_sp_degrees(4, ulysses_degree=3)


def test_sglang_alignment_example():
    # Matches sglang set_seq_parallel_pg_by_sp_groups: sp=4,u=2,r=2
    _, _, sp_groups, ulysses_groups, ring_groups = sp_subgroups(4, 4, 2)
    assert sp_groups == [[0, 1, 2, 3]]
    assert ulysses_groups == [[0, 1], [2, 3]]
    assert ring_groups == [[0, 2], [1, 3]]


@pytest.mark.parametrize(
    "world,sp,u,r",
    [
        (2, 2, 2, 1),
        (4, 2, 2, 1),
        (4, 4, 2, 2),
        (4, 4, 4, 1),
        (8, 4, 2, 2),
        (16, 8, 2, 4),
        (64, 8, 8, 1),
        (256, 16, 8, 2),
        (1024, 8, 8, 1),
    ],
)
def test_layout_invariants_any_scale(world, sp, u, r):
    dp_size, sp_size, sp_groups, ulysses_groups, ring_groups = sp_subgroups(world, sp, u)
    assert sp_size == sp == u * r  # r in the parametrize row is the expected derived ring degree
    assert dp_size * sp_size == world
    assert len(sp_groups) == dp_size and all(len(g) == sp for g in sp_groups)
    # ulysses groups: contiguous, size u, exact cover
    assert all(len(g) == u and g == list(range(g[0], g[0] + u)) for g in ulysses_groups)
    assert sorted(x for g in ulysses_groups for x in g) == list(range(world))
    # ring groups: strided, size r, exact cover
    assert all(len(g) == r for g in ring_groups)
    assert sorted(x for g in ring_groups for x in g) == list(range(world))


@pytest.mark.parametrize("world,sp,u,r", [(8, 4, 2, 2), (16, 8, 2, 4), (256, 16, 8, 2)])
def test_locate_rank_consistent_with_subgroups(world, sp, u, r):
    _, _, _, ulysses_groups, ring_groups = sp_subgroups(world, sp, u)
    for rank in range(world):
        dp_rank, sp_rank, u_rank, r_rank = locate_rank(rank, sp, u)
        assert dp_rank == rank // sp and sp_rank == rank % sp
        my_u_group = next(g for g in ulysses_groups if rank in g)
        my_r_group = next(g for g in ring_groups if rank in g)
        assert my_u_group[u_rank] == rank
        assert my_r_group[r_rank] == rank


def test_validate_rejects_illegal():
    with pytest.raises(ValueError):
        validate_sp_config(world_size=6, sequence_parallel_size=4)
    with pytest.raises(ValueError):
        validate_sp_config(world_size=8, sequence_parallel_size=4, ulysses_degree=3)
    assert validate_sp_config(world_size=4, sequence_parallel_size=2) == (2, 2, 1)
    assert validate_sp_config(world_size=4, sequence_parallel_size=4) == (4, 4, 1)
