"""Sequence-parallel rank layout: sp = ulysses * ring, global rank = dp_rank * sp + sp_rank.

Pure functions, no distributed init required. Group layout matches sglang's USP
(Ulysses ranks contiguous within an SP group, Ring ranks strided).
"""


def resolve_sp_degrees(sequence_parallel_size, ulysses_degree=0, ring_degree=0):
    """Normalize (sp, ulysses, ring). degree=0 means auto: ulysses fills sp, ring=1."""
    sp = max(1, sequence_parallel_size)
    u = ulysses_degree or sp
    r = ring_degree or (sp // u)
    if u * r != sp:
        raise ValueError(f"ulysses_degree({u}) * ring_degree({r}) != sequence_parallel_size({sp})")
    return sp, u, r


def validate_sp_config(world_size, sequence_parallel_size, ulysses_degree=0, ring_degree=0):
    """Validate at startup. Returns (sp, ulysses, ring).

    The num_heads % ulysses check lives in apply_sequence_parallel, where the
    real model config is available.
    """
    sp, u, r = resolve_sp_degrees(sequence_parallel_size, ulysses_degree, ring_degree)
    if world_size % sp != 0:
        raise ValueError(f"world_size({world_size}) is not divisible by sequence_parallel_size({sp})")
    return sp, u, r


def sp_subgroups(world_size, sequence_parallel_size, ulysses_degree=0, ring_degree=0):
    """Return (dp_size, sp_size, sp_groups, ulysses_groups, ring_groups); groups are global-rank lists."""
    sp, u, r = resolve_sp_degrees(sequence_parallel_size, ulysses_degree, ring_degree)
    dp_size = world_size // sp
    sp_groups = [list(range(d * sp, (d + 1) * sp)) for d in range(dp_size)]
    ulysses_groups = [g[i * u : (i + 1) * u] for g in sp_groups for i in range(r)]
    ring_groups = [g[i::u] for g in sp_groups for i in range(u)]
    return dp_size, sp, sp_groups, ulysses_groups, ring_groups


def locate_rank(rank, sequence_parallel_size, ulysses_degree=0, ring_degree=0):
    """Locate a global rank's (dp_rank, sp_rank, ulysses_rank, ring_rank)."""
    sp, u, _ = resolve_sp_degrees(sequence_parallel_size, ulysses_degree, ring_degree)
    dp_rank, sp_rank = divmod(rank, sp)
    ring_rank, ulysses_rank = divmod(sp_rank, u)
    return dp_rank, sp_rank, ulysses_rank, ring_rank
