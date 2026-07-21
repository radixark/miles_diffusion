"""Sequence-parallel rank layout: sp = ulysses * ring, global rank = dp_rank * sp + sp_rank.

Pure functions, no distributed init required. Group layout matches sglang's USP
(Ulysses ranks contiguous within an SP group, Ring ranks strided).
"""


def resolve_sp_degrees(sequence_parallel_size, ulysses_degree=0):
    """Normalize to (sp, ulysses, ring); ring = sp // ulysses. 0 means auto: ulysses fills sp."""
    if sequence_parallel_size < 1:
        raise ValueError(f"sequence_parallel_size must be positive, got {sequence_parallel_size}")
    if ulysses_degree < 0:
        raise ValueError(f"ulysses_degree must be non-negative, got {ulysses_degree}")
    sp = sequence_parallel_size
    u = ulysses_degree or sp
    if sp % u:
        raise ValueError(f"sequence_parallel_size({sp}) is not divisible by ulysses_degree({u})")
    return sp, u, sp // u


def validate_sp_config(world_size, sequence_parallel_size, ulysses_degree=0):
    """Validate at startup. Returns (sp, ulysses, ring).

    The num_heads % ulysses check lives in apply_sequence_parallel, where the
    real model config is available.
    """
    sp, u, r = resolve_sp_degrees(sequence_parallel_size, ulysses_degree)
    if world_size % sp != 0:
        raise ValueError(f"world_size({world_size}) is not divisible by sequence_parallel_size({sp})")
    return sp, u, r


def sp_subgroups(world_size, sequence_parallel_size, ulysses_degree=0):
    """Return (dp_size, sp_size, sp_groups, ulysses_groups, ring_groups); groups are global-rank lists.

    Inverse of locate_rank: coordinates -> full member list per group.
    """
    sp, u, r = validate_sp_config(world_size, sequence_parallel_size, ulysses_degree)
    dp_size = world_size // sp
    sp_groups = [list(range(d * sp, (d + 1) * sp)) for d in range(dp_size)]
    ulysses_groups = [g[i * u : (i + 1) * u] for g in sp_groups for i in range(r)]
    ring_groups = [g[i::u] for g in sp_groups for i in range(u)]
    return dp_size, sp, sp_groups, ulysses_groups, ring_groups


def locate_rank(rank, sequence_parallel_size, ulysses_degree=0):
    """Locate a global rank's (dp_rank, sp_rank, ulysses_rank, ring_rank).

    Inverse of sp_subgroups: global rank -> its index on each axis.
    """
    sp, u, _ = resolve_sp_degrees(sequence_parallel_size, ulysses_degree)
    dp_rank, sp_rank = divmod(rank, sp)
    ring_rank, ulysses_rank = divmod(sp_rank, u)
    return dp_rank, sp_rank, ulysses_rank, ring_rank
