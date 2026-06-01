"""序列并行（SP）的 rank 布局与子组划分。

纯函数、不依赖分布式初始化，便于单测且不假定卡数（覆盖 2~1000+ 卡）。
对齐 sglang-d / FastVideo 的 USP：sp = ulysses_degree × ring_degree；
全局 rank = dp_rank * sp + sp_rank；SP 组内 Ulysses 连续切、Ring 跨步切。
"""


def resolve_sp_degrees(sequence_parallel_size, ulysses_degree=0, ring_degree=0):
    """归一化 (sp, ulysses, ring)。degree=0 表示自动：ulysses 吃满 sp、ring=1。"""
    sp = max(1, sequence_parallel_size)
    u = ulysses_degree or sp
    r = ring_degree or (sp // u)
    if u * r != sp:
        raise ValueError(f"ulysses_degree({u}) * ring_degree({r}) != sequence_parallel_size({sp})")
    return sp, u, r


def validate_sp_config(world_size, sequence_parallel_size, ulysses_degree=0, ring_degree=0, num_heads=None):
    """启动期合法性校验（唯一必要的报警点）。返回 (sp, ulysses, ring)。"""
    sp, u, r = resolve_sp_degrees(sequence_parallel_size, ulysses_degree, ring_degree)
    if world_size % sp != 0:
        raise ValueError(f"world_size({world_size}) 不能被 sp({sp}) 整除")
    if num_heads is not None and num_heads % u != 0:
        raise ValueError(f"num_heads({num_heads}) 不能被 ulysses_degree({u}) 整除（Ulysses 不适合 GQA/MQA）")
    return sp, u, r


def sp_subgroups(world_size, sequence_parallel_size, ulysses_degree=0, ring_degree=0):
    """返回 (dp_size, sp_size, sp_groups, ulysses_groups, ring_groups)，组均为全局 rank 列表。"""
    sp, u, r = resolve_sp_degrees(sequence_parallel_size, ulysses_degree, ring_degree)
    dp_size = world_size // sp
    sp_groups = [list(range(d * sp, (d + 1) * sp)) for d in range(dp_size)]
    ulysses_groups = [g[i * u : (i + 1) * u] for g in sp_groups for i in range(r)]
    ring_groups = [g[i::u] for g in sp_groups for i in range(u)]
    return dp_size, sp, sp_groups, ulysses_groups, ring_groups


def locate_rank(rank, sequence_parallel_size, ulysses_degree=0, ring_degree=0):
    """定位 rank 的 (dp_rank, sp_rank, ulysses_rank, ring_rank)。"""
    sp, u, _ = resolve_sp_degrees(sequence_parallel_size, ulysses_degree, ring_degree)
    dp_rank, sp_rank = divmod(rank, sp)
    ring_rank, ulysses_rank = divmod(sp_rank, u)  # sp_rank = ring_rank * u + ulysses_rank
    return dp_rank, sp_rank, ulysses_rank, ring_rank
