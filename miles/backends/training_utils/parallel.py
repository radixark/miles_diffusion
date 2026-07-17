from dataclasses import dataclass

import torch.distributed as dist


@dataclass
class ParallelState:
    """Core parallel state shared across all backends.
    Required by the general training utils.
    """

    dp_rank: int
    dp_src_rank: int
    dp_size: int
    dp_group: dist.ProcessGroup | None
    # Sequence Parallelism (USP = Ulysses x Ring)
    sp_rank: int
    sp_size: int
    sp_group: dist.ProcessGroup | None
    ulysses_degree: int
    ring_degree: int
    ulysses_group: dist.ProcessGroup | None
    ring_group: dist.ProcessGroup | None
    # dp x sp spans every training rank
    dp_sp_rank: int
    dp_sp_size: int
    dp_sp_group_gloo: dist.ProcessGroup | None
    tp_size: int
    tp_rank: int
    tp_group: dist.ProcessGroup | None
    is_pp_last_stage: bool = True
    vpp_size: int | None = 1
    microbatch_group_size_per_vp_stage: int | None = None
