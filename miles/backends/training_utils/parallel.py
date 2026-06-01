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
    cp_rank: int
    cp_size: int
    dp_cp_rank: int
    dp_cp_size: int
    dp_group: dist.ProcessGroup | None
    dp_cp_group: dist.ProcessGroup | None
    dp_cp_group_gloo: dist.ProcessGroup | None
    cp_group: dist.ProcessGroup | None
    tp_size: int
    tp_rank: int
    tp_group: dist.ProcessGroup | None
    is_pp_last_stage: bool = True
    vpp_size: int | None = 1
    microbatch_group_size_per_vp_stage: int | None = None
    # Sequence Parallelism (USP = Ulysses × Ring)。CP≡SP，上面的 cp_* 为兼容别名（= sp_*）。
    sp_rank: int = 0
    sp_size: int = 1
    sp_group: dist.ProcessGroup | None = None
    ulysses_degree: int = 1
    ring_degree: int = 1
    ulysses_group: dist.ProcessGroup | None = None
    ring_group: dist.ProcessGroup | None = None
