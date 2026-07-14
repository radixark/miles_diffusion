"""Pin allocation-invariant kernels so rollout numerics don't depend on topology.

cuBLAS bf16 GEMM results depend on input pointer alignment (a 2-byte offset
changes the selected kernel variant and thus the bitwise output), and cuDNN
conv algo choice is similarly context-dependent. Engine topologies with
different tensor sizes (sp=1 vs sp>1 shards) produce different allocator
layouts, so the same mathematical forward diverges bitwise between them.

Fixes:
- sglang's batch-invariant matmul kernels (fixed Triton mm/addmm/bmm)
  replace cuBLAS for every Linear in the DiT.
- cuDNN pinned to deterministic, TF32-off algos for the VAE convs.

Must be applied to ALL rollout engines regardless of sp_degree — it changes
rollout numerics, so a sp=1 engine without it will not match a sp=2 engine
with it. Registered CI standards need re-recording when this lands.
"""

import torch


def apply() -> None:
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
        enable_batch_invariant_mode,
    )

    enable_batch_invariant_mode()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
