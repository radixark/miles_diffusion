"""LTX sequence-parallel and FSDP wrapping declarations.

LTX does not support sequence parallelism yet; ``sequence_parallel_plan`` is
the authoritative runtime check until an LTX plan is implemented.
"""

from __future__ import annotations

import torch

from miles.backends.fsdp_utils.sequence_parallel.plan import SequenceParallelPlan

FSDP_NO_SPLIT_MODULES = ["BasicAVTransformerBlock"]


def sequence_parallel_plan(model: torch.nn.Module) -> SequenceParallelPlan:
    raise NotImplementedError("LTX does not support sequence parallelism yet")
