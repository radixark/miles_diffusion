"""FLUX.3 image FSDP wrapping; sequence parallelism is not supported."""

from __future__ import annotations

import torch

from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan
from miles.backends.fsdp_utils.sequence_parallel.plan import SequenceParallelPlan

FSDP_PARALLEL_PLAN = FSDPParallelPlan(no_split_modules=("Flux3ImageBlock",))


def sequence_parallel_plan(model: torch.nn.Module) -> SequenceParallelPlan:
    raise NotImplementedError("FLUX.3 image does not support sequence parallelism")
