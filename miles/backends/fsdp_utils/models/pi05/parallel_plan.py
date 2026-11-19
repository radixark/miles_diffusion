"""π₀.₅ FSDP wrapping and sequence-parallel support."""

from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan

# Joint prefix/action attention calls layer children, bypassing
# GemmaDecoderLayer.forward. Wrap only children invoked via __call__;
# embeddings and tied LM heads stay together in the root FSDP group.
FSDP_PARALLEL_PLAN = FSDPParallelPlan(no_split_modules=("SiglipEncoderLayer", "GemmaMLP", "Pi05RMSNorm"))


def sequence_parallel_plan(model):
    raise NotImplementedError("π₀.₅ training currently supports FSDP/DP with sequence-parallel-size=1")
