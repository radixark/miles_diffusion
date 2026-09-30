"""FSDP wrapping; this first experiment uses data parallelism, not SP."""

from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan

FSDP_PARALLEL_PLAN = FSDPParallelPlan(
    no_split_modules=("WanTransformerBlock",),
    param_dtype_patterns={
        "*.norm2.*": "fp32",
        "*scale_shift_table": "fp32",
        "*.condition_embedder.time_embedder.*": "fp32",
    },
)


def sequence_parallel_plan(model):
    raise NotImplementedError("Wan pose-control sequence parallelism is not implemented; use SP=1")


def install_sequence_parallel_attention(model, parallel_state):
    raise NotImplementedError("Wan pose-control sequence parallelism is not implemented; use SP=1")
