from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan


FSDP_PARALLEL_PLAN = FSDPParallelPlan(
    param_dtype_patterns={
        "*proj_in.*": "fp32",
        "*time_embedder.*": "fp32",
        "*proj_out.*": "fp32",
    },
)
