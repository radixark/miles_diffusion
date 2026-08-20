from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan


FSDP_PARALLEL_PLAN = FSDPParallelPlan(
    param_dtype_patterns={
        "*.norm2.*": "fp32",
    },
)

# No local boundaries: the diffusers model's own _cp_plan is authoritative.
CP_PLAN = None
