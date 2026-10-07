from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan


# H3 ships a mixed-precision checkpoint: its patch projections, timestep MLP and output heads
# (diffusers' _keep_in_fp32_modules) compute in fp32 while the block stack runs at the param dtype.
FSDP_PARALLEL_PLAN = FSDPParallelPlan(
    param_dtype_patterns={
        "*proj_in.*": "fp32",
        "*time_embedder.*": "fp32",
        "*proj_out.*": "fp32",
    },
)
