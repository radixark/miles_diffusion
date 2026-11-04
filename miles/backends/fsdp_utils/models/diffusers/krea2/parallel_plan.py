from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan

# Krea2RMSNorm scales by `weight + 1.0` in the weight's dtype, so a bf16 all-gather rounds
# small learned offsets away. Mirrors the model's `_keep_in_fp32_modules`, which FSDP2 does
# not read on its own.
FSDP_PARALLEL_PLAN = FSDPParallelPlan(
    param_dtype_patterns={
        "*.norm.weight": "fp32",
        "*.norm1.weight": "fp32",
        "*.norm2.weight": "fp32",
        "*.norm_q.weight": "fp32",
        "*.norm_k.weight": "fp32",
    },
)
