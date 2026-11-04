from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan

# Krea2RMSNorm computes `weight + 1.0` in the weight's dtype; bf16 rounds small offsets away.
FSDP_PARALLEL_PLAN = FSDPParallelPlan(
    param_dtype_patterns={
        "*.norm.weight": "fp32",
        "*.norm1.weight": "fp32",
        "*.norm2.weight": "fp32",
        "*.norm_q.weight": "fp32",
        "*.norm_k.weight": "fp32",
    },
)
