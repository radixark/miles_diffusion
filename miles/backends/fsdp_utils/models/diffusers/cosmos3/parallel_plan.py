from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan


FSDP_PARALLEL_PLAN = FSDPParallelPlan(
    param_dtype_patterns={
        # diffusers declares `_keep_in_fp32_modules = ["time_embedder"]` and
        # sglang-d pins the same module to fp32 at load; a blanket FSDP bf16
        # gather silently downgraded it on the train side. Gather it at fp32 to
        # restore the upstream precision contract (log-prob parity with the
        # rollout engine's fp32 MLP).
        "*time_embedder*": "fp32",
    },
)
