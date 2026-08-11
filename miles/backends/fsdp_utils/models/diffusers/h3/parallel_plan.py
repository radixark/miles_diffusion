from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan


# Empty: MiniMaxH3Transformer3DModel already declares its _no_split_modules, and no
# submodule needs to be pinned off the global param dtype.
FSDP_PARALLEL_PLAN = FSDPParallelPlan()
