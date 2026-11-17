# the H3 patterns cover H3 Image's fp32 modules: proj_in, time_embedder, pdd_proj_out
from miles.backends.fsdp_utils.models.diffusers.h3.parallel_plan import FSDP_PARALLEL_PLAN

__all__ = ["FSDP_PARALLEL_PLAN"]
