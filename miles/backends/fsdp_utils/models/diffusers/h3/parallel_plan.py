from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan


# Empty: MiniMaxH3Transformer3DModel already declares its _no_split_modules, and no
# submodule needs to be pinned off the global param dtype.
FSDP_PARALLEL_PLAN = FSDPParallelPlan()

# Copied from MiniMaxH3Transformer3DModel._cp_plan on diffusers main; the commit
# requirements.txt pins predates it. Drop this once the pin advances past it.
CP_PLAN = {
    "rope": {
        0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
    },
    "transformer_blocks.0": {
        "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
    },
    # hidden_states is split once and flows on already sharded, but every block reads
    # adaln_indices from forward's unsharded tensor, so each splits its own copy.
    "transformer_blocks.*": {
        "adaln_indices": ContextParallelInput(split_dim=0, expected_dims=1, split_output=False),
    },
    "norm_out": {
        "timestep_indices": ContextParallelInput(split_dim=0, expected_dims=1, split_output=False),
    },
    "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    "audio_proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
}
