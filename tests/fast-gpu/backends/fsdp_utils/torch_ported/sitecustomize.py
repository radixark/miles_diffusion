import torch
from torch.testing._internal.distributed.multi_threaded_pg import ThreadLocalWorld

from miles.backends.fsdp_utils.monkey_patches.fsdp_param_dtype_patch import apply_param_dtype_map_patch

apply_param_dtype_map_patch()
torch.use_deterministic_algorithms(True)

# torch 2.11's destroy_process_group reads _world.comms, which that release's threaded test world lacks: every thread
# of a threaded test raises in its teardown, and the interpreter has segfaulted while they printed their tracebacks.
if not hasattr(ThreadLocalWorld, "comms"):
    ThreadLocalWorld.comms = property(lambda self: [])
