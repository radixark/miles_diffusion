import torch
from torch.distributed.fsdp import FSDPModule
from torch.utils._pytree import tree_map


@torch.no_grad()
def move_model(model: torch.nn.Module, device: str | torch.device) -> None:
    """Move ``model``'s parameters and buffers to ``device``, keeping FSDP bindings and tied buffers.

    A move to the CPU lands in pinned memory.
    """
    target_device = torch.device(device)
    fsdp_parameters = [
        fsdp_parameter
        for module in model.modules()
        if isinstance(module, FSDPModule)
        for fsdp_parameter_group in [module._get_fsdp_state()._fsdp_param_group]
        if fsdp_parameter_group is not None
        for fsdp_parameter in fsdp_parameter_group.fsdp_params
    ]
    # FSDP re-pads these uneven shards on the CPU as soon as _apply() hands them over.
    repadded_parameter_ids = {
        id(fsdp_parameter.sharded_param)
        for fsdp_parameter in fsdp_parameters
        if fsdp_parameter.sharded_param.to_local().size() != fsdp_parameter.padded_sharded_param_size
    }
    # _apply() visits a shared tensor once per registering module; reusing the first move keeps it shared.
    original_and_moved_by_id = {}

    def move(tensor: torch.Tensor) -> torch.Tensor:
        if id(tensor) not in original_and_moved_by_id:
            # A non_blocking device-to-host copy allocates its CPU output in pinned memory.
            moved_tensor = tensor.to(target_device, non_blocking=True)
            if id(tensor) in repadded_parameter_ids and tensor.is_cuda:
                torch.cuda.current_stream().synchronize()
            original_and_moved_by_id[id(tensor)] = (tensor, moved_tensor)
        return original_and_moved_by_id[id(tensor)][1]

    original_pin_memory_by_fsdp_parameter = {
        fsdp_parameter: fsdp_parameter.pin_memory for fsdp_parameter in fsdp_parameters
    }
    try:
        if target_device.type == "cpu" and torch.cuda.is_available():
            # FSDP pins the storage it allocates for re-padding only when pin_memory is set.
            for fsdp_parameter in fsdp_parameters:
                fsdp_parameter.pin_memory = True
        model._apply(move)
    finally:
        for fsdp_parameter, original_pin_memory in original_pin_memory_by_fsdp_parameter.items():
            fsdp_parameter.pin_memory = original_pin_memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def move_optimizer(optimizer: torch.optim.Optimizer, device: str | torch.device) -> None:
    # A non_blocking device-to-host copy allocates its CPU output in pinned memory.
    for parameter, state in optimizer.state.items():
        optimizer.state[parameter] = tree_map(
            lambda value: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value, state
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
