import os
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor

from miles.backends.fsdp_utils.actor import move_torch_model, move_torch_optimizer
from miles.backends.fsdp_utils.ema import EmaShadow


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 7), torch.nn.Linear(7, 8)])
        self.register_buffer("scale", torch.arange(8, dtype=torch.float32))
        self.register_buffer("transposed", torch.arange(16, dtype=torch.float32).reshape(4, 4).T)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = torch.nn.functional.relu(layer(inputs))
        return inputs * self.scale


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _assert_device(tensors, device_type, *, pinned=False):
    for tensor in tensors:
        local = _local(tensor)
        assert local.device.type == device_type, (
            f"expected {device_type}, got tensor={type(tensor).__name__} "
            f"device={tensor.device} local_device={local.device}"
        )
        if device_type == "cpu":
            assert local.is_pinned() is pinned


def _optimizer_tensors(optimizer):
    return [value for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor)]


def _fsdp_params(model):
    params = []
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        state = module._get_fsdp_state()
        if state._fsdp_param_group is not None:
            params.extend(state._fsdp_param_group.fsdp_params)
    return params


def _layout_signature(tensor):
    local = _local(tensor)
    dtensor_layout = (
        (
            tuple(tensor.placements),
            tensor.device_mesh.device_type,
            tuple(tensor.device_mesh.shape),
            tensor.device_mesh.mesh_dim_names,
            tuple(tensor.device_mesh.mesh.flatten().cpu().tolist()),
        )
        if isinstance(tensor, DTensor)
        else None
    )
    return (
        type(tensor),
        tensor.layout,
        tuple(tensor.shape),
        tensor.stride(),
        local.layout,
        tuple(local.shape),
        local.stride(),
        local.storage_offset(),
        dtensor_layout,
    )


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    device = torch.device("cuda", local_rank)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    model = _Model().to(device)
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EmaShadow(model.parameters())

    inputs = torch.randn(4, 8, device=device)
    model(inputs).sum().backward()
    optimizer.step()
    assert any(param.grad is not None for param in model.parameters())

    param_ids = [id(param) for param in model.parameters()]
    param_layouts = [_layout_signature(param) for param in model.parameters()]
    buffer_layouts = [_layout_signature(buffer) for buffer in model.buffers()]
    state_layouts = [_layout_signature(tensor) for tensor in _optimizer_tensors(optimizer)]
    expected_params = [_local(param).detach().cpu().clone() for param in model.parameters()]
    expected_buffers = [_local(buffer).detach().cpu().clone() for buffer in model.buffers()]
    expected_states = [tensor.detach().cpu().clone() for tensor in _optimizer_tensors(optimizer)]
    expected_shadow = [tensor.detach().cpu().clone() for tensor in ema.shadow]
    fsdp_params = _fsdp_params(model)
    storage_layouts = [_layout_signature(param._sharded_param_data) for param in fsdp_params]
    expected_storage = [param._sharded_param_data.detach().cpu().clone() for param in fsdp_params]
    has_padding = torch.tensor(
        any(param._sharded_param_data.numel() > param.sharded_param.to_local().numel() for param in fsdp_params),
        device=device,
    )
    dist.all_reduce(has_padding, op=dist.ReduceOp.MAX)
    assert has_padding.item()
    sleep_times = []
    wake_times = []
    awake_allocated = torch.cuda.memory_allocated()
    sleep_allocated = awake_allocated

    for _ in range(2):
        start = time.perf_counter()
        move_torch_model(model, "cpu", pin_memory=True)
        move_torch_optimizer(optimizer, "cpu", pin_memory=True)
        ema.to("cpu", non_blocking=True, pin_memory=True)
        torch.cuda.synchronize()
        sleep_times.append(time.perf_counter() - start)
        sleep_allocated = torch.cuda.memory_allocated()
        assert sleep_allocated < awake_allocated

        assert [id(param) for param in model.parameters()] == param_ids
        assert [_layout_signature(param) for param in model.parameters()] == param_layouts
        assert [_layout_signature(buffer) for buffer in model.buffers()] == buffer_layouts
        assert [_layout_signature(tensor) for tensor in _optimizer_tensors(optimizer)] == state_layouts
        assert [_layout_signature(param._sharded_param_data) for param in fsdp_params] == storage_layouts
        assert all(param.grad is None for param in model.parameters())
        _assert_device(model.parameters(), "cpu", pinned=True)
        _assert_device(model.buffers(), "cpu", pinned=True)
        _assert_device(_optimizer_tensors(optimizer), "cpu", pinned=True)
        _assert_device(ema.shadow, "cpu", pinned=True)
        _assert_device((param._sharded_param_data for param in fsdp_params), "cpu", pinned=True)

        start = time.perf_counter()
        move_torch_model(model, device)
        move_torch_optimizer(optimizer, device)
        ema.to(device, non_blocking=True)
        torch.cuda.synchronize()
        wake_times.append(time.perf_counter() - start)

        assert [id(param) for param in model.parameters()] == param_ids
        assert [_layout_signature(param) for param in model.parameters()] == param_layouts
        assert [_layout_signature(buffer) for buffer in model.buffers()] == buffer_layouts
        assert [_layout_signature(tensor) for tensor in _optimizer_tensors(optimizer)] == state_layouts
        assert [_layout_signature(param._sharded_param_data) for param in fsdp_params] == storage_layouts
        _assert_device(model.parameters(), "cuda")
        _assert_device(model.buffers(), "cuda")
        _assert_device(_optimizer_tensors(optimizer), "cuda")
        _assert_device(ema.shadow, "cuda")

    for actual, expected in zip(model.parameters(), expected_params, strict=True):
        torch.testing.assert_close(_local(actual).cpu(), expected)
    for actual, expected in zip(model.buffers(), expected_buffers, strict=True):
        torch.testing.assert_close(_local(actual).cpu(), expected)
    for actual, expected in zip(_optimizer_tensors(optimizer), expected_states, strict=True):
        torch.testing.assert_close(actual.cpu(), expected)
    for actual, expected in zip(ema.shadow, expected_shadow, strict=True):
        torch.testing.assert_close(actual.cpu(), expected)
    for param, expected in zip(fsdp_params, expected_storage, strict=True):
        torch.testing.assert_close(param._sharded_param_data.cpu(), expected)

    optimizer.zero_grad(set_to_none=True)
    model(inputs).sum().backward()
    optimizer.step()
    torch.cuda.synchronize()

    dist.destroy_process_group()
    if local_rank == 0:
        print(
            f"sleep_ms={sum(sleep_times) / len(sleep_times) * 1e3:.3f} "
            f"wake_ms={sum(wake_times) / len(wake_times) * 1e3:.3f} "
            f"awake_allocated={awake_allocated} sleep_allocated={sleep_allocated}"
        )
        print("OK")


if __name__ == "__main__":
    main()
