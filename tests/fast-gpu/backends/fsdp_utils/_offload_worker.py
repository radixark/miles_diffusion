"""Real CUDA counterpart to the CPU offload contract.

    same initial model --> control CUDA forward / AdamW
              |
       FSDP forward / AdamW --> pinned sleep --> wake --> equal next forward / AdamW

The 2 x 2 x 2 matrix covers frozen references, cached gathered parameters, and
native CPUOffloadPolicy. Uneven local shards must remain pinned after FSDP padding;
optimizer Parameter bindings and tied/nonpersistent buffer aliases must survive.

    GPU/CPU actor x GPU/CPU EMA --> EMA inference --> actor forward/backward
               |                                      |
           pinned sleep <--------- EMA update --------+
               |
        DCP save/load --> same EMA tensors, placement, and update count
"""

import copy
import importlib.util
import itertools
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor

_MODULE_PATH = Path(__file__).resolve().parents[4] / "miles/backends/fsdp_utils/offload.py"
_SPEC = importlib.util.spec_from_file_location("fsdp_offload_under_test", _MODULE_PATH)
offload = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(offload)
sys.modules["miles.backends.fsdp_utils.offload"] = offload
_EMA_SPEC = importlib.util.spec_from_file_location("fsdp_ema_under_test", _MODULE_PATH.with_name("ema.py"))
ema_module = importlib.util.module_from_spec(_EMA_SPEC)
_EMA_SPEC.loader.exec_module(ema_module)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(3, 5)
        self.output = torch.nn.Linear(5, 3)
        self.register_buffer("scale", torch.tensor([1.25]))
        self.input.register_buffer("scale_alias", self.scale, persistent=False)
        self.register_buffer("offset", torch.tensor([0.25]), persistent=False)

    def forward(self, inputs):
        return self.output(self.input(inputs).tanh()) * self.scale + self.offset


def local_tensor(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def check_model_offload_round_trip(mesh, frozen, reshard_after_forward, cpu_offload):
    torch.manual_seed(19)
    model = Tiny().cuda()
    # Build aliases after cuda(), whose default _apply does not preserve buffer aliases.
    model.input.scale_alias = model.scale
    control = copy.deepcopy(model)
    model.requires_grad_(not frozen)
    control.requires_grad_(not frozen)
    model.train(not frozen)
    control.train(not frozen)
    policy = CPUOffloadPolicy() if cpu_offload else OffloadPolicy()
    fully_shard(model.input, mesh=mesh, reshard_after_forward=reshard_after_forward, offload_policy=policy)
    fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward, offload_policy=policy)
    parameters = dict(model.named_parameters())
    optimizer = None if frozen else torch.optim.AdamW(model.parameters(), lr=0.01)
    control_optimizer = None if frozen else torch.optim.AdamW(control.parameters(), lr=0.01)
    inputs = torch.arange(12, dtype=torch.float32, device="cuda").reshape(4, 3) / 12

    for _ in range(2):
        with torch.set_grad_enabled(not frozen):
            actual = model(inputs)
            expected = control(inputs)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            if not frozen:
                actual.square().mean().backward()
                expected.square().mean().backward()
                optimizer.step()
                control_optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                control_optimizer.zero_grad(set_to_none=True)
        offload.offload_model(model)
        assert model.scale is model.input.scale_alias
        assert "input.scale_alias" not in model.state_dict()
        assert "offset" not in model.state_dict()
        for name, parameter in model.named_parameters():
            assert parameter is parameters[name]
            assert local_tensor(parameter).device.type == "cpu" and local_tensor(parameter).is_pinned(), name
        assert all(buffer.device.type == "cpu" and buffer.is_pinned() for buffer in model.buffers())
        if optimizer is not None:
            # Only state copied off the GPU lands in pinned memory; CPU-resident state (step, native offload) stays.
            copied_from_cuda = {
                (id(parameter), key): local_tensor(value).is_cuda
                for parameter, state in optimizer.state.items()
                for key, value in state.items()
            }
            offload.move_optimizer(optimizer, "cpu")
            for parameter, state in optimizer.state.items():
                assert any(parameter is original for original in parameters.values())
                for key, value in state.items():
                    assert local_tensor(value).device.type == "cpu"
                    assert local_tensor(value).is_pinned() or not copied_from_cuda[(id(parameter), key)], key

        offload.onload_model(model, cpu_offload=cpu_offload)
        assert model.scale is model.input.scale_alias
        expected_parameter_device = "cpu" if cpu_offload else "cuda"
        assert all(
            local_tensor(parameter).device.type == expected_parameter_device for parameter in model.parameters()
        )
        assert all(buffer.device.type == "cuda" for buffer in model.buffers())
        if optimizer is not None:
            offload.move_optimizer(optimizer, expected_parameter_device)
        assert all(parameter is parameters[name] for name, parameter in model.named_parameters())


def check_ema_offload_and_checkpoint(mesh, cpu_offload):
    torch.manual_seed(31)
    model = Tiny().cuda()
    model.input.scale_alias = model.scale
    model.input.requires_grad_(False)
    control = copy.deepcopy(model)
    policy = CPUOffloadPolicy() if cpu_offload else OffloadPolicy()
    fully_shard(model.input, mesh=mesh, reshard_after_forward=False, offload_policy=policy)
    fully_shard(model, mesh=mesh, reshard_after_forward=False, offload_policy=policy)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.01)
    control_optimizer = torch.optim.AdamW(
        (parameter for parameter in control.parameters() if parameter.requires_grad), lr=0.01
    )
    ema = ema_module.EMAOptimizer(model, decay=0.5, flat_steps=10)
    expected_ema = {
        name: parameter.detach().clone() for name, parameter in control.named_parameters() if parameter.requires_grad
    }
    inputs = torch.arange(12, dtype=torch.float32, device="cuda").reshape(4, 3) / 12
    for _ in range(2):
        averaged_control = copy.deepcopy(control)
        with torch.no_grad():
            for name, parameter in averaged_control.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(expected_ema[name])
        with torch.no_grad(), ema.use_weights(model):
            torch.testing.assert_close(model(inputs), averaged_control(inputs), atol=2e-6, rtol=2e-6)

        actor_output = model(inputs)
        control_output = control(inputs)
        torch.testing.assert_close(actor_output, control_output, atol=2e-6, rtol=2e-6)
        actor_output.square().mean().backward()
        control_output.square().mean().backward()
        optimizer.step()
        control_optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        control_optimizer.zero_grad(set_to_none=True)
        assert ema.step() == 0.5
        for name, parameter in control.named_parameters():
            if parameter.requires_grad:
                expected_ema[name].lerp_(parameter.detach(), 0.5)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                average = ema.state[parameter]["ema"]
                assert isinstance(average, DTensor)
                expected_device = "cpu" if cpu_offload else "cuda"
                assert local_tensor(average).device.type == expected_device
                torch.testing.assert_close(average.to("cuda").full_tensor(), expected_ema[name], atol=2e-6, rtol=2e-6)
            else:
                assert parameter not in ema.state

        offload.offload_model(model)
        offload.move_optimizer(optimizer, "cpu")
        offload.move_optimizer(ema, "cpu")
        # With native CPU offload the EMA never leaves the CPU, so only GPU-resident EMA becomes pinned.
        assert all(local_tensor(state["ema"]).is_pinned() or cpu_offload for state in ema.state.values())
        offload.onload_model(model, cpu_offload=cpu_offload)
        offload.move_optimizer(optimizer, "cpu" if cpu_offload else "cuda")
        offload.move_optimizer(ema, "cpu" if cpu_offload else "cuda")

    checkpoint_path = [tempfile.mkdtemp(prefix="stage1_ema_") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(checkpoint_path, src=0)
    try:
        dcp.save({"ema": get_optimizer_state_dict(model, ema)}, checkpoint_id=checkpoint_path[0])
        resumed = ema_module.EMAOptimizer(model, decay=0.1)
        state_dict = {"ema": get_optimizer_state_dict(model, resumed)}
        dcp.load(state_dict, checkpoint_id=checkpoint_path[0])
        set_optimizer_state_dict(model, resumed, state_dict["ema"])
        assert resumed.update_count == ema.update_count == 2
        assert resumed.param_groups[0]["decay"] == 0.5
        for parameter, state in resumed.state.items():
            average = state["ema"]
            expected = ema.state[parameter]["ema"]
            assert isinstance(average, DTensor) and average.placements == expected.placements
            assert local_tensor(average).device == local_tensor(expected).device
            torch.testing.assert_close(local_tensor(average), local_tensor(expected))
    finally:
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(checkpoint_path[0])


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    for frozen, reshard_after_forward, cpu_offload in itertools.product((False, True), repeat=3):
        check_model_offload_round_trip(mesh, frozen, reshard_after_forward, cpu_offload)
    for cpu_offload in (False, True):
        check_ema_offload_and_checkpoint(mesh, cpu_offload)
    dist.barrier()
    if dist.get_rank() == 0:
        print("OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
