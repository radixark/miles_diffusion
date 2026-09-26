"""Phase offload preserves training state and module tensor aliases.

    FSDP actor + AdamW --> clear gradients --> offload --> same Parameter objects + CPU state
            |                                                          |
       control step                                               next actor step
            +-------------------------- equal -------------------------+

CPU coverage checks FSDP bindings, cleared gradients, tied/nonpersistent buffers, and
optimizer state. The CUDA worker checks actual pinned allocations and transfers.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=["fsdp"])

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.testing._internal.distributed.fake_pg  # noqa: F401
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

_MODULE_PATH = Path(__file__).resolve().parents[4] / "miles/backends/fsdp_utils/offload.py"
_SPEC = importlib.util.spec_from_file_location("fsdp_offload_under_test", _MODULE_PATH)
offload = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(offload)


@pytest.fixture
def cpu_mesh():
    dist.init_process_group("fake", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def test_offload_preserves_training_bindings_and_buffer_aliases(cpu_mesh):
    torch.manual_seed(9)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3)).double()
    model.register_buffer("marker", torch.tensor([7.0]))
    model[0].register_buffer("marker_alias", model.marker, persistent=False)
    control = copy.deepcopy(model)
    fully_shard(model[0], mesh=cpu_mesh)
    fully_shard(model, mesh=cpu_mesh)
    parameters = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.01)
    inputs = torch.arange(12, dtype=torch.float64).reshape(4, 3) / 12

    for _ in range(2):
        for module, optim in ((model, optimizer), (control, control_optimizer)):
            optim.zero_grad(set_to_none=True)
            module(inputs).square().mean().backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
        offload.offload_model(model)
        offload.move_optimizer(optimizer, "cpu")
        assert model.marker is model[0].marker_alias
        assert "0.marker_alias" not in model.state_dict()
        for name, parameter in model.named_parameters():
            assert parameter is parameters[name]
            assert parameter.grad is None
            assert isinstance(parameter, DTensor)
            torch.testing.assert_close(parameter.full_tensor(), control.get_parameter(name))
        assert all(parameter in optimizer.state for parameter in parameters.values())


def test_optimizer_move_preserves_nested_state_and_metadata():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    optimizer.state[parameter] = {
        "momentum_buffer": parameter.detach().clone(),
        "schedule": {"step": 7, "history": [torch.tensor(3.0), "constant"]},
    }
    offload.move_optimizer(optimizer, "cpu")
    assert optimizer.param_groups[0]["params"][0] is parameter
    assert optimizer.state[parameter]["schedule"]["step"] == 7
    assert optimizer.state[parameter]["schedule"]["history"][1] == "constant"
    torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], parameter)
    torch.testing.assert_close(optimizer.state[parameter]["schedule"]["history"][0], torch.tensor(3.0))
