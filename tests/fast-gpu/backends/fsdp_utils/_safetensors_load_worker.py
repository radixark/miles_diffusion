"""Two ranks read their own halves of a safetensors checkpoint, or fall back to rank 0's broadcast together.

    files hold the state dict --> both ranks on meta --> rows [0:2] on rank 0, rows [2:4] on rank 1, GPU or CPU offload
    another name in the files --> rank 0 builds again with weights --> the broadcast value on both ranks
    LoRA, saved or new        --> the adapter shards and the generator the broadcast leaves, on both ranks
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tests.fast.backends.fsdp_utils.test_reference_models import (
    TinyPipelineConfig,
    make_model_loader_args,
    save_component_lora_adapters,
)
from tests.fast.backends.fsdp_utils.test_safetensors_load import (
    BROADCAST_VALUE,
    FileBackend,
    save_checkpoint,
    saved_weight,
)
from torch.distributed.device_mesh import init_device_mesh

from miles.backends.fsdp_utils.model_loader import load_fsdp_models


def load(mesh, path, cpu_offload, *, lora_init=None, **kwargs):
    parallel = SimpleNamespace(get_mesh=lambda name: mesh, get_optional_mesh=lambda name: None)
    args, backend = make_model_loader_args(), FileBackend()
    args.use_lora = lora_init is not None
    args.lora_rank, args.lora_alpha, args.lora_target_modules, args.lora_init_weights = 2, 4, ["proj"], lora_init
    models = load_fsdp_models(
        args, backend, TinyPipelineConfig(), parallel, checkpoint_path=str(path), cpu_offload=cpu_offload, **kwargs
    )
    for model in models.values():
        base = model.base_model.model if kwargs else model
        weight = base.block.proj.base_layer.weight if kwargs else base.block.proj.weight
        shard = weight.to_local()
        assert shard.shape == (2, 3) and shard.dtype == torch.float32
        assert shard.device.type == ("cpu" if cpu_offload else "cuda")
        assert base.forward_count.device.type == "cuda" and base.forward_count.item() == 0
    return models, backend.materialize_requests


def adapter_shards(models):
    return {
        (component, name): parameter.to_local().cpu()
        for component, model in models.items()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


def check_lora(mesh, root, rank, cpu_offload, **kwargs):
    # read per rank
    torch.manual_seed(0)
    models, materialize_requests = load(mesh, root / "state", cpu_offload, **kwargs)
    assert materialize_requests == [False, False]
    for component, model in models.items():
        base = model.base_model.model.block.proj.base_layer.weight.to_local().cpu()
        assert torch.equal(base, saved_weight(component).float()[2 * rank : 2 * rank + 2])
    adapters, generator = adapter_shards(models), torch.get_rng_state()
    assert len(adapters) == 4 and any(shard.abs().sum() > 0 for shard in adapters.values())
    # rank 0 builds with weights, then the real broadcast
    torch.manual_seed(0)
    models, materialize_requests = load(mesh, root / "renamed", cpu_offload, **kwargs)
    assert materialize_requests == ([False, True, False, True] if rank == 0 else [False, False, False, False])
    broadcast = adapter_shards(models)
    assert adapters.keys() == broadcast.keys() and all(torch.equal(adapters[key], broadcast[key]) for key in adapters)
    assert torch.equal(torch.get_rng_state(), generator)


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    rank, root = dist.get_rank(), Path(sys.argv[1])
    if rank == 0:
        save_checkpoint(root / "state")
        save_checkpoint(root / "renamed", name="block.proj.kernel")
        save_component_lora_adapters(root / "adapters")
    dist.barrier()
    inputs = torch.ones(2, 3, device="cuda")
    for cpu_offload in (False, True):
        models, materialize_requests = load(mesh, root / "state", cpu_offload)
        assert materialize_requests == [False, False]
        for component, model in models.items():
            expected = saved_weight(component).float()
            assert torch.equal(model.block.proj.weight.to_local().cpu(), expected[2 * rank : 2 * rank + 2])
            with torch.no_grad():
                torch.testing.assert_close(model(inputs).cpu(), (inputs.cpu() @ expected.T).tanh())

        models, materialize_requests = load(mesh, root / "renamed", cpu_offload)
        assert materialize_requests == ([False, True, False, True] if rank == 0 else [False, False])
        for model in models.values():
            assert torch.all(model.block.proj.weight.to_local() == BROADCAST_VALUE)

        adapter_paths = [str(root / "adapters" / component) for component in ("transformer", "transformer_2")]
        check_lora(mesh, root, rank, cpu_offload, lora_adapter_paths=adapter_paths)
        check_lora(mesh, root, rank, cpu_offload, lora_init="gaussian", trainable=True)
    dist.barrier()
    if rank == 0:
        print("OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
