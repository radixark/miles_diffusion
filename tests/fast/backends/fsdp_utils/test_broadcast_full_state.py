"""broadcast_full_state_to_fsdp gives every rank rank 0's params and buffers (CPU, gloo, FSDP2)."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(8, 16)
        self.b = torch.nn.Linear(16, 4, bias=False)
        self.register_buffer("persistent", torch.arange(6, dtype=torch.float32))
        self.register_buffer("table", torch.arange(5, dtype=torch.float32) * 0.5, persistent=False)


def _reference():
    torch.manual_seed(0)
    net = _Net()
    with torch.no_grad():
        net.persistent.mul_(3)
        net.table.add_(7)
    return net


def _worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        from miles.backends.fsdp_utils.checkpoint import broadcast_full_state_to_fsdp

        if rank == 0:
            model = _reference()
            full_state = model.state_dict()
        else:
            from accelerate import init_empty_weights

            with init_empty_weights(include_buffers=False):
                model = _Net()  # params on meta, buffers hold constructor values, not rank 0's
            full_state = {}
        mesh = init_device_mesh("cpu", (world,))
        fully_shard(model, mesh=mesh)
        broadcast_full_state_to_fsdp(model, full_state, cpu_offload=False, device="cpu")
        ref = _reference()
        params_ok = all(
            torch.equal(p.full_tensor(), dict(ref.named_parameters())[n]) for n, p in model.named_parameters()
        )
        buffers_ok = all(torch.equal(b, dict(ref.named_buffers())[n]) for n, b in model.named_buffers())
        out[rank] = (params_ok, buffers_ok)
    finally:
        dist.destroy_process_group()


def test_every_rank_gets_rank0_params_and_buffers():
    pytest.importorskip("accelerate")
    world = 2
    port = 29500 + os.getpid() % 1000
    with mp.Manager() as manager:
        out = manager.dict()
        mp.spawn(_worker, args=(world, port, out), nprocs=world, join=True)
        assert dict(out) == {0: (True, True), 1: (True, True)}


class _CkptNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(6, 10)  # weight + bias straight from the checkpoint (bias stored as F32)
        self.extra = torch.nn.Linear(6, 3, bias=False)  # not in the checkpoint, like a LoRA adapter
        self.renamed = torch.nn.Linear(6, 4, bias=False)  # stored under another key
        self.permuted = torch.nn.Linear(6, 8, bias=False)  # same key and shape, but the loader reorders rows
        self.register_buffer("table", torch.arange(3, dtype=torch.float32), persistent=False)


def _write_checkpoint(root):
    from safetensors.torch import save_file

    torch.manual_seed(1)
    raw = {
        "a.weight": torch.randn(10, 6).to(torch.bfloat16),
        "a.bias": torch.randn(10, dtype=torch.float32),
        "old_name.weight": torch.randn(4, 6).to(torch.bfloat16),
        "permuted.weight": torch.randn(8, 6).to(torch.bfloat16),
    }
    comp = os.path.join(root, "transformer")
    os.makedirs(comp, exist_ok=True)
    save_file({k: raw[k] for k in ("a.weight", "a.bias")}, os.path.join(comp, "model-1.safetensors"))
    save_file({k: raw[k] for k in ("old_name.weight", "permuted.weight")}, os.path.join(comp, "model-2.safetensors"))
    weight_map = {"a.weight": "model-1.safetensors", "a.bias": "model-1.safetensors"}
    weight_map.update({"old_name.weight": "model-2.safetensors", "permuted.weight": "model-2.safetensors"})
    with open(os.path.join(comp, "model.safetensors.index.json"), "w") as f:
        import json

        json.dump({"weight_map": weight_map}, f)
    return raw


def _loaded_reference(raw):
    """What rank 0's loader produces: bf16 params, a converted tensor, a fresh adapter."""
    torch.manual_seed(2)
    net = _CkptNet().to(torch.bfloat16)
    with torch.no_grad():
        net.a.weight.copy_(raw["a.weight"])
        net.a.bias.copy_(raw["a.bias"].to(torch.bfloat16))
        net.renamed.weight.copy_(raw["old_name.weight"])
        net.permuted.weight.copy_(raw["permuted.weight"].flip(0))
        net.table.add_(5)
    return net


def _ckpt_worker(rank, world, port, root, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        from miles.backends.fsdp_utils import checkpoint as ck
        from miles.backends.fsdp_utils import sharded_checkpoint_fill
        from miles.backends.fsdp_utils.checkpoint import broadcast_full_state_to_fsdp

        seen = {}
        raw_fill, checked_fill = sharded_checkpoint_fill.fill_from_checkpoint, ck._fill_from_checkpoint_files

        def record_raw(*a, **k):
            seen["raw"] = sorted(raw_fill(*a, **k))
            return set(seen["raw"])

        def record_checked(*a, **k):
            seen["checked"] = sorted(checked_fill(*a, **k))
            return set(seen["checked"])

        sharded_checkpoint_fill.fill_from_checkpoint = record_raw
        ck._fill_from_checkpoint_files = record_checked

        raw = _write_checkpoint(root) if rank == 0 else None
        dist.barrier()
        if raw is None:
            from safetensors.torch import load_file

            raw = {}
            for name in ("model-1.safetensors", "model-2.safetensors"):
                raw.update(load_file(os.path.join(root, "transformer", name)))
        if rank == 0:
            model = _loaded_reference(raw)
            full_state = model.state_dict()
        else:
            from accelerate import init_empty_weights

            with init_empty_weights(include_buffers=False):
                model = _CkptNet().to(torch.bfloat16)
            full_state = {}
        fully_shard(model, mesh=init_device_mesh("cpu", (world,)))
        files = sharded_checkpoint_fill.resolve_component_files(root, "transformer")
        broadcast_full_state_to_fsdp(model, full_state, cpu_offload=False, device="cpu", checkpoint_files=files)
        ref = _loaded_reference(raw)
        params_ok = {
            n: torch.equal(p.full_tensor(), dict(ref.named_parameters())[n]) for n, p in model.named_parameters()
        }
        buffers_ok = all(torch.equal(b, dict(ref.named_buffers())[n]) for n, b in model.named_buffers())
        out[rank] = (params_ok, buffers_ok, len(files), seen["raw"], seen["checked"])
    finally:
        dist.destroy_process_group()


def test_checkpoint_fill_matches_rank0_load_and_falls_back_for_the_rest(tmp_path):
    pytest.importorskip("accelerate")
    pytest.importorskip("safetensors")
    world = 2
    port = 30500 + os.getpid() % 1000
    with mp.Manager() as manager:
        out = manager.dict()
        mp.spawn(_ckpt_worker, args=(world, port, str(tmp_path), out), nprocs=world, join=True)
        result = dict(out)
    for rank in range(world):
        params_ok, buffers_ok, n_files, raw, checked = result[rank]
        assert n_files == 2
        assert raw == ["a.bias", "a.weight", "permuted.weight"], raw  # name and shape matched
        assert checked == ["a.bias", "a.weight"], checked  # rank 0's spot check rejected the converted one
        assert all(params_ok.values()), (rank, params_ok)
        assert buffers_ok
