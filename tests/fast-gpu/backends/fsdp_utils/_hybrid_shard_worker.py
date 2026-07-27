"""Worker for test_hybrid_shard, one configuration per launch."""

import argparse
import os
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import HSDPMeshInfo
from torch.distributed.tensor import DTensor, Replicate

from miles.backends.fsdp_utils.actor import load_sharded_model
from miles.backends.fsdp_utils.checkpoint import ModelState
from miles.backends.fsdp_utils.parallel import create_fsdp_parallel_state
from miles.backends.fsdp_utils.sequence_parallel.plan import SequenceParallelPlan, apply_sequence_parallel
from miles.utils.distributed_utils import init_gloo_group

DIM = 64
OPTIMIZER_STEPS = 5
SEQUENCE_LENGTH = 8

MINIMAL_SP_PLAN = SequenceParallelPlan(
    boundaries={
        "input": {"input": ContextParallelInput(split_dim=1, expected_dims=3)},
        "output": ContextParallelOutput(gather_dim=1, expected_dims=3),
    },
    num_attention_heads=4,
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x):
        return self.fc(x).relu()


class Tiny(nn.Module):
    def __init__(self, depth=4):
        super().__init__()
        self.input = nn.Identity()
        self.blocks = nn.ModuleList(Block() for _ in range(depth))
        self.output = nn.Identity()

    def forward(self, x):
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def check_fully_shard_topology(model, dp_replicate, world_size):
    """fully_shard should use hybrid sharding exactly when replicas are requested."""
    # The root wrapper owns no parameters of its own, so read the info off a block.
    mesh_info = model.blocks[0]._get_fsdp_state()._fsdp_param_group.mesh_info
    uses_hybrid_shard = isinstance(mesh_info, HSDPMeshInfo)
    assert uses_hybrid_shard == (
        dp_replicate > 1
    ), f"dp_replicate={dp_replicate} but fully_shard used {type(mesh_info).__name__}"
    if uses_hybrid_shard:
        assert (mesh_info.replicate_mesh_dim, mesh_info.shard_mesh_dim) == (0, 1)
        assert mesh_info.replicate_mesh_size == dp_replicate
        assert mesh_info.shard_mesh_size == world_size // dp_replicate


def check_gradient_is_world_mean(model, reference, x, world_size):
    """FSDP must divide by dp_shard x sp x dp_replicate, not just the shard axis."""
    model(x).sum().backward()
    grad = model.blocks[0].fc.weight.grad
    got = (grad.full_tensor() if isinstance(grad, DTensor) else grad).clone()

    plain = Tiny().cuda()
    plain.load_state_dict(reference)
    plain(x).sum().backward()
    expected = plain.blocks[0].fc.weight.grad.clone()
    dist.all_reduce(expected)
    expected /= world_size
    torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-5)


def check_weight_sync_gather(model, reference):
    """The shape the weight-sync updater uses to rebuild a full parameter."""
    param = model.blocks[0].fc.weight
    full = param.redistribute(placements=[Replicate()] * param.device_mesh.ndim).to_local()
    torch.testing.assert_close(full.cpu(), reference["blocks.0.fc.weight"])


def check_dcp_round_trip(model, reference, rank):
    path = [tempfile.mkdtemp(dir="/tmp", prefix="hybrid_shard_ckpt_") if rank == 0 else None]
    dist.broadcast_object_list(path, src=0)
    ckpt_dir = path[0]
    try:
        dcp.save({"model_state": ModelState(model)}, checkpoint_id=ckpt_dir)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        dcp.load({"model_state": ModelState(model)}, checkpoint_id=ckpt_dir)
        for name, param in model.named_parameters():
            torch.testing.assert_close(param.full_tensor().cpu(), reference[name])
    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(ckpt_dir, ignore_errors=True)


def check_replicas_do_not_drift(model, state):
    """full_tensor() would agree by construction, so compare raw local shards after steps."""
    replicate_mesh = state.get_mesh("fsdp")["dp_replicate"]
    data_rank = state.get_mesh("dp").get_local_rank()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for step in range(OPTIMIZER_STEPS):
        optimizer.zero_grad()
        torch.manual_seed(1000 * step + data_rank)
        model(torch.randn(4, SEQUENCE_LENGTH, DIM, device="cuda")).sum().backward()
        optimizer.step()
        for name, param in model.named_parameters():
            shard = param.to_local().contiguous()
            peers = [torch.empty_like(shard) for _ in range(replicate_mesh.size())]
            dist.all_gather(peers, shard, group=replicate_mesh.get_group())
            for peer in peers[1:]:
                torch.testing.assert_close(peers[0], peer, rtol=0, atol=0, msg=f"{name} drifted at step {step}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-replicate-size", type=int, default=1)
    parser.add_argument("--sequence-parallel-size", type=int, default=1)
    parser.add_argument("--ulysses-degree", type=int, default=0)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    init_gloo_group()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    state = create_fsdp_parallel_state(args)

    torch.manual_seed(0)
    model = Tiny().cuda()
    # Doubles as the rank0 full state dict, which load_sharded_model expects on CPU.
    reference = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    for block in model.blocks:
        fully_shard(block, mesh=state.get_mesh("fsdp"), mp_policy=mp_policy)
    fully_shard(model, mesh=state.get_mesh("fsdp"), mp_policy=mp_policy)
    if state.get_optional_mesh("sp") is not None:
        apply_sequence_parallel(model, state, MINIMAL_SP_PLAN, lambda *_: None)

    check_fully_shard_topology(model, args.dp_replicate_size, world_size)

    # set_model_state_dict moves rank0's tensors onto the device, so hand it a copy.
    load_sharded_model(model, {k: v.clone() for k, v in reference.items()} if rank == 0 else {}, cpu_offload=False)
    for name, param in model.named_parameters():
        torch.testing.assert_close(param.full_tensor().cpu(), reference[name])

    torch.manual_seed(100 + state.get_mesh("dp").get_local_rank())
    check_gradient_is_world_mean(
        model,
        reference,
        torch.randn(4, SEQUENCE_LENGTH, DIM, device="cuda"),
        world_size,
    )
    check_weight_sync_gather(model, reference)
    check_dcp_round_trip(model, reference, rank)
    if args.dp_replicate_size > 1:
        check_replicas_do_not_drift(model, state)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("OK", flush=True)


if __name__ == "__main__":
    main()
