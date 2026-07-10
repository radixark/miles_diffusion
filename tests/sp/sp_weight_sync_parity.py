"""Weight-sync parity under FSDP + SP: every rank's reconstructed full weights must
hash identically and match a single-process reference.

Reconstruction uses the same redistribute([Replicate()]) path as update_weights;
the checksum is sglang's compute_weights_checksum (name + dtype + shape + bytes),
the same function the online verify uses. Also asserts shape/dtype sensitivity.

Usage:
    torchrun --standalone --nproc_per_node=4 tests/sp/sp_weight_sync_parity.py --sp 2 --ulysses 2
    torchrun --standalone --nproc_per_node=4 tests/sp/sp_weight_sync_parity.py --sp 4 --ulysses 4
    torchrun --standalone --nproc_per_node=4 tests/sp/sp_weight_sync_parity.py --sp 4 --ulysses 2 --ring 2
"""

import argparse

import torch
import torch.distributed as dist
from diffusers import WanTransformer3DModel
from sglang.multimodal_gen.runtime.loader.weight_utils import compute_weights_checksum
from torch.distributed.fsdp import fully_shard

from miles.backends.fsdp_utils.parallel import create_fsdp_parallel_state
from miles.utils.distributed_utils import init_gloo_group

DTYPE = torch.bfloat16


def build_model(device):
    torch.manual_seed(0)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=8,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=1024,
        num_layers=2,
        rope_max_seq_len=1024,
    ).to(device=device, dtype=DTYPE)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)
    return model


def full_state_pairs(model):
    """Mirror update_weights' redistribute([Replicate()]).to_local() reconstruction."""
    from torch.distributed.tensor import DTensor, Replicate

    pairs = []
    for name, param in model.state_dict().items():
        param = param.cuda()
        if isinstance(param, DTensor):
            param = param.redistribute(placements=[Replicate()] * param.device_mesh.ndim).to_local()
        pairs.append((name, param.detach().cpu().contiguous()))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sp", type=int, default=2)
    p.add_argument("--ulysses", type=int, default=2)
    p.add_argument("--ring", type=int, default=0)
    p.add_argument("--shard-mode", choices=("dp", "dp_sp"), default="dp_sp")
    cli = p.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.cuda.current_device()
    init_gloo_group()

    args = argparse.Namespace(
        sequence_parallel_size=cli.sp,
        ulysses_degree=cli.ulysses,
        ring_degree=cli.ring,
        fsdp_shard_mode=cli.shard_mode,
    )
    ps = create_fsdp_parallel_state(args)

    ref_model = build_model(device)
    ref_sum = compute_weights_checksum(
        [(n, pa.detach().cpu().contiguous()) for n, pa in ref_model.state_dict().items()]
    )

    model = build_model(device)
    for blk in model.blocks:
        fully_shard(blk, mesh=ps.fsdp_mesh)
    fully_shard(model, mesh=ps.fsdp_mesh)

    my_sum = compute_weights_checksum(full_state_pairs(model))

    gathered = [None] * world
    dist.all_gather_object(gathered, my_sum)

    if rank == 0:
        all_equal = all(s == gathered[0] for s in gathered)
        match_ref = gathered[0] == ref_sum
        print(
            f"[WEIGHT-SYNC] world={world} mode={cli.shard_mode} dp{ps.dp_size}xsp{ps.sp_size}(u{ps.ulysses_degree}r{ps.ring_degree})"
        )
        print(f"[WEIGHT-SYNC] all ranks equal={all_equal}  == single-process ref={match_ref}")
        assert all_equal, "reconstructed full weights differ across ranks"
        assert match_ref, "reconstructed full weights != single-process reference"

        # reshape keeps bytes identical, so only the shape term can catch it
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        assert compute_weights_checksum([("w", t)]) != compute_weights_checksum([("w", t.reshape(4, 3))])
        assert compute_weights_checksum([("w", t)]) != compute_weights_checksum([("w", t.to(torch.float64))])
        print("[SP-WEIGHT-SYNC OK] bitwise-identical reconstruction + dtype/shape sensitivity")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
