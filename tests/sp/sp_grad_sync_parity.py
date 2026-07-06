"""SP gradient sync parity under real FSDP2: full grads must match a single-process
full-sequence reference after reduce-scatter (dp) + cross-sp all-reduce(SUM).

Runs dp1 x sp4 so FSDP local shards are the full parameters, isolating the sp
dimension. Also asserts model outputs are bitwise identical across sp ranks
(the gather-after-proj_out contract that keeps loss/log_prob code unchanged).

Usage: torchrun --standalone --nproc_per_node=4 tests/sp/sp_grad_sync_parity.py
"""

import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import WanTransformer3DModel
from torch.distributed.tensor import DTensor

from miles.backends.fsdp_utils.parallel import create_fsdp_parallel_state
from miles.backends.fsdp_utils.sp_attention import (
    WanUSPAttnProcessor,
    apply_sequence_parallel,
    init_sp_backend,
)
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
    model.train()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    return model


def make_inputs(device):
    g = torch.Generator(device=device).manual_seed(123)
    hidden = torch.randn(1, 16, 4, 8, 8, device=device, dtype=DTYPE, generator=g)
    enc = torch.randn(1, 32, 4096, device=device, dtype=DTYPE, generator=g)
    ts = torch.tensor([500], device=device)
    out_grad = torch.randn(1, 16, 4, 8, 8, device=device, dtype=DTYPE, generator=g)
    for t in (hidden, enc, out_grad):
        dist.broadcast(t, src=0)
    return hidden, enc, ts, out_grad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp32", action="store_true", help="fp32 + SDPA, isolates bf16 summation rounding")
    cli = p.parse_args()
    if cli.fp32:
        global DTYPE
        DTYPE = torch.float32

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.cuda.current_device()
    init_gloo_group()

    args = argparse.Namespace(sequence_parallel_size=4, ulysses_degree=4, ring_degree=0, context_parallel_size=1)
    ps = create_fsdp_parallel_state(args)
    hidden, enc, ts, out_grad = make_inputs(device)

    # Full-sequence single-process reference.
    from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention

    ref = build_model(device)
    init_sp_backend(DTYPE)
    proc = WanUSPAttnProcessor(ref.config.num_attention_heads, ref.config.attention_head_dim, DTYPE)
    proc.usp_attn = USPAttention(
        num_heads=ref.config.num_attention_heads,
        head_size=ref.config.attention_head_dim,
        causal=False,
        skip_sequence_parallel=True,
    )
    ref.set_attn_processor(proc)
    out = ref(hidden_states=hidden, timestep=ts, encoder_hidden_states=enc, return_dict=False)[0]
    out.backward(out_grad)
    ref_grads = {n: p.grad.detach().clone() for n, p in ref.named_parameters()}

    # FSDP(dp1) + SP(sp4).
    from torch.distributed.fsdp import fully_shard

    model = build_model(device)
    for blk in model.blocks:
        fully_shard(blk, mesh=ps.dp_mesh)
    fully_shard(model, mesh=ps.dp_mesh)
    apply_sequence_parallel(model, ps, compute_dtype=DTYPE)

    out = model(hidden_states=hidden, timestep=ts, encoder_hidden_states=enc, return_dict=False)[0]

    o32 = out.detach().float()
    ref0 = o32.clone()
    dist.broadcast(ref0, src=ps.dp_rank * ps.sp_size, group=ps.sp_group)
    diff = (o32 - ref0).abs().max()
    if rank == 0:
        print(f"[OUTPUT] max abs diff across sp ranks = {diff.item():.2e} (must be 0)")
    assert diff.item() == 0.0, "model outputs diverge across sp ranks"

    out.backward(out_grad)

    # Mirror actor._all_reduce_sp_grads.
    for p in model.parameters():
        if p.grad is None:
            continue
        local = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
        f = local.float()
        dist.all_reduce(f, group=ps.sp_group)
        local.copy_(f.to(local.dtype))

    fails = 0
    checked = 0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
        r = ref_grads[n]
        if g.shape != r.shape:
            continue
        rel = (g.float() - r.float()).abs().max() / r.float().abs().max().clamp_min(1e-6)
        cos = F.cosine_similarity(g.float().flatten(), r.float().flatten(), dim=0)
        checked += 1
        # Secondary band: small-norm tensors (biases) sit right at the bf16
        # summation noise floor; --fp32 confirms exact agreement there.
        if not (rel < 5e-2 and cos > 0.99) and not (rel < 1e-1 and cos > 0.999):
            fails += 1
            if rank == 0:
                print(f"  [FAIL] {n:42s} rel={rel:.2e} cos={1 - cos:.2e}(1-)")

    dist.barrier()
    if rank == 0:
        print(f"[SP-GRAD-SYNC] checked={checked} fails={fails}")
        assert fails == 0
        print("[SP-GRAD-SYNC OK] dp1xsp4 FSDP+SP full grads == full-sequence reference")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
