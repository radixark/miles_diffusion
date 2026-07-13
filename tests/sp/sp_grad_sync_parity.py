"""SP gradient parity under real FSDP2: full grads must match a single-process
full-sequence reference, for both parameter placements.

dp_sp (production): FSDP shards over the flattened dp x sp mesh; the sequence
gather's sum_grad backward makes FSDP's own reduce restore full grads.
dp (validation anchor, test-only): params shard over dp only (replicated
across sp), slice-backward gather, explicit cross-sp grad all-reduce here in
the test — the obvious mechanism cross-checking the subtle one.
Inputs are broadcast to all ranks, so the reference gradient is topology-free.
Also asserts model outputs are bitwise identical across sp ranks.

Usage: torchrun --standalone --nproc_per_node=4 tests/sp/sp_grad_sync_parity.py \
    [--sp S --ulysses U] [--shard-mode dp|dp_sp] [--fp32]
"""

import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import WanTransformer3DModel
from torch.distributed.tensor import DTensor

from miles.backends.fsdp_utils.configs.wan2_2 import Wan2_2TrainPipelineConfig, WanUSPAttnProcessor
from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend
from miles.backends.fsdp_utils.parallel import create_fsdp_parallel_state
from miles.backends.fsdp_utils.sp_attention import apply_sequence_parallel
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


def make_inputs(device, seed=123):
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(1, 16, 4, 8, 8, device=device, dtype=DTYPE, generator=g)
    enc = torch.randn(1, 32, 4096, device=device, dtype=DTYPE, generator=g)
    ts = torch.tensor([500], device=device)
    out_grad = torch.randn(1, 16, 4, 8, 8, device=device, dtype=DTYPE, generator=g)
    return hidden, enc, ts, out_grad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sp", type=int, default=4)
    p.add_argument("--ulysses", type=int, default=0)
    p.add_argument("--shard-mode", choices=("dp", "dp_sp"), default="dp_sp")
    p.add_argument("--lora", action="store_true", help="train LoRA params only, like the RL recipe")
    p.add_argument("--distinct-dp", action="store_true", help="different data per dp rank, like real RL")
    p.add_argument("--accum", type=int, default=1, help="gradient-accumulation microbatches")
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

    args = argparse.Namespace(
        sequence_parallel_size=cli.sp,
        ulysses_degree=cli.ulysses,
    )
    ps = create_fsdp_parallel_state(args)
    # dp anchor: shard over the dp submesh (sp-replicated params) + slice-backward gather.
    fsdp_mesh = ps.dp_mesh if cli.shard_mode == "dp" else ps.fsdp_mesh
    sum_grad = cli.shard_mode == "dp_sp"
    # One dataset per (microbatch, dp group); seeds are rank-independent so the
    # reference can rebuild every dataset locally.
    datasets = []
    for mb in range(cli.accum):
        seeds = [1000 + (d * 100 if cli.distinct_dp else 0) + mb for d in range(ps.dp_size)]
        datasets.append([make_inputs(device, seed=s) for s in seeds])

    def maybe_lora(m):
        if not cli.lora:
            return m
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(7)
        return get_peft_model(
            m, LoraConfig(r=8, lora_alpha=16, target_modules=["to_q", "to_k", "to_v"], init_lora_weights=False)
        )

    # Full-sequence single-process reference (plain SDPA self-attention). Inputs
    # are broadcast, so every dp replica computes the same full gradient.
    ref = maybe_lora(build_model(device))
    ref.set_attn_processor(WanUSPAttnProcessor(None))
    for mbsets in datasets:
        for hidden, enc, ts, out_grad in mbsets:
            out = ref(hidden_states=hidden, timestep=ts, encoder_hidden_states=enc, return_dict=False)[0]
            out.backward(out_grad / ps.dp_size)
    ref_grads = {n: p.grad.detach().clone() for n, p in ref.named_parameters() if p.grad is not None}

    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    # fp32 reduce matches production apply_fsdp2 (--fsdp-reduce-dtype fp32).
    mp_policy = MixedPrecisionPolicy(param_dtype=DTYPE, reduce_dtype=torch.float32)
    model = maybe_lora(build_model(device))
    for blk in model.blocks:
        fully_shard(blk, mesh=fsdp_mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=fsdp_mesh, mp_policy=mp_policy)
    plan = DiffusersModelBackend(Wan2_2TrainPipelineConfig()).sequence_parallel_plan(model)
    apply_sequence_parallel(model, ps, plan, sum_grad=sum_grad)

    for i, mbsets in enumerate(datasets):
        hidden, enc, ts, out_grad = mbsets[ps.dp_rank]
        out = model(hidden_states=hidden, timestep=ts, encoder_hidden_states=enc, return_dict=False)[0]
        if i == 0:
            o32 = out.detach().float()
            ref0 = o32.clone()
            dist.broadcast(ref0, src=ps.dp_rank * ps.sp_size, group=ps.sp_group)
            diff = (o32 - ref0).abs().max()
            if rank == 0:
                print(f"[OUTPUT] max abs diff across sp ranks = {diff.item():.2e} (must be 0)")
            assert diff.item() == 0.0, "model outputs diverge across sp ranks"
        out.backward(out_grad)

    if cli.shard_mode == "dp":
        # The anchor's explicit cross-sp grad sum; in dp_sp mode FSDP's own
        # reduce-scatter already restores full grads (sum_grad gather).
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
        g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
        r = ref_grads[n]
        assert g.shape == r.shape, f"{n}: {g.shape} != {r.shape}"
        rel = (g.float() - r.float()).abs().max() / r.float().abs().max().clamp_min(1e-6)
        cos = F.cosine_similarity(g.float().flatten(), r.float().flatten(), dim=0)
        checked += 1
        # Secondary band: small-norm tensors (biases) sit at the bf16 noise
        # floor of this tiny test model (ring backward pushes cross-attn K
        # biases to ~7e-2 rel; dp and dp_sp modes produce bit-identical
        # values there, so it is accumulation noise, not placement).
        if not (rel < 5e-2 and cos > 0.99) and not (rel < 1e-1 and cos > 0.995):
            fails += 1
            if rank == 0:
                print(f"  [FAIL] {n:42s} rel={rel:.2e} cos={1 - cos:.2e}(1-)")

    # clip_grad_norm_ must report the same total norm as the reference.
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
    total = total.full_tensor() if isinstance(total, DTensor) else total
    ref_total = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.float()) for g in ref_grads.values()])
    )
    norm_rel = ((total.float() - ref_total) / ref_total).abs()

    dist.barrier()
    if rank == 0:
        print(f"[GRAD-NORM] clip={total.item():.6e} ref={ref_total.item():.6e} rel={norm_rel.item():.2e}")
        assert norm_rel.item() < 5e-2, "clip_grad_norm_ reports a wrong total norm"
        print(
            f"[SP-GRAD-SYNC] mode={cli.shard_mode} dp{ps.dp_size}xsp{ps.sp_size}"
            f"(u{ps.ulysses_degree}r{ps.ring_degree}) checked={checked} fails={fails}"
        )
        assert fails == 0
        print("[SP-GRAD-SYNC OK] full grads == full-sequence reference")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
