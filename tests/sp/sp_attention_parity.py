"""SP parity on a small real Wan DiT: USP attention + shard/gather vs full-sequence reference.

Both paths use the same local attention kernel (SDPA), so any diff comes from
the SP collectives' float summation order and must stay within bf16 tolerance.
Checks forward output, input grad, and per-block self-attn + proj_out weight
grads, with and without gradient checkpointing.

Usage: torchrun --standalone --nproc_per_node=N tests/sp/sp_attention_parity.py \
    --sp S [--ulysses U] [--ckpt] [--fp32]
"""

import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import WanTransformer3DModel

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
    )
    model = model.to(device=device, dtype=DTYPE)
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


def _set_ref_processor(model):
    # parallel_state=None: same processor, plain SDPA self-attention.
    model.set_attn_processor(WanUSPAttnProcessor(None))


def _run(model, hidden, enc, ts, out_grad, ckpt):
    if ckpt:
        model.enable_gradient_checkpointing()
    else:
        model.disable_gradient_checkpointing()
    inp = hidden.clone().requires_grad_(True)
    out = model(hidden_states=inp, timestep=ts, encoder_hidden_states=enc, return_dict=False)[0]
    out.backward(out_grad)
    return out.detach(), inp.grad.detach()


def _report(name, a, b, rtol, ctol):
    rel = (a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)
    cos = F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)
    ok = rel < rtol and cos > ctol
    if dist.get_rank() == 0:
        print(f"  [{'OK' if ok else 'FAIL'}] {name:28s} rel={rel:.2e} cos={1 - cos:.2e}(1-)")
    assert ok, f"{name}: rel={rel:.3e} cos={cos:.5f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sp", type=int, default=4)
    p.add_argument("--ulysses", type=int, default=0)
    p.add_argument("--ckpt", action="store_true")
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

    attn_weight_names = [
        f"blocks.{i}.attn1.{proj}.weight" for i in range(2) for proj in ("to_q", "to_k", "to_v", "to_out.0")
    ]
    attn_weight_names.append("proj_out.weight")

    model = build_model(device)
    hidden, enc, ts, out_grad = make_inputs(device)
    _set_ref_processor(model)
    out_ref, gin_ref = _run(model, hidden, enc, ts, out_grad, cli.ckpt)
    gw_ref = {n: dict(model.named_parameters())[n].grad.detach().clone() for n in attn_weight_names}
    model.zero_grad(set_to_none=True)

    plan = DiffusersModelBackend(Wan2_2TrainPipelineConfig()).sequence_parallel_plan(model)
    # operator-level parity uses plain slice-backward gather semantics
    apply_sequence_parallel(model, ps, plan, sum_grad=False)
    out_sp, gin_sp = _run(model, hidden, enc, ts, out_grad, cli.ckpt)

    # Each rank backprops only its 1/sp of the tokens; sum across sp restores full grads.
    dist.all_reduce(gin_sp, group=ps.sp_group)
    if rank == 0:
        print(f"[PARITY] sp={cli.sp} ulysses={ps.ulysses_degree} ring={ps.ring_degree} ckpt={cli.ckpt}")
    _report("forward(out)", out_sp, out_ref, rtol=2e-2, ctol=0.9990)
    _report("grad(input)", gin_sp, gin_ref, rtol=4e-2, ctol=0.9980)
    params = dict(model.named_parameters())
    for n in attn_weight_names:
        assert params[n].grad is not None, f"{n} grad is None — all-to-all backward did not propagate"
        g = params[n].grad.detach().clone()
        dist.all_reduce(g, group=ps.sp_group)
        _report(f"grad({n})", g, gw_ref[n], rtol=5e-2, ctol=0.9950)

    dist.barrier()
    if rank == 0:
        print(f"[PARITY OK] sp={cli.sp} u={ps.ulysses_degree} r={ps.ring_degree} ckpt={cli.ckpt}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
