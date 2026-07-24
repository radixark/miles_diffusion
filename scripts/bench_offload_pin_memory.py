"""Head-to-head: pageable (model.cpu()/cuda()) vs pinned offload for sleep/wake_up.

Builds the real SD3.5 DiT under FSDP2 (matching the train actor's apply_fsdp2),
populates AdamW state, and for each path measures GPU memory freed, sleep/wake
time (median of N), and bit-exact round-trip. Exercises the production code:
model.cpu()/move_torch_optimizer for the baseline, PinnedCPUOffload for the new.
"""

import os
import statistics as st
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from miles.backends.fsdp_utils.actor import apply_fsdp2, move_torch_optimizer
from miles.backends.fsdp_utils.offload_utils import PinnedCPUOffload, optimizer_state_cells

MODEL = os.environ.get("SD3_MODEL", "stabilityai/stable-diffusion-3.5-medium")
N = int(os.environ.get("BENCH_ITERS", "10"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))
GB = 1024**3


def r0(*a):
    if dist.get_rank() == 0:
        print(*a, flush=True)


def local(t):
    return t._local_tensor if hasattr(t, "_local_tensor") else t


def fingerprint(model, opt):
    fp = []
    for p in model.parameters():
        fp.append(float(local(p).double().sum().item()))
    for _k, get, _s in optimizer_state_cells(opt):
        fp.append(float(local(get()).double().sum().item()))
    return fp


def main():
    dist.init_process_group("nccl")
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dev = torch.cuda.current_device()
    world = dist.get_world_size()
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))

    from diffusers import SD3Transformer2DModel

    r0(f"loading {MODEL} transformer (bf16) on {world} ranks ...")
    model = SD3Transformer2DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=torch.bfloat16)
    model.train()
    args = SimpleNamespace(diffusion_forward_dtype="bf16", fsdp_reduce_dtype="fp32", gradient_checkpointing=False)
    apply_fsdp2(model, mesh=mesh, cpu_offload=False, args=args)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p) * 0.01
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    ref = fingerprint(model, opt)
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated() / GB
    r0(f"resident (weights+optim) = {resident:.2f} GB/rank\n")

    mgr = PinnedCPUOffload()

    def sleep_pinned():
        mgr.sleep(model, opt)

    def wake_pinned():
        mgr.wake_up(model, opt, dev)

    def sleep_pageable():
        model.cpu()
        move_torch_optimizer(opt, "cpu")

    def wake_pageable():
        model.cuda()
        move_torch_optimizer(opt, dev)

    def run(name, sleep_fn, wake_fn):
        for _ in range(WARMUP):
            sleep_fn()
            wake_fn()
        dist.barrier()
        s_t, w_t, freed = [], [], None
        for _ in range(N):
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            sleep_fn()
            dist.barrier()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            torch.cuda.empty_cache()
            freed = resident - torch.cuda.memory_allocated() / GB
            wake_fn()
            dist.barrier()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            s_t.append(t1 - t0)
            w_t.append(t2 - t1)
        s, w = st.median(s_t), st.median(w_t)
        ok = fingerprint(model, opt) == ref
        r0(
            f"[{name:8s}] sleep {s*1e3:7.1f} ms | wake {w*1e3:7.1f} ms | cycle {(s+w)*1e3:7.1f} ms "
            f"| freed {freed:.2f} GB/rank | roundtrip_bit_exact={ok}"
        )
        return s, w

    r0(f"=== {N} iters (median), warmup {WARMUP} ===")
    g_s, g_w = run("pageable", sleep_pageable, wake_pageable)
    p_s, p_w = run("pinned", sleep_pinned, wake_pinned)

    r0("\n=== speedup (pageable / pinned) ===")
    r0(f"sleep : {g_s/p_s:5.2f}x   ({g_s*1e3:7.1f} -> {p_s*1e3:7.1f} ms)")
    r0(f"wake  : {g_w/p_w:5.2f}x   ({g_w*1e3:7.1f} -> {p_w*1e3:7.1f} ms)")
    r0(f"cycle : {(g_s+g_w)/(p_s+p_w):5.2f}x   ({(g_s+g_w)*1e3:7.1f} -> {(p_s+p_w)*1e3:7.1f} ms)")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
