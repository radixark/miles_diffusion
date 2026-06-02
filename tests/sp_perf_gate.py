"""AC-9 训练步 perf 对照闸（torchrun，4 卡，无 reward）。

对照三档训练并行形态在**真实 Wan2.2 per-layer 维度**（dim=5120=40×128, ffn=13824,
qk_norm across_heads, text_dim=4096）下的 fwd+bwd：
  ddp     : dp4, sp1, 参数复制不分片（DDP）
  fsdp    : dp4, sp1, 参数 FSDP 分片
  sp_dp2  : dp2×sp2, 参数 FSDP 分片(dp2) + 序列并行(sp2)
  sp_dp4  : dp1×sp4, 参数不分片(dp1) + 序列并行(sp4)  —— 容量测试

口径：gradient checkpointing 开、bf16、**只 fwd+bwd 不建 Adam**（隔离激活显存这一
SP 的核心收益；optimizer state footprint 另作分析，对 ddp/sp_dp4 会额外 OOM）。
测：峰值显存(torch.cuda.max_memory_allocated)、单步时间(CUDA event)、SP 通信占比
(torch.profiler 汇总集合通信 kernel 自时间)。numerical 护栏由 parity 测试已证(此处不重复)。

为迭代速度默认 num_layers 取代表性子集；ckpt 下峰值≈单层激活，SP 的 1/sp 收益与层数无关。

用法（每档单独进程，避免显存/进程组残留）:
  torchrun --standalone --nproc_per_node=4 sp_perf_gate.py --band fsdp   --seq-frames 32
  torchrun --standalone --nproc_per_node=4 sp_perf_gate.py --band sp_dp2 --seq-frames 32
"""
import argparse
import contextlib

import torch
import torch.distributed as dist
from diffusers import WanTransformer3DModel
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.parallel import DistributedDataParallel

from miles.backends.fsdp_utils.parallel import create_fsdp_parallel_state
from miles.backends.fsdp_utils.sp_attention import apply_sequence_parallel
from miles.utils.distributed_utils import init_gloo_group

DTYPE = torch.bfloat16

# 真实 Wan2.2-A14B per-layer 维度（layers 可调以控迭代速度）
WAN_KW = dict(
    patch_size=(1, 2, 2), num_attention_heads=40, attention_head_dim=128,
    in_channels=16, out_channels=16, text_dim=4096, freq_dim=256, ffn_dim=13824,
    qk_norm="rms_norm_across_heads", cross_attn_norm=True, rope_max_seq_len=1024,
)

BANDS = {
    "ddp":    dict(sp=1, ulysses=0, ring=0),
    "fsdp":   dict(sp=1, ulysses=0, ring=0),
    "sp_dp2": dict(sp=2, ulysses=2, ring=0),
    "sp_dp4": dict(sp=4, ulysses=4, ring=0),
}


def build_model(device, num_layers):
    torch.manual_seed(0)
    model = WanTransformer3DModel(num_layers=num_layers, **WAN_KW).to(device=device, dtype=DTYPE)
    model.enable_gradient_checkpointing()
    model.train()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    return model


def make_inputs(device, frames, hw):
    # latent [B,16,T,H,W]; patch(1,2,2) → seq = T*(H/2)*(W/2)
    g = torch.Generator(device=device).manual_seed(123)
    hidden = torch.randn(1, 16, frames, hw, hw, device=device, dtype=DTYPE, generator=g)
    enc = torch.randn(1, 512, 4096, device=device, dtype=DTYPE, generator=g)
    ts = torch.tensor([500], device=device)
    for t in (hidden, enc):
        dist.broadcast(t, src=0)
    seq = frames * (hw // 2) * (hw // 2)
    return hidden, enc, ts, seq


def run_step(model, hidden, enc, ts):
    out = model(hidden_states=hidden, timestep=ts, encoder_hidden_states=enc, return_dict=False)[0]
    out.sum().backward()
    for p in model.parameters():
        p.grad = None
    return out


def comm_fraction(model, hidden, enc, ts):
    """通信分解：profiler 汇总集合通信 kernel device 自时间，分 SP(ulysses a2a/ring send-recv)
    与 FSDP(all-gather/reduce-scatter) 两类，各除总 device 自时间。"""
    from torch.profiler import ProfilerActivity, profile

    def dev_us(e):  # torch≥2.x 用 self_device_time_total（self_cuda_time_total 已弃→返回0）
        return float(getattr(e, "self_device_time_total", 0.0) or getattr(e, "self_cuda_time_total", 0.0))

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            run_step(model, hidden, enc, ts)
    torch.cuda.synchronize()
    sp_us = fsdp_us = total_us = 0.0
    for e in prof.key_averages():
        cu = dev_us(e)
        total_us += cu
        n = e.key.lower()
        if "nccl" not in n:
            continue
        if any(k in n for k in ("alltoall", "all_to_all", "sendrecv", "send_recv", "p2p")):
            sp_us += cu
        elif any(k in n for k in ("allgather", "all_gather", "reducescatter", "reduce_scatter")):
            fsdp_us += cu
    t = total_us or 1.0
    return sp_us / t, fsdp_us / t, sp_us / 1e3, fsdp_us / 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--band", required=True, choices=list(BANDS))
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--seq-frames", type=int, default=32, help="latent T；seq=T*(hw/2)^2")
    p.add_argument("--hw", type=int, default=44, help="latent H=W")
    p.add_argument("--iters", type=int, default=5)
    cli = p.parse_args()
    cfg = BANDS[cli.band]

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.cuda.current_device()
    init_gloo_group()

    args = argparse.Namespace(sequence_parallel_size=cfg["sp"], ulysses_degree=cfg["ulysses"],
                              ring_degree=cfg["ring"], context_parallel_size=1)
    ps = create_fsdp_parallel_state(args)
    hidden, enc, ts, seq = make_inputs(device, cli.seq_frames, cli.hw)

    torch.cuda.reset_peak_memory_stats()
    model = build_model(device, cli.num_layers)

    if cli.band == "ddp":
        model = DistributedDataParallel(model, device_ids=[device])
    else:
        mp = MixedPrecisionPolicy(param_dtype=DTYPE, reduce_dtype=torch.float32)
        for blk in model.blocks:
            fully_shard(blk, mesh=ps.dp_mesh, mp_policy=mp)
        fully_shard(model, mesh=ps.dp_mesh, mp_policy=mp)
        if cfg["sp"] > 1:
            apply_sequence_parallel(model, ps, compute_dtype=DTYPE)

    oom = False
    try:
        for _ in range(2):  # warmup
            run_step(model, hidden, enc, ts)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(cli.iters):
            run_step(model, hidden, enc, ts)
        ender.record()
        torch.cuda.synchronize()
        step_ms = starter.elapsed_time(ender) / cli.iters
        peak_gb = torch.cuda.max_memory_allocated() / 1e9

        sp_f = fsdp_f = sp_ms = fsdp_ms = 0.0
        if cfg["sp"] > 1:
            sp_f, fsdp_f, sp_ms, fsdp_ms = comm_fraction(model, hidden, enc, ts)
    except torch.cuda.OutOfMemoryError:
        oom = True

    dist.barrier()
    if rank == 0:
        tag = f"{cli.band}(dp{ps.dp_size}×sp{ps.sp_size})"
        if oom:
            print(f"[PERF] {tag:18s} L={cli.num_layers} seq={seq} → **OOM**")
        else:
            extra = (f" SPcomm={sp_f*100:.1f}%({sp_ms:.1f}ms) FSDPcomm={fsdp_f*100:.1f}%({fsdp_ms:.1f}ms)"
                     if cfg["sp"] > 1 else "")
            print(f"[PERF] {tag:18s} L={cli.num_layers} seq={seq} "
                  f"step={step_ms:.1f}ms peak={peak_gb:.2f}GB{extra}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
