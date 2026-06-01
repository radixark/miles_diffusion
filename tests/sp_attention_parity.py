"""阶段2 SP 算子 parity（torchrun，需 NCCL + flash-attn）。AC-3 / AC-4。

在真实 Wan DiT（小配置）上比对：序列并行（USPAttention + 切分契约）vs 全序列参考。
两路都用同一份 FlashAttention 内核（参考路用 skip_sequence_parallel 的 USPAttention 跑全序列），
故 forward/grad 差异只来自 SP 集合通信的浮点累加序，bf16 下应在紧 tolerance 内。

校验项：
- self-attn forward 输出（全序列重建）对参考满足 tolerance；
- 输入梯度（跨 sp all-reduce sum，因每 rank 只回传 1/sp token）对参考满足 tolerance；
- 各 block self-attn 的 to_q/k/v/to_out 权重梯度（跨 sp sum）对参考满足 tolerance；
- gradient checkpointing 开/关均通过。

用法: torchrun --standalone --nproc_per_node=N sp_attention_parity.py --sp S [--ulysses U --ring R] [--ckpt]
"""
import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import WanTransformer3DModel

from miles.backends.fsdp_utils.parallel import create_fsdp_parallel_state
from miles.backends.fsdp_utils.sp_attention import WanUSPAttnProcessor, apply_sequence_parallel
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
    for p in model.parameters():  # 各 rank 权重一致
        dist.broadcast(p.data, src=0)
    return model


def make_inputs(device):
    g = torch.Generator(device=device).manual_seed(123)
    hidden = torch.randn(1, 16, 4, 8, 8, device=device, dtype=DTYPE, generator=g)
    enc = torch.randn(1, 32, 4096, device=device, dtype=DTYPE, generator=g)
    ts = torch.tensor([500], device=device)
    dist.broadcast(hidden, src=0)
    dist.broadcast(enc, src=0)
    out_grad = torch.randn(1, 16, 4, 8, 8, device=device, dtype=DTYPE, generator=g)
    dist.broadcast(out_grad, src=0)
    return hidden, enc, ts, out_grad


def _set_ref_processor(model):
    """全序列参考：USPAttention 走 skip_sequence_parallel（=本地 FA），不切序列。"""
    from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention

    proc = WanUSPAttnProcessor(model.config.num_attention_heads, model.config.attention_head_dim)
    proc.usp_attn = USPAttention(
        num_heads=model.config.num_attention_heads,
        head_size=model.config.attention_head_dim,
        causal=False,
        skip_sequence_parallel=True,
    )
    model.set_attn_processor(proc)


def _run(model, hidden, enc, ts, out_grad, ckpt):
    # forward_context 由 init_sp_backend（_set_ref_processor / apply_sequence_parallel 内）持久化设置。
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
    p.add_argument("--ring", type=int, default=0)
    p.add_argument("--ckpt", action="store_true")
    p.add_argument("--fwd-only", action="store_true",
                   help="只校验 forward + 输入梯度（Ring 训练反向 dK/dV 在 torch _templated_ring_attention + fa4 下尚不正确，见报告）")
    cli = p.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.cuda.current_device()
    init_gloo_group()

    args = argparse.Namespace(
        sequence_parallel_size=cli.sp, ulysses_degree=cli.ulysses, ring_degree=cli.ring, context_parallel_size=1,
    )
    ps = create_fsdp_parallel_state(args)

    attn_weight_names = [
        f"blocks.{i}.attn1.{proj}.weight" for i in range(2) for proj in ("to_q", "to_k", "to_v", "to_out.0")
    ]

    # ---- 参考：全序列 FA ----
    model = build_model(device)
    hidden, enc, ts, out_grad = make_inputs(device)
    _set_ref_processor(model)
    out_ref, gin_ref = _run(model, hidden, enc, ts, out_grad, cli.ckpt)
    gw_ref = {n: dict(model.named_parameters())[n].grad.detach().clone() for n in attn_weight_names}
    model.zero_grad(set_to_none=True)

    # ---- SP ----
    apply_sequence_parallel(model, ps)
    out_sp, gin_sp = _run(model, hidden, enc, ts, out_grad, cli.ckpt)

    # 输入/权重梯度：每 rank 只回传 1/sp 的 token 贡献，跨 sp sum 还原全量。
    dist.all_reduce(gin_sp, group=ps.sp_group)
    if rank == 0:
        print(f"[PARITY] sp={cli.sp} ulysses={ps.ulysses_degree} ring={ps.ring_degree} ckpt={cli.ckpt} "
              f"S=64 H=8 D=128")
    _report("forward(out)", out_sp, out_ref, rtol=2e-2, ctol=0.9990)
    _report("grad(input)", gin_sp, gin_ref, rtol=4e-2, ctol=0.9980)
    if not cli.fwd_only:
        params = dict(model.named_parameters())
        for n in attn_weight_names:
            assert params[n].grad is not None, f"{n} grad is None —— Ulysses all-to-all 反向未传"
            g = params[n].grad.detach().clone()
            dist.all_reduce(g, group=ps.sp_group)
            _report(f"grad({n})", g, gw_ref[n], rtol=5e-2, ctol=0.9950)

    dist.barrier()
    if rank == 0:
        print(f"[PARITY OK] sp={cli.sp} u={ps.ulysses_degree} r={ps.ring_degree} ckpt={cli.ckpt}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
