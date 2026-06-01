"""阶段4 权重同步 parity（torchrun，需 NCCL）。AC-8。

验证 Option B + SP 下训练侧权重打包的正确性（rollout 本期非 SP，不在本测试范围；
rollout 接收端 checksum 一致性由运行期 MILES_VERIFY_WEIGHT_SYNC=1 在 smoke 跑里校验）：

  1. 每个参数经 FSDP(dp 维) reduce 后用 update_weights 的同款 redistribute([Replicate()])
     还原为全量张量；
  2. 各 rank（dp 与 sp 一并）重建的全量模型 checksum 必须**逐位一致**——这正是
     "单代表 rank 去重无冗余" 成立的前提：sp replica 不发散，任一 rank 都是合法发送源；
  3. 且 == 单进程全模型参考 checksum（证明 redistribute 真的还原了完整参数、未漏 shard）;
  4. checksum 覆盖 dtype/shape 语义：转置/reshape 后字节相同但 checksum 必变。

checksum 与 sglang-d compute_weights_checksum 同算法（name+dtype+shape+bytes）。

用法（两档都应跑）:
    torchrun --standalone --nproc_per_node=4 sp_weight_sync_parity.py --sequence_parallel_size 2 --ulysses_degree 2   # dp2×sp2
    torchrun --standalone --nproc_per_node=4 sp_weight_sync_parity.py --sequence_parallel_size 4 --ulysses_degree 4   # dp1×sp4
"""
import argparse

import torch
import torch.distributed as dist
from diffusers import WanTransformer3DModel
from torch.distributed.fsdp import fully_shard

from miles.backends.fsdp_utils.diffusion_update_weight_utils import DiffusionUpdateWeightFromTensor
from miles.backends.fsdp_utils.parallel import create_fsdp_parallel_state
from miles.utils.distributed_utils import init_gloo_group

DTYPE = torch.bfloat16

sha256 = DiffusionUpdateWeightFromTensor._sha256_named_tensors


def build_model(device):
    torch.manual_seed(0)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=8, attention_head_dim=128,
        in_channels=16, out_channels=16, text_dim=4096, freq_dim=256, ffn_dim=1024,
        num_layers=2, rope_max_seq_len=1024,
    ).to(device=device, dtype=DTYPE)
    # 让所有 rank 参数完全一致（模拟训练中各 rank 同一份权重）
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)
    return model


def full_state_pairs(model):
    """复刻 update_weights 的 redistribute([Replicate()]).to_local() 全量还原。"""
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
    p.add_argument("--sequence_parallel_size", type=int, default=2)
    p.add_argument("--ulysses_degree", type=int, default=2)
    p.add_argument("--ring_degree", type=int, default=0)
    cli = p.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.cuda.current_device()
    init_gloo_group()

    args = argparse.Namespace(
        sequence_parallel_size=cli.sequence_parallel_size,
        ulysses_degree=cli.ulysses_degree,
        ring_degree=cli.ring_degree,
        context_parallel_size=1,
    )
    ps = create_fsdp_parallel_state(args)

    # ---- 单进程全模型参考 checksum（FSDP 之前，全量参数）----
    ref_model = build_model(device)
    ref_sum = sha256([(n, pa.detach().cpu().contiguous()) for n, pa in ref_model.state_dict().items()])

    # ---- FSDP(dp 维 shard) + Option B：sp 维参数复制 ----
    model = build_model(device)
    for blk in model.blocks:
        fully_shard(blk, mesh=ps.dp_mesh)
    fully_shard(model, mesh=ps.dp_mesh)

    pairs = full_state_pairs(model)          # 各 rank 用 update_weights 同款路径还原全量
    my_sum = sha256(pairs)

    # 跨全部 rank（dp×sp）收集 checksum：必须全部一致 == 参考
    gathered = [None] * world
    dist.all_gather_object(gathered, my_sum)

    if rank == 0:
        all_equal = all(s == gathered[0] for s in gathered)
        match_ref = gathered[0] == ref_sum
        print(f"[AC-8] world={world} dp{ps.dp_size}×sp{ps.sp_size}"
              f"(u{ps.ulysses_degree}r{ps.ring_degree})")
        print(f"[AC-8] 各 rank checksum 全一致={all_equal}  == 单进程参考={match_ref}")
        print(f"[AC-8] ref={ref_sum[:16]}  ranks={[s[:8] for s in gathered]}")
        assert all_equal, "SP/DP 各 rank 还原的全量权重 checksum 不一致 → 发送源不可去重"
        assert match_ref, "redistribute 还原的全量权重 != 单进程参考 → 漏 shard 或 dtype/shape 错"

        # ---- checksum 语义覆盖：转置后字节相同但 shape 不同 → checksum 必变 ----
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        s_a = sha256([("w", t)])
        s_b = sha256([("w", t.t().contiguous())])      # 同元素不同 shape
        s_c = sha256([("w", t.to(torch.float64))])      # 同形状不同 dtype（字节变）
        print(f"[AC-8] shape 敏感: {s_a[:8]} vs {s_b[:8]} (须不同); dtype 敏感: {s_c[:8]}")
        assert s_a != s_b, "checksum 忽略 shape 语义（转置未被检出）"
        assert s_a != s_c, "checksum 忽略 dtype 语义"
        print("[SP-WEIGHT-SYNC OK] 全量重建逐位一致 + 覆盖 dtype/shape 语义")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
