# Ask Codex Input

## Question

对 miles_diffusion 仓库 feat/usp 分支的 FSDP+USP 序列并行实现做独立代码审查(diff base = usp-impl-base, base_commit e5711c23b672041ef993cc19afaca1d374efeb55, 当前 HEAD=ee9264f)。重点审查从未被 review 过的 stage 2/3/4。请尤其挑战以下结论的正确性,不要轻易认同:

1. 阶段4 权重同步(刚写,最需审): 结论是'Option B + colocate 下现有 weight-sync 路径对 SP 结构性正确、单代表 rank 去重天然满足、无需新增去重代码'。论据:(a) 所有 train rank 经 dp 维 redistribute([Replicate()]) 后持有相同全量参数(FSDP 只 shard dp、sp 维复制);(b) connect_rollout_engines 仅按全局 rank + rollout_num_gpus_per_engine 连续分组,与 SP 逻辑划分无关;(c) gather 给每个 rollout worker 恰好一个同卡 train rank 的张量,无跨卡冗余 IPC。请验证这个推理在 dp×sp 拓扑与 colocate 物理映射下是否真成立? 有没有 rollout_num_gpus_per_engine 不整除、多 engine、或 tp_rank 索引越界/错配的反例? gather_object 的 rank 顺序与 rollout TP worker 的 tp_rank 选择(_select_rank_scoped_payload[tp_rank])是否真对齐? 文件 miles/backends/fsdp_utils/diffusion_update_weight_utils.py。

2. checksum 改动: 把 _verify_weight_sync/_sha256_named_tensors 从 LoRA 子类提升到基类 DiffusionUpdateWeightFromTensor; checksum 从 name+bytes 扩展为 name+dtype+shape+bytes(两侧对称: train 侧 + sglang loader/weight_utils.py 的 compute_weights_checksum)。验证两侧是否真对称、是否引入恒不匹配,以及 LoRA 子类继承后是否仍正确(它有 name remapping)。

3. 阶段2 sglang 两处 autograd 修复(sglang 仓 wan-strict-mode 分支,路径 /workspace/809a2940-8360-4812-81c2-c7383f3f43e7/sglang): usp.py 给 all-to-all 包 _AllToAllSingle(对合,反向=同一 all-to-all); 给 ring 包 _RingFlashAttention(fwd=_templated_ring_attention, bwd=_templated_ring_attention_backward),按 torch.is_grad_enabled() 分流训练/推理。验证反向数学是否正确、推理路径是否真逐字不变。

4. 阶段3 SP 梯度同步: FSDP reduce-scatter 后跨 sp all-reduce(SUM, fp32); 为何 SUM 而非 mean; gather 点从 blocks[-1] 后移到 proj_out 后使所有参数偏梯度统一。文件 actor.py 的 _all_reduce_sp_grads、sp_attention.py。

按 [P0]~[P3] 标注严重度。代码风格硬约束: 极简、禁止防御性编程、断言仅 debug、不在热路径堆校验——据此评估,不要把'缺少防御性校验'当缺陷。

## Configuration

- Model: gpt-5.5
- Effort: high
- Timeout: 5400s
- Timestamp: 2026-06-01_23-28-44
- Tool: codex
