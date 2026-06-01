# Goal Tracker

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal
为 miles_diffusion 的 diffusion GRPO 训练（FSDP2 后端，Wan2.2-T2V 视频 DiT）引入**序列并行（SP）**，注意力算子采用 **USP**（Ulysses + Ring），与 sglang-diffusion 算子精度对齐；SP 不达标则评估回退 Mooncake。五阶段：阶段0 隔离 LLM CP 死代码 → 阶段1 解锁 SP + Option B 复合并行 → 阶段2 复用 sglang-d USP 算子 → 阶段3 diffusion 训练逻辑 SP-aware → 阶段4 权重同步适配，最后 10 步 perf 闸定 go/no-go。实现不假定卡数（2~1000+，ulysses×ring 可配）。

### Acceptance Criteria
<!-- 完整定义见 .humanize/plans/usp-video-sp-plan.md，此处为可独立验证的摘要 -->
- **AC-1**：阶段0 隔离 LLM CP 残留（`training_utils/{loss,data,cp_utils,log_utils}.py`），不破坏现有纯 DP 训练；防引用测试 + 纯 DP 回归门；物理删除推迟到 AC-2~6 全绿后。
- **AC-2**：解锁 `context_parallel_size==1` 断言，建 Option B 复合并行（FSDP 仅 dp 维 shard + 独立 sp_group + 参数 SP 维复制）；不假定卡数，`dp×ulysses×ring=world` 任意合法组合可初始化；非法组合（`num_heads%ulysses!=0` 等）启动即报错。
- **AC-3**：训练侧复用 sglang-d USPAttention，forward/backward/checkpoint/混精对 sp=1 参考 parity 通过。
- **AC-4**：序列 patchify 后切分 + RoPE 全局 position offset 接口约束。
- **AC-5**：SP 梯度同步协议（reduce-scatter 后 shard-grad 同步）+ DDP parity。
- **AC-6**：diffusion loss/advantage/log_prob 的 SP 归约逐项等价（sum+global_count，非 local-mean 再平均）。
- **AC-7**：RNG 三级一致性（样本/token/noise）。
- **AC-8**：权重同步单代表 rank 去重 + checksum 覆盖 dtype/shape/replica 语义（rollout 本期非 SP）。
- **AC-9**：10 步 RL perf 三档对照（DDP/纯FSDP/FSDP+SP）+ 通信分解 + go/no-go（决定是否触发 Mooncake）。

---

## MUTABLE SECTION

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |

#### Active Tasks
<!-- 本轮 mainline 聚焦阶段0（AC-1）：纯代码 + 单元测试，不依赖多 GPU 环境 -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| （本轮 mainline 阶段0+1 已完成，见 Completed；阶段2+ 待后续 round） | - | - | - | - | - |

### Blocking Side Issues
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|

### Queued Side Issues
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| AC-3/5/9 的 parity 与 10 步 perf 需真实多 GPU 环境（4+ 卡 + Wan2.2-A14B 权重/数据） | 0 | 阶段0/1 为纯代码地基，不依赖 GPU 运行即可推进 | 进入阶段2+ 或运行 perf 闸时 |

### Completed and Verified
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|
| AC-1 | task1: 4 个死代码模块加 `__deprecated__` 标记 | 0 | pending | docstring + 标记已加 |
| AC-1 | task2: 防引用守卫测试 `tests/test_cp_deadcode_isolation.py` | 0 | pending | `pytest` 5 passed；并发现死代码已 import-broken（缺 flops_utils） |
| AC-2 | task3: 解锁 `context_parallel_size==1` 断言 + 加 SP args（CP→SP 兼容） | 0 | pending | arguments 已改 |
| AC-2 | sp_mesh 纯函数 + test_sp_mesh（rank 映射/子组划分，对齐 sglang，不假定卡数） | 0 | pending | `pytest` 22 passed |
| AC-2 | task4: ParallelState sp 字段 + create_fsdp_parallel_state Option B 接线（复用 sglang 建 ulysses/ring 组） | 0 | pending | 多卡 smoke 8×B200 5 配置通过 |

### Explicitly Deferred
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
