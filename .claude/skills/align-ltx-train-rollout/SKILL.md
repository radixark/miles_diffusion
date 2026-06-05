---
name: align-ltx-train-rollout
description: Diagnose and fix train-vs-rollout forward numerical misalignment for LTX-2.3 diffusion GRPO on sglang rollout, driving model_output_cosine_sim to ~0.9998 and log_prob_mean_abs_diff small. Use when GRPO clipfrac stays near 1.0, train/log_prob_mean_abs_diff is large, model_output_cosine_sim is low, or when aligning the miles FSDP train forward with the sglang rollout forward for LTX video diffusion.
---

# align-ltx-train-rollout

把 miles 训练侧重算的 `log_prob` / `model_output` 与 sglang live rollout 对齐,使 GRPO 的 importance ratio 可信(`clipfrac` 不再长期 = 1.0)。

## 核心心智模型(先读这一条)

**对齐 gap 几乎从不是 checkpoint / 权重错。** 按以下三层顺序定位,不要一上来换权重:

1. **log_prob 公式层** — SDE dynamics 类型、`sigma_min` 用错 → `log_prob_mean_abs_diff` 巨大(2~15)
2. **DiT forward 层** — temb/AdaLN 的 shape 与语义、算子 parity、AV cross-attn、attention backend → `cosine` 低(~0.96)
3. **live pipeline 后处理层** — guider(cfg/stg/modality/rescale)改了 velocity 语义 → offline 高但 live 低(~0.94)

## 诊断决策树

开 `--diffusion-debug-mode`,看训练 log 的 `train/model_output_cosine_sim`、`train/log_prob_mean_abs_diff`、`train/clipfrac`(计算位置 `miles/backends/fsdp_utils/actor.py:836-847`)。然后:

```
cosine 高(>0.999) 但 log_prob_diff 大?
  └─→ 第①层:SDE 公式 / sigma_min。查 sde_log_prob.py + sglang scheduler_rl_mixin。

cosine 低(<0.99)?
  └─ 先做 offline injected 对比(同 latent/kwargs 喂两侧 DiT,排除 checkpoint):
       bash scripts/capture-and-compare-ltx23-forward.sh
     ├─ injected 也低  → 第②层:temb/parity/AV/attention(见下方必备开关)
     └─ injected 高(>0.999) 但 live 低 → 第③层:guider 后处理(identity guider)
```

## 对齐必备开关(缺一项就掉精度)

| 开关 | 作用 | 对应层 |
|------|------|--------|
| `MILES_APPLY_LTX2_LTXCORE_PARITY=1` | temb expand `[B,1,D]→[B,T,D]` + AdaLN/RMSNorm/RoPE/SDPA 数值对齐 | ② |
| `MILES_LTX_IDENTITY_GUIDER=1` | 强制 stage1 guider 为 identity(cfg=1/stg=0/modality=1/rescale=0) | ③ |
| `--ltx-disable-av-cross-attn` (+ `MILES_LTX_DISABLE_AV_CROSS=1`) | train video-only 与 rollout 算子图一致 | ② |
| `SGLANG_ATTENTION_BACKEND=torch_sdpa` | 避免 FlashAttention vs SDPA 数值差 | ② |
| train 与 sglang 用**同一份 `.safetensors`** | 避免 HF materialized overlay 与单文件差 ~1e-4(由 `configs/ltx.py` resolve) | ② |
| train 与 rollout **同 `--diffusion-num-steps` / `--ltx-dynamics-type` / `--ltx-sigma-min`** | 步数/动力学/σ_min 一致 | ①② |

**经验**:`dev` ckpt + 24 步比 `distilled` + 8 步对齐好得多(完整路径 velocity 更平滑)。优先用 dev 验证。

## 验收标准

| 指标 | 达标 | dev/512×768×57f/24步实测 |
|------|------|--------------------------|
| `model_output_cosine_sim` | ≥ 0.999 | 0.9995 ~ 0.9999 |
| `log_prob_mean_abs_diff` | < 5e-3 | 6e-6 ~ 1.7e-3 |
| `clipfrac` | 明显 < 1.0 | 0 ~ 0.125 |

## 快速命令

```bash
cd /sgl-workspace/master_miles/miles_diffusion
export PYTHONPATH=/sgl-workspace/master_sglang/sglang/python${PYTHONPATH:+:$PYTHONPATH}

# 纯前向对齐验证(不更新权重,最快看 cosine / log_prob_diff)
CUDA_VISIBLE_DEVICES=<free_gpu> MILES_DIFFUSION_DEBUG=1 \
LTX_DISABLE_AV_CROSS_ATTN=1 MILES_LTX_IDENTITY_GUIDER=1 USE_LORA=1 SKIP_OPTIMIZER=1 \
NUM_ROLLOUT=3 ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=2 GLOBAL_BATCH_SIZE=2 \
nohup bash scripts/run-diffusion-grpo-ltx23-sglang-dev-flowsde.sh > logs/align_verify.log 2>&1 &

# 离线 capture + compare(定位 gap 在 DiT raw 还是后处理)
bash dist/scripts/capture-and-compare-ltx23-forward.sh
```

监控:`grep -E 'model_output_cosine_sim|log_prob_mean_abs_diff|clipfrac' logs/*.log`

## 常见陷阱(实战踩过)

- **temb 错 1000 倍**:AdaLN 输入应是 `σ×1000` 而非 `σ`;错了 cosine≈0(与"权重错"是不同量级)。
- **temb shape**:ltx_core 走 `[B,T,D]`,sglang 构 `[B,1,D]`,即使 σ 均匀,block 内 broadcast 行为不同 → 必须 `expand_temb_for_hidden`。
- **guider 默认值**:LTX23 one-stage 默认 `video_cfg_scale=3 / modality=3 / rescale=0.7`,在 x0 上改 velocity 语义 → live cosine 0.94。train `forward_velocity` 等价 cfg=1/无STG/无modality/无rescale。
- **identity guider 不能经 `extra_sampling_params` 传**:sglang `SamplingParams` 基类拒绝 LTX23 guider 字段(`400 unexpected keyword 'video_cfg_scale'`)→ 必须用 monkey patch `patch_ltx2_identity_guider.py` override `_get_ltx2_stage1_guider_params`。
- **Flow-SDE 公式 / sigma_min**:`--ltx-dynamics-type Flow-SDE` 时 rollout 误用 SD3 `sde` 公式、或误用 `sigmas[-2]=0.1` 作 σ_min → `log_prob_mean_abs_diff` 2~15。sglang 侧需 `flow_sde` + `rollout_sigma_min`。
- **guider 修复后必须重新 capture dump 再 compare**:旧 dump 是默认 guider 下生成的。

## 参考文档(深度细节，本地 `dist/docs/`，不进 git)

- [dist/docs/ltx23_changes_overview.md](../../../dist/docs/ltx23_changes_overview.md) — 全部代码改动按 DEBUG/TRAIN/ROLLOUT 三分类
- [dist/docs/ltx23_train_rollout_alignment_journey.md](../../../dist/docs/ltx23_train_rollout_alignment_journey.md) — Phase A/B/C 排查历程(SDE→temb→guider)
- [dist/docs/ltx23_sglang_rollout_train_troubleshooting.md](../../../dist/docs/ltx23_sglang_rollout_train_troubleshooting.md) — 两侧工程问题 P1–P13 + 跨边界 C1–C8
- [dist/docs/ltx23_forward_alignment_test_report.md](../../../dist/docs/ltx23_forward_alignment_test_report.md) — 数值实验矩阵与 block-wise 二分
