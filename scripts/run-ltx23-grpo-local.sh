#!/usr/bin/env bash
# LTX-2.3 GRPO 本地启动脚本（/data/wenhao 环境）
#
# 仅需 --diffusion-model Lightricks/LTX-2.3（train + rollout 共用 sglang overlay）。
# 首次运行会自动 materialize 到 SGLANG_DIFFUSION_CACHE_ROOT。
#
# 前置条件：
#   - venv:  /data/wenhao/.venvs/miles-diffusion
#   - sglang: /data/wenhao/master_sglang/sglang/python
#   - miles:  /data/wenhao/master_miles/miles_diffusion
#
# 用法：
#   # 前台调试
#   bash scripts/run-ltx23-grpo-local.sh
#
#   # 冒烟测试（1 rollout，跳过 optimizer）
#   NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=2 NUM_STEPS_PER_ROLLOUT=1 \
#     SKIP_OPTIMIZER=1 bash scripts/run-ltx23-grpo-local.sh
#
# 可覆盖的环境变量：
#   CUDA_VISIBLE_DEVICES  GPU 编号（默认 2）
#   NUM_ROLLOUT           rollout 总数（默认 100）
#   MILES_DIFFUSION_DEBUG 1=打印对齐指标（默认 1）
#   SKIP_OPTIMIZER        1=冒烟模式，不更新权重（默认 0）
#   LTX_DEV_SAFETENSORS   可选：dev .safetensors 覆盖 overlay 默认 DiT 权重

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILES_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── 路径 ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export SGLANG_PYTHON="${SGLANG_PYTHON:-/data/wenhao/master_sglang/sglang/python}"
export PYTHONPATH="${SGLANG_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}"

export MILES_DATA_ROOT="${MILES_DATA_ROOT:-/data/wenhao/master_miles/miles_diffusion}"
export MILES_DATA_DISK_ROOT="${MILES_DATA_DISK_ROOT:-/data/wenhao/miles_diffusion}"
export LTX_HF_MODEL="${LTX_HF_MODEL:-Lightricks/LTX-2.3}"
export PROMPT_DATA="${PROMPT_DATA:-${MILES_DATA_ROOT}/data/vidgen/train.jsonl}"
export HF_HOME="${HF_HOME:-/data/wenhao/hf_home}"
export SGLANG_DIFFUSION_CACHE_ROOT="${SGLANG_DIFFUSION_CACHE_ROOT:-/data/wenhao/sgl_diffusion_cache}"

# ── 训练规模 ──────────────────────────────────────────────────────────────
export USE_LORA="${USE_LORA:-1}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

# ── 对齐开关 ──────────────────────────────────────────────────────────────
export MILES_DIFFUSION_DEBUG="${MILES_DIFFUSION_DEBUG:-1}"
export LTX_DISABLE_AV_CROSS_ATTN="${LTX_DISABLE_AV_CROSS_ATTN:-1}"
export MILES_APPLY_LTX2_LTXCORE_PARITY="${MILES_APPLY_LTX2_LTXCORE_PARITY:-1}"

# ── 冒烟 / 恢复 ───────────────────────────────────────────────────────────
export SKIP_OPTIMIZER="${SKIP_OPTIMIZER:-0}"

# ── 激活环境 ──────────────────────────────────────────────────────────────
VENV="${VENV:-/data/wenhao/.venvs/miles-diffusion}"
if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV}/bin/activate"
else
  echo "[warn] venv not found: ${VENV}" >&2
fi

echo "============================================"
echo " LTX-2.3 GRPO Local Launch"
echo "============================================"
echo " GPU:        ${CUDA_VISIBLE_DEVICES}"
echo " Rollouts:   ${NUM_ROLLOUT}"
echo " Model:      ${LTX_DEV_SAFETENSORS:-${LTX_HF_MODEL}}"
echo " Debug:      ${MILES_DIFFUSION_DEBUG}"
echo " Skip optim: ${SKIP_OPTIMIZER}"
echo " Logs:       ${MILES_DATA_DISK_ROOT}/logs/"
echo " Ckpt:       ${MILES_DATA_DISK_ROOT}/ckpt/"
echo "============================================"

cd "${MILES_ROOT}"
exec bash scripts/run-diffusion-grpo-ltx23-sglang.sh
