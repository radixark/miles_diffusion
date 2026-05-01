#!/usr/bin/env bash
# 4-GPU QwenImage OCR GRPO training, fully aligned with miles 4-GPU formal
# training, with rollout-side LoRA MERGED forward to test the bf16 merge-vs-
# additive hypothesis.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOW_GRPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${FLOW_GRPO_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Pull wandb key from ~/.netrc
if [ -z "${WANDB_API_KEY:-}" ]; then
    WANDB_API_KEY="$(awk '/api\.wandb\.ai/{getline; getline; print $2}' "${HOME}/.netrc")"
    export WANDB_API_KEY
fi
export WANDB_MODE="${WANDB_MODE:-online}"

UNIQUE_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-ocr_4gpu_merge_lora_aligned_${UNIQUE_TAG}}"
LOG_DIR="${FLOW_GRPO_DIR}/logs/ocr/qwenimage_4gpu_merge_lora_aligned"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

echo "===== flow_grpo qwen-image OCR (4 GPU, miles-aligned, rollout_merge_lora=true) ====="
echo "GPUs:    ${CUDA_VISIBLE_DEVICES}"
echo "Run:     ${RUN_NAME}"
echo "Log:     ${LOG_FILE}"
echo "Wandb:   $([ -n "${WANDB_API_KEY:-}" ] && echo SET || echo MISSING) (mode=${WANDB_MODE})"
echo "==================================================================================="

torchrun --standalone --nproc_per_node=4 --master_port=19513 \
  scripts/train_qwenimage.py \
  --config config/grpo.py:ocr_qwenimage_4gpu \
  --config.run_name="${RUN_NAME}" \
  --config.save_dir="${LOG_DIR}" \
  --rollout_merge_lora=true \
  --simulate_offpolicy=0.0 \
  2>&1 | tee "${LOG_FILE}"
