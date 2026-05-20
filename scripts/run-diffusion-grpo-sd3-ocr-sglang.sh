#!/usr/bin/env bash
# SD3.5 medium + OCR GRPO training via master_sglang /rollout/generate.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2,3 WANDB_API_KEY=<key> \
#     nohup bash scripts/run-diffusion-grpo-sd3-pickscore-sglang.sh \
#       > logs/sd3_ocr_sglang_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# GPU layout (default, override with CUDA_VISIBLE_DEVICES):
#   slot 0 (e.g. physical GPU 2) → FSDP actor
#   slot 1 (e.g. physical GPU 3) → sglang-diffusion rollout engine
#
# The script kills only processes whose cwd is inside this Miles workspace,
# so co-located experiments on other GPUs are not disturbed.

MILES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[kill] hunting for stale miles processes under cwd=${MILES_ROOT}"
for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  link=$(readlink "/proc/${pid}/cwd" 2>/dev/null) || continue
  exe=$(readlink  "/proc/${pid}/exe" 2>/dev/null) || continue
  case "${link}" in
    "${MILES_ROOT}"|"${MILES_ROOT}"/*)
      case "${exe}" in
        */python*|*/ray*)
          echo "[kill] ${pid} (${exe}) cwd=${link}"
          kill -9 "${pid}" 2>/dev/null || true
          ;;
      esac
      ;;
  esac
done
sleep 3

# Reap zombie parents.
ps -eo ppid,state,comm --no-headers \
  | awk '$2=="Z" && $1!=1 && $3~/ray|python|sglang/ {print $1}' \
  | sort -u | xargs -r kill -9 2>/dev/null || true
sleep 2

set -euo pipefail

ROOT_DIR="${MILES_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Use master_sglang (with native SD3 /rollout/generate support) instead of
# the default installed sglang. Prepending to PYTHONPATH shadows the editable
# install at /sgl-workspace/sglang.
MASTER_SGLANG_PYTHON="/sgl-workspace/master_sglang/sglang/python"
export PYTHONPATH="${MASTER_SGLANG_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Model path ────────────────────────────────────────────────────────────────
# SD3.5 is a gated model; HF_TOKEN is required for sglang to download
# model_index.json from the hub (even if weights are already cached locally).
export HF_TOKEN="${HF_TOKEN:-}"
SD3_MODEL="${SD3_MODEL:-stabilityai/stable-diffusion-3.5-medium}"

# ── Run name / weight dir ─────────────────────────────────────────────────────
RUN_NAME="diffusion_grpo_sd3_ocr_sglang_$(date +%Y%m%d_%H%M%S)"
ROLLOUT_WEIGHT_DIR="/tmp/miles_sd3_rollout_weights_${RUN_NAME}"
rm -rf "${ROLLOUT_WEIGHT_DIR}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100000}"

DEBUG_ARGS=()
if [[ "${MILES_DEBUG_ALIGNMENT:-0}" == "1" ]]; then
  export MILES_VERIFY_WEIGHT_SYNC="${MILES_VERIFY_WEIGHT_SYNC:-1}"
  DEBUG_ARGS+=(
    --diffusion-debug-mode
    --debug-skip-optimizer-step
  )
fi

# ── WandB ──────────────────────────────────────────────────────────────────────
WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project miles-diffusion-grpo
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --diffusion-log-images 8
    --diffusion-log-image-interval 10
    --disable-wandb-random-suffix
  )
fi

# ── Prepare prompt data ────────────────────────────────────────────────────────
python "${ROOT_DIR}/tools/prepare_ocr_jsonl.py"

# ── Training ───────────────────────────────────────────────────────────────────
# Key differences vs local-rollout OCR script:
#   - rollout-function-path → sglang_diffusion_rollout  (uses /rollout/generate)
#   - use-miles-router      → Miles spawns & owns the sglang server
#   - no-offload-rollout    → rollout lives in sglang, not in actor process
#   - no --debug-rollout-only / --custom-generate-function-path
python -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint gpt2 \
  --prompt-data "${ROOT_DIR}/data/ocr/train.jsonl" \
  --input-key input \
  --rollout-batch-size 8 \
  --n-samples-per-prompt 16 \
  --num-rollout "${NUM_ROLLOUT}" \
  --micro-batch-size-sample 8 \
  --micro-batch-size-tstep 5 \
  --gradient-checkpointing \
  --actor-num-gpus-per-node 1 \
  --rollout-num-gpus 1 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 2 \
  --no-offload-rollout \
  --colocate \
  --use-miles-router \
  --sglang-server-concurrency 4 \
  --use-lora \
  --lora-rank 32 \
  --lora-alpha 64 \
  --diffusion-init-lora-weight gaussian \
  --lr 3e-4 \
  --adam-beta2 0.999 \
  --diffusion-clip-range 1e-4 \
  --weight-decay 1e-4 \
  --diffusion-kl-beta 0.04 \
  --diffusion-model "${SD3_MODEL}" \
  --diffusion-reward ocr:1.0 \
  --advantage-estimator grpo \
  --globalize-reward-std \
  --rm-type ocr \
  --diffusion-forward-dtype fp16 \
  --sglang-dit-precision fp16 \
  --sglang-vae-slicing \
  --diffusion-num-steps 10 \
  --diffusion-eval-num-steps 40 \
  --update-weight-buffer-size 2147483648 \
  --diffusion-guidance-scale 4.5 \
  --diffusion-noise-level 0.7 \
  --diffusion-ignore-last 1 \
  --diffusion-height 512 \
  --diffusion-width 512 \
  --global-batch-size 64 \
  --save "${ROLLOUT_WEIGHT_DIR}" \
  "${DEBUG_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
