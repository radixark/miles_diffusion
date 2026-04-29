#!/usr/bin/env bash
# ps -ef | grep train_diffusion.py | grep -v grep
# nohup bash /data/zhiheng/miles/scripts/run-diffusion-grpo-sd3-ocr.sh > /data/zhiheng/miles/logs/diffusion_grpo_sd3_ocr_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# pkill -f "/data/zhiheng/miles/train_diffusion.py"
# Local SD3 rollout uses 1 GPU and FSDP trainer uses 1 GPU.

# For rerun the task: only kill this workspace's Miles processes so other
# co-located experiments are not disturbed.
MILES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[kill] hunting for stale miles processes under cwd=${MILES_ROOT}"
for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  link=$(readlink "/proc/${pid}/cwd" 2>/dev/null) || continue
  exe=$(readlink "/proc/${pid}/exe" 2>/dev/null) || continue
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

# pkill cannot reap zombies; kill their live parents so init reaps them.
ps -eo ppid,state,comm --no-headers | awk '$2=="Z" && $1!=1 && $3~/ray|python|sglang/ {print $1}' | sort -u | xargs -r kill -9 2>/dev/null || true
sleep 2

set -euo pipefail
ROOT_DIR="${MILES_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,7}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# WandB: enable if WANDB_API_KEY is present.
RUN_NAME="diffusion_grpo_sd3_ocr_$(date +%Y%m%d_%H%M%S)"
ROLLOUT_WEIGHT_DIR="/tmp/miles_sd3_rollout_weights_${RUN_NAME}"
rm -rf "${ROLLOUT_WEIGHT_DIR}"

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

# Prepare OCR prompts into JSONL expected by Miles data loader.
python "${ROOT_DIR}/tools/prepare_ocr_jsonl.py"

# SD3.5 diffusion GRPO run with OCR reward, aligned with flow_grpo SD3 settings.
python -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --diffusion-train \
  --rollout-function-path miles.rollout.diffusion_rollout.generate_rollout \
  --hf-checkpoint gpt2 \
  --prompt-data "${ROOT_DIR}/data/ocr/train.jsonl" \
  --input-key input \
  --rollout-batch-size 8 \
  --n-samples-per-prompt 16 \
  --num-rollout 100000 \
  --diffusion-timestep-batch 5 \
  --gradient-checkpointing \
  --actor-num-gpus-per-node 1 \
  --rollout-num-gpus 1 \
  --num-gpus-per-node 2 \
  --use-lora \
  --lora-rank 32 \
  --lora-alpha 64 \
  --diffusion-init-lora-weight gaussian \
  --lr 3e-4 \
  --adam-beta2 0.999 \
  --diffusion-clip-range 1e-4 \
  --weight-decay 1e-4 \
  --diffusion-kl-beta 0.04 \
  --diffusion-model stabilityai/stable-diffusion-3.5-medium \
  --diffusion-reward ocr:1.0 \
  --reward-key avg \
  --advantage-estimator grpo \
  --globalize-reward-std \
  --rm-type ocr \
  --diffusion-dtype fp16 \
  --diffusion-num-steps 10 \
  --diffusion-eval-num-steps 40 \
  --diffusion-gradient-accumulation-steps 64 \
  --diffusion-guidance-scale 4.5 \
  --diffusion-noise-level 0.7 \
  --diffusion-ignore-last 1 \
  --diffusion-height 512 \
  --diffusion-width 512 \
  --global-batch-size 128 \
  --save "${ROLLOUT_WEIGHT_DIR}" \
  "${WANDB_ARGS[@]}"
