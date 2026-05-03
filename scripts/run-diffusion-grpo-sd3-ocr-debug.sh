#!/usr/bin/env bash
# SD3 local-rollout precision alignment sanity script.
#
# Freezes weights (--debug-skip-optimizer-step) so log_prob_new ~= log_prob_old
# is purely a function of the local diffusers rollout/train numerical floor.

# for rerun the task
pkill -9 sgl* || true
sleep 3
ray stop --force
pkill -9 ray* || true
pkill -9 python* || true
sleep 3
pkill -9 ray* || true
pkill -9 python* || true

# pkill cannot reap zombies; kill their live parents so init reaps them.
ps -eo ppid,state,comm --no-headers | awk '$2=="Z" && $1!=1 && $3~/ray|python|sglang/ {print $1}' | sort -u | xargs -r kill -9 2>/dev/null || true
sleep 2

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="diffusion_sd3_align_$(date +%Y%m%d_%H%M%S)"
ROLLOUT_WEIGHT_DIR="/tmp/miles_sd3_rollout_weights_${RUN_NAME}"
rm -rf "${ROLLOUT_WEIGHT_DIR}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project miles-diffusion-align
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --disable-wandb-random-suffix
  )
fi

# Prepare OCR prompts into JSONL expected by Miles data loader.
python "${ROOT_DIR}/tools/prepare_ocr_jsonl.py"

python -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --rollout-function-path miles.rollout.diffusion_rollout.generate_rollout \
  --hf-checkpoint gpt2 \
  --prompt-data "${ROOT_DIR}/data/ocr/train.jsonl" \
  --input-key input \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 8 \
  --num-rollout 100000 \
  --micro-batch-size-tstep 10 \
  --gradient-checkpointing \
  --actor-num-gpus-per-node 1 \
  --rollout-num-gpus 1 \
  --num-gpus-per-node 2 \
  --use-lora \
  --lora-rank 32 \
  --lora-alpha 64 \
  --diffusion-init-lora-weight gaussian \
  --diffusion-model stabilityai/stable-diffusion-3.5-medium \
  --diffusion-reward ocr:1.0 \
  --reward-key avg \
  --advantage-estimator grpo \
  --globalize-reward-std \
  --rm-type ocr \
  --diffusion-forward-dtype bf16 \
  --diffusion-num-steps 10 \
  --diffusion-guidance-scale 4.5 \
  --diffusion-noise-level 0.7 \
  --diffusion-ignore-last 1 \
  --diffusion-height 256 \
  --diffusion-width 256 \
  --global-batch-size 8 \
  --save "${ROLLOUT_WEIGHT_DIR}" \
  --diffusion-debug-mode \
  --debug-skip-optimizer-step \
  "${WANDB_ARGS[@]}"
