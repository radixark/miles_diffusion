#!/usr/bin/env bash
# 2-GPU OCR run, flow_grpo math-equivalent.
# Per-rollout (= flow_grpo per-epoch):
#   * 32 unique prompts × k=16 = 512 items / rollout
#   * num_steps_per_rollout = 2 (= flow_grpo num_optim_per_epoch)
#   * global_batch_size = 256 (auto-derived: 512 / 2)
#   * local_batch_size  = 128 (per rank: 256 / 2 GPUs)
#   * sample_microbatch = 4  (= flow_grpo train_batch_size)
#   * tstep_microbatch  = 2  (= SDE window size, full window per forward)
#   * gradient accumulation per optim step per rank = 128/4 = 32 forwards
# Master + forward dtype = fp32 / bf16 (= flow_grpo's mixed precision).
# SDE window range 3,5 covers steps {3,4} (per the flow_grpo bug we mirror).
# Checkpoint: LoRA-only (auto via use_lora), every 20 rollouts.
# Eval on test split every 50 rollouts.

pkill -9 sgl* 2>/dev/null
sleep 3
ray stop --force 2>&1 | tail -1
pkill -9 ray* 2>/dev/null
pkill -9 python* 2>/dev/null
sleep 3
pkill -9 ray* 2>/dev/null
pkill -9 python* 2>/dev/null
ps -eo ppid,state,comm --no-headers | awk '$2=="Z" && $1!=1 && $3~/ray|python|sglang/ {print $1}' | sort -u | xargs -r kill -9 2>/dev/null || true
sleep 2

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=4,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_NAME="diffusion_grpo_ocr_2gpu_fg_aligned_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${ROOT_DIR}/logs/${RUN_NAME}/ckpt"

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

python "${ROOT_DIR}/tools/prepare_ocr_jsonl.py"

python -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint Qwen/Qwen-Image \
  --prompt-data "${ROOT_DIR}/data/ocr/train.jsonl" \
  --input-key input \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 16 \
  --num-rollout 100000 \
  --diffusion-microgroup-size 16 \
  --micro-batch-size-sample 4 \
  --micro-batch-size-tstep 2 \
  --diffusion-train-iter-order sample_major \
  --gradient-checkpointing \
  --actor-num-gpus-per-node 2 \
  --rollout-num-gpus 2 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 2 \
  --colocate \
  --use-lora \
  --lora-rank 64 \
  --lora-alpha 128 \
  --diffusion-init-lora-weight gaussian \
  --lr 3e-4 \
  --adam-beta2 0.999 \
  --diffusion-clip-range 1e-4 \
  --weight-decay 1e-4 \
  --use-miles-router \
  --sglang-server-concurrency 4 \
  --update-weight-buffer-size 2147483648 \
  --diffusion-model Qwen/Qwen-Image \
  --diffusion-reward ocr:1.0 \
  --advantage-estimator grpo \
  --globalize-reward-std \
  --rm-type ocr \
  --fsdp-master-dtype fp32 \
  --diffusion-forward-dtype bf16 \
  --diffusion-num-steps 10 \
  --diffusion-eval-num-steps 50 \
  --num-steps-per-rollout 2 \
  --diffusion-guidance-scale 4.0 \
  --diffusion-true-cfg-scale 4.0 \
  --diffusion-noise-level 1.2 \
  --diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window \
  --diffusion-sde-window-size 2 \
  --diffusion-sde-window-range 3,5 \
  --apply-qwen-image-sgl-d-patch \
  --diffusion-height 512 \
  --diffusion-width 512 \
  --save "${SAVE_DIR}" \
  --save-interval 20 \
  --eval-prompt-data ocr_test "${ROOT_DIR}/data/ocr/test.jsonl" \
  --eval-interval 50 \
  --skip-eval-before-train \
  "${WANDB_ARGS[@]}"
