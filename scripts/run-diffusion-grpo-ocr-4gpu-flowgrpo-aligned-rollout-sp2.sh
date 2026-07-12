#!/usr/bin/env bash
# 4-GPU OCR training, same hyperparameters as
# run-diffusion-grpo-ocr-4gpu-flowgrpo-aligned.sh, but with rollout-side
# sequence parallelism: 2 sgl-d engines x 2 GPUs, Ulysses sp_degree=2.
#
# Rollout topology:
#   --rollout-num-gpus 4 / --rollout-num-gpus-per-engine 2 -> 2 engines
#   --sglang-sp-degree 2 -> each engine shards the image-token sequence
#   across its 2 GPUs (text prefix replicated, Ulysses all-to-all attention).
# Training side is unchanged (4-rank FSDP colocate); weight sync sends one
# CUDA-IPC payload per training rank in each engine's GPU span.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_NAME="diffusion_grpo_ocr_4gpu_rollout_sp2_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${ROOT_DIR}/logs/${RUN_NAME}/ckpt"
# Per-run metric recording; registerable as CI standard (tests/ci/e2e_metrics_registry.py).
export MILES_METRICS_JSONL="${MILES_METRICS_JSONL:-${ROOT_DIR}/logs/${RUN_NAME}/metrics.jsonl}"
mkdir -p "$(dirname "${MILES_METRICS_JSONL}")"

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

PYTHON_BIN="${PYTHON_BIN:-python}"

DATASETS_DIR="/root/datasets/miles-diffusion-datasets"
hf download --repo-type dataset rockdu/miles-diffusion-datasets \
  --include "flowgrpo_ocr/**" \
  --local-dir "${DATASETS_DIR}"

"${PYTHON_BIN}" -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  ${DETERMINISTIC_MODE:+--deterministic-mode} \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint Qwen/Qwen-Image \
  --prompt-data "${DATASETS_DIR}/flowgrpo_ocr/train.jsonl" \
  --input-key input \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 16 \
  --num-rollout "${NUM_ROLLOUT:-100000}" \
  --diffusion-microgroup-size 16 \
  --micro-batch-size-sample 4 \
  --micro-batch-size-tstep 2 \
  --diffusion-train-iter-order sample_major \
  --gradient-checkpointing \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 2 \
  --sglang-sp-degree 2 \
  --num-gpus-per-node 4 \
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
  --fsdp-reduce-dtype fp32 \
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
  --diffusion-debug-mode \
  --apply-sgld-monkey-patches \
  --diffusion-height 512 \
  --diffusion-width 512 \
  --save "${SAVE_DIR}" \
  --save-interval 10 \
  --eval-prompt-data ocr_test "${DATASETS_DIR}/flowgrpo_ocr/test.jsonl" \
  --eval-interval 30 \
  --skip-eval-before-train \
  "${WANDB_ARGS[@]}"
