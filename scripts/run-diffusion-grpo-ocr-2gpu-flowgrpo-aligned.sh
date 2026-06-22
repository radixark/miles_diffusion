#!/usr/bin/env bash
# 2-GPU OCR training aligned with flow_grpo qwen-image OCR config (4-GPU
# scaled down: rollout_batch_size halved, all other knobs identical).
#
# Per-rollout math:
#   rollout_batch_size=16 prompts × n_samples=16 = 256 items/rollout
#   num_steps_per_rollout=2 → 128 items/optim step (global)
#   ÷ 2 train gpus = 64 items/rank/optim step (same as 4-GPU script)
#   tile = (sample=4, tstep=2) = 8 cells, 8 tiles/rank/optim step.
#
# Other knobs (matching flow_grpo qwen-image OCR):
#   lr=3e-4, adam_beta2=0.999, weight_decay=1e-4, max_grad_norm=1.0,
#   clip_range=1e-4, beta=0 (no KL), ema=False, same_latent=False,
#   global_std=True + per-prompt mean, noise_level=1.2, num_steps=10,
#   eval_steps=50, guidance=4, resolution=512, sde_window_size=2.
#   sde_window_range=3,5 → effective SDE indices [3,4] (mirror flow_grpo bug).
#   LoRA: r=64, alpha=128, init=gaussian; mixed precision (master fp32 / forward bf16).

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_NAME="diffusion_grpo_ocr_2gpu_flowgrpo_aligned_$(date +%Y%m%d_%H%M%S)"
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

PYTHON_BIN="${PYTHON_BIN:-python}"

DATASETS_DIR="/root/datasets/miles-diffusion-datasets"
hf download --repo-type dataset rockdu/miles-diffusion-datasets \
  --include "flowgrpo_ocr/**" \
  --local-dir "${DATASETS_DIR}"

"${PYTHON_BIN}" -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint Qwen/Qwen-Image \
  --prompt-data "${DATASETS_DIR}/flowgrpo_ocr/train.jsonl" \
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
