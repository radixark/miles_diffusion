#!/usr/bin/env bash
# 4-GPU train + 1-GPU pickscore reward, aligned with flow_grpo `pickscore_qwenimage`:
#   pretrained = Qwen/Qwen-Image, resolution=512, num_steps=10, eval_steps=50,
#   guidance=4, noise_level=1.2, sde_window_size=2.
#   sde_window_range=3,5 → effective SDE indices [3,4] (matches flow_grpo's
#   bug where they hard-code (0, num_steps//2) but their actual training
#   covers steps 3-4 only; we mirror).
#   beta=0 (no KL), ema=False, same_latent=False, global_std=True, per-prompt mean.
#   train: lr=3e-4, adam_beta2=0.999, weight_decay=1e-4, clip_range=1e-4, max_grad_norm=1.0,
#          mixed precision (master fp32 / forward bf16), gradient checkpointing.
#   LoRA: r=64, alpha=128, init=gaussian.
#
# Per rollout: 32 prompts × 16 samples = 512 items.
#   num_steps_per_rollout=2 → 256 items/optim step (matches flow_grpo's
#   pickscore_qwenimage 32-GPU: train_batch_size=4 × 32 GPU × 2 grad_accum = 256).
#   ÷ 4 train gpus = 64 items/rank/optim step.
#   --micro-batch-size 8 → 8 train pairs/forward, 8 forwards/rank/optim step.
#
# Layout: first 4 GPUs in CUDA_VISIBLE_DEVICES = train+sgld colocate,
# the 5th GPU = pickscore reward worker (1 GPU dedicated).
# Default: GPU 4,5,6,7 for train+sgld; GPU 1 for pickscore.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7,1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_NAME="diffusion_grpo_pickscore_4gpu_flowgrpo_aligned_$(date +%Y%m%d_%H%M%S)"
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
  --include "flowgrpo_pickscore/**" \
  --local-dir "${DATASETS_DIR}"

"${PYTHON_BIN}" -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint Qwen/Qwen-Image \
  --prompt-data "${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl" \
  --input-key input \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 16 \
  --num-rollout 100000 \
  --diffusion-microgroup-size 8 \
  --micro-batch-size 8 \
  --gradient-checkpointing \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 5 \
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
  --diffusion-reward pickscore:1.0 \
  --advantage-estimator grpo \
  --globalize-reward-std \
  --rm-type pickscore \
  --pickscore-num-workers 1 \
  --pickscore-num-gpus-per-worker 1.0 \
  --pickscore-batch-size 8 \
  --pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --pickscore-model-path yuvalkirstain/PickScore_v1 \
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
  --apply-sgld-monkey-patches \
  --diffusion-height 512 \
  --diffusion-width 512 \
  --save "${SAVE_DIR}" \
  --save-interval 10 \
  --eval-prompt-data pickscore_test "${DATASETS_DIR}/flowgrpo_pickscore/test.jsonl" \
  --eval-interval 30 \
  --skip-eval-before-train \
  "${WANDB_ARGS[@]}"
