#!/usr/bin/env bash
# 4-GPU train + 1-GPU pickscore reward, Cosmos3-Nano T2I GRPO:
#   pretrained = nvidia/Cosmos3-Nano (16B MoT: 8B UND tower frozen, 8B GEN tower
#   trained via LoRA), resolution=832x480, single frame (T2I),
#   num_steps=16, eval_steps=35, guidance=1.0 (CFG-free training; checkpoints
#   still transfer to CFG deployment — merged LoRA sampled at g=4 beats the
#   base model at g=4), Flow-SDE noise_level=0.7,
#   KL beta=1e-3, global reward std, per-prompt mean.
#   train: lr=3e-4, adam_beta2=0.95, weight_decay=1e-4, clip_range=1e-3,
#          clip_grad=2e-3, mixed precision (master fp32 / forward bf16).
#   LoRA: r=64, alpha=128, init=gaussian.
#
# SDE schedule: epoch_global_window draws a 2-step window per rollout from
#   --diffusion-sde-candidate-steps 4-15. The Cosmos3 checkpoint ships a Karras
#   flow-sigma grid whose head steps 1-3 sit at sigma>0.96 with |dt|<0.02 and
#   train nothing; step numbers are NOT transferable across sigma-grid
#   families — re-derive candidates from |dt| when changing model/grid.
#
# Ratio/stability choices:
#   --diffusion-recompute-old-log-prob: the trainer recomputes old log-probs at
#     rollout ingestion so the PPO ratio is implementation-self-consistent
#     (rollout fa kernels vs train SDPA would otherwise leak into the ratio).
#   --adam-beta2 0.95 + --clip-grad 2e-3: absorb Adam-preconditioner spikes
#     after quiet stretches (single-step policy jumps that the PPO loss clip
#     cannot stop).
#
# Per rollout: 48 prompts × 16 samples = 768 items.
#   num_steps_per_rollout=2 → 384 items/optim step ÷ 4 train gpus = 96 items/rank.
#   --diffusion-microgroup-size 1: the Cosmos3 transformer is a packed-sequence
#   single-sample interface; one request cannot batch multiple outputs.
#
# Layout: first 4 GPUs in CUDA_VISIBLE_DEVICES = train+sgld colocate,
# the 5th GPU = pickscore reward worker. Default: GPU 0,1,2,3 + GPU 4.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# RL rollout scores raw samples; skip the serving-side guardrail models.
export SGLANG_DISABLE_COSMOS3_GUARDRAILS=1
RUN_NAME="diffusion_grpo_cosmos3_pickscore_t2i_5gpu_$(date +%Y%m%d_%H%M%S)"
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
if [[ ! -f "${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl" ]]; then
  hf download --repo-type dataset rockdu/miles-diffusion-datasets \
    --include "flowgrpo_pickscore/**" \
    --local-dir "${DATASETS_DIR}"
fi

"${PYTHON_BIN}" -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint nvidia/Cosmos3-Nano \
  --prompt-data "${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl" \
  --input-key input \
  --rollout-batch-size 48 \
  --n-samples-per-prompt 16 \
  --num-rollout 10000 \
  --num-steps-per-rollout 2 \
  --diffusion-microgroup-size 1 \
  --micro-batch-size 1 \
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
  --adam-beta2 0.95 \
  --clip-grad 2e-3 \
  --diffusion-clip-range 1e-3 \
  --weight-decay 1e-4 \
  --use-miles-router \
  --sglang-server-concurrency 8 \
  --sglang-attention-backend fa \
  --update-weight-buffer-size 2147483648 \
  --update-weight-target-module transformer \
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
  --diffusion-num-steps 16 \
  --diffusion-eval-num-steps 35 \
  --diffusion-output-num-frames 1 \
  --diffusion-guidance-scale 1.0 \
  --diffusion-noise-level 0.7 \
  --diffusion-height 480 \
  --diffusion-width 832 \
  --diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_window \
  --diffusion-sde-window-size 2 \
  --diffusion-sde-candidate-steps 4,5,6,7,8,9,10,11,12,13,14,15 \
  --diffusion-recompute-old-log-prob \
  --diffusion-kl-beta 1e-3 \
  --diffusion-debug-mode \
  --save "${SAVE_DIR}" \
  --save-interval 5 \
  --eval-prompt-data pickscore_test "${DATASETS_DIR}/flowgrpo_pickscore/test.jsonl" \
  --eval-interval 30 \
  --skip-eval-before-train \
  "${WANDB_ARGS[@]}"
