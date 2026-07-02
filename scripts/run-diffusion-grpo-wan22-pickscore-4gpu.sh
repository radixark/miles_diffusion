#!/usr/bin/env bash
# Wan2.2-T2V-A14B 1-frame PickScore GRPO recipe: 4-GPU train+rollout colocate
# + 1-GPU pickscore reward.
#
# Knobs aligned with Flow-Factory's Wan2.2 LoRA GRPO recipe:
#   pretrained = Wan-AI/Wan2.2-T2V-A14B-Diffusers, resolution=480, num_steps=10,
#   guidance=4.0 (high-noise) / 3.0 (low-noise), Flow-SDE noise_level=0.9,
#   LoRA r=64/alpha=128 (self-attn + cross-attn + FFN),
#   lr=1e-4, adam_beta2=0.999, weight_decay=1e-4, clip_range=1e-4.
#   One SDE step drawn per epoch from the high-noise indices 1,2,3
#   (wan_ff_global_window, window_size=1, shared across the batch like
#   Flow-Factory) → only the high-noise expert ("transformer") trains.
#
# Layout: first 4 GPUs in CUDA_VISIBLE_DEVICES = train+sgld colocate,
# the 5th GPU = pickscore reward worker. Default GPUs 0-4.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
RUN_NAME="diffusion_grpo_wan22_pickscore_4gpu_$(date +%Y%m%d_%H%M%S)"
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

# Wan2.2 DiT LoRA targets: self-attn (attn1), cross-attn (attn2), and FFN.
WAN_LORA_TARGET_MODULES=(
  attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0
  attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0
  ffn.net.0.proj ffn.net.2
)

"${PYTHON_BIN}" -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --diffusion-model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt-data "${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl" \
  --input-key input \
  --rollout-batch-size 48 \
  --n-samples-per-prompt 16 \
  --num-rollout 10000 \
  --num-steps-per-rollout 2 \
  --diffusion-microgroup-size 8 \
  --micro-batch-size-sample 1 \
  --micro-batch-size-tstep 1 \
  --diffusion-train-iter-order sample_major \
  --gradient-checkpointing \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 5 \
  --colocate \
  --use-lora \
  --lora-rank 64 \
  --lora-alpha 128 \
  --lora-target-modules "${WAN_LORA_TARGET_MODULES[@]}" \
  --diffusion-init-lora-weight gaussian \
  --lr 1e-4 \
  --adam-beta2 0.999 \
  --diffusion-clip-range 1e-4 \
  --weight-decay 1e-4 \
  --use-miles-router \
  --sglang-server-concurrency 8 \
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
  --diffusion-num-steps 10 \
  --diffusion-eval-num-steps 28 \
  --diffusion-output-num-frames 1 \
  --diffusion-guidance-scale 4.0 \
  --diffusion-guidance-scale-2 3.0 \
  --diffusion-noise-level 0.9 \
  --diffusion-height 480 \
  --diffusion-width 480 \
  --diffusion-step-strategy-path miles.rollout.step_strategy_hub.wan_ff_global_window \
  --diffusion-sde-window-size 1 \
  --diffusion-sde-candidate-steps 1,2,3 \
  --save "${SAVE_DIR}" \
  --save-interval 10 \
  --eval-prompt-data pickscore_test "${DATASETS_DIR}/flowgrpo_pickscore/test.jsonl" \
  --eval-interval 30 \
  --skip-eval-before-train \
  "${WANDB_ARGS[@]}"
