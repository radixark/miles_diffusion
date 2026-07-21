#!/usr/bin/env bash
# 4-GPU train + 1-GPU pickscore reward, Wan2.2-T2V-A14B dual-expert 5-frame video GRPO:
#   pretrained = Wan-AI/Wan2.2-T2V-A14B-Diffusers, resolution=480, num_frames=5,
#   num_steps=10, eval_steps=28, flow_shift=3.0 (engine-launch override of the
#   sgl-d serving default 12.0), guidance=4.0 (high-noise) / 3.0 (low-noise expert),
#   Flow-SDE noise_level=0.9, beta=0 (no KL), per-prompt mean/std.
#   train: lr=1e-4, adam_beta2=0.999, weight_decay=1e-4, clip_range=1e-4,
#          mixed precision (master fp32 / forward bf16).
#   LoRA: r=64, alpha=128, init=gaussian, self-attn + cross-attn + FFN of both DiTs.
#
# SDE schedule: epoch_global_random_choice draws ONE step per rollout (shared across the
#   batch) from --diffusion-sde-candidate-steps 1,2,3. At flow_shift=3.0 the
#   dual-expert boundary is t=875: steps 1,2 train "transformer" (high-noise),
#   step 3 trains "transformer_2" (low-noise), so both experts get gradient
#   stochastically and --update-weight-target-module syncs both.
#
# Per rollout: 48 prompts × 16 samples = 768 items.
#   num_steps_per_rollout=2 → 384 items/optim step ÷ 4 train gpus = 96 items/rank.
#   --micro-batch-size 2: the one-step-per-rollout schedule keeps every
#   micro-batch phase-pure (one DiT, one CFG scale); mbs=4 OOMs on H200, 2 fits.
#
# NOTE: gradient checkpointing stays OFF. Wan2.2 under FSDP2 mixed precision hits
#   torch.utils.checkpoint CheckpointError (fp32 RoPE freqs buffers; fix pending
#   in a separate PR). If you OOM, lower --rollout-batch-size,
#   --n-samples-per-prompt, or --diffusion-microgroup-size.
#
# Layout: first 4 GPUs in CUDA_VISIBLE_DEVICES = train+sgld colocate,
# the 5th GPU = pickscore reward worker. Default: GPU 0,1,2,3 + GPU 4.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
RUN_NAME="diffusion_grpo_wan22_pickscore_5gpu_$(date +%Y%m%d_%H%M%S)"
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
  --micro-batch-size 2 \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 5 \
  --colocate \
  --use-lora \
  --lora-ipc-weight-sync \
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
  --update-weight-target-module transformer,transformer_2 \
  --diffusion-reward pickscore:1.0 \
  --advantage-estimator grpo \
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
  --diffusion-output-num-frames 5 \
  --diffusion-guidance-scale 4.0 \
  --diffusion-guidance-scale-2 3.0 \
  --diffusion-noise-level 0.9 \
  --diffusion-height 480 \
  --diffusion-width 480 \
  --diffusion-flow-shift 3.0 \
  --diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_random_choice \
  --diffusion-num-sde-steps 1 \
  --diffusion-sde-candidate-steps 1,2,3 \
  --diffusion-debug-mode \
  --save "${SAVE_DIR}" \
  --save-interval 10 \
  --eval-prompt-data pickscore_test "${DATASETS_DIR}/flowgrpo_pickscore/test.jsonl" \
  --eval-interval 30 \
  --skip-eval-before-train \
  "${WANDB_ARGS[@]}"
