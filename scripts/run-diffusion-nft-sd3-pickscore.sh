#!/usr/bin/env bash
# SD3.5 DiffusionNFT training with PickScore.

MILES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail

ROOT_DIR="${MILES_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_TOKEN="${HF_TOKEN:-}"
unset RAY_ADDRESS HF_HUB_OFFLINE TRANSFORMERS_OFFLINE 2>/dev/null || true
rm -f /tmp/ray/session_latest

SD3_MODEL="${SD3_MODEL:-stabilityai/stable-diffusion-3.5-medium}"
SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
  NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
else
  NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
fi
RUN_NAME="diffusion_nft_sd3_pickscore_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${ROOT_DIR}/logs/${RUN_NAME}/ckpt"
mkdir -p "${SAVE_DIR}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project miles-diffusion-nft
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --diffusion-log-images 8
    --diffusion-log-image-interval 10
    --disable-wandb-random-suffix
  )
fi

DATASETS_DIR="${DATASETS_DIR:-/root/datasets/miles-diffusion-datasets}"
if [[ "${SMOKE}" == "1" ]]; then
  hf download --repo-type dataset rockdu/miles-diffusion-datasets \
    --include "flowgrpo_ocr/**" \
    --local-dir "${DATASETS_DIR}"
  PROMPT_DATA="${DATASETS_DIR}/flowgrpo_ocr/train.jsonl"
  REWARD_ARGS=(
    --diffusion-reward ocr:1.0
    --rm-type ocr
  )
  # 2-GPU smoke: tiny batch, no dedicated reward GPU.
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
  BATCH_ARGS=(
    --rollout-batch-size 2
    --n-samples-per-prompt 2
    --num-rollout "${NUM_ROLLOUT}"
    --num-steps-per-rollout 1
    --micro-batch-size 2
    --diffusion-microgroup-size 2
    --actor-num-gpus-per-node 2
    --rollout-num-gpus 2
    --rollout-num-gpus-per-engine 1
    --num-gpus-per-node 2
  )
else
  hf download --repo-type dataset rockdu/miles-diffusion-datasets \
    --include "flowgrpo_pickscore/**" \
    --local-dir "${DATASETS_DIR}"
  PROMPT_DATA="${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl"
  REWARD_ARGS=(
    --diffusion-reward pickscore:1.0
    --rm-type pickscore
    --pickscore-num-workers 1
    --pickscore-num-gpus-per-worker 1.0
    --pickscore-batch-size 8
    --pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K
    --pickscore-model-path yuvalkirstain/PickScore_v1
  )
  # Match UniRL 100-roll override: 8×8 prompts/samples, micro=4.
  BATCH_ARGS=(
    --rollout-batch-size 8
    --n-samples-per-prompt 8
    --num-rollout "${NUM_ROLLOUT}"
    --num-steps-per-rollout 1
    --micro-batch-size 4
    --diffusion-microgroup-size 8
    --actor-num-gpus-per-node 2
    --rollout-num-gpus 2
    --rollout-num-gpus-per-engine 1
    --num-gpus-per-node 3
    --eval-prompt-data pickscore_test "${DATASETS_DIR}/flowgrpo_pickscore/test.jsonl"
    --eval-interval 30
  )
fi

echo "RUN=${RUN_NAME}" | tee "${ROOT_DIR}/logs/${RUN_NAME}.log"

python -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint "${SD3_MODEL}" \
  --prompt-data "${PROMPT_DATA}" \
  --input-key input \
  "${BATCH_ARGS[@]}" \
  --gradient-checkpointing \
  --colocate \
  --use-miles-router \
  --sglang-server-concurrency 8 \
  --use-lora \
  --lora-ipc-weight-sync \
  --lora-rank 32 \
  --lora-alpha 64 \
  --diffusion-init-lora-weight gaussian \
  --lr 3e-4 \
  --adam-beta2 0.999 \
  --weight-decay 1e-4 \
  --clip-grad 1.0 \
  --loss-type nft \
  --diffusion-nft-beta 1.0 \
  --diffusion-nft-adv-clip-max 5.0 \
  --diffusion-nft-timestep-fraction 0.99 \
  --ref-mode ema \
  --ema-shadow \
  --ema-rollout-policy ema \
  --ema-decay 0.001 \
  --ema-uprate 0.001 \
  --ema-uphold 0.5 \
  --ema-flat-steps 0 \
  --advantage-estimator grpo \
  --globalize-reward-std \
  --diffusion-model "${SD3_MODEL}" \
  "${REWARD_ARGS[@]}" \
  --diffusion-forward-dtype fp16 \
  --sglang-dit-precision fp16 \
  --sglang-vae-slicing \
  --diffusion-num-steps 10 \
  --diffusion-eval-num-steps 50 \
  --update-weight-buffer-size 2147483648 \
  --diffusion-guidance-scale 1.0 \
  --diffusion-noise-level 0.0 \
  --diffusion-sde-type ode \
  --diffusion-height 512 \
  --diffusion-width 512 \
  --save "${SAVE_DIR}" \
  --save-interval 20 \
  --skip-eval-before-train \
  "${WANDB_ARGS[@]}" \
  2>&1 | tee -a "${ROOT_DIR}/logs/${RUN_NAME}.log"
