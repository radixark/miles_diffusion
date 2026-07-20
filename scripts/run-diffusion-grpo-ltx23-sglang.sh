#!/usr/bin/env bash
# LTX-2.3 video PickScore GRPO: 4-GPU FSDP train + sglang rollout colocate, 1-GPU pickscore.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_NAME="diffusion_grpo_ltx23_pickscore_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${ROOT_DIR}/logs/${RUN_NAME}/ckpt"
mkdir -p "${SAVE_DIR}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project miles-diffusion-grpo
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --diffusion-log-images 4
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
  --deterministic-mode \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --diffusion-model Lightricks/LTX-2.3 \
  --hf-checkpoint gpt2 \
  --prompt-data "${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl" \
  --input-key input \
  --rollout-batch-size 8 \
  --n-samples-per-prompt 8 \
  --num-steps-per-rollout 2 \
  --num-rollout "${NUM_ROLLOUT:-200}" \
  --micro-batch-size-sample 1 \
  --micro-batch-size-tstep 1 \
  --diffusion-train-iter-order sample_major \
  --diffusion-microgroup-size 1 \
  --gradient-checkpointing \
  --colocate \
  --actor-num-gpus-per-node 4 \
  --actor-num-nodes 1 \
  --num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 1 \
  --use-miles-router \
  --rollout-health-check-interval 120 \
  --miles-router-health-check-failure-threshold 30 \
  --sglang-server-concurrency 4 \
  --sglang-attention-backend torch_sdpa \
  --use-lora \
  --lora-rank 64 \
  --lora-alpha 128 \
  --diffusion-init-lora-weight gaussian \
  --lr 2e-4 \
  --adam-beta2 0.999 \
  --weight-decay 1e-4 \
  --diffusion-clip-range 1e-5 \
  --diffusion-kl-beta 0.0 \
  --diffusion-num-steps 24 \
  --diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_random_choice \
  --diffusion-num-sde-steps 3 \
  --diffusion-sde-candidate-steps 0,1,2,3,4,5,6,7,8,9 \
  --fsdp-attention-backend sdpa_math \
  --diffusion-sde-type cps \
  --diffusion-noise-level 0.8 \
  --diffusion-guidance-scale 1.0 \
  --diffusion-height 512 \
  --diffusion-width 768 \
  --diffusion-output-num-frames 57 \
  --diffusion-fps 24 \
  --diffusion-forward-dtype bf16 \
  --fsdp-master-dtype bf16 \
  --fsdp-reduce-dtype bf16 \
  --sglang-dit-precision bf16 \
  --advantage-estimator grpo \
  --globalize-reward-std \
  --rm-type pickscore \
  --diffusion-reward pickscore:1.0 \
  --pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --pickscore-model-path yuvalkirstain/PickScore_v1 \
  --pickscore-num-frames 3 \
  --pickscore-num-gpus-per-worker 1.0 \
  --pickscore-num-workers 1 \
  --rollout-parser-num-workers 8 \
  --pickscore-batch-size 8 \
  --update-weight-buffer-size 2147483648 \
  --save "${SAVE_DIR}" \
  --save-interval 50 \
  "${WANDB_ARGS[@]}" \
  "$@"
