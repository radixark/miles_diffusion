#!/usr/bin/env bash
# Wan2.2-T2V-A14B 1-frame PickScore GRPO recipe for a 4-GPU colocated run.
#
# Data handling follows the existing Miles PickScore scripts:
#   rockdu/miles-diffusion-datasets/flowgrpo_pickscore/{train,test}.jsonl
#
# Training knobs are aligned with Flow-Factory's Wan2.2 LoRA GRPO recipe where
# practical in Miles:
#   - LoRA r=64/alpha=128
#   - lr=1e-4, beta2=0.999, weight_decay=1e-4, clip_range=1e-4
#   - 10 rollout steps, guidance 4.0 / 3.0
#   - Flow-SDE noise_level=0.9
#   - one SDE step sampled from Wan high-noise indices 1,2,3
#   - Wan LoRA targets include self-attn, cross-attn, and FFN

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,7}"
export HF_HOME="${HF_HOME:-/data/andyye/.cache/huggingface}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/data/andyye/.cache/flashinfer}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-/data/andyye/miniforge3/envs/miles-diffusion/bin/python}"
HF_BIN="${HF_BIN:-$(dirname "${PYTHON_BIN}")/hf}"
RUN_NAME="${RUN_NAME:-wan22_pickscore_4gpu_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/logs/${RUN_NAME}/ckpt}"
DATASETS_DIR="${DATASETS_DIR:-/data/andyye/datasets/miles-diffusion-datasets}"

"${HF_BIN}" download --repo-type dataset rockdu/miles-diffusion-datasets \
  --include "flowgrpo_pickscore/**" \
  --local-dir "${DATASETS_DIR}"

WAN_LORA_TARGET_MODULES=(
  attn1.to_q
  attn1.to_k
  attn1.to_v
  attn1.to_out.0
  attn2.to_q
  attn2.to_k
  attn2.to_v
  attn2.to_out.0
  ffn.net.0.proj
  ffn.net.2
)

WANDB_ARGS=()
if [[ "${USE_WANDB:-0}" == "1" || -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-mode "${WANDB_MODE:-online}"
    --wandb-project "${WANDB_PROJECT:-pickscore}"
    --wandb-group "${RUN_NAME}"
    --diffusion-log-images "${DIFFUSION_LOG_IMAGES:-8}"
    --diffusion-log-image-interval "${DIFFUSION_LOG_IMAGE_INTERVAL:-1}"
    --disable-wandb-random-suffix
  )
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_API_KEY}")
  fi
  if [[ -n "${WANDB_TEAM:-}" ]]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
  fi
  if [[ -n "${WANDB_DIR:-}" ]]; then
    WANDB_ARGS+=(--wandb-dir "${WANDB_DIR}")
  fi
fi

EVAL_ARGS=()
if [[ "${ENABLE_EVAL:-0}" == "1" ]]; then
  EVAL_ARGS+=(
    --diffusion-eval-num-steps "${DIFFUSION_EVAL_NUM_STEPS:-28}"
    --eval-prompt-data pickscore_test "${DATASETS_DIR}/flowgrpo_pickscore/test.jsonl"
    --eval-interval "${EVAL_INTERVAL:-20}"
    --skip-eval-before-train
  )
fi

CHECKPOINT_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then
  CHECKPOINT_ARGS+=(--gradient-checkpointing)
fi

REWARD_NORM_ARGS=()
if [[ "${GLOBALIZE_REWARD_STD:-0}" == "1" ]]; then
  REWARD_NORM_ARGS+=(--globalize-reward-std)
fi

"${PYTHON_BIN}" -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --diffusion-model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt-data "${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl" \
  --input-key input \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-16}" \
  --num-rollout "${NUM_ROLLOUT:-1000}" \
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT:-2}" \
  --diffusion-microgroup-size "${DIFFUSION_MICROGROUP_SIZE:-1}" \
  --micro-batch-size-sample "${MICRO_BATCH_SIZE_SAMPLE:-1}" \
  --micro-batch-size-tstep "${MICRO_BATCH_SIZE_TSTEP:-1}" \
  --diffusion-train-iter-order sample_major \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node "${NUM_GPUS_PER_NODE:-4}" \
  --colocate \
  --use-lora \
  --lora-rank "${LORA_RANK:-64}" \
  --lora-alpha "${LORA_ALPHA:-128}" \
  --lora-target-modules "${WAN_LORA_TARGET_MODULES[@]}" \
  --diffusion-init-lora-weight gaussian \
  --lr "${LR:-1e-4}" \
  --adam-beta2 0.999 \
  --diffusion-clip-range 1e-4 \
  --weight-decay 1e-4 \
  --use-miles-router \
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-1}" \
  --update-weight-buffer-size 2147483648 \
  --diffusion-reward pickscore:1.0 \
  --advantage-estimator grpo \
  --rm-type pickscore \
  --pickscore-num-workers "${PICKSCORE_NUM_WORKERS:-1}" \
  --pickscore-num-gpus-per-worker "${PICKSCORE_NUM_GPUS_PER_WORKER:-0}" \
  --pickscore-batch-size "${PICKSCORE_BATCH_SIZE:-8}" \
  --pickscore-processor-path "${PICKSCORE_PROCESSOR_PATH:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}" \
  --pickscore-model-path "${PICKSCORE_MODEL_PATH:-yuvalkirstain/PickScore_v1}" \
  --fsdp-master-dtype fp32 \
  --fsdp-reduce-dtype fp32 \
  --diffusion-forward-dtype bf16 \
  --diffusion-num-steps "${DIFFUSION_NUM_STEPS:-10}" \
  --diffusion-output-num-frames "${DIFFUSION_OUTPUT_NUM_FRAMES:-1}" \
  --diffusion-guidance-scale "${DIFFUSION_GUIDANCE_SCALE:-4.0}" \
  --diffusion-guidance-scale-2 "${DIFFUSION_GUIDANCE_SCALE_2:-3.0}" \
  --diffusion-noise-level "${DIFFUSION_NOISE_LEVEL:-0.9}" \
  --diffusion-height "${DIFFUSION_HEIGHT:-480}" \
  --diffusion-width "${DIFFUSION_WIDTH:-480}" \
  --diffusion-step-strategy-path "${DIFFUSION_STEP_STRATEGY_PATH:-miles.rollout.step_strategy_hub.wan_high_window}" \
  --diffusion-sde-window-size "${DIFFUSION_SDE_WINDOW_SIZE:-1}" \
  --diffusion-sde-window-range "${DIFFUSION_SDE_WINDOW_RANGE:-1,4}" \
  --save "${SAVE_DIR}" \
  --save-interval "${SAVE_INTERVAL:-10}" \
  "${REWARD_NORM_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
