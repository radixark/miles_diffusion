#!/usr/bin/env bash
# LTX-2.3 video PickScore GRPO: sglang rollout + FSDP train (colocate).
#
# Default: 1-GPU colocate (train FSDP + sglang rollout). Override NUM_GPUS for
# multi-GPU colocate. CPS dynamics, 3 SDE steps from candidates 0–9, clip 1e-4.
#
# Layout mirrors other scripts/run-diffusion-grpo-*.sh recipes:
#   train+rollout share the first NUM_GPUS in CUDA_VISIBLE_DEVICES;
#   optional pickscore worker uses additional GPUs when configured.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="diffusion_grpo_ltx23_pickscore_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${ROOT_DIR}/logs/${RUN_NAME}/ckpt"
mkdir -p "${SAVE_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

DATASETS_DIR="${DATASETS_DIR:-/root/datasets/miles-diffusion-datasets}"
if [[ ! -f "${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl" ]]; then
  hf download --repo-type dataset rockdu/miles-diffusion-datasets \
    --include "flowgrpo_pickscore/**" \
    --local-dir "${DATASETS_DIR}"
fi

DIFFUSION_MODEL="${DIFFUSION_MODEL:-Lightricks/LTX-2.3}"
NUM_GPUS="${NUM_GPUS:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-miles-diffusion-grpo}"
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --diffusion-log-images "${WANDB_LOG_IMAGES:-4}"
    --diffusion-log-image-interval "${WANDB_LOG_IMAGE_INTERVAL:-10}"
    --disable-wandb-random-suffix
  )
fi

LORA_ARGS=()
if [[ "${USE_LORA:-1}" == "1" ]]; then
  LORA_ARGS+=(
    --use-lora
    --lora-rank 64
    --lora-alpha 128
    --diffusion-init-lora-weight gaussian
  )
fi

LTX_AV_ARGS=()
if [[ "${LTX_DISABLE_AV_CROSS_ATTN:-0}" == "1" ]]; then
  LTX_AV_ARGS+=(--ltx-disable-av-cross-attn)
fi

"${PYTHON_BIN}" -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --diffusion-model "${DIFFUSION_MODEL}" \
  --diffusion-model-type ltx \
  --hf-checkpoint gpt2 \
  --prompt-data "${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl" \
  --input-key input \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}" \
  --num-rollout "${NUM_ROLLOUT}" \
  --micro-batch-size-sample "${MICRO_BATCH_SIZE_SAMPLE:-1}" \
  --micro-batch-size-tstep "${MICRO_BATCH_SIZE_TSTEP:-1}" \
  --gradient-checkpointing \
  --colocate \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --actor-num-nodes 1 \
  --num-gpus-per-node "${NUM_GPUS}" \
  --rollout-num-gpus "${NUM_GPUS}" \
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}" \
  --use-miles-router \
  --rollout-health-check-interval "${ROLLOUT_HEALTH_CHECK_INTERVAL:-120}" \
  --miles-router-health-check-failure-threshold "${MILES_ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD:-30}" \
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-1}" \
  --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND:-torch_sdpa}" \
  "${LORA_ARGS[@]}" \
  "${LTX_AV_ARGS[@]}" \
  --lr 2e-4 \
  --adam-beta2 0.999 \
  --weight-decay 1e-4 \
  --diffusion-clip-range "${CLIP_RANGE:-1e-4}" \
  --diffusion-kl-beta 0.0 \
  --diffusion-num-steps "${NUM_STEPS:-24}" \
  --diffusion-step-strategy-path miles.rollout.step_strategy_hub.ltx_sde_candidates \
  --ltx-num-sde-steps "${LTX_NUM_SDE_STEPS:-3}" \
  --ltx-sde-step-candidates "${LTX_SDE_STEP_CANDIDATES:-0,1,2,3,4,5,6,7,8,9}" \
  --ltx-dynamics-type CPS \
  --diffusion-noise-level 0.8 \
  --ltx-sigma-min 0.001 \
  --diffusion-guidance-scale 1.0 \
  --diffusion-height "${HEIGHT:-512}" \
  --diffusion-width "${WIDTH:-768}" \
  --ltx-frames "${FRAMES:-57}" \
  --ltx-fps "${LTX_FPS:-24}" \
  --diffusion-forward-dtype bf16 \
  --fsdp-master-dtype bf16 \
  --fsdp-reduce-dtype bf16 \
  --sglang-dit-precision bf16 \
  --advantage-estimator grpo \
  --globalize-reward-std \
  --rm-type pickscore \
  --diffusion-reward "pickscore:1.0" \
  --reward-key avg \
  --pickscore-processor-path "${PICKSCORE_PROCESSOR:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}" \
  --pickscore-model-path "${PICKSCORE_MODEL:-yuvalkirstain/PickScore_v1}" \
  --pickscore-num-frames "${PICKSCORE_NUM_FRAMES:-3}" \
  --pickscore-num-gpus-per-worker "${PICKSCORE_NUM_GPUS_PER_WORKER:-0}" \
  --pickscore-batch-size 8 \
  --update-weight-buffer-size 2147483648 \
  --save "${SAVE_DIR}" \
  --save-interval "${SAVE_INTERVAL}" \
  "${WANDB_ARGS[@]}" \
  "$@"
