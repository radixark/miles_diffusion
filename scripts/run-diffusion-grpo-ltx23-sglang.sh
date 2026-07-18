#!/usr/bin/env bash
# LTX-2.3 video PickScore GRPO: sglang rollout + FSDP train (colocate).
#
# Default: 2-GPU colocate on CUDA 6,7 (train FSDP DP + one sglang engine / GPU).
# Override with CUDA_VISIBLE_DEVICES / NUM_GPUS. CPS dynamics, 3 SDE steps from
# candidates 0–9, clip 1e-4.
#
# Examples:
#   # formal 2-GPU (default)
#   bash scripts/run-diffusion-grpo-ltx23-sglang.sh
#
#   # smoke
#   CUDA_VISIBLE_DEVICES=6,7 NUM_GPUS=2 \
#     ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=2 NUM_ROLLOUT=1 NUM_STEPS_PER_ROLLOUT=1 \
#     bash scripts/run-diffusion-grpo-ltx23-sglang.sh
#
#   # single-GPU
#   CUDA_VISIBLE_DEVICES=6 NUM_GPUS=1 bash scripts/run-diffusion-grpo-ltx23-sglang.sh
#
# Layout: train+rollout share the first NUM_GPUS in CUDA_VISIBLE_DEVICES;
# optional pickscore worker uses additional GPUs when configured.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${NUM_GPUS:-}" ]]; then
  IFS=',' read -ra _VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_GPUS="${#_VISIBLE_GPUS[@]}"
fi
NUM_GPUS="${NUM_GPUS}"

RUN_NAME="diffusion_grpo_ltx23_pickscore_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${ROOT_DIR}/logs/${RUN_NAME}/ckpt"
mkdir -p "${SAVE_DIR}"
# Per-run metric recording; registerable as CI standard (tests/ci/e2e_metrics_registry.py).
export MILES_METRICS_JSONL="${MILES_METRICS_JSONL:-${ROOT_DIR}/logs/${RUN_NAME}/metrics.jsonl}"

PYTHON_BIN="${PYTHON_BIN:-python}"

DATASETS_DIR="${DATASETS_DIR:-/root/datasets/miles-diffusion-datasets}"
if [[ ! -f "${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl" ]]; then
  hf download --repo-type dataset rockdu/miles-diffusion-datasets \
    --include "flowgrpo_pickscore/**" \
    --local-dir "${DATASETS_DIR}"
fi

DIFFUSION_MODEL="${DIFFUSION_MODEL:-Lightricks/LTX-2.3}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
# One engine per GPU by default; concurrency scales with engines.
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-${NUM_GPUS}}"

echo "[ltx23] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS} engines=$((NUM_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE))"
echo "[ltx23] batch=${ROLLOUT_BATCH_SIZE}x${N_SAMPLES_PER_PROMPT} rollouts=${NUM_ROLLOUT} save_interval=${SAVE_INTERVAL}"
echo "[ltx23] run=${RUN_NAME}"

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

"${PYTHON_BIN}" -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  ${DETERMINISTIC_MODE:+--deterministic-mode} \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --diffusion-model "${DIFFUSION_MODEL}" \
  --hf-checkpoint gpt2 \
  --prompt-data "${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl" \
  --input-key input \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}" \
  --num-rollout "${NUM_ROLLOUT}" \
  --micro-batch-size-sample "${MICRO_BATCH_SIZE_SAMPLE:-1}" \
  --micro-batch-size-tstep "${MICRO_BATCH_SIZE_TSTEP:-1}" \
  --diffusion-train-iter-order "${TRAIN_ITER_ORDER:-sample_major}" \
  --diffusion-microgroup-size "${MICROGROUP_SIZE:-1}" \
  --gradient-checkpointing \
  --colocate \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --actor-num-nodes 1 \
  --num-gpus-per-node "${NUM_GPUS}" \
  --rollout-num-gpus "${NUM_GPUS}" \
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
  --use-miles-router \
  --rollout-health-check-interval "${ROLLOUT_HEALTH_CHECK_INTERVAL:-120}" \
  --miles-router-health-check-failure-threshold "${MILES_ROUTER_HEALTH_CHECK_FAILURE_THRESHOLD:-30}" \
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}" \
  --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND:-torch_sdpa}" \
  "${LORA_ARGS[@]}" \
  --lr 2e-4 \
  --adam-beta2 0.999 \
  --weight-decay 1e-4 \
  --diffusion-clip-range "${CLIP_RANGE:-1e-5}" \
  --diffusion-kl-beta 0.0 \
  --diffusion-num-steps "${NUM_STEPS:-24}" \
  --diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_random_choice \
  --diffusion-num-sde-steps "${LTX_NUM_SDE_STEPS:-3}" \
  --diffusion-sde-candidate-steps "${LTX_SDE_STEP_CANDIDATES:-0,1,2,3,4,5,6,7,8,9}" \
  --fsdp-attention-backend "${FSDP_ATTENTION_BACKEND:-sdpa_math}" \
  --diffusion-sde-type cps \
  --diffusion-noise-level 0.8 \
  --diffusion-guidance-scale 1.0 \
  --diffusion-height "${HEIGHT:-512}" \
  --diffusion-width "${WIDTH:-768}" \
  --diffusion-output-num-frames "${FRAMES:-57}" \
  --diffusion-fps "${LTX_FPS:-24}" \
  --diffusion-forward-dtype bf16 \
  --fsdp-master-dtype bf16 \
  --fsdp-reduce-dtype bf16 \
  --sglang-dit-precision bf16 \
  --advantage-estimator grpo \
  --globalize-reward-std \
  --rm-type pickscore \
  --diffusion-reward "pickscore:1.0" \
  --pickscore-processor-path "${PICKSCORE_PROCESSOR:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}" \
  --pickscore-model-path "${PICKSCORE_MODEL:-yuvalkirstain/PickScore_v1}" \
  --pickscore-num-frames "${PICKSCORE_NUM_FRAMES:-3}" \
  --pickscore-num-gpus-per-worker "${PICKSCORE_NUM_GPUS_PER_WORKER:-0}" \
  --pickscore-num-workers "${PICKSCORE_NUM_WORKERS:-1}" \
  --pickscore-batch-size "${PICKSCORE_BATCH_SIZE:-8}" \
  --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE:-2147483648}" \
  --save "${SAVE_DIR}" \
  --save-interval "${SAVE_INTERVAL}" \
  "${WANDB_ARGS[@]}" \
  "$@"
