#!/usr/bin/env bash
# LTX-2.3 sglang-rollout GRPO — dev checkpoint, 512x768x57f, 24 steps, CPS.
#
# Mirrors the legacy trainer-rollout reward run
# (/sgl-workspace/miles/scripts/run-diffusion-grpo-ltx23-trainer-rollout.sh):
#   CPS dynamics, 3 SDE steps from candidates 0–9, clip-range 1e-4.
# Rollout goes through sglang with weight sync; train/rollout forward alignment
# fixes stay on (ltxcore parity + AV-off + identity guider).
#
# GPU layout: single physical GPU colocate (train FSDP world_size=1 and sglang
#   rollout time-share one GPU via offload). Set NUM_GPUS>1 for multi-GPU
#   colocate if 512x768x57f OOMs on one card.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 USE_LORA=1 NUM_ROLLOUT=200 \
#     LTX_DISABLE_AV_CROSS_ATTN=1 \
#     nohup bash scripts/run-diffusion-grpo-ltx23-sglang.sh \
#     > logs/ltx23_dev_cps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Key overridable env:
#   LTX_MODEL_PATH          — dev 22B safetensors (train + rollout DiT via transformer_weights_path)
#   MILES_LTX_ROLLOUT_MODEL_PATH — optional; default Lightricks/LTX-2.3 (sglang overlay)
#   MILES_LTX_MODEL_ID      — optional; default LTX-2.3
#   HEIGHT WIDTH FRAMES     — 512 768 57
#   NUM_STEPS               — 24
#   LTX_NUM_SDE_STEPS       — 3
#   LTX_SDE_STEP_CANDIDATES — 0,1,2,3,4,5,6,7,8,9
#   CLIP_RANGE              — 1e-4
#   ROLLOUT_BATCH_SIZE      — unique prompts per rollout (default: 8)
#   N_SAMPLES_PER_PROMPT    — GRPO group size (default: 8)
#   NUM_STEPS_PER_ROLLOUT   — optimizer steps per rollout (default: 2 → gbs=32)
#   NUM_ROLLOUT             — 200
#   SAVE_INTERVAL           — 50

MILES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[kill] hunting for stale miles processes under cwd=${MILES_ROOT}"
for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  # timeout guards against readlink hanging on a process whose cwd points at a
  # stale/unresponsive mount — otherwise this loop can wedge the whole shell.
  link=$(timeout 2 readlink "/proc/${pid}/cwd" 2>/dev/null) || continue
  exe=$(timeout 2 readlink "/proc/${pid}/exe" 2>/dev/null) || continue
  case "${link}" in
    "${MILES_ROOT}"|"${MILES_ROOT}"/*)
      case "${exe}" in
        */python*|*/ray*)
          echo "[kill] ${pid} (${exe}) cwd=${link}"
          kill -9 "${pid}" 2>/dev/null || true
          ;;
      esac
      ;;
  esac
done
sleep 3

ps -eo ppid,state,comm --no-headers \
  | awk '$2=="Z" && $1!=1 && $3~/ray|python|sglang/ {print $1}' \
  | sort -u | xargs -r kill -9 2>/dev/null || true
sleep 2

set -euo pipefail

ROOT_DIR="${MILES_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SGLANG_PYTHON="${SGLANG_PYTHON:-/sgl-workspace/master_sglang/sglang/python}"
export PYTHONPATH="${SGLANG_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}"

# All heavy I/O lives on /data — workspace overlay is small and often full.
MILES_DATA_DISK_ROOT="${MILES_DATA_DISK_ROOT:-/data/wenhao/miles_diffusion}"
RAY_BIG_TMP="${RAY_BIG_TMP:-/data/wenhao/miles_ray_tmp}"
TMP_BIG="${TMP_BIG:-/data/wenhao/tmp}"
SGL_DIFF_CACHE="${SGLANG_DIFFUSION_CACHE_ROOT:-/data/wenhao/sgl_diffusion_cache}"
HF_HOME="${HF_HOME:-/data/wenhao/hf_home}"
LOG_DIR="${LOG_DIR:-${MILES_DATA_DISK_ROOT}/logs}"
WANDB_DIR="${WANDB_DIR:-${MILES_DATA_DISK_ROOT}/wandb}"
CKPT_ROOT="${CKPT_ROOT:-${MILES_DATA_DISK_ROOT}/ckpt}"
mkdir -p "${MILES_DATA_DISK_ROOT}" "${RAY_BIG_TMP}" "${TMP_BIG}" "${SGL_DIFF_CACHE}" \
  "${HF_HOME}" "${LOG_DIR}" "${WANDB_DIR}" "${CKPT_ROOT}"
export RAY_TMPDIR="${RAY_BIG_TMP}"
export TMPDIR="${TMP_BIG}"
export SGLANG_DIFFUSION_CACHE_ROOT="${SGL_DIFF_CACHE}"
export HF_HOME
export WANDB_DIR
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HUGGINGFACE_HUB_CACHE}"
export MILES_APPLY_LTX2_LTXCORE_PARITY="${MILES_APPLY_LTX2_LTXCORE_PARITY:-1}"
export RAY_object_spilling_config="$(python -c "import json,os; print(json.dumps({'type':'filesystem','params':{'directory_path':[os.environ['RAY_TMPDIR']]}}))")"
ray stop --force 2>/dev/null || true
sleep 2

# ── dev checkpoint (borrowed from legacy reward run) ─────────────────────
LTX_MODEL_PATH="${LTX_MODEL_PATH:-/sgl-workspace/rollout_compare/models/LTX-2.3/ltx-2.3-22b-dev.safetensors}"
# sglang text_encoder: use materialized Lightricks overlay (local gemma_for_ltx23
# symlinks often point at stale HF cache and break rollout startup).
LTX_MATERIALIZED_ROOT="${LTX_MATERIALIZED_ROOT:-/data/wenhao/sgl_diffusion_cache/materialized_models/Lightricks__LTX-2.3-10cce1713d7efa14}"
GEMMA_ROOT="${GEMMA_ROOT:-${LTX_MATERIALIZED_ROOT}/text_encoder}"
MILES_DATA_ROOT="${MILES_DATA_ROOT:-/sgl-workspace/miles}"
PROMPT_DATA="${PROMPT_DATA:-${MILES_DATA_ROOT}/data/vidgen/train.jsonl}"

NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
SAMPLES_PER_ROLLOUT=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
DERIVED_GLOBAL_BATCH_SIZE=$((SAMPLES_PER_ROLLOUT / NUM_STEPS_PER_ROLLOUT))
MICRO_BATCH_SIZE_SAMPLE="${MICRO_BATCH_SIZE_SAMPLE:-1}"
MICRO_BATCH_SIZE_TSTEP="${MICRO_BATCH_SIZE_TSTEP:-1}"

# ── borrowed-from-legacy generation config ───────────────────────────────
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-768}"
FRAMES="${FRAMES:-57}"
NUM_STEPS="${NUM_STEPS:-24}"
# ── trainer-rollout SDE config (CPS + candidate sampling) ────────────────
LTX_NUM_SDE_STEPS="${LTX_NUM_SDE_STEPS:-3}"
LTX_SDE_STEP_CANDIDATES="${LTX_SDE_STEP_CANDIDATES:-0,1,2,3,4,5,6,7,8,9}"
CLIP_RANGE="${CLIP_RANGE:-1e-4}"

NUM_GPUS="${NUM_GPUS:-1}"
# Multi-GPU colocate: one sglang engine PER GPU (each card runs both the sglang
# rollout engine and an FSDP trainer shard). per-engine=1 => num_engines=NUM_GPUS.
# Set ROLLOUT_NUM_GPUS_PER_ENGINE=NUM_GPUS instead for a single TP-sharded engine.
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
# Periodic checkpoint (LoRA adapter) so the run is resumable via LOAD_CKPT.
# (The earlier run had no --save-interval, so nothing was ever saved.)
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

if [[ ! -f "${PROMPT_DATA}" ]]; then
  python "${MILES_DATA_ROOT}/tools/prepare_vidgen_jsonl.py"
fi

RUN_NAME="ltx23_dev_cps_${NUM_ROLLOUT}step_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${CKPT_ROOT}/${RUN_NAME}"
mkdir -p "${SAVE_DIR}"

echo "[run] dev+cps CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS}"
echo "[run] dit=${LTX_MODEL_PATH}"
echo "[run] gemma=${GEMMA_ROOT}"
echo "[run] log=${LOG_DIR}"
echo "[run] wandb=${WANDB_DIR}"
echo "[run] save=${SAVE_DIR}"
echo "[run] ${HEIGHT}x${WIDTH}x${FRAMES}f steps=${NUM_STEPS} sde_steps=${LTX_NUM_SDE_STEPS} candidates=${LTX_SDE_STEP_CANDIDATES} clip=${CLIP_RANGE}"
echo "[run] batch: rollout=${ROLLOUT_BATCH_SIZE} n_samples=${N_SAMPLES_PER_PROMPT} samples/rollout=${SAMPLES_PER_ROLLOUT} optim_steps/rollout=${NUM_STEPS_PER_ROLLOUT} gbs=${DERIVED_GLOBAL_BATCH_SIZE} save_interval=${SAVE_INTERVAL}"

DEBUG_ARGS=()
if [[ "${MILES_DIFFUSION_DEBUG:-0}" == "1" ]]; then
  DEBUG_ARGS+=(--diffusion-debug-mode)
fi

DUMP_ARGS=()
if [[ -n "${LTX_FORWARD_DUMP_ROOT:-}" ]]; then
  mkdir -p "${LTX_FORWARD_DUMP_ROOT}"
  DUMP_ARGS+=(--dump-details "${LTX_FORWARD_DUMP_ROOT}")
fi

LTX_AV_ARGS=()
if [[ "${LTX_DISABLE_AV_CROSS_ATTN:-0}" == "1" ]]; then
  LTX_AV_ARGS+=(--ltx-disable-av-cross-attn)
  export MILES_LTX_DISABLE_AV_CROSS=1
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

SKIP_OPT_ARGS=()
if [[ "${SKIP_OPTIMIZER:-0}" == "1" ]]; then
  SKIP_OPT_ARGS+=(--debug-skip-optimizer-step)
fi

# Resume: point LOAD_CKPT at a previously saved --save dir (LoRA adapter).
LOAD_ARGS=()
if [[ -n "${LOAD_CKPT:-}" ]]; then
  LOAD_ARGS+=(--load "${LOAD_CKPT}")
fi

# WandB: enabled when WANDB_API_KEY is set. Mirrors the legacy reward run so the
# reward curve is directly comparable.
WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-dir "${WANDB_DIR}"
    --wandb-project "${WANDB_PROJECT:-miles-diffusion-grpo}"
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --diffusion-log-images "${WANDB_LOG_IMAGES:-4}"
    --diffusion-log-image-interval "${WANDB_LOG_IMAGE_INTERVAL:-5}"
    --disable-wandb-random-suffix
  )
fi

python -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --diffusion-model "${LTX_MODEL_PATH}" \
  --diffusion-model-type ltx \
  --ltx-gemma-path "${GEMMA_ROOT}" \
  --hf-checkpoint gpt2 \
  --prompt-data "${PROMPT_DATA}" \
  --input-key input \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}" \
  --num-rollout "${NUM_ROLLOUT}" \
  --micro-batch-size-sample "${MICRO_BATCH_SIZE_SAMPLE}" \
  --micro-batch-size-tstep "${MICRO_BATCH_SIZE_TSTEP}" \
  --gradient-checkpointing \
  --colocate \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --actor-num-nodes 1 \
  --num-gpus-per-node "${NUM_GPUS}" \
  --rollout-num-gpus "${NUM_GPUS}" \
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
  --use-miles-router \
  --rollout-health-check-interval 120 \
  --miles-router-health-check-failure-threshold 30 \
  --sglang-server-concurrency 1 \
  --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND:-torch_sdpa}" \
  "${LORA_ARGS[@]}" \
  "${LTX_AV_ARGS[@]}" \
  --lr 2e-4 \
  --adam-beta2 0.999 \
  --weight-decay 1e-4 \
  --diffusion-clip-range "${CLIP_RANGE}" \
  --diffusion-kl-beta 0.0 \
  --diffusion-num-steps "${NUM_STEPS}" \
  --diffusion-step-strategy-path miles.rollout.step_strategy_hub.ltx_sde_candidates \
  --ltx-num-sde-steps "${LTX_NUM_SDE_STEPS}" \
  --ltx-sde-step-candidates "${LTX_SDE_STEP_CANDIDATES}" \
  --ltx-dynamics-type CPS \
  --diffusion-noise-level 0.8 \
  --ltx-sigma-min 0.001 \
  --diffusion-guidance-scale 1.0 \
  --diffusion-height "${HEIGHT}" \
  --diffusion-width "${WIDTH}" \
  --ltx-frames "${FRAMES}" \
  --ltx-fps 24 \
  --diffusion-forward-dtype bf16 \
  --fsdp-master-dtype bf16 \
  --fsdp-reduce-dtype bf16 \
  --sglang-dit-precision bf16 \
  --advantage-estimator grpo \
  --globalize-reward-std \
  --rm-type pickscore \
  --diffusion-reward "pickscore:1.0" \
  --reward-key avg \
  --pickscore-processor-path "${PICKSCORE_PROCESSOR:-/data/wenhao/hf_home/pickscore}" \
  --pickscore-model-path "${PICKSCORE_MODEL:-/data/wenhao/hf_home/pickscore}" \
  --pickscore-num-frames 3 \
  --pickscore-batch-size 8 \
  --pickscore-num-gpus-per-worker 0 \
  --update-weight-buffer-size 2147483648 \
  --save "${SAVE_DIR}" \
  --save-interval "${SAVE_INTERVAL}" \
  "${LOAD_ARGS[@]}" \
  "${DEBUG_ARGS[@]}" \
  "${DUMP_ARGS[@]}" \
  "${SKIP_OPT_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "$@"
