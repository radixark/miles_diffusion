#!/usr/bin/env bash
# Wan2.2-T2V-A14B PickScore GRPO —— AC-9 perf 闸用 4 卡变体（reward 跑 CPU）。
#
# 与 8gpu recipe 的差异：
#   - 4 卡全给 colocate train+rollout（actor/rollout/num-gpus-per-node=NUM_GPUS，默认 4）；
#   - PickScore reward 跑 **CPU**（PICKSCORE_NUM_GPUS_PER_WORKER=0 → ray num_gpus=0 →
#     actor get_gpu_ids() 空 → device=cpu），不再独占第 5 张卡；
#   - SP 可配：SEQUENCE_PARALLEL_SIZE/ULYSSES_DEGREE/RING_DEGREE（不设=纯 FSDP dp4 基线）；
#   - 短跑：NUM_ROLLOUT 默认 5（×NUM_STEPS_PER_ROLLOUT=2 = 10 train steps）；
#   - 帧数可调：DIFFUSION_OUTPUT_NUM_FRAMES（默认 1=图像；SP 收益需调大到长序列）。
# 约束：global_batch = rollout_batch*n_samples/num_steps_per_rollout 须被 dp_size 整除
#   （dp_size = NUM_GPUS / sequence_parallel_size）。默认 4*8/2=16，被 4/2/1 均整除。
#
# 用法：
#   纯 FSDP dp4 基线: NUM_ROLLOUT=5 bash scripts/run-...-4gpu-perfgate.sh
#   FSDP+SP dp2×sp2 : SEQUENCE_PARALLEL_SIZE=2 ULYSSES_DEGREE=2 bash scripts/...
#   FSDP+SP dp1×sp4 : SEQUENCE_PARALLEL_SIZE=4 ULYSSES_DEGREE=4 bash scripts/...

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export HF_HOME="${HF_HOME:-/workspace/809a2940-8360-4812-81c2-c7383f3f43e7/models}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/workspace/809a2940-8360-4812-81c2-c7383f3f43e7/.cache/flashinfer}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/809a2940-8360-4812-81c2-c7383f3f43e7/miniforge3/envs/miles-diffusion/bin/python}"
HF_BIN="${HF_BIN:-$(dirname "${PYTHON_BIN}")/hf}"
NUM_GPUS="${NUM_GPUS:-4}"
RUN_NAME="${RUN_NAME:-wan22_pickscore_perfgate_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/logs/${RUN_NAME}/ckpt}"
DATASETS_DIR="${DATASETS_DIR:-/workspace/809a2940-8360-4812-81c2-c7383f3f43e7/datasets/miles-diffusion-datasets}"

"${HF_BIN}" download --repo-type dataset rockdu/miles-diffusion-datasets \
  --include "flowgrpo_pickscore/**" \
  --local-dir "${DATASETS_DIR}"

WAN_LORA_TARGET_MODULES=(
  attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0
  attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0
  ffn.net.0.proj ffn.net.2
)

# SP 配置：仅当 SEQUENCE_PARALLEL_SIZE>1 时追加（不设=纯 FSDP dp4 基线）。
SP_ARGS=()
if [[ "${SEQUENCE_PARALLEL_SIZE:-1}" != "1" ]]; then
  SP_ARGS+=(--sequence-parallel-size "${SEQUENCE_PARALLEL_SIZE}")
  [[ -n "${ULYSSES_DEGREE:-}" ]] && SP_ARGS+=(--ulysses-degree "${ULYSSES_DEGREE}")
  [[ -n "${RING_DEGREE:-}" ]] && SP_ARGS+=(--ring-degree "${RING_DEGREE}")
fi

CHECKPOINT_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING:-1}" == "1" ]]; then
  CHECKPOINT_ARGS+=(--gradient-checkpointing)
fi

WANDB_ARGS=()
if [[ "${USE_WANDB:-0}" == "1" || -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(--use-wandb --wandb-mode "${WANDB_MODE:-offline}"
    --wandb-project "${WANDB_PROJECT:-pickscore-perfgate}" --wandb-group "${RUN_NAME}"
    --disable-wandb-random-suffix)
  [[ -n "${WANDB_API_KEY:-}" ]] && WANDB_ARGS+=(--wandb-key "${WANDB_API_KEY}")
fi

"${PYTHON_BIN}" -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint /workspace/809a2940-8360-4812-81c2-c7383f3f43e7/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --diffusion-model /workspace/809a2940-8360-4812-81c2-c7383f3f43e7/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt-data "${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl" \
  --input-key input \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}" \
  --num-rollout "${NUM_ROLLOUT:-5}" \
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT:-2}" \
  --diffusion-microgroup-size "${DIFFUSION_MICROGROUP_SIZE:-8}" \
  --micro-batch-size-sample "${MICRO_BATCH_SIZE_SAMPLE:-1}" \
  --micro-batch-size-tstep "${MICRO_BATCH_SIZE_TSTEP:-1}" \
  --diffusion-train-iter-order sample_major \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --rollout-num-gpus "${NUM_GPUS}" \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  "${SP_ARGS[@]}" \
  --use-lora \
  --lora-rank "${LORA_RANK:-64}" \
  --lora-alpha "${LORA_ALPHA:-128}" \
  --lora-target-modules "${WAN_LORA_TARGET_MODULES[@]}" \
  --diffusion-init-lora-weight gaussian \
  --lr "${LR:-1e-4}" \
  --adam-beta2 0.999 \
  --diffusion-clip-range "${DIFFUSION_CLIP_RANGE:-1e-4}" \
  --weight-decay 1e-4 \
  --use-miles-router \
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-8}" \
  --update-weight-buffer-size 2147483648 \
  --diffusion-reward pickscore:1.0 \
  --advantage-estimator grpo \
  --rm-type pickscore \
  --pickscore-num-workers "${PICKSCORE_NUM_WORKERS:-2}" \
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
  --diffusion-debug-mode \
  --save "${SAVE_DIR}" \
  --save-interval "${SAVE_INTERVAL:-1000}" \
  "${CHECKPOINT_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
