#!/usr/bin/env bash
# Real-RL numeric guardrail: run the Wan2.2 PickScore recipe for a few steps on
# 4 GPUs (train+rollout colocate, PickScore reward on CPU) and compare train
# metrics between bands. With identical seeds the rollout data is identical, so
# step-1 adv_abs_mean / log_prob_old_idx_0 must match bitwise across bands;
# log_prob_new / loss / grad_norm may differ at bf16 summation level.
#
# Usage:
#   bash tests/sp/run_rl_guardrail.sh                            # FSDP dp4 baseline
#   SP_SIZE=2 ULYSSES_DEGREE=2 bash tests/sp/run_rl_guardrail.sh # dp2 x sp2
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PICKSCORE_NUM_GPUS_PER_WORKER=0

PYTHON_BIN="${PYTHON_BIN:-python}"
SP_SIZE="${SP_SIZE:-1}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-0}"
RING_DEGREE="${RING_DEGREE:-0}"
NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
NUM_FRAMES="${NUM_FRAMES:-5}"
GRAD_CKPT="${GRAD_CKPT:-0}"
RUN_NAME="${RUN_NAME:-sp_guardrail_sp${SP_SIZE}_$(date +%Y%m%d_%H%M%S)}"
DATASETS_DIR="/root/datasets/miles-diffusion-datasets"

EXTRA_ARGS=()
[[ "${GRAD_CKPT}" == "1" ]] && EXTRA_ARGS+=(--gradient-checkpointing)

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
  --rollout-batch-size 8 \
  --n-samples-per-prompt 8 \
  --num-rollout "${NUM_ROLLOUT}" \
  --num-steps-per-rollout 2 \
  --diffusion-microgroup-size 8 \
  --micro-batch-size 2 \
  --actor-num-gpus-per-node 4 \
  --sequence-parallel-size "${SP_SIZE}" \
  --ulysses-degree "${ULYSSES_DEGREE}" \
  --ring-degree "${RING_DEGREE}" \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 4 \
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
  --update-weight-target-module transformer,transformer_2 \
  --diffusion-reward pickscore:1.0 \
  --advantage-estimator grpo \
  --rm-type pickscore \
  --pickscore-num-workers 1 \
  --pickscore-num-gpus-per-worker 0 \
  --pickscore-batch-size 8 \
  --pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --pickscore-model-path yuvalkirstain/PickScore_v1 \
  --fsdp-master-dtype fp32 \
  --fsdp-reduce-dtype fp32 \
  --diffusion-forward-dtype bf16 \
  --diffusion-num-steps 10 \
  --diffusion-eval-num-steps 28 \
  --diffusion-output-num-frames "${NUM_FRAMES}" \
  --diffusion-guidance-scale 4.0 \
  --diffusion-guidance-scale-2 3.0 \
  --diffusion-noise-level 0.9 \
  --diffusion-height 480 \
  --diffusion-width 480 \
  --diffusion-flow-shift 3.0 \
  --diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_window \
  --diffusion-sde-window-size 1 \
  --diffusion-sde-candidate-steps 1,2,3 \
  --diffusion-debug-mode \
  --save "${ROOT_DIR}/logs/${RUN_NAME}/ckpt" \
  --save-interval 100 \
  --skip-eval-before-train \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${ROOT_DIR}/logs/${RUN_NAME}.log"
