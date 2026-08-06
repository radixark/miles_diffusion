#!/usr/bin/env bash
# 4-GPU Wan2.2-T2V-A14B dual-expert LoRA SFT on a (video, prompt) jsonl dataset.
# No sglang engines: the sft_rollout plugin lazily encodes each round's cache
# misses via a colocated encoder actor pool, writing one content-addressed file
# per sample into .sft_cache/ next to the jsonl. Epoch 2+ is all cache hits.
#
# Dataset rows: {"prompt": "...", "metadata": {"video": "/abs/path.mp4"}}
#
# Per rollout step: 64 samples, num_steps_per_rollout=4
#   -> 16 samples/optim step / 4 dp ranks = 4 samples/rank at mbs=1.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RUN_NAME="${RUN_NAME:-diffusion_sft_wan22_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${ROOT_DIR}/logs/${RUN_NAME}/ckpt"

SFT_DATA_JSONL="${SFT_DATA_JSONL:?set SFT_DATA_JSONL to a jsonl with prompt + metadata.video per line}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project miles-diffusion-sft
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --disable-wandb-random-suffix
  )
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

RESUME_ARGS=()
if [[ -n "${RESUME_CKPT:-}" ]]; then
  RESUME_ARGS+=(--load "${RESUME_CKPT}")
  [[ -n "${START_ROLLOUT:-}" ]] && RESUME_ARGS+=(--start-rollout-id "${START_ROLLOUT}")
fi

WAN_LORA_TARGET_MODULES=(
  attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0
  attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0
  ffn.net.0.proj ffn.net.2
)

"${PYTHON_BIN}" -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --loss-type sft_loss \
  --train-only \
  --rollout-function-path miles.rollout.sft_rollout.generate_rollout \
  --custom-convert-samples-to-train-data-path miles.rollout.sft_rollout.convert_samples_to_train_data \
  --custom-rollout-log-function-path miles.rollout.sft_rollout.log_rollout_data \
  --custom-prepare-train-batch-path miles.backends.fsdp_utils.loss_hub.sft.prepare_sft_batch \
  --custom-loss-function-path miles.backends.fsdp_utils.loss_hub.sft.sft_loss_formula \
  --hf-checkpoint Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --diffusion-model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --sft-encoder-checkpoint Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt-data "${SFT_DATA_JSONL}" \
  --input-key prompt \
  --diffusion-height 480 \
  --diffusion-width 832 \
  --diffusion-output-num-frames 81 \
  --sft-frame-stride 2 \
  --rollout-batch-size 64 \
  --num-epoch 3 \
  --num-steps-per-rollout 4 \
  --micro-batch-size 1 \
  --actor-num-gpus-per-node 4 \
  --num-gpus-per-node 4 \
  --use-lora \
  --lora-rank 64 \
  --lora-alpha 128 \
  --lora-target-modules "${WAN_LORA_TARGET_MODULES[@]}" \
  --diffusion-init-lora-weight gaussian \
  --lr 1e-4 \
  --adam-beta2 0.999 \
  --weight-decay 1e-4 \
  --update-weight-target-module transformer,transformer_2 \
  --fsdp-master-dtype fp32 \
  --fsdp-reduce-dtype fp32 \
  --diffusion-forward-dtype bf16 \
  --fsdp-flow-shift 3.0 \
  --save "${SAVE_DIR}" \
  --save-interval 20 \
  "${RESUME_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
