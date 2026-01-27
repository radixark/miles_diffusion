# nohup bash /data/zhiheng/miles/scripts/run-diffusion-grpo-ocr.sh > /data/zhiheng/miles/logs/diffusion_grpo_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# ps -ef | grep train.py | grep -v grep
# rollout needs 1 gpu for now, or there's going to be precision issue.
# parameter rollout-num-gpus and --rollout-num-gpus-per-engine  only makes sense in sglang diffusion case.
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
# Prepare OCR prompts into JSONL expected by Miles data loader.
python "${ROOT_DIR}/tools/prepare_ocr_jsonl.py"

# Minimal diffusion GRPO run, aligned with flow_grpo single-node settings.

#hf-checkpoint can be any text generation model from HuggingFace, used to generate initial prompts for diffusion model.
python "${ROOT_DIR}/train.py" \
  --train-backend fsdp \
  --diffusion-train \
  --rollout-function-path miles.rollout.diffusion_rollout.generate_rollout \
  --hf-checkpoint gpt2 \
  --prompt-data "${ROOT_DIR}/data/ocr/train.jsonl" \
  --input-key input \
  --rollout-batch-size 8 \
  --n-samples-per-prompt 16 \
  --num-rollout 100000 \
  --diffusion-train-batch-size 2 \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 0 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 5 \
  --colocate \
  --diffusion-model stabilityai/stable-diffusion-3.5-medium \
  --diffusion-dtype fp32 \
  --diffusion-num-steps 10 \
  --diffusion-guidance-scale 4.5 \
  --diffusion-noise-level 0.7 \
  --diffusion-height 512 \
  --diffusion-width 512 \
  --diffusion-reward pickscore \
  --sglang-disable-cuda-graph \
  --sglang-mem-fraction-static 0.7 \
  --sglang-cuda-graph-max-bs 16 \
  --global-batch-size 128
