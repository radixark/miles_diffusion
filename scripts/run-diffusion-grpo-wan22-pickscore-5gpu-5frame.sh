#!/usr/bin/env bash
# Wan2.2-T2V-A14B 5-FRAME PickScore GRPO recipe: 4-GPU train+rollout colocate
# + 1-GPU pickscore reward (5 GPUs total, "4+1").
#
# Differs from run-diffusion-grpo-wan22-pickscore-4gpu.sh (the 1-frame recipe)
# in three ways:
#   1. Trains 5-frame video clips (--diffusion-output-num-frames 5). PickScore
#      already mean-pools over all generated frames (rm_hub/pickscore.py), and
#      wandb now logs the full clip as mp4 instead of frame 0 (ray/rollout.py
#      _log_images), so reward and media both reflect all 5 frames.
#   2. Trains BOTH Wan2.2 experts. The SDE schedule (wan_ff_global_window)
#      draws ONE step per rollout, shared across the batch, from the candidate
#      set --diffusion-sde-candidate-steps 1,2,3 (via torch.randperm seeded by
#      epoch + rollout_seed), exactly mirroring Flow-Factory's
#      FlowMatchEulerDiscreteSDEScheduler with num_sde_steps=1, sde_steps=[1,2,3].
#      With num_steps=10 and --diffusion-flow-shift 3.0 the boundary is t=875:
#      idx 1,2 (t>=875) -> "transformer" (high-noise expert), idx 3 (t<875) ->
#      "transformer_2" (low-noise expert). So over rollouts both DiTs get
#      gradient stochastically (~2/3 high, ~1/3 low), and
#      --update-weight-target-module transformer,transformer_2 syncs both.
#      NOTE: this replaced wan_dual_2high_1low, which trained ALL of [1,2,3]
#      deterministically EVERY rollout (= 3x the sample*step training pairs of
#      FF's 1-step draw). That tripled per-rollout gradient coverage, fitting
#      ~3x faster but overfitting the 3 fixed trajectory points and saturating
#      early at a lower reward ceiling. wan_ff_global_window restores FF's
#      slower-but-higher trajectory.
#
#   3. flow_shift=3.0 (NOT sgl-d's hardcoded 12.0). 3.0 is the diffusers
#      Wan2.2-T2V-A14B scheduler_config default that FF inherits, and the
#      historical 480p Wan shift. --diffusion-flow-shift sends a per-request
#      sigma schedule that composes out sgl-d's built-in 12.0 so the rollout
#      denoises at an effective shift of 3.0 (verified: matches a plain shift=3
#      Euler schedule to <1.5e-4). The step strategy reads the same flow_shift,
#      so step selection and rollout stay consistent. Without this the boundary
#      sits at idx 5/6 and the SDE steps would be [1,2,6] on a shift=12 traj.
#
# Other knobs mirror the 1-frame recipe (Flow-Factory Wan2.2 LoRA GRPO):
#   pretrained = Wan-AI/Wan2.2-T2V-A14B-Diffusers, resolution=480, num_steps=10,
#   guidance=4.0 (high) / 3.0 (low), Flow-SDE noise_level=0.9,
#   LoRA r=64/alpha=128 (self-attn + cross-attn + FFN),
#   lr=1e-4, adam_beta2=0.999, weight_decay=1e-4, clip_range=1e-4.
#
# NOTE: --micro-batch-size can be >1 with wan_ff_global_window. That strategy
# draws ONE SDE step per rollout, shared across the whole batch, so every train
# pair in a rollout has the same timestep -> same phase -> same DiT + same CFG
# scale. actor.py only refuses a micro-batch that MIXES phases (one forward = one
# DiT, one CFG); a phase-pure batch of any size is fine, so we batch pairs for
# GPU efficiency. micro_batch_size is a pure systems knob here: the loss is
# normalized by total pairs per optim step (loss_sum / num_local_pairs) and
# grad-accumulated, so the update is identical regardless of batch size. Set to 2:
# mbs=4 OOMs on H200 (~134/140 GiB) at 5-frame with grad-ckpt OFF; 2 fits.
# CAVEAT: only safe while the strategy is single-phase-per-rollout. A multi-step /
# cross-phase strategy (e.g. wan_dual_2high_1low) would mix phases -> requires
# --micro-batch-size 1 again. The deprecated --micro-batch-size-sample/-tstep 2D
# grouping (PR #10) cannot express phase routing; use the flat --micro-batch-size.
#
# NOTE: gradient checkpointing is intentionally OFF. With it on, Wan2.2 dual-expert
# training hits torch.utils.checkpoint CheckpointError ("recomputed values have
# different metadata") in the backward recompute. Flow-Factory's lora/wan22 recipe
# also runs enable_gradient_checkpointing=false. If you OOM without it, lower
# --rollout-batch-size, --n-samples-per-prompt, or --diffusion-microgroup-size.
#
# NOTE: 5 frames + 3 trainable SDE steps cost ~5x latent memory and ~3x the
# train steps vs the 1-frame recipe.
#
# Layout: first 4 GPUs in CUDA_VISIBLE_DEVICES = train+sgld colocate,
# the 5th GPU = pickscore reward worker. Default GPUs 0-4.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
RUN_NAME="diffusion_grpo_wan22_pickscore_5gpu_5frame_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${ROOT_DIR}/logs/${RUN_NAME}/ckpt"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project miles-diffusion-grpo
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --diffusion-log-images 8
    --diffusion-log-image-interval 10
    --disable-wandb-random-suffix
  )
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

DATASETS_DIR="/root/datasets/miles-diffusion-datasets"
# Idempotent download. Invoke the hf CLI via ${PYTHON_BIN} rather than the `hf`
# console script, whose shebang points at the env's original build path
# (/cluster-storage/.../miniforge3) and fails with "required file not found"
# when the env is mounted elsewhere (/personal/miniforge3). Skips if present.
if [[ ! -f "${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl" ]]; then
  "${PYTHON_BIN}" -c "import sys; from huggingface_hub.cli.hf import main; sys.argv=['hf','download','--repo-type','dataset','rockdu/miles-diffusion-datasets','--include','flowgrpo_pickscore/**','--local-dir','${DATASETS_DIR}']; sys.exit(main())"
fi

# Wan2.2 DiT LoRA targets: self-attn (attn1), cross-attn (attn2), and FFN.
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
  --rollout-batch-size 48 \
  --n-samples-per-prompt 16 \
  --num-rollout 10000 \
  --num-steps-per-rollout 2 \
  --diffusion-microgroup-size 8 \
  --micro-batch-size 2 \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 5 \
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
  --pickscore-num-gpus-per-worker 1.0 \
  --pickscore-batch-size 8 \
  --pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --pickscore-model-path yuvalkirstain/PickScore_v1 \
  --fsdp-master-dtype fp32 \
  --fsdp-reduce-dtype fp32 \
  --diffusion-forward-dtype bf16 \
  --diffusion-num-steps 10 \
  --diffusion-eval-num-steps 28 \
  --diffusion-output-num-frames 5 \
  --diffusion-guidance-scale 4.0 \
  --diffusion-guidance-scale-2 3.0 \
  --diffusion-noise-level 0.9 \
  --diffusion-height 480 \
  --diffusion-width 480 \
  --diffusion-flow-shift 3.0 \
  --diffusion-step-strategy-path miles.rollout.step_strategy_hub.wan_ff_global_window \
  --diffusion-sde-window-size 1 \
  --diffusion-sde-candidate-steps 1,2,3 \
  --diffusion-debug-mode \
  --save "${SAVE_DIR}" \
  --save-interval 10 \
  --eval-prompt-data pickscore_test "${DATASETS_DIR}/flowgrpo_pickscore/test.jsonl" \
  --eval-interval 30 \
  --skip-eval-before-train \
  "${WANDB_ARGS[@]}"
