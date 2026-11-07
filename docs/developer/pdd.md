# PDD training in Miles

This implementation adds Parallel Decoding Distillation (PDD) for Qwen-Image 2.1
to `train_diffusion.py`. SGLang rolls out the student, the FSDP actor trains it
against a frozen teacher, and the existing FSDP-to-engine weight sync closes the
loop. It reuses the Ray actors, FSDP2 loader, prepare/loss hooks, optimizer,
dataset cursor and distributed checkpoints.

## Supported configuration

- Qwen-Image 2.1 text-to-image, 1024 × 1024, CFG = 1.
- 32 output heads, four intervals per block: eight student forwards per image.
- Full student weights, FP32 parameters/reduction and BF16 autocast.
- Frozen BF16 teacher loaded from the same base model.
- FSDP with SP = 1, microbatch = 1, eight optimizer updates per rollout.
- SGLang engines with one GPU each, TP = SP = 1, FP32 DiT weights, deterministic
  8-step ODE rollout, one sample per request.
- No LoRA, EMA, RL log probabilities, reference-policy loss, dynamic batching,
  Cache-DiT or compiled/graph execution.

## How it works

1. SGLang generates each prompt's 8-block trajectory with fixed student weights and
   returns all nine states plus the text conditioning.
2. After data-parallel splitting, each trajectory expands into its eight blocks.
3. Update j recomputes the 32 heads from the rollout state at block j, selects one
   of its four intervals, rolls the student's earlier heads to it (detached),
   obtains the frozen teacher's midpoint target and steps once. Block j's state was
   produced j updates before the weights it trains.
4. After the eight updates the driver synchronizes the student back to SGLang.

The student output projection is replicated before FSDP wrapping and optimizer
creation; all heads start from the base model's output projection.

The `pdd_qwen_image21` rollout patch group is selected automatically. It replaces
the native DiT's output projection with the 32 heads, uses the trainer's
32-interval grid at block boundaries and returns the interval-weighted block-average
velocity, so the native ODE scheduler advances the latent and records the normal
trajectory response. The native Transformer forward, timestep embeddings, norms and
RoPE stay intact; native fused fast paths are disabled because they are verified
only for BF16 engine weights. Engine timesteps are chosen so that the native DiT's
BF16 time-embedding input equals the trainer's `bf16(sigma)`; when no engine
timestep reaches it, the trainer feeds the student the engine's embedded value, so
both sides always embed the same time. The trainer applies the existing Qwen-Image
2.1 RoPE alignment hook to the student and teacher.

## Train and resume

Validated on 2 × H200 with colocated training and rollout:

```bash
export PYTHONPATH=.
export MODEL=/path/to/qwen-image-2.1-snapshot
export PROMPTS=/path/to/train.jsonl
export OUTPUT=/scratch/pdd/run
python train_diffusion.py \
  --loss-type pdd --train-backend fsdp \
  --hf-checkpoint "$MODEL" --sglang-model-id Qwen-Image-2.1 \
  --diffusion-model-family qwen_image21 --diffusion-task t2i \
  --prompt-data "$PROMPTS" --input-key input \
  --rollout-batch-size 8 --n-samples-per-prompt 1 --num-steps-per-rollout 8 \
  --rollout-microgroup-size 1 --num-rollout 400 --micro-batch-size 1 --gradient-checkpointing \
  --fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 --diffusion-forward-dtype bf16 \
  --diffusion-height 1024 --diffusion-width 1024 \
  --diffusion-num-steps 8 --diffusion-eval-num-steps 8 \
  --diffusion-sde-type ode --diffusion-noise-level 0 \
  --diffusion-guidance-scale 1 --diffusion-true-cfg-scale 1 \
  --lr 5e-7 --weight-decay 0 --adam-beta2 0.999 --seed 42 \
  --sglang-dit-precision fp32 --use-miles-router --update-weight-buffer-size 1073741824 \
  --actor-num-gpus-per-node 2 --num-gpus-per-node 2 \
  --rollout-num-gpus 2 --rollout-num-gpus-per-engine 1 --colocate \
  --save "$OUTPUT/ckpt" --save-interval 100
```

Optionally add `--eval-prompt-data <name> <jsonl> --eval-interval 50` to render
fixed prompts with the current student in SGLang. The rollout's reward field is a
zero placeholder, **not** a quality metric.

To resume, rerun the command with `--load "$OUTPUT/ckpt"` and a larger
`--num-rollout`, keeping the model, data and training configuration unchanged.
Checkpoints contain model, optimizer, scheduler, per-rank Torch/CUDA RNG, protocol
signature and dataset cursor, and are written only at rollout boundaries. Resume
rejects missing state or a changed protocol/data/topology.

`train/pdd_next_latent_mae` and `train/pdd_update_relative_error` compare the
trainer's next block state with the one SGLang recorded. Block 0 measures pure
engine/trainer parity; later blocks also include rollout staleness.

## Verification and limits

```bash
python -m pytest tests/fast/distillation
python -m pytest tests/fast-gpu/backends/fsdp_utils/test_pdd.py
python -m pytest tests/fast-gpu/backends/sglang_diffusion_utils/test_pdd_sglang_loop.py
```

CPU tests cover detached teacher supervision, selected-head gradients, signed
nonuniform grids, eight block updates from rollout states against an independent
optimizer loop, the engine head reduction, trajectory validation, timestep matching
and unsupported configurations. The FSDP GPU test compares tiny-model updates with
an unsharded reference and verifies checkpoint restoration and the next update on
one and two ranks (two-rank case skips with fewer GPUs). The SGLang GPU smoke runs
real native stages, the response parser, one-rank FSDP, the PDD loss and the Miles
weight iterator into SGLang's weight receiver: one rollout, eight updates and a
sync, then checks every block of the next rollout against the trainer within 0.5%.
It does not exercise the HTTP/Ray driver, pretrained image quality or multi-GPU
placement. All GPU cases skip when CUDA is unavailable.

These are correctness checks, not image-quality claims. Automatic image metrics,
historical datasets/reports and experiment orchestration are outside this change.
