# PDD training in Miles

This implementation adds Parallel Decoding Distillation (PDD) to
`train_diffusion.py` for Qwen-Image 2.1 text-to-image. SGLang rolls out the student,
the FSDP actor trains it against a frozen teacher, and the existing FSDP-to-engine
weight sync closes the loop. It reuses
the Ray actors, FSDP2 loader, loss hub, optimizer, dataset cursor and distributed
checkpoints.

## Supported configuration

- 32 output heads per latent stream, four intervals per block: eight student
  forwards per sample.
- A LoRA student: adapters on the backbone, the new heads trained in full; FP32 master
  weights, BF16 compute; a frozen teacher loaded from the same base model.
- The engine applies the adapter unmerged (`--lora-ipc-weight-sync` when colocated,
  `--sglang-lora-merge-mode dynamic`): merging it into BF16 weights rounds most of a
  small adapter update away and drifts from the trainer's forward.
- FSDP with SP = 1, microbatch = 1, eight optimizer updates per rollout.
- SGLang engines with one GPU each, TP = SP = 1, deterministic 8-step ODE rollout,
  one sample per request, CFG = 1.
- No EMA, RL log probabilities, reference-policy loss, dynamic batching, Cache-DiT or
  compiled/graph execution.

| Family | Task | Streams | Grid | Engine weights | FSDP all-gather |
|---|---|---|---|---|---|
| `qwen_image21` | t2i, 1024 × 1024 | visual | the scheduler's shifted grid | FP32 | FP32, BF16 autocast |

## How it works

1. SGLang generates each prompt's trajectory with fixed student weights. The engine
   steps all 32 intervals: the first interval of each block runs the DiT and caches
   the 32 heads, the other three step on the cache, so the trajectory carries every
   interval's start state (33 states per stream) for eight DiT forwards.
2. After data-parallel splitting, each trajectory expands into its eight blocks.
3. Update j recomputes the heads from the rollout state at block j and samples one
   of its four intervals. The prepare hook asks the actor for the frozen teacher's
   midpoint velocity from the rollout's state at that interval, one joint teacher call
   per midpoint stage for all streams; the loss regresses the selected head of every
   stream on it and steps once. Block j's states were produced j updates before the
   weights it trains.
4. After the eight updates the driver synchronizes the student back to SGLang: the
   LoRA adapter, then the heads as plain weights.

The student's output layers are replicated into heads after LoRA is applied and before
FSDP wrapping and optimizer creation, so the heads train in full beside the adapters and
LoRA checkpoints keep them; all heads start from the base model's output layer.

`--loss-type pdd` registers its prepare/loss/train-input hooks in the loss hub
(`LOSS_HOOKS[("pdd", "actor")]`) and fills defaults for generic Miles hooks
(`set_pdd_defaults` in `miles/distillation/integration.py`); the trainer has no PDD
branches. What differs per model family lives in `PDD_FAMILIES`:

| Hook | PDD default |
|---|---|
| `--custom-train-model-transform-path` | `install_student`: each stream's output layer as 32 heads, after LoRA |
| loss hub train input | `expand_train_pairs`: trajectory → eight block pairs |
| loss hub prepare / formula | `loss_hub/pdd.py`; the teacher runs in the actor through `PreparedBatch.teacher_solver` |
| `--custom-convert-samples-to-train-data-path` / `--custom-rm-path` | `miles/rollout/pdd_rollout.py` |
| `--rollout-patch-group` | `pdd_qwen_image21` |
| `--fsdp-param-dtype` | `fp32` |

`validate_pdd_args` then rejects any configuration outside the validated scope.

### Rollout patch group

`pdd_qwen_image21` keeps the native Transformer forward, embeddings, norms and RoPE,
and replaces only the output projection with the live heads that weight sync updates.
It uses the trainer's 32-interval grid. Its timesteps are chosen so that the native
DiT's BF16 time-embedding input equals the trainer's `bf16(sigma)`; when none reaches
it, the trainer feeds the student the engine's embedded value. Native fused fast paths
are disabled because they are verified only for BF16 engine weights.

## Train and resume

### Qwen-Image 2.1, validated on 2 × H200 with colocated training and rollout

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
  --use-lora --lora-rank 64 --lora-alpha 128 --lora-ipc-weight-sync --sglang-lora-merge-mode dynamic \
  --lr 2e-5 --weight-decay 0 --adam-beta2 0.999 --seed 42 \
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
PDD uses the standard FSDP checkpoint (model, optimizer, scheduler, RNG, dataset
cursor), written at rollout boundaries; interval selection is a hash of seed,
rollout, lane and block, so it does not depend on RNG state.

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
optimizer loop (the test plays the actor's teacher solve), the engine's cached
interval steps, trajectory validation, timestep matching and unsupported configurations. The FSDP
GPU test compares tiny-model updates with an unsharded reference on one and two
ranks, loading the student through the model-transform hook (two-rank case skips
with fewer GPUs). The SGLang GPU smoke runs real Qwen-Image 2.1 native stages, the
response parser, one-rank FSDP, the PDD loss and the Miles weight iterator into
SGLang's weight receiver: one rollout, eight updates and a sync, then checks every
interval of the next rollout against the trainer within 0.5%. It does not exercise
the HTTP/Ray driver, pretrained quality or multi-GPU placement.
All GPU cases skip when CUDA is unavailable.

These are correctness checks, not quality claims. Automatic image/video metrics,
historical datasets/reports and experiment orchestration are outside this change.
