# PDD training in Miles

This implementation adds Parallel Decoding Distillation (PDD) to
`train_diffusion.py` for Qwen-Image 2.1 text-to-image and MiniMax H3 text-to-audio-video.
SGLang rolls out the student,
the FSDP actor trains it against a frozen teacher, and the existing FSDP-to-engine
weight sync closes the loop. It reuses
the Ray actors, FSDP2 loader, loss hub, optimizer, dataset cursor and distributed
checkpoints.

## Supported configuration

- 32 output heads per latent stream, four intervals per block: eight student
  forwards per sample. H3 trains a video and an audio head stack.
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

What differs per model family lives in `PDD_FAMILIES` (`miles/distillation/integration.py`):

| Family | Task | Streams | Grid | Engine weights | FSDP all-gather |
|---|---|---|---|---|---|
| `qwen_image21` | t2i, 1024 × 1024 | visual | the scheduler's shifted grid | FP32 | FP32, BF16 autocast |
| `h3` | t2va, short edge 768 | visual, audio | H3 shift 12 (video) / 3 (audio) | BF16, FP32 heads | the H3 FSDP plan (BF16, FP32 heads) |

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
branches:

| Hook | PDD default |
|---|---|
| `--custom-train-model-transform-path` | `install_student`: each stream's output layer as 32 heads, after LoRA |
| loss hub train input | `expand_train_pairs`: trajectory → eight block pairs |
| loss hub prepare / formula | `loss_hub/pdd.py`; the teacher runs in the actor through `PreparedBatch.teacher_solver` |
| `--custom-convert-samples-to-train-data-path` / `--custom-rm-path` | `miles/rollout/pdd_rollout.py` |
| `--rollout-patch-group` | `pdd_qwen_image21` / `pdd_h3` |
| `--fsdp-param-dtype` | `fp32` (Qwen-Image 2.1) / the family plan (H3) |

`validate_pdd_args` then rejects any configuration outside the validated scope.

### Rollout patch groups

Both groups keep the native Transformer forward, embeddings, norms and RoPE, and
replace only the output projections with the live heads that weight sync updates.

`pdd_qwen_image21` uses the trainer's 32-interval grid. Its timesteps are chosen so that the native
DiT's BF16 time-embedding input equals the trainer's `bf16(sigma)`; when none reaches
it, the trainer feeds the student the engine's embedded value. Native fused fast paths
are disabled because they are verified only for BF16 engine weights.

`pdd_h3` turns an 8-step request into the 32-interval request (block boundaries are
every fourth point of H3's shifted grid, as for a native 8-step request). The native
rollout collector records video target rows; the patch also records the audio rows
after each step and appends them, zero-padded to the video width, to the trajectory,
which the family's `split_states` cuts back apart. It also appends the engine's audio grid
to the recorded video grid, and the converter checks both against the trainer's grids. Native fused PDD heads
(`pdd_fused_heads`) are rejected: training needs the live, unfused heads.

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

### MiniMax H3 t2va on 8 × H200: four training GPUs and two rollout engines

Each H3 engine holds a 48 GB text encoder, the 62 GB DiT and the VAEs; colocated, the
sleeping engines alone need about 1 TB of host memory, so the engines stay resident on
their own GPUs instead. The H3 FSDP plan keeps the heads, input projections and timestep
MLP in FP32, which needs the Miles FSDP param-dtype patch for the installed torch version.

```bash
export MODEL=/path/to/MiniMax-H3-snapshot
python train_diffusion.py \
  --loss-type pdd --train-backend fsdp \
  --hf-checkpoint "$MODEL" --diffusion-model-family h3 --diffusion-task t2va \
  --prompt-data "$PROMPTS" --input-key input \
  --rollout-batch-size 8 --n-samples-per-prompt 1 \
  --rollout-microgroup-size 1 --num-rollout 400 --micro-batch-size 1 --gradient-checkpointing \
  --fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 --diffusion-forward-dtype bf16 \
  --diffusion-h3-aspect-ratio 1:1 --diffusion-h3-duration-seconds 4 \
  --diffusion-flow-shift 12 --diffusion-audio-flow-shift 3 \
  --diffusion-num-steps 8 --diffusion-eval-num-steps 8 \
  --diffusion-sde-type ode --diffusion-noise-level 0 --diffusion-guidance-scale 1 \
  --use-lora --lora-rank 64 --lora-alpha 128 --sglang-lora-merge-mode dynamic \
  --lr 5e-6 --weight-decay 0 --adam-beta2 0.999 --seed 42 \
  --sglang-dit-precision bf16 --sglang-tp-size 1 --sglang-model-variant fl2va \
  --use-miles-router --update-weight-buffer-size 2147483648 \
  --actor-num-gpus-per-node 4 --num-gpus-per-node 6 \
  --rollout-num-gpus 2 --rollout-num-gpus-per-engine 1 \
  --save "$OUTPUT/ckpt" --save-interval 50
```

To serve an H3 student with SGLang's native PDD path, export the LoRA adapter and the
32 heads from the checkpoint, fuse the heads with
`python -m sglang.multimodal_gen.tools.fuse_minimax_h3_pdd_heads <dir>`, and run 8 steps
with `--lora-path <adapter> --lora-merge-mode dynamic` and
`SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS=<dir>/pdd_fused_heads.safetensors`.

Optionally add `--eval-prompt-data <name> <jsonl> --eval-interval 50` to render
fixed prompts with the current student in SGLang. The rollout's reward field is a
zero placeholder, **not** a quality metric.

To resume, rerun the command with `--load "$OUTPUT/ckpt"` and a larger
`--num-rollout`, keeping the model, data and training configuration unchanged.
PDD uses the standard FSDP checkpoint (model, optimizer, scheduler, RNG, dataset
cursor), written at rollout boundaries; interval selection is a hash of seed,
rollout, lane and block, so it does not depend on RNG state.

`train/pdd_next_latent_mae` and `train/pdd_update_relative_error` compare the
trainer's next block state with the one SGLang recorded. They keep their Qwen names for
the video stream; H3 adds `train/pdd_audio_next_latent_mae` and
`train/pdd_audio_update_relative_error`, and logs each stream's share of `train/loss` as
`train/pdd_visual_loss` / `train/pdd_audio_loss`. The streams' per-element mean interval
losses are summed without weights; H3's audio loss is about 30× smaller than its video
loss, so the video stream dominates the gradient. Block 0 measures pure
engine/trainer parity; later blocks also include rollout staleness.

## Verification and limits

```bash
python -m pytest tests/fast/distillation/test_pdd.py
python -m pytest tests/fast-gpu/backends/sglang_diffusion_utils/test_pdd_sglang_loop.py
```

The CPU tests cover the interval math on signed nonuniform grids, the detached
midpoint teacher target, the engine's cached-head interval steps, timestep matching,
trajectory validation, the LoRA student's trainable and checkpointed parameters,
unsupported configurations, eight block updates from rollout states against an
independent optimizer loop (the test plays the actor's teacher solve), and H3's grids,
stream splitting, two head stacks and joint two-stream teacher call. The GPU smoke
runs a tiny Qwen-Image 2.1 through SGLang's native stages, the response parser,
one-rank FSDP with a LoRA student, the PDD loss and the Miles weight iterator into
SGLang's weight receiver: one rollout, eight updates and a sync, then checks every
interval of the next rollout against the trainer within 0.5%. It does not exercise the
HTTP/Ray driver, H3's native stages, pretrained quality or multi-GPU placement, and
skips without CUDA.

These are correctness checks, not quality claims. Automatic image/video metrics,
historical datasets/reports and experiment orchestration are outside this change.
