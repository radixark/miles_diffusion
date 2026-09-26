# FSDP model roles and phase offload

The FSDP trainer owns one trainable actor, optional independent frozen reference
and teacher models, and an optional EMA optimizer. Each model contains the same
configured components; Wan2.2 can therefore have both `transformer` and
`transformer_2` in every role.

```text
train worker
  actor components -------- AdamW + scheduler
         |
         +----------------- EMAOptimizer.state[parameter]["ema"]
  reference components ---- frozen, eval, independent checkpoint
  teacher components ------ frozen, eval, independent checkpoint
```

Only actor parameters belong to AdamW. EMA stores averages for trainable actor
parameters in a separate `torch.optim.Optimizer`; frozen base parameters and all
buffers are excluded from averaging. There is no tensor snapshot manager.

## Model sources

Use an independent reference with:

```bash
--hf-checkpoint /models/student \
--ref-mode ref --ref-load /models/reference \
--teacher-load /models/teacher
```

All three sources use the same family backend, component names, input preparation,
and device mesh. This provides separate model instances, not cross-family
distillation or a new DMD2 loss. The existing loss receives the reference prediction.
Custom prepare/loss hooks can access `ctx.reference_models` and `ctx.teacher_models`
by component name. Teacher inference should run inside `torch.no_grad()`.

Each source may load a pretrained PEFT LoRA adapter:

```bash
--use-lora --lora-adapter-path /adapters/student --lora-rank 64 --lora-alpha 128 \
--ref-mode ref --ref-load /models/reference \
--ref-lora-adapter-path /adapters/reference \
--teacher-load /models/teacher \
--teacher-lora-adapter-path /adapters/teacher
```

Without a role's adapter path, that frozen role loads ordinary full weights, even
when the actor uses LoRA. Full weights and base-only weights use the same loader.
An adapter source is a PEFT checkpoint; for multiple components it contains a PEFT
checkpoint under each component subdirectory. `--lora-rank` and `--lora-alpha` must
match the actor adapter's `r` and `lora_alpha`. Actor adapter weights are loaded
before a training checkpoint is restored. Independent reference/teacher weights
are always reconstructed from their configured sources, so retain those sources
and their adapter checkpoints when resuming training.

`--ref-mode lora_base` keeps the existing low-memory alternative: temporarily
disable the actor adapter for a gradient-free base forward. It does not construct
another base model. The reference forward runs before the actor forward. FSDP is
resharded before the adapters are re-enabled, so PEFT restores adapter gradients on
the shards rather than on gathered copies.

## EMA and checkpointing

`--use-ema` creates the EMA optimizer. `--ref-mode ema` evaluates its weights.
EMA updates once after a rollout's training completes, retaining the existing
DiffusionNFT update cadence. Publishing or retrying a weight update never advances
EMA. EMA evaluation and publication swap the EMA values into the actor's shards and
swap them back afterward, dropping FSDP's gathered copies on both sides so no forward
reads stale weights; reference evaluation precedes the actor's autograd graph.

EMA state lives on the same device as its actor parameter. Phase sleep moves it to
pinned CPU memory together with AdamW state, and wake moves both back.

Rollout publication pushes the current actor weights unless
`--rollout-weights ema` asks for the EMA weights.

```text
iter_0000001/
  model/         actor model state (adapter-only for LoRA)
  optimizer/     AdamW state
  lr_scheduler/  scheduler state
  ema/           EMA optimizer state, decay schedule, and update count
```

EMA uses the existing distributed optimizer-state checkpoint wrapper, including
parameter names and DTensor sharding. `--no-save-optim` and `--no-load-optim` apply
to AdamW; EMA remains independently saved and restored. Loading a legacy checkpoint
without EMA seeds EMA from the loaded actor and logs that its averaging history
starts over. Disabling EMA simply omits this state from loading.

## Two offload lifecycles

| Setting | Parameter residence during training | Transfer boundary |
|---|---|---|
| No offload | GPU | None |
| `--offload-train` | GPU while awake | Training phase sleep/wake |
| `--fsdp-cpu-offload` | Actor shards on CPU | FSDP unshard |
| `--ref-cpu-offload` | Reference shards on CPU | FSDP unshard |
| `--teacher-cpu-offload` | Teacher shards on CPU | FSDP unshard |

Reference and teacher CPU offload are opt-in. Phase sleep covers actor, reference,
teacher, their buffers, AdamW state, and EMA state. Sleep clears completed-step
gradients instead of transferring them. Transfers use pinned CPU
storage and enqueue asynchronous copies before synchronization; Parameter objects
and optimizer bindings are preserved. Wake respects each model's native FSDP
offload policy: native-offloaded parameter shards stay on CPU while buffers return
to GPU.

Phase offload retains the model's own CPU tensors, without a second actor backup.
The helper lets FSDP rebuild its shard views and padding through `Module._apply`;
uneven shards remain pinned. This requires a small FSDP-internal adaptation of its
pinning flag and is covered by the registered CUDA test on supported PyTorch.

## Focused validation

CPU CI covers EMA averaging, DCP resume, model-source selection, loaded LoRA,
argument validation, and offload tensor identity. CUDA CI covers real two-rank
FSDP offload with uneven shards, reshard settings, frozen/trainable models, native
CPU offload, EMA residence, and checkpoint state.

```bash
python -m pytest tests/fast/backends/fsdp_utils/test_ema_optimizer.py \
  tests/fast/backends/fsdp_utils/test_reference_models.py \
  tests/fast/backends/fsdp_utils/test_offload.py \
  tests/fast/utils/test_reference_arguments.py
python -m pytest tests/fast-gpu/backends/fsdp_utils/test_offload.py \
  tests/fast-gpu/backends/fsdp_utils/test_reference_models.py
```
