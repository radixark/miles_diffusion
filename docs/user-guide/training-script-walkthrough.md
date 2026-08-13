---
title: Training Script Walkthrough
description: How a miles-diffusion launch script is built — the argument groups, the batch-size arithmetic, and the GPU layout.
---
Every recipe in `scripts/` is a Python launcher. Each one assembles a flat string of flags for
`train_diffusion.py` and submits it as a Ray job. Read one launcher and you can read all of them;
this page walks the structure.

Reference recipe throughout: **`scripts/run_diffusion_grpo_pickscore_5gpu_flowgrpo_aligned.py`**
(Qwen-Image, PickScore, 4 train GPUs + 1 reward GPU).

## 1. The launcher shape

```python
@dataclass
class ScriptArgs(U.ExecuteTrainConfig):     # inherits output_dir, cuda_visible_devices, num_nodes, ...
    cuda_visible_devices: str = "4,5,6,7,1"
    num_rollout: int = 400
    extra_args: str = ""                     # your escape hatch — appended verbatim

def prepare(args) -> str:                    # download the dataset, return its local path
    ...

def execute(args, data_dir) -> None:
    ckpt_args      = "..."                   # a dozen named groups of flags
    rollout_args   = "..."
    ...
    U.execute_train(train_args=" ".join(groups), num_gpus_per_node=5, config=args)

@U.dataclass_cli
def main(args): execute(args, prepare(args))
```

Run it with no arguments for the tuned defaults, or override any `ScriptArgs` field from the CLI
(the `dataclass_cli` decorator wires the dataclass to Typer):

```bash
python3 scripts/run_diffusion_grpo_pickscore_5gpu_flowgrpo_aligned.py
python3 scripts/run_diffusion_grpo_pickscore_5gpu_flowgrpo_aligned.py \
    --num-rollout 50 --extra-args "--diffusion-kl-beta 0.02"
```

<Note>

Those Typer flags are also the CI interface. An e2e test names the recipe and passes its knobs:
`script="scripts/run_diffusion_grpo_sd3_ocr_sglang.py", args=["--num-rollout", "2", ...]`. The
older `run-*.sh` wrappers were removed — the `.py` launcher is the only entry point.

</Note>

`U.execute_train` (`miles/utils/external_utils/command_utils.py`) does four things:

1. Kills stale `sglang` / `ray` / `miles` processes.
2. `ray start --head` — unless `MILES_SCRIPT_EXTERNAL_RAY=1`, for when Slurm/k8s already built
   the cluster.
3. Builds a runtime env (NCCL socket vars, `MASTER_ADDR`, `CUDA_VISIBLE_DEVICES`, `PYTHONPATH`).
4. `ray job submit -- python3 train_diffusion.py <flags>`.

The job is *submitted*, not run directly, so the driver lives inside the cluster and sees every
node's GPUs.

## 2. The argument groups

The groups are a convention, not a framework feature — the parser sees one flat list. They are
grouped **by concern**, not by flag prefix, so a `--diffusion-*` flag can live in `rollout_args`
and a `--micro-batch-size-*` flag in `perf_args`. Look for the concern, not the prefix.

### `ckpt_args` — what to load and where to save

```bash
--hf-checkpoint Qwen/Qwen-Image
--save $OUT/ckpt --save-interval 10
```

`--hf-checkpoint` is **required**, and it is the single most important flag: one value serves
three readers that must not disagree.

1. The training side loads the pipeline components and scheduler from it.
2. The sglang-d rollout engine serves it.
3. The **model family** is matched from its name — `qwen-image` → the `qwen_image` config.

Only the architecture has to be right; weights are pushed from the trainer to the engine before
step 0 either way. If your checkpoint is a local directory whose name carries no family hint, name
the family explicitly with `--diffusion-model-family qwen_image`. Add `--load` + `--ckpt-step` to
resume.

### `rollout_args` — sampling: how many, and which steps get trained

This is the biggest group, and it covers both the batch shape and the sampler.

```bash
--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout
--prompt-data $DATA/train.jsonl --input-key input
--rollout-batch-size 32                # prompts per rollout
--n-samples-per-prompt 16              # GRPO group size
--num-rollout 400
--num-steps-per-rollout 2              # optimizer steps per rollout
--rollout-microgroup-size 8            # samples per HTTP request to sglang-d
--train-dp-split-mode stride           # how pairs are dealt to DP ranks
--diffusion-train-iter-order sample_major
--diffusion-num-steps 10               # denoising steps per sample
--diffusion-guidance-scale 4.0 --diffusion-true-cfg-scale 4.0
--diffusion-noise-level 1.2            # SDE noise injected during rollout
--diffusion-height 512 --diffusion-width 512
--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window
--diffusion-num-sde-steps 2            # how many steps become train pairs
--diffusion-sde-window-range 3,5
--rollout-patch-group sgld             # numeric-parity patches on the engine
```

`--rollout-microgroup-size` splits a prompt's `n_samples_per_prompt` group into concurrent
`POST /rollout/generate` calls, each asking sglang-d for that many outputs from one prompt. Larger
= fewer round trips and better engine batching; smaller = finer-grained reward overlap.

**The step strategy decides what training even sees.** A rollout runs `--diffusion-num-steps`
denoising steps, but only the steps the strategy selects are run as SDE and turned into train
pairs; the rest run ODE. Two ship in `miles/rollout/step_strategy_hub.py`:

| Strategy | Selection | Needs |
|---|---|---|
| `sde_window` | Random contiguous window per request, drawn inside `--diffusion-sde-window-range` | `--diffusion-num-sde-steps` |
| `epoch_global_random_choice` | Random subset drawn once per epoch, shared by every sample in it | `--diffusion-sde-candidate-steps` |

Write your own by pointing `--diffusion-step-strategy-path` at any
`(args, sample, num_steps, seed) -> (sde_indices, return_indices)` function.

### `eval_args`

```bash
--eval-prompt-data pickscore_test $DATA/test.jsonl   # <name> <path>
--eval-interval 30
--diffusion-eval-num-steps 50          # eval usually wants more steps than rollout
--skip-eval-before-train
```

Eval requests set `rollout=False`: pure generation, no trajectory, no log-probs.

### `grpo_args` — the objective

```bash
--advantage-estimator grpo
--globalize-reward-std                 # batch-wide std instead of per-group
--diffusion-clip-range 1e-4            # PPO clip; note how much tighter than LLM RL
```

`--globalize-reward-mean` and `--globalize-reward-std` are independent. flow_grpo's
`PerPromptStatTracker` uses a per-prompt mean with a global std, which is exactly
`--globalize-reward-std` alone.

### `optimizer_args`

```bash
--lr 3e-4 --adam-beta2 0.999 --weight-decay 1e-4
```

Only AdamW is implemented. `--adam-beta2` defaults to `0.999` (not the LLM-side `0.95`) so a
forgotten flag doesn't silently break flow_grpo comparability.

### `lora_args`

```bash
--use-lora --lora-rank 64 --lora-alpha 128 --lora-init-weights gaussian
--lora-ipc-weight-sync
```

Target modules default per model family (`TrainPipelineConfig.lora_target_modules`); override with
`--lora-target-modules`. `--lora-ipc-weight-sync` pushes only `lora_A`/`lora_B` to the engine and
lets it merge locally — far less traffic than shipping merged weights.

### `reward_args`

```bash
--rm-type pickscore
--pickscore-num-workers 1 --pickscore-num-gpus-per-worker 1.0 --pickscore-batch-size 8
--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K
--pickscore-model-path yuvalkirstain/PickScore_v1
```

Built-ins are `pickscore` and `ocr`. Reward actors are separate Ray actors; give them their own
GPU (as here) or squeeze them onto the rollout GPUs with `--colocate-reward`.

### `sglang_args` — the rollout engine

```bash
--use-miles-router                     # required: miles-diffusion only supports the miles router
--sglang-server-concurrency 4
--sglang-attention-backend torch_sdpa
--update-weight-buffer-size 2147483648
```

sglang-d's own CLI arguments are auto-registered with a `--sglang-` prefix (a short skip list
covers the ones miles sets itself: `model_path`, ports, `base_gpu_id`, …). At engine launch, every
`ServerArgs` field with a matching `--sglang-<field>` is forwarded by name, and
`--sglang-dit-precision` / `--sglang-vae-slicing` are forwarded separately as `PipelineConfig`
overrides — only when you changed them, so a subclass default is never clobbered.

When you set them explicitly, `--sglang-tp-size` × `--sglang-sp-degree` must equal
`--rollout-num-gpus-per-engine`; that product is validated at parse time.

### `train_backend_args` — precision and which modules train

```bash
--train-backend fsdp
--fsdp-master-dtype fp32               # sharded master weights + optimizer state
--fsdp-reduce-dtype fp32               # gradient reduce-scatter
--diffusion-forward-dtype bf16         # DiT forward, both sides
```

`--diffusion-forward-dtype` is applied in three places at once: the sglang-d engine, FSDP's
`param_dtype`, and the training-side input cast. That triple use is the point — see
[Dtype Control](/advanced/dtype-control).

Dual-expert models add `--update-weight-target-module transformer,transformer_2` here: it names
which pipeline modules are trained *and* pushed back to the engine.

### `perf_args` — throughput and micro-batch tiling

```bash
--gradient-checkpointing
--micro-batch-size-sample 8 --micro-batch-size-tstep 1
```

This is where the DiT-forward tiling lives. See
[the arithmetic](#3-the-batch-size-arithmetic) — the choice between `--micro-batch-size` and the
sample × timestep pair is the main decision here. `--rollout-parser-num-workers` also belongs to
this group when the response tensors are large enough to bottleneck deserialization.

### `misc_args` — the GPU layout and run-level switches

```bash
--actor-num-gpus-per-node 4            # FSDP DP=4
--rollout-num-gpus 4                   # 4 sglang-d engines...
--rollout-num-gpus-per-engine 1        # ...one GPU each
--num-gpus-per-node 5                  # 5 GPUs total on this node
--colocate
--deterministic-mode
```

Read it together with `cuda_visible_devices = "4,5,6,7,1"`: physical GPUs 4-7 hold both the
trainer and the engines (time-multiplexed by `--colocate`), and GPU 1 is left for the PickScore
actor. Under `--colocate`, `--offload-train` and `--offload-rollout` are forced on, so the
trainer sleeps to CPU while rollout runs and vice versa.

## 3. The batch-size arithmetic

This is the part that bites. Diffusion adds a dimension LLM RL does not have: one sample expands
into **several train pairs**, one per trained denoising step.

```
samples per rollout   = rollout_batch_size × n_samples_per_prompt
train pairs           = samples × (number of SDE step indices)
```

The sample-level knobs are locked by one identity, checked at parse time:

```
global_batch_size = rollout_batch_size × n_samples_per_prompt ÷ num_steps_per_rollout
```

Pass **one** of `--global-batch-size` / `--num-steps-per-rollout`; passing both with
inconsistent values is a hard error. `global_batch_size` counts samples and must divide evenly by
`dp_size` (= train world size ÷ `--sequence-parallel-size`).

### Cutting the pairs into forwards

Below the sample level, two ways to decide what shares one DiT forward:

**Flat** — `--micro-batch-size N` cuts the flat pair list into contiguous chunks of N.

**2D tiling** — `--micro-batch-size-sample S` and `--micro-batch-size-tstep T`, set together,
group pairs into S samples × T timesteps per forward. More expressive than the flat knob, which
can only cut the list: this controls *how many samples* and *how many timesteps* share a forward
independently. `--diffusion-train-iter-order` (`sample_major` / `timestep_major`) then sets the
order those tiles are visited. Setting the pair overrides `--micro-batch-size` with `S × T`.

Most shipped recipes use the 2D form (Qwen-Image `8 × 1`, SD3.5 `16 × 5`, LTX-2.3 `1 × 1`),
because the natural grouping is "these many samples' SDE steps in one forward", which the flat
knob cannot express when the window length and the desired sample count differ. Wan2.2 is the
exception: its binding constraint is that a forward must not straddle the high/low-noise expert
boundary, and `--micro-batch-size 2` says that directly.

| Knob | Unit | Meaning |
|---|---|---|
| `--num-steps-per-rollout` | optimizer steps | Windows the pairs are cut into per rollout |
| `--micro-batch-size` | train pairs | Flat: pairs per DiT forward |
| `--micro-batch-size-sample` × `--micro-batch-size-tstep` | samples × timesteps | 2D tile per DiT forward |
| `--diffusion-train-iter-order` | — | Tile visit order (2D only) |
| `--train-dp-split-mode` | — | `contiguous` or `stride` dealing of pairs to DP ranks |

Worked example, the reference recipe:

```
32 prompts × 16 samples             = 512 samples per rollout
512 samples × 2 SDE steps           = 1024 train pairs
1024 pairs ÷ 4 DP ranks             = 256 pairs per rank
256 ÷ num_steps_per_rollout(2)      = 128 pairs per optimizer step per rank
tile = 8 samples × 1 timestep       = 8 pairs per DiT forward → 16 forwards per step
```

Two failure modes to recognize:

- `num_pairs_shard=N not divisible by num_steps_per_rollout=M` — the per-rank pair count doesn't
  split evenly into windows. Adjust `rollout_batch_size`, `n_samples_per_prompt`, or
  `num_steps_per_rollout`.
- `Micro-batch mixes denoising phases` — Wan2.2 only. A micro-batch spans the high/low-noise
  expert boundary; set `--micro-batch-size 1` (or a tiling that keeps each forward phase-pure).

## 4. What one rollout iteration does

`train_diffusion.py` is 90 lines; the whole loop is visible there.

```
generate  → RolloutManager fans prompts out to sglang-d engines
             each request returns images + the DiT trajectory + rollout log-probs
reward    → reward actors score each microgroup as soon as it lands
convert   → rewards normalised into advantages, samples expanded into train pairs,
             pairs split into per-DP-rank shards
train     → FSDP actor: prepare → DiT forward → SDE log-prob → PPO-clip → step
save      → every --save-interval
sync      → trainer pushes updated weights back into the engines
eval      → every --eval-interval
```

Under `--colocate` the offload/onload calls around `generate` and `train` are what make the two
halves fit on the same GPUs.

## 5. Adapting a recipe

Start from the closest existing script and change one group at a time.

| Goal | What to change |
|---|---|
| Different reward | `reward_args` — swap `--rm-type`, or point `--custom-rm-path` at your own `async def rm(args, samples) -> list[float]` |
| Different prompts | `--prompt-data`, `--input-key`, and the `prepare()` download |
| Different checkpoint | `--hf-checkpoint`, plus `--diffusion-model-family` if the name carries no family hint |
| Smaller GPU budget | Lower `--rollout-batch-size`, `--n-samples-per-prompt`, `--rollout-microgroup-size`; keep the batch identity satisfied |
| Train more denoising steps | Raise `--diffusion-num-sde-steps` — remember this multiplies train pairs |
| Try a flag without editing the file | `--extra-args "--flag value"` |

For a one-off experiment, `--extra-args` is appended last and therefore wins on argparse
duplicates.

## Next

- [CLI Reference](/user-guide/cli-reference) — every flag, grouped.
- [Dtype Control](/advanced/dtype-control) — what the three dtype flags actually do.
- [Deterministic Training](/advanced/deterministic) — what `--deterministic-mode` covers.
