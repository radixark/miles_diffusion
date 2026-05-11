# Batch sizes in miles-diffusion

A complete catalog of every batch-shape parameter in miles-diffusion, organized by where it lives (CLI flags → derived in `arguments.py` → in-memory in `actor.py` / rollout). Each entry lists its **unit** (prompt / sample / denoising-step / DP rank) and the relationships to other entries.

Anchor commit: miles `diffusion_RL_v0.1`.

---

## At a glance

```
                       ── CLI level ──
   ┌──────────────────────────────────────────────────────┐
   │ --actor-num-nodes × --actor-num-gpus-per-node = dp_size                      │
   │ --rollout-batch-size           (prompts per rollout)                         │
   │ --n-samples-per-prompt         (samples per prompt)                          │
   │ --num-steps-per-rollout        (optimizer steps per rollout)                 │
   │ --global-batch-size            (samples per grad step, total across DP)      │
   │ --micro-batch-size-sample      (DiT forward sample-axis tile)                │
   │ --micro-batch-size-tstep       (DiT forward tstep-axis tile)                 │
   │ --diffusion-microgroup-size    (rollout-side sub-batch per sgl-d request)    │
   │ --over-sampling-batch-size     (prompt-axis sampling granularity in rollout) │
   └──────────────────────────────────────────────────────┘
                              │
                              ▼
                    ── arguments.py derivation ──
   ┌──────────────────────────────────────────────────────┐
   │ samples_per_rollout    = rollout_batch_size × n_samples_per_prompt           │
   │ args.global_batch_size = samples_per_rollout ÷ num_steps_per_rollout         │
   └──────────────────────────────────────────────────────┘
                              │
                              ▼
                    ── actor.py runtime ──
   ┌──────────────────────────────────────────────────────┐
   │ num_rollout_samples         = len(denoising_envs)    (== global_batch_size ÷ dp_size × num_optim_steps_per_rollout)
   │ num_optim_steps_per_rollout = args.num_steps_per_rollout
   │ num_samples_per_optim_step  = num_rollout_samples ÷ num_optim_steps_per_rollout
   │ num_samples_in_window       = traj_end − traj_start  (actual; ≤ nominal at tail)
   │ sample_microbatch           = clamped(--micro-batch-size-sample, num_samples_in_window)
   │ tstep_microbatch            = clamped(--micro-batch-size-tstep, sde_window_size)
   │ tile_sample_count           = sample_indices.numel()  (≤ sample_microbatch)
   │ tile_tstep_count            = tstep_indices.numel()   (≤ tstep_microbatch)
   └──────────────────────────────────────────────────────┘
```

---

## CLI flags (user-facing knobs)

All defined in `miles/utils/arguments.py`.

| Flag | Type | Default | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `--rollout-batch-size` | int | **required** | prompt | Prompts handled by one rollout. |
| `--n-samples-per-prompt` | int | 1 | sample/prompt | Trajectories sampled per prompt (group size for GRPO advantage normalization). |
| `--num-steps-per-rollout` | int | None | (count) | How many optimizer `.step()` calls share one rollout. When set, drives the derivation of `global_batch_size`. |
| `--global-batch-size` | int | None | sample | Samples per grad step, summed across all DP ranks. Auto-derived from the above when `--num-steps-per-rollout` is set; setting both raises if they disagree. |
| `--micro-batch-size-sample` | int | None | sample | DiT forward's sample-axis tile size. `None` ⇒ use the full window (`num_samples_in_window`). |
| `--micro-batch-size-tstep` | int | 1 | denoising-step | DiT forward's denoising-step axis tile size. |
| `--diffusion-microgroup-size` | int | 1 | sample | Rollout-side: sub-batch of samples (per prompt) packed into one sgl-d HTTP `/rollout/generate` request. Independent of training-side micro-batches. |
| `--over-sampling-batch-size` | int | None | prompt | Granularity of one rollout sampling cycle (used by partial-rollout / filtered-rollout paths). Defaults to `--rollout-batch-size`. |
| `--actor-num-nodes` | int | 1 | (count) | Number of training nodes. |
| `--actor-num-gpus-per-node` | int | 8 | (count) | Number of training GPUs per node. Product with `--actor-num-nodes` gives `dp_size`. |
| `--diffusion-train-iter-order` | enum | `"sample_major"` | — | Outer-loop axis when iterating tiles (`sample_major` / `timestep_major`). Doesn't change batch sizes, only iteration order. |

(Tstep-axis fields like `--diffusion-num-steps`, `--diffusion-sde-window-size`, `--diffusion-sde-window-range` describe rollout step counts, not batch sizes — see `docs/developer_guide/adding_a_new_diffusion_model.md` for those.)

---

## Derived in `arguments.py`

These are not user-set; they are computed inside `arguments.py` after CLI parse.

| Field | Computed as | Unit | Notes |
| --- | --- | --- | --- |
| `args.global_batch_size` | `rollout_batch_size × n_samples_per_prompt ÷ num_steps_per_rollout` (when `--num-steps-per-rollout` is set; else user-supplied; else falls back to `dp_size`) | sample | Total samples per grad step across all DP ranks. Validated to be divisible by `dp_size`. |
| `args.train_iters` (`lr_scheduler.py`) | `args.num_rollout × args.rollout_batch_size × args.n_samples_per_prompt ÷ args.global_batch_size`, equivalent to `num_rollout × num_steps_per_rollout` | (count) | Total optimizer steps the LR scheduler should plan for. |

---

## In-memory variables in `actor.py`

Live only on the training side, computed at the entry of `train()` and propagated through `_build_train_grids` → `_run_optim_window` → `_forward_tile`.

| Variable | Computed as | Unit | Equivalent CLI / derived |
| --- | --- | --- | --- |
| `num_rollout_samples` | `len(denoising_envs)` | sample (per-DP) | `args.global_batch_size ÷ dp_size × num_optim_steps_per_rollout` |
| `num_optim_steps_per_rollout` | `args.num_steps_per_rollout` | (count) | `--num-steps-per-rollout` |
| `num_samples_per_optim_step` | `num_rollout_samples ÷ num_optim_steps_per_rollout` | sample (per-DP) | `args.global_batch_size ÷ dp_size` |
| `num_samples_in_window` | `traj_end − traj_start`, returned as `grids["num_samples_in_window"]` | sample | Equals `num_samples_per_optim_step` for full windows; smaller for the trailing window when `num_rollout_samples` is not divisible by `num_optim_steps_per_rollout`. |
| `sample_microbatch` | `min(args.micro_batch_size_sample if not None else num_samples_in_window, num_samples_in_window)` | sample | `--micro-batch-size-sample` clamped to actual window. |
| `tstep_microbatch` | `min(args.micro_batch_size_tstep if not None else 1, sde_window_size)` | denoising-step | `--micro-batch-size-tstep` clamped to actual SDE window. |
| `tile_sample_count` | `sample_indices.numel()` | sample | Actual samples in one DiT forward tile (≤ `sample_microbatch`). |
| `tile_tstep_count` | `tstep_indices.numel()` | denoising-step | Actual denoising steps in one DiT forward tile (≤ `tstep_microbatch`). |
| `sde_window_size` | `grids["sde_window_size"]`, ultimately set by the rollout step strategy | denoising-step | Length of the active SDE window inside one trajectory. Per-sample but assumed equal across the batch. |
| `num_rollout_steps` | `dit_trajectories[0].timesteps.shape[0]` | denoising-step | Total denoising steps the rollout actually ran (= `--diffusion-num-steps`). Different from `num_train_timesteps` (the scheduler config constant, e.g. 1000). |

---

## Worked example: OCR FlowGRPO 4 GPU

From `scripts/run-diffusion-grpo-ocr-fg-aligned.sh`:

```bash
--rollout-batch-size 32          # 32 prompts per rollout
--n-samples-per-prompt 16         # 16 trajectories per prompt
--num-steps-per-rollout 2         # 2 optim steps per rollout
--micro-batch-size-sample 4
--micro-batch-size-tstep 2
--diffusion-microgroup-size 16
--actor-num-gpus-per-node 4       # dp_size = 4
```

Derived in `arguments.py`:

```
samples_per_rollout    = 32 × 16  = 512
args.global_batch_size = 512 ÷ 2  = 256          (samples per grad step, all DP)
args.train_iters       = num_rollout × 2
```

In `actor.py` per-rollout (per DP rank):

```
num_rollout_samples         = 128                     (= 512 ÷ dp_size = 4)
                              # = 64 × 2 = (global_batch_size ÷ dp_size) × num_optim_steps_per_rollout
num_optim_steps_per_rollout = 2
num_samples_per_optim_step  = 128 / 2 = 64            (= global_batch_size ÷ dp_size)

# inside the optim loop
num_samples_in_window       = 64                       (full window, 128 % 2 == 0)
sample_microbatch           = min(4, 64) = 4
tstep_microbatch            = min(2, sde_window_size = 2) = 2

# inside one DiT forward tile
tile_sample_count = 4
tile_tstep_count  = 2
batch dim = 4 × 2 = 8 samples flattened into one DiT call
```

Total tiles per optim step = `(num_samples_in_window / sample_microbatch) × (sde_window_size / tstep_microbatch)` = `(64/4) × (2/2)` = 16 tiles. Each tile contributes `loss / 16` to the gradient.

---

## Cross-reference: name vs scope

This table answers "I see `X` in some file — what unit is it in?":

| Name appears as | Unit | Scope |
| --- | --- | --- |
| `rollout_batch_size` | prompt | one rollout, all DP |
| `n_samples_per_prompt` | sample/prompt | per prompt |
| `global_batch_size` / `args.global_batch_size` | sample | one grad step, all DP |
| `num_rollout_samples` | sample | one rollout, per DP |
| `num_samples_per_optim_step` | sample | one grad step, per DP |
| `num_samples_in_window` | sample | one optim step's actual window, per DP |
| `sample_microbatch` / `tile_sample_count` | sample | one DiT forward, per DP |
| `tstep_microbatch` / `tile_tstep_count` | denoising-step | one DiT forward |
| `sde_window_size` | denoising-step | one trajectory |
| `num_rollout_steps` / `--diffusion-num-steps` | denoising-step | one trajectory |
| `num_train_timesteps` (scheduler config) | denoising-step | model constant (typically 1000) |
| `dp_size` | (count) | global |

---

## Reference: the `actor.py` call chain

```
train()                                   # entry per rollout
├── num_rollout_samples         = len(denoising_envs)
├── num_optim_steps_per_rollout = args.num_steps_per_rollout
├── num_samples_per_optim_step  = num_rollout_samples // num_optim_steps_per_rollout
└── for step_id in range(num_optim_steps_per_rollout):
        traj_start = step_id * num_samples_per_optim_step
        traj_end   = min(num_rollout_samples,
                         traj_start + num_samples_per_optim_step)
        grids = _build_train_grids(traj_start, traj_end, ...)
        # grids["num_samples_in_window"] == traj_end - traj_start
        num_samples_in_window = grids["num_samples_in_window"]
        sample_microbatch = min(args.micro_batch_size_sample or num_samples_in_window,
                                num_samples_in_window)
        tstep_microbatch  = min(args.micro_batch_size_tstep or 1,
                                grids["sde_window_size"])
        _run_optim_window(grids, sample_microbatch, tstep_microbatch, ...)

_run_optim_window(grids, ...)
├── num_samples_in_window = grids["num_samples_in_window"]
├── sde_window_size       = grids["sde_window_size"]
├── sample_chunks = _chunked_indices(num_samples_in_window, sample_microbatch, device)
├── tstep_chunks  = _chunked_indices(sde_window_size,        tstep_microbatch,  device)
└── for outer in outer_chunks:
        for inner in inner_chunks:
            _forward_tile(sample_indices, tstep_indices, grids, ...)

_forward_tile(sample_indices, tstep_indices, grids, ...)
├── tile_sample_count     = sample_indices.numel()
├── tile_tstep_count      = tstep_indices.numel()
├── num_samples_in_window = grids["num_samples_in_window"]   # used only for CFG indexing
└── DiT forward → PPO loss (mean over tile_sample_count × tile_tstep_count cells)
```
