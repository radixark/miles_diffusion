# Dense FA3 training and pure Ulysses validation

The operator regression targets `DiffusersModelBackend`, the `_flash_3` backend, and
Miles pure Ulysses (`ring_degree=1`) on H100. It exercises real FA3 without
downloading model weights. Passing it establishes attention-level correctness
on the tested build/shapes, not full-model, FSDP2, or rollout/train equivalence.
The separate tiny Wan gate below tests the production FSDP2/model integration;
both its native CPU control and two-H100 FA3/SP2 training run passed.

## Why the adapter exists

At the Diffusers pin in `requirements.txt`
(`f53d552036a0d1bd5570782a39cd40cfabf112bc`), `_flash_attention_3` filters out
`deterministic` and calls a forward-only custom op. That op passes an explicit
`deterministic=False`, overriding a `functools.partial(..., deterministic=True)`
default. An inner autograd-capable function does not provide an autograd formula
for the outer custom op.

`miles/backends/fsdp_utils/fa3_attention.py` registers an adapter that calls the
public FA3 autograd function directly, with `num_splits=1`. The registry decorator
also refreshes the cached accepted argument names. Both deterministic-on and
deterministic-off explicit `_flash_3` selection use the adapter. Forced actor mode
or PyTorch deterministic mode takes precedence over a per-call false value.
With deterministic mode enabled and the CLI backend unset, the model backend
resolves Diffusers' active backend, including environment-based selection.

Only the selected kernel can satisfy driver validation: installed FA2 cannot
validate FA3. Hub and varlen FA3 paths have no integration here. Native model
spellings such as LTX `flash_attention_3` retain their separate package hook.

The registry is process-wide. Install before execution/compilation; separate
processes are required for models needing different forced modes. Masks,
nonzero dropout, split counts other than one, and Diffusers context-parallel
configurations are rejected. Miles Ulysses moves tensors before the local call
and clears that configuration. Ring remains unsupported. Optional LSE has
Diffusers' `[B,S,H]` layout and is detached: upstream FA3 does not differentiate it.

## Environment and preflight

Use the project's complete GPU image, with its matching PyTorch/CUDA/FA3 wheel.
The GPU regression requires SM90 or newer and actual support in the installed
FA3 build. The results reported here were measured on H100; this test gate does
not establish support for every later GPU architecture.
`fsdp_utils` currently imports the train actor eagerly, so a minimal installation
of only Torch, Diffusers and FA3 is insufficient for normal Miles imports.
Do not rebuild the full image or download a model as the first experiment.

Record the Miles commit, image digest, driver, CUDA, NCCL, Python, PyTorch,
Diffusers commit, FA3 wheel filename/hash, actual `flash_attn_interface.__file__`,
SGLang commit, GPU topology, commands, seeds and results. Docker's SGLang branch
is not an immutable pin; record the commit actually installed.

From the checkout root:

```bash
mkdir -p artifacts/fa3
git rev-parse HEAD > artifacts/fa3/miles-commit.txt
python -m pip freeze > artifacts/fa3/packages.txt
nvidia-smi > artifacts/fa3/nvidia-smi.txt
nvidia-smi topo -m > artifacts/fa3/topology.txt
python - <<'PY'
import inspect
import torch
import flash_attn_interface as fa3
import diffusers.models.attention_dispatch as ad
print('torch/cuda:', torch.__version__, torch.version.cuda)
print('kernel file:', fa3.__file__)
print('kernel signature:', inspect.signature(fa3.flash_attn_func))
assert callable(ad.flash_attn_3_func), 'Diffusers did not load FA3'
assert torch.cuda.get_device_capability()[0] >= 9, 'Expected SM90 or newer'
print('Diffusers kernel:', ad.flash_attn_3_func)
PY
```

If imports fail, stop and fix the image before running more ranks. Do not replace
FA3 with SDPA or mark an unavailable-kernel result as passing.

## CPU interface regression

In the complete development/CI environment:

```bash
python -m pytest -q \
  tests/fast/backends/test_fsdp_deterministic_args.py \
  tests/fast/backends/fsdp_utils/test_model_backend.py \
  tests/fast/backends/fsdp_utils/test_fa3_attention.py
```

These use real Diffusers dispatch and Torch autograd with a differentiable CPU
kernel spy. They establish argument routing and graph preservation, not CUDA
kernel behavior. They also cover the actor's enable-before-set order, active
backend selection, registry isolation, mode-off behavior and detached LSE.

The full normal CPU suite was also run at `b0ff911` on macOS arm64 with Python
3.12.12: **317 passed, 0 skipped, 20 warnings in 98.98 seconds**. Both project and
CPU-CI requirement files were installed without pin overrides, with real SGLang
at `859a278eb882dddbc4ccab1d0a3697da04a66f28` and the repository's CPU stubs.
No tests or package initializers were bypassed. Source imports used `PYTHONPATH`
because the existing Miles editable wheel emits an invalid macOS platform tag;
this is a local CPU result, not a hosted Linux or GPU CI result.

```bash
PYTHONPATH="$PWD" python -m pytest tests/fast -x -q
```

All configured pre-commit hooks and `train_diffusion.py --help` also passed.

## GPU gates: stop at the first failure

First run one GPU, using a small input. `timeout` below is GNU coreutils on Linux.
The worker's process-group timeout is an additional protection against hangs.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
timeout --kill-after=10s 300s python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  tests/fast-gpu/backends/fsdp_utils/sequence_parallel/_fa3_attention_worker.py \
  --seq-len 128 --repeats 20 --output-json artifacts/fa3/bf16-sp1.json \
  > artifacts/fa3/bf16-sp1.log 2>&1
```

Repeat the single-GPU test with `--seq-len 1024 --repeats 20` to cover multiple
key tiles contributing to dQ. Stop and isolate upstream FA3 if exact repeatability
fails, even when every value is numerically close.

Only after these pass, repeat with `--nproc_per_node=2`, then `4`, and distinct
`bf16-sp2` / `bf16-sp4` output filenames. Keep seed, full sequence length, batch,
head count and head dimension unchanged. All cases compare their local shards
against a full-sequence FA3 reference and independent FP32 math SDPA. No model or
Ray service needs to be launched.

Next exercise a longer shape and a second dtype:

```bash
timeout --kill-after=10s 300s python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  tests/fast-gpu/backends/fsdp_utils/sequence_parallel/_fa3_attention_worker.py \
  --seq-len 1024 --head-dim 128 --repeats 20 \
  --output-json artifacts/fa3/bf16-sp2-long.json \
  > artifacts/fa3/bf16-sp2-long.log 2>&1
timeout --kill-after=10s 300s python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  tests/fast-gpu/backends/fsdp_utils/sequence_parallel/_fa3_attention_worker.py \
  --dtype float16 --repeats 20 --output-json artifacts/fa3/fp16-sp1.json \
  > artifacts/fa3/fp16-sp1.log 2>&1
```

These shapes are initial probes; add the actual target model's head dimensions
and sequence lengths after confirming the small cases. Long-video shapes can
make the independent math reference expensive, so increase sizes deliberately.

The strict CI grid (BF16 SP1/2/4, FP16 SP1; S=1024 and 20 repeats) is:

```bash
python -m pytest -q tests/fast-gpu/backends/fsdp_utils/sequence_parallel/test_fa3_attention.py
```

It requires four visible GPUs. On a one-GPU host select `-k sp1`; on a two-GPU
host select `-k 'not sp4'`. Report the omitted topology as untested. CPU skips
are not GPU passes. CUDA CI fails when devices or FA3 are missing.

## Acceptance and evidence

- The spy sees real FA3, `num_splits=1`, and the requested deterministic mode;
  no fallback path can satisfy the test.
- Output and dQ/dK/dV are finite and match both references within logged bounds.
- Repeating fixed inputs/topology gives bitwise-equal output and dQ/dK/dV in
  deterministic mode. Compare repeats within one topology, not bitwise across SP.
- A tiny shared QKV projection matches the full-sequence loss, summed parameter
  gradient and one SGD update. This uses explicit gradient summation, not FSDP2.
- Mode-off forward/backward also matches its direct FA3 reference; mode-off is
  not required to show nondeterminism.

Initial cross-reference bounds are `rtol=2*eps(dtype)`, `atol=eps(dtype)/16`.
They are proposed test bounds, not measured GPU results. Do not widen them to
turn a failure green. Save error magnitudes, input shape and installed versions;
investigate scale, reduction order, layout and kernel choice first. Same-topology
repeat equality has zero tolerance.

The worker writes aggregate mismatch counts, errors, timings, memory and kernel
calls to JSON, plus rank-specific failure JSON. Preserve logs and exit codes.
Its time is total validation runtime, not an attention throughput benchmark.
Run the gates in a fresh process again to check restart reproducibility as needed;
this worker checks bitwise repeats within each process, not persisted cross-run
tensor hashes.

## Tiny Wan FSDP2 + Ulysses training validation

This separate regression uses the genuine `WanTransformer3DModel` with random
initial weights and synthetic inputs. It requires no pretrained checkpoint or
downloads. It exercises `DiffusersModelBackend`, `Wan2_2TrainPipelineConfig`,
`actor.apply_fsdp2`, the Wan GEMM patchification materialize hook, its existing
`_cp_plan`, and the production sequence-parallel boundary/attention hooks.
No new model support or package-import bypass is introduced.

The fixture has two blocks, four attention heads of width 64, and global batch
two. Latents/targets have shape `[2,4,2,32,64]`: two frames produce 1024 tokens
after `(1,2,2)` patchification. SP2 splits across the frames, exercising temporal
as well as spatial RoPE. Text length is 17, deliberately indivisible by two:
text cross-attention must retain complete conditioning while self-attention
uses Ulysses. Timesteps are 1D per example; the TI2V per-token timestep path is
outside this fixture.

Both layouts use two ranks: DP2/SP1 processes one distinct example per rank;
DP1/SP2 processes the same global batch on both ranks. Every case starts from
the same FP32 parameters and uses the same two precomputed batches, timesteps,
and targets. Two SGD steps run with checkpointing off and on, each repeated
twice. Master parameters and gradient reduction are FP32; CUDA computation uses
BF16 autocast with FP32 boundary inputs and the production Wan `norm2` FP32
override. The FP32 local-batch MSE equals the production SFT per-pair loss sum
divided by the local pair count. Production gather backward and FSDP averaging
provide the SP normalization; no extra gradient multiplier is applied.

Acceptance criteria are fixed before GPU execution:

- Compare complete output, global reporting loss, all 69 trainable parameter
  gradients, and cumulative parameter deltas from the initial state after each
  update. Require finite values and nonzero aggregate gradients/step updates.
- Cross-topology and checkpoint comparisons require output/loss relative L2
  error at most `0.02`, gradient/delta relative L2 at most `0.05`, and normalized
  maximum error at most `0.10`. Near-zero per-tensor denominators have declared
  floors of `group L2 * 1e-6` and `group peak * 1e-4`; complete concatenated
  vectors have no group-relative floor. Actual per-parameter errors are saved.
- Fixed-topology/configuration repetitions require bitwise equality. Numerical
  negative controls must reject doubled updates, zero updates, and a missing
  second update. Do not widen bounds after a failing GPU result.
- Every attention layer is traced at the real public FA3 function, asserting
  BF16 Q/K/V on the same CUDA device, expected self/cross-attention shapes,
  `deterministic=True`, and `num_splits=1`. The default matrix expects 96 call
  entries per rank: 64 original-forward entries and 32 checkpoint-recompute
  entries during backward. A separate `completed` flag records actual returns;
  non-reentrant checkpoint early-stop can interrupt a recompute call. Original
  forward entries must complete. Entry counts are not return counts.
- Live `norm2` hooks require FP32 weights in 48 observations per rank, including
  checkpoint recomputation. The matrix produces sixteen update records and
  2304 comparisons per rank, including 1152 bitwise repeat checks.

Run the registered test with two compatible GPUs in the complete Miles image:

```bash
PYTHONPATH="$PWD" python -m pytest -x -q \
  tests/fast-gpu/backends/fsdp_utils/sequence_parallel/test_fa3_wan.py
```

It is registered in `stage-c-5-gpu-h200` with the `fsdp` label. CUDA CI fails for
missing GPUs/FA3. The launcher kills its complete worker process group on a
360-second timeout; the worker's distributed-operation timeout is 180 seconds.
Preserve both rank JSON reports and the pytest exit/JUnit result.

The explicit native FP32 CPU control passed on two Gloo ranks with real FSDP2:
**2304 checks per rank, all 1152 repeat checks bitwise equal**, and all three
negative controls rejected. The final two-frame S1024 fixture took about 10.25
seconds per rank after setup. Its stricter CPU budgets were relative L2 `1e-4`
and normalized maximum `1e-3`; observed maxima were `5.786e-5` and `2.578e-4`.
This CPU result does not validate FA3 or BF16. Its invocation is:

```bash
PYTHONPATH="$PWD" python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29571 \
  tests/fast-gpu/backends/fsdp_utils/sequence_parallel/_fa3_wan_worker.py \
  --cpu-sanity --backend native --seq-len 1024 \
  --output-json artifacts/fa3/wan-sp2-cpu.json
```

The **two-H100 GPU run passed on 2026-09-23 (PT)** through the normal pytest
entry point: one test, zero failures/errors/skips, 42.41 seconds. Each rank
completed 2304 numerical comparisons and sixteen update records over eight
cases, checking all 69 trainable parameter tensors. All 1152 fixed-configuration
repeat comparisons were bitwise equal. Checkpoint on/off comparisons were also
bitwise equal in this run. All three numerical negative controls were rejected.

Observed SP1/SP2 relative L2 maxima (the declared limits were unchanged):

| Quantity | Full-tensor/vector maximum | Worst individual parameter maximum |
| --- | ---: | ---: |
| Output | 0.002740 | — |
| Loss | 0.00002426 | — |
| Gradients | 0.001843 | 0.02044 |
| Parameter deltas | 0.001701 | 0.01799 |

The full-vector comparisons have no denominator floor; the parameter-wise
column uses the documented group-relative floors. All SP1/SP2 normalized maximum
errors were below 0.01876, within the fixed 0.10 budget. Each rank recorded
96 real FA3 entries and 96 successful returns: 64 original forwards and 32
checkpoint recomputations, all BF16 with `deterministic=True, num_splits=1`.
All 48 live `norm2` observations per rank retained FP32 weights.

The environment was two NVIDIA H100 80GB HBM3 GPUs connected by NVLink,
Python 3.12.3, Torch 2.11.0+cu129, CUDA 12.9, NCCL 2.28.9 and Diffusers
0.40.0.dev0 in the pinned complete Miles image below. The FA3 interface SHA256
was `a85d9b9b8ecf4284e967f3bb6da7522dfe2f80cbd4ae7e4752d054e1ff6b7685`.
The verified 144-file runtime snapshot matched test/source commit `634b591`;
unrelated test files remained at the recorded upstream base. The output archive
SHA256 was `d6264380c25ebda2eb95d6ba43c4298e2167e1cdd06ea945bde6bc01228e3695`.

This establishes this tiny real architecture's FSDP2/FA3/SP2 training integration.
It does not establish pretrained Wan quality/convergence, the complete Ray
actor lifecycle, SGLang rollout, NFT/GRPO, Krea SP2, model SP4, or performance.

## H100 validation (2026-09-23)

The operator matrix used H100 80GB HBM3 (SM90, driver 595.91.07), Python 3.13.5,
Torch 2.11.0+cu129, CUDA runtime 12.9, NCCL 2.28.9 and the Diffusers pin above.
The FA3 wheel was `flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl` from
`yueming-yuan/miles-wheels`, SHA256
`b0f4d97418aa129522cd4b4e65ce516ddf8af64815f4ce040cb38a6d94cef971`.

All 12 operator/control cases passed. Standalone BF16 and FP16 S=1024 passed all 19 repeated
comparisons for output, dQ, dK and dV. Miles BF16 SP1/2/4 and FP16 SP1 passed at
both S=128 and S=1024, with 20 runs each. Every Miles case completed 91 checks,
including numerical references, bitwise repeats, a projection SGD update and
deterministic-off backward. The four-GPU SDPA control and 48 focused CPU tests
also passed.

These runs used the real operators with only the eager actor package initializer
bypassed. They establish the measured H100 attention behavior, not successful
import of the complete actor, rollout/training equivalence, performance, or
bitwise equality across GPU architectures.

## Complete-image and Krea2 short E2E (2026-09-23)

The complete official Miles image subsequently passed the normal pytest entry
points: 48 focused CPU tests and all four strict GPU cases, without the operator
bootstrap. On H100, pretrained Krea2 completed a short DP=2/SP=1 rollout/train/sync
run: four NFT training steps and checkpoints, 16 generated samples with OCR
rewards, EMA steps 1 through 5 and real LoRA IPC weight updates. The run exited 0.

The trace checker passed over 2,106 records in 33 files. It observed 62 sampled
main-training FA3 returns, all with `deterministic=True, num_splits=1`, and 110
successful rollout FA3 v3 returns, with no traced masks or kernel errors. This
confirms actual execution beyond backend flags; the trace is sampled, not a
complete count of every attention call. Teardown emitted a loky resource-tracker
warning, retained in the logs.

The image was
`radixark/miles_diffusion@sha256:ed008ae25cc0e80ef0ccebee7137b5c498bf86d4a6a5a79147cdc60ad6a6a2ce`.
Launching required an external Ray instance limited to 16 CPUs and
`expandable_segments:False` for CUDA IPC in this container. These were runtime
configuration changes, not attention changes. This one short E2E run does not
establish convergence, model-level SP=2/4, whole-run repeatability or numerical
parity between frozen rollout and training paths.

## Remaining rollout parity and model gates

This patch changes the Diffusers training adapter. SGLang diffusion rollout has a
separate attention implementation. A rollout flag or the name "FA3" alone does
not establish that both sides executed compatible kernels.

For a new model or topology, agree on the model, checkpoint revision, container
and attention backend. The completed Krea2 smoke covers an actual update/sync
loop at SP=1; frozen rollout/train parity and model-level Ulysses still need:

1. Instrument the actual SGLang diffusion attention call. Verify FA3/`fa_ver=3`,
   input layout, dtype, scale, mask, sequence lengths and split count. Its varlen
   inference interface differs from the dense training autograd interface.
2. With frozen weights, capture one attention layer's Q/K/V after positional
   transforms. Replay identical tensors through rollout and training entrypoints;
   use numerical parity, not an assumption of bitwise equality between kernels.
3. Replay identical latents, timesteps and conditioning through the frozen
   transformer. Compare predictions and the log-probability quantity actually
   used by the selected Miles objective.
4. Run one actual FSDP2 update and weight synchronization. Check parameter/LoRA
   synchronization and finite losses/gradients before a short training smoke.

Native LTX, hub/varlen training backends, ring, compile/CUDA graphs, arbitrary
masks/GQA, and full-model repeatability are separate validation scopes. Do not
claim them from this attention-level test.
