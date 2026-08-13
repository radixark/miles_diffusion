---
title: Installation
description: Install miles-diffusion — Docker is the recommended path; a verified installer script covers bare metal.
---
There are three ways to install miles-diffusion. Docker is recommended: the rollout engine is
sglang built from **main** (release wheels do not carry the diffusion rollout support), the
trainer pins **torch 2.11.0** for an FSDP2 patch, and FlashAttention-3 comes as a prebuilt
per-CUDA wheel — the image carries all of that pre-assembled.

Every command block on this page was executed verbatim on a fresh machine before being
written down.

## Method 1: Docker (recommended)

```bash
docker pull rockdu/miles_diffusion:latest

docker run --rm \
  --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -it rockdu/miles_diffusion:latest /bin/bash
```

`latest` tracks sglang main and is rebuilt every few days; dated tags
(`dev-cu129-sglang-main-YYYYMMDD`) pin a specific build if you need reproducibility across
pulls.

The image ships with:

- PyTorch 2.11.0 on CUDA 12.9
- sglang built from main at `/sgl-workspace/sglang` (editable), including the
  `sglang.multimodal_gen` diffusion engine
- FlashAttention-3 (`flash_attn_interface`), `sglang-kernel`, `torch_memory_saver`
- `diffusers`, `peft`, `transformers`, `ray`, `wandb`, and `ltx-core` per `requirements.txt`
- miles-diffusion installed editable at `/root/miles_diffusion`
- PaddleOCR's English det/rec/cls models pre-downloaded, so OCR-reward actors never
  race-download them at runtime
- `nccl-tests` binaries on `PATH` for link diagnostics

To run your own working tree instead of the baked copy, bind-mount it over the same path:

```bash
docker run ... -v $PWD:/root/miles_diffusion -it rockdu/miles_diffusion:latest /bin/bash
cd /root/miles_diffusion && pip install -e . --no-deps
```

### Building the image yourself

`docker/Dockerfile` is the recipe:

```bash
# CUDA 12.9 (default)
docker build -f docker/Dockerfile -t miles-diffusion:$(cat docker/version.txt) .

# CUDA 13.0
docker build -f docker/Dockerfile -t miles-diffusion:$(cat docker/version-cu13.txt) \
  --build-arg SGLANG_IMAGE_TAG=v0.5.14-cu130 \
  --build-arg FA3_WHEELS_TAG=cu130-x86_64 .
```

| Arg | Default | What it does |
|---|---|---|
| `SGLANG_IMAGE_TAG` | `v0.5.12-cu129` | Base `lmsysorg/sglang` image. |
| `SGLANG_DIFFUSION_BRANCH` | `main` | sglang branch the rollout engine is built from. |
| `SGLANG_DIFFUSION_COMMIT` | `none` | Pin a sglang sha; `none` follows the branch tip. |
| `FA3_WHEELS_TAG` | `cu129-x86_64` | Which prebuilt FlashAttention-3 wheel to pull. |
| `MILES_DIFFUSION_COMMIT` | `main` | Ref of miles-diffusion baked into the image. |

## Method 2: Bare metal, from source

For a machine that cannot run the image. Requires **CUDA 12.9** and **Ubuntu 24.04** — the
installer reproduces the official image's package set on the host, and its apt-sourced
pieces assume that distribution.

```bash
apt-get update && apt-get install -y --no-install-recommends git ca-certificates
git clone https://github.com/radixark/miles_diffusion.git
cd miles_diffusion
bash .claude/skills/install-miles-diffusion/install.sh
```

About 20 minutes cold, ~4 with a warm pip cache. The script runs six idempotent steps
(apt, pip, package replay, sglang from source, miles, verify) and `--from STEP` resumes a
failed run. The final verify step diffs every installed package against the image's
snapshot and fails on drift — a clean run ends with:

```
matched 369   mismatched 0   missing 0   expected-absent 4   extra 0
environment matches the official image
```

To check whether an existing machine still matches the image without installing anything:

```bash
python3 .claude/skills/install-miles-diffusion/verify_env.py
```

<Warning>

Plain `pip install -r requirements.txt && pip install -e . --no-deps` also works, but
produces a **trainer-only** environment: `import miles` and CUDA torch are fine, and CPU
tests and `--train-only` SFT run, but there is no rollout engine — `sglang.multimodal_gen`
and FlashAttention-3 are not on PyPI and not in `requirements.txt`. Do not expect an RL run
out of it. Resolving the full environment through pip alone is not possible (the pinned set
is only consistent under `--no-deps` replay), which is exactly what the installer script
handles.

</Warning>

## Method 3: Update an existing container

If you already run the image and want the latest code:

```bash
cd /root/miles_diffusion
git pull --rebase
pip install -e . --no-deps
```

No Ray restart is needed — the launch scripts stop and restart the Ray cluster themselves
on every run.

## Verify

Whichever method you used:

```bash
# the trainer package
python3 -c "import miles; print('miles OK')"

# the rollout engine
python3 -c "from sglang.multimodal_gen.runtime.server_args import ServerArgs; print('sglang-d OK')"

# GPUs visible
nvidia-smi

# arguments parse
python3 train_diffusion.py --help | head

# CPU test suite (seconds, catches most environment breakage)
pytest tests/fast -x -q
```

## Hardware

| | |
|---|---|
| Validated GPUs | H200 — the CI runners, and what every shipped recipe was tuned on. H100-class Hopper works the same; the default image pulls a Hopper FA3 wheel. |
| Minimum for a real run | 2 GPUs — the SD3.5 GRPO recipe runs train + rollout colocated on two. |
| Typical | 4 train GPUs + 1 dedicated reward GPU (the Qwen-Image / Wan2.2 / LTX recipes). |

Under `--colocate` the trainer and the rollout engines time-share the same GPUs, so the
floor is set by whichever needs more memory, not by their sum.

## Environment variables

| Variable | When you need it |
|---|---|
| `HF_TOKEN` | Gated checkpoints. SD3.5 needs it **even when the weights are cached** — sglang still fetches `model_index.json` from the hub at startup. |
| `WANDB_API_KEY` | Without it the launch scripts silently drop all `--use-wandb` flags. |
| `MILES_SCRIPT_EXTERNAL_RAY=1` | A scheduler (Slurm, k8s) already built the Ray cluster — skips `ray stop` / `ray start` in the launcher. |

## Next

- [Quick Start](/getting-started/quick-start) — from this container to a running Flow-GRPO
  job on SD3.5.
- [Training Script Walkthrough](/user-guide/training-script-walkthrough) — what the launch
  scripts actually pass, group by group.
