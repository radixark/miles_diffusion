---
title: Installation
description: Get a working miles-diffusion environment — Docker (recommended) or from source.
---
miles-diffusion trains a diffusion DiT with FSDP2 while [sglang-diffusion](https://github.com/sgl-project/sglang)
serves the rollout. The two halves must agree numerically, so the pinned versions matter more
than usual: the rollout engine tracks **sglang main**, not a release tag, and the training side
pins **torch 2.11.0** (an FSDP monkey patch is version-gated on it).

Use Docker unless you have a reason not to.

## Method 1: Docker (recommended)

<Warning>

The image is still **experimental** and there is no published release tag yet
(`docker/README.md`: *Release rule — TBD*). Build it yourself from `docker/Dockerfile`.

</Warning>

### Build

```bash
git clone https://github.com/radixark/miles_diffusion.git
cd miles_diffusion

# CUDA 12.9 (default)
docker build -f docker/Dockerfile -t miles-diffusion:$(cat docker/version.txt) .

# CUDA 13.0
docker build -f docker/Dockerfile -t miles-diffusion:$(cat docker/version-cu13.txt) \
  --build-arg SGLANG_IMAGE_TAG=v0.5.14-cu130 \
  --build-arg FA3_WHEELS_TAG=cu130-x86_64 .
```

Useful build args:

| Arg | Default | What it does |
|---|---|---|
| `SGLANG_IMAGE_TAG` | `v0.5.12-cu129` | Base `lmsysorg/sglang` image. |
| `SGLANG_DIFFUSION_BRANCH` | `main` | sglang branch the rollout engine is built from. |
| `SGLANG_DIFFUSION_COMMIT` | `none` | Pin a sglang sha; `none` follows the branch tip. |
| `FA3_WHEELS_TAG` | `cu129-x86_64` | Which prebuilt FlashAttention-3 wheel to pull. |
| `MILES_DIFFUSION_COMMIT` | `main` | Ref of miles_diffusion baked into the image. |

### Run

```bash
docker run --rm \
  --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -v /your/datasets:/root/datasets \
  -e HF_TOKEN=$HF_TOKEN \
  -it miles-diffusion:<tag> /bin/bash
```

The image ships with:

- PyTorch 2.11.0 and the sglang base image's CUDA stack
- sglang built from main (`/sgl-workspace/sglang`, editable) with `sglang.multimodal_gen`
- FlashAttention-3 (`flash_attn_interface`), `sglang-kernel==0.4.5`, `torch_memory_saver==0.0.9`
- `diffusers`, `peft`, `transformers`, `ray`, `wandb`, and `ltx-core` from `requirements.txt`
- miles_diffusion installed editable at `/root/miles_diffusion`
- PaddleOCR's English det/rec/cls weights pre-downloaded (the OCR reward would otherwise
  race-download them at runtime)
- `nccl-tests` binaries on `PATH` for link diagnostics

To run your own working tree instead of the baked copy, bind-mount it and reinstall:

```bash
docker run ... -v $PWD:/root/miles_diffusion -it miles-diffusion:<tag> /bin/bash
cd /root/miles_diffusion && pip install -e . --no-deps
```

## Method 2: From source

Only worth it if you already have a working sglang-main environment.

```bash
git clone https://github.com/radixark/miles_diffusion.git
cd miles_diffusion
pip install -r requirements.txt
pip install -e . --no-deps
```

Requires **Python ≥ 3.12**. `requirements.txt` pins `torch==2.11.0`, `diffusers==0.38.0`,
`transformers==5.5.4`, `peft==0.18.1`, and `ray==2.53.0`, among others.

<Warning>

`pip install -r requirements.txt` does **not** give you a rollout engine. You additionally need
sglang built from `main` with the `sglang.multimodal_gen` package, plus FlashAttention-3 and
`torch_memory_saver`. `docker/Dockerfile` is the authoritative recipe for that half — read it
before assembling an environment by hand.

</Warning>

## Verify

```bash
# package imports
python -c "import miles; print('miles-diffusion import OK')"

# rollout engine is present
python -c "from sglang.multimodal_gen.runtime.server_args import ServerArgs; print('sglang-d OK')"

# GPUs visible
nvidia-smi

# arguments parse
python3 train_diffusion.py --help | head
```

Then run the CPU test suite — seconds, and it catches most environment breakage:

```bash
pytest tests/fast -x -q
```

On a machine with no sglang GPU kernels installed, install the CPU stubs first, the way CI does:
`uv pip install tests/ci/cpu_stubs`.

## Hardware

| | |
|---|---|
| Validated GPUs | H200 — what the CI runners are, and what every shipped recipe was tuned on. The default image pulls a Hopper FlashAttention-3 wheel. |
| Minimum for a real run | 2 GPUs — SD3.5-medium LoRA GRPO colocated (`scripts/run_diffusion_grpo_sd3_ocr_sglang.py`) |
| Typical | 4 train GPUs + 1 dedicated reward GPU (the Qwen-Image / Wan2.2 / LTX recipes) |

Under `--colocate` the training actor and the rollout engines time-share the same GPUs,
so the floor is set by whichever of the two needs more memory, not by their sum.

## Environment variables

| Variable | When you need it |
|---|---|
| `HF_TOKEN` | Gated checkpoints. SD3.5 needs it **even when the weights are cached** — sglang still fetches `model_index.json` from the hub at startup. |
| `WANDB_API_KEY` | Without it the launch scripts silently drop all `--use-wandb` flags. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Set by every launch script; reduces fragmentation OOMs. |
| `MILES_SCRIPT_EXTERNAL_RAY=1` | A scheduler (Slurm, k8s) already built the Ray cluster — skips `ray stop` / `ray start` in the launcher. |
| `MILES_DIFFUSION_MODEL_FAMILY` | Escape hatch for family auto-detection. Prefer the `--diffusion-model-family` flag; the env var still wins over both. |

## Next

- [Training Script Walkthrough](/user-guide/training-script-walkthrough) — what the launch scripts
  actually pass, group by group.
- [CLI Reference](/user-guide/cli-reference) — every flag.
