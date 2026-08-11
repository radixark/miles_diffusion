---
name: install-miles-diffusion
description: Fallback installer for miles_diffusion on a bare CUDA 12.9 Linux GPU box, reproducing the official rockdu/miles_diffusion image's package versions and verifying them. Docker is the supported way to run miles_diffusion and this is not recommended — use it only when the image cannot be pulled, or to check whether a machine's env still matches the image.
---

# install-miles-diffusion

> **Docker is the official way to run miles_diffusion — pull `OFFICIAL_IMAGE` and use it.**
> This skill is a fallback for machines that cannot, and it is not recommended: it reconstructs
> the image's package set, it is not the image. `deep_ep` and `flash_mla` cannot be reproduced at
> all, and every apt-sourced dist depends on the host being Ubuntu 24.04.

Reproduces the official Docker environment on a box that has only CUDA 12.9 and Ubuntu 24.04.
The target is not "an env that works" but "the image's env": `snapshot/` is a capture of
`OFFICIAL_IMAGE`, `install.sh` replays it, and `verify_env.py` fails if the result drifts.

## Files

| | |
|---|---|
| `snapshot/packages.txt` | the image's `pip freeze`; `#skip[reason]` marks what we don't install |
| `snapshot/apt.txt` | the apt packages install.sh needs |
| `snapshot/pins.env` | pins pip can't express — image tag, sglang/miles branch and commit, indexes |
| `snapshot/kernels.lock` | the image's resolved FA3 kernel hashes |
| `install.sh` | six steps: apt, pip, packages, sglang, miles, verify |
| `verify_env.py` | per-package diff against `packages.txt`; non-zero exit on drift |
| `refresh.sh` | re-capture `snapshot/` when the image moves |

## Run

A bare CUDA image has no git, and some clusters hand out a resolver that can't answer for
outside names:

```bash
grep -q nameserver /etc/resolv.conf || echo "nameserver 8.8.8.8" >> /etc/resolv.conf
apt-get update -qq && apt-get install -y --no-install-recommends git ca-certificates
git clone https://github.com/radixark/miles_diffusion.git && cd miles_diffusion
bash .claude/skills/install-miles-diffusion/install.sh
```

~20 min cold, ~4 with a warm pip cache. Run it in the background and poll the log.
`--from STEP` resumes; every step is idempotent. To check an existing box instead:
`python3 .claude/skills/install-miles-diffusion/verify_env.py`.

## Why it replays instead of resolving

The image does not satisfy its own dependency metadata — `pip check` there reports ~20
violations, because sglang is installed `--no-deps` from a commit newer than the base image's
package set, and nvidia-modelopt wants `setuptools>=80` against a pinned 70.2.0. Resolving the
freeze returns `ResolutionImpossible`, and any relaxation that resolves lands on versions the
image does not have. So `install.sh` installs `packages.txt` with `--no-deps`.

Two packages cannot be reproduced at all: `deep_ep` and `flash_mla` are compiled inside
`lmsysorg/sglang` and published nowhere. Neither is on a miles_diffusion import path.

`flash_attn_3` comes from a wheel the image installs from a local copy, so `pip freeze` reports
an unresolvable `file://` path. `refresh.sh` rewrites it to the release asset the Dockerfile
pulls from (`FA3_WHEELS_REPO` / `FA3_WHEELS_TAG`), so pip installs it like any other pin.

## sglang and miles are anchored to main

Both come from `main` upstream, as in the Dockerfile (`SGLANG_DIFFUSION_BRANCH=main`,
`MILES_DIFFUSION_COMMIT=main`). The difference is that the Dockerfile follows the branch tip at
build time while this pins a commit and checks it is an ancestor of the branch. That check
matters: the first capture pinned a local cherry-pick that exists in no remote.

Because the ancestor check needs real history, the checkout is a blob-filtered full clone rather
than the image's `--depth=1`, so setuptools_scm reports a different commit count for the same
sha; `verify_env.py` compares sglang and miles by presence, not by that string.

## Refreshing the snapshot

```bash
bash .claude/skills/install-miles-diffusion/refresh.sh <devbox-on-the-new-image>
```

Capture from a box running the image itself, not from whichever tag a devbox happens to be on —
the first snapshot here was taken from a devbox two image releases behind, which pinned
`diffusers 0.37.0` against a repo that had moved to `0.38.0`.

A devbox people have worked on is also not a pristine image. `refresh.sh` guards both drift
classes it has hit: the sglang commit comes from setuptools_scm rather than `git rev-parse HEAD`,
and packages dated after the image's build day are marked `#skip[capture-drift]`. If the box ran
`:latest`, replace `OFFICIAL_IMAGE` with the concrete tag it resolved to.

## What goes wrong

- **"Cannot uninstall X, RECORD file not found"** — apt dists have no RECORD. `step_pip`
  pre-seeds pip, PyJWT and wheel with `--ignore-installed`; a fourth needs the same flag.
- **"Device or resource busy" on a file in `/usr/local/bin`** — the host mounted a binary
  read-only there (rx devboxes do this for `uv`, `gh`, `claude`). `step_packages` drops the
  matching packages and `verify` treats them as expected absences.
- **Not Ubuntu 24.04** — the apt-sourced python dists then come from different pockets and
  `verify` reports drift on them.

## Running SD3 after install

```bash
export HF_TOKEN=...              # SD3.5 is gated
export NCCL_NVLS_ENABLE=0        # partial-node containers have no IMEX multicast capability
python3 scripts/run_diffusion_grpo_sd3_ocr_sglang.py --cuda-visible-devices 0,1
```

Without `NCCL_NVLS_ENABLE=0`, `execute_train` enables NVLink SHARP whenever it sees NVLink and
NCCL dies in `FSDPTrainRayActor.init` with `Failed to bind NVLink SHARP (NVLS) Multicast memory`.
On a fresh HF cache the SD3.5 download pulls both `model.safetensors` and `model.fp16.safetensors`
for the CLIP encoders; sglang's fast loader refuses the duplicate tensor names, logs a traceback
and falls back to the native loader. The run continues.

Verified on `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04`, 4×H100, no python preinstalled: 369
matched / 0 mismatched / 0 missing, then a completed GRPO step of the SD3 recipe.
