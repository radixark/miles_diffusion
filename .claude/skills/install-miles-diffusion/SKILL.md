---
name: install-miles-diffusion
description: One-click install of miles-diffusion on a fresh Linux GPU machine. Sets up the conda env, sglang-diffusion (PR #20464), miles package, flow_grpo OCR reward deps, then smoke-tests train_diffusion. Use when the user asks to install / bootstrap / set up miles-diffusion on a new machine.
---

# install-miles-diffusion

Drives the install helper at `.claude/skills/install-miles-diffusion/install.sh` and verifies the result. The helper is idempotent — re-running it skips steps that already succeeded (conda env exists, sglang at pinned commit, etc.).

**All package versions are pinned**, including torch, sglang (commit SHA), transformers, accelerate, ray, diffusers, peft, paddleocr, paddlepaddle-gpu, and the entire flow_grpo dep set. Pins reflect the currently-validated working environment.

## When to invoke

User says anything like: "install miles-diffusion", "set up this repo on a new machine", "bootstrap the env for run-diffusion-grpo-ocr.sh", "装一下", etc.

## What it installs

End goal: `bash scripts/run-diffusion-grpo-ocr.sh` boots cleanly on this host with **bit-reproducible** package versions.

Components (every version pinned):

1. **apt system libs** — `libglib2.0-0 libgl1` (paddleocr / cv2 runtime).
2. **Conda env** — Python `3.11` (configurable via `PY_VER`).
3. **Tooling** — `pip==26.0.1`, `wheel==0.45.1`, `setuptools==82.0.1` (resolver behaviour depends on these).
4. **PyTorch** — `torch==2.9.1` on `cu129` (override via `TORCH_VER` / `CUDA_VER`).
5. **sglang-diffusion** — clones **`Rockdu/sglang` @ `sglang-diffusion-rollout-test`** into `$SGLANG_DIR` (default `../sglang`) and `git checkout --detach $SGLANG_COMMIT` (default `0372158dd66bc7cb0740c733bd60047db790ec7d`). Installed editable as `python[all]`. Pinning to a SHA (not just the branch tip) is required for bit-exact rollout reproducibility. Override `SGLANG_REPO` / `SGLANG_BRANCH` / `SGLANG_COMMIT` only if you know what you're doing.
6. **miles package** — `pip install -r requirements.txt` (all `==`-pinned: transformers 5.5.4, accelerate 1.12.0, ray 2.53.0, datasets 4.4.2, safetensors 0.7.0, wandb 0.23.1, …) plus `pip install -e . --no-deps`.
7. **flow_grpo OCR deps** — runs `flow_grpo/setup.sh` (every line `--no-deps` and `==`-pinned: paddleocr 2.9.1, paddlepaddle-gpu 2.6.2, peft 0.18.1, diffusers 0.37.0, opencv 4.11.0.86, etc.).
8. **torch_memory_saver** — pinned to `0.0.9`, skipped silently on failure.
9. **Smoke test** — `nvidia-smi`, then `python -c "import train_diffusion"`.

## How to run

Before doing anything, surface these to the user and let them override:

- `ENV_NAME` (default `miles-diffusion`)
- `PY_VER` (default `3.11`)
- `SGLANG_DIR` (default `$(dirname "$PWD")/sglang`)
- `SGLANG_REPO` (default `https://github.com/Rockdu/sglang.git`)
- `SGLANG_BRANCH` (default `sglang-diffusion-rollout-test`)
- `SGLANG_COMMIT` (default `0372158dd66bc7cb0740c733bd60047db790ec7d`)
- `CUDA_VER` (default `12.9`)
- `TORCH_VER` (default `2.9.1`)

Then:

```bash
bash .claude/skills/install-miles-diffusion/install.sh
```

It's long — run with `run_in_background: true` and stream with Monitor, or let it block in foreground if the user is OK with a 5–15 min wait.

## What can go wrong

- **No conda/mamba** — helper aborts with a message telling the user to install miniforge. Don't auto-install conda.
- **No CUDA toolkit / no GPU** — `nvidia-smi` fails at smoke-test; install still succeeds but warn the user.
- **sglang branch missing / renamed** — if `Rockdu/sglang` no longer has `sglang-diffusion-rollout-test`, the clone/fetch fails. Do not silently fall back to upstream sgl-project/sglang: the required changes (multimodal_gen + weight-sync RPC) only live on the Rockdu fork. Surface the failure to the user and ask which branch to pin to instead.
- **paddlepaddle-gpu wheel mismatch** — pinned to 2.6.2 in flow_grpo/setup.sh. If the machine's CUDA is too new, you may need to swap the pin. Report the mismatch; don't silently change the pin.
- **System apt missing sudo** — fall back to `apt-get` without sudo (works in containers). If both fail, tell the user which .so is missing.

## After install

Tell the user:
1. `conda activate <ENV_NAME>`
2. `export WANDB_API_KEY=...` (optional)
3. `bash scripts/run-diffusion-grpo-ocr.sh`

Do **not** kick off the training run yourself — the script does `pkill -9 python` at the top, which would kill the Claude Code session.
