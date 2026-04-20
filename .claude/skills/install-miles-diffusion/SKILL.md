---
name: install-miles-diffusion
description: One-click install of miles-diffusion on a fresh Linux GPU machine. Sets up the conda env, sglang-diffusion (PR #20464), miles package, flow_grpo OCR reward deps, then smoke-tests train_diffusion. Use when the user asks to install / bootstrap / set up miles-diffusion on a new machine.
---

# install-miles-diffusion

Drives the install helper at `.claude/skills/install-miles-diffusion/install.sh` and verifies the result. The helper is idempotent — re-running it skips steps that already succeeded (conda env exists, sglang checkout present, etc.).

## When to invoke

User says anything like: "install miles-diffusion", "set up this repo on a new machine", "bootstrap the env for run-diffusion-grpo-ocr.sh", "装一下", etc.

## What it installs

End goal: `bash scripts/run-diffusion-grpo-ocr.sh` boots cleanly on this host.

Components:

1. **apt system libs** — `libglib2.0-0 libgl1` (paddleocr / cv2 runtime).
2. **Conda env** — Python 3.11 (configurable via `PY_VER`).
3. **PyTorch** — CUDA 12.4 build matched to the sglang-diffusion branch.
4. **sglang-diffusion** — clones **`Rockdu/sglang` @ `sglang-diffusion-rollout-test`** into `$SGLANG_DIR` (default `../sglang` sibling of miles) and installs `python[all]` in editable mode. That branch is the sglang-diffusion fork miles-diffusion depends on (multimodal_gen + `update_weights_from_tensor` for RL weight sync). Override via `SGLANG_REPO` / `SGLANG_BRANCH` only if you know what you're doing.
5. **miles package** — `pip install -e .` plus `requirements.txt`.
6. **flow_grpo OCR deps** — runs `flow_grpo/setup.sh` (paddleocr 2.9.1, peft 0.10.0, diffusers 0.33.1, etc., all `--no-deps` to avoid the usual paddlepaddle dep-hell).
7. **torch_memory_saver** — optional, skipped silently on failure.
8. **Smoke test** — `nvidia-smi`, then `python -c "import train_diffusion"`.

## How to run

Before doing anything, surface these to the user and let them override:

- `ENV_NAME` (default `miles-diffusion`)
- `PY_VER` (default `3.11`)
- `SGLANG_DIR` (default `$(dirname "$PWD")/sglang`)
- `SGLANG_REPO` (default `https://github.com/Rockdu/sglang.git`)
- `SGLANG_BRANCH` (default `sglang-diffusion-rollout-test`)
- `CUDA_VER` (default `12.4`)

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
