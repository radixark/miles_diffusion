# Qwen-Image OCR with 2 GPUs

This example runs miles-diffusion with Qwen-Image, FSDP training, LoRA updates,
the built-in diffusion rollout path, and the OCR reward.

## Environment Setup

First complete the base environment setup in
[Quick Start](../get_started/quick_start.md).

Then install the OCR task dependencies:

```bash
conda activate miles-diffusion
cd /path/to/miles
```

Follow [Task Dependencies: OCR Dependencies](../get_started/task_dependencies.md#ocr-dependencies).
The important check is:

```bash
python -c "from paddleocr import PaddleOCR; from Levenshtein import distance; import miles.rollout.rm_hub.ocr; print('OCR deps OK')"
```

The example uses 2 NVIDIA GPUs. It downloads the training and evaluation data
from Hugging Face during startup, so the machine must be able to access Hugging
Face.

Optionally enable Weights & Biases logging:

```bash
export WANDB_API_KEY=...
```

## Run Training

Execute the 2-GPU script:

```bash
cd /path/to/miles
conda activate miles-diffusion
bash scripts/run-diffusion-grpo-ocr-2gpu-flowgrpo-aligned.sh
```

By default, the script uses:

```bash
CUDA_VISIBLE_DEVICES=2,3
```

Override it if your available GPUs are different:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run-diffusion-grpo-ocr-2gpu-flowgrpo-aligned.sh
```

The script writes checkpoints under:

```bash
logs/diffusion_grpo_ocr_2gpu_flowgrpo_aligned_<timestamp>/ckpt
```

## Data and Model

The script downloads the OCR dataset to:

```bash
/root/datasets/miles-diffusion-datasets
```

using:

```bash
hf download --repo-type dataset rockdu/miles-diffusion-datasets \
  --include "flowgrpo_ocr/**" \
  --local-dir /root/datasets/miles-diffusion-datasets
```

Training reads:

```bash
/root/datasets/miles-diffusion-datasets/flowgrpo_ocr/train.jsonl
```

Evaluation reads:

```bash
/root/datasets/miles-diffusion-datasets/flowgrpo_ocr/test.jsonl
```

The model is loaded from:

```bash
Qwen/Qwen-Image
```

## Parameter Introduction

Here, we briefly introduce the main parts of
`scripts/run-diffusion-grpo-ocr-2gpu-flowgrpo-aligned.sh`.

### Diffusion Rollout

The script uses the diffusion rollout function:

```bash
--train-backend fsdp
--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout
--hf-checkpoint Qwen/Qwen-Image
--diffusion-model Qwen/Qwen-Image
```

### LoRA Training

The example trains LoRA weights instead of full model weights:

```bash
--use-lora
--lora-rank 64
--lora-alpha 128
--diffusion-init-lora-weight gaussian
```

### OCR Reward

The reward is configured through both the diffusion reward string and the miles
reward type:

```bash
--diffusion-reward ocr:1.0
--rm-type ocr
--advantage-estimator grpo
```

### Colocated Resources

Training and rollout share the same 2 GPUs:

```bash
--actor-num-gpus-per-node 2
--rollout-num-gpus 2
--rollout-num-gpus-per-engine 1
--num-gpus-per-node 2
--colocate
```

## Batch and Step Math

The 2-GPU script is scaled down from the 4-GPU FlowGRPO-aligned OCR recipe.
Per rollout:

```text
rollout_batch_size = 16 prompts
n_samples_per_prompt = 16 samples per prompt
samples_per_rollout = 16 * 16 = 256 samples
num_steps_per_rollout = 2 optimizer steps
global_batch_size = 256 / 2 = 128 samples per optimizer step
```

With 2 training GPUs, each rank receives:

```text
128 / 2 = 64 samples per optimizer step
```

The DiT forward is tiled as:

```bash
--micro-batch-size-sample 4
--micro-batch-size-tstep 2
```

so one forward tile covers `4 * 2 = 8` sample/timestep cells.

For a deeper explanation of these batch-shape parameters, see
[Batch sizes in miles-diffusion](../developer_guide/batch_sizes_in_miles_d.md).

## Diffusion Sampling Settings

The example mirrors the Qwen-Image OCR FlowGRPO settings:

```bash
--diffusion-num-steps 10
--diffusion-eval-num-steps 50
--diffusion-guidance-scale 4.0
--diffusion-true-cfg-scale 4.0
--diffusion-noise-level 1.2
--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window
--diffusion-sde-window-size 2
--diffusion-sde-window-range 3,5
--diffusion-height 512
--diffusion-width 512
```

The active SDE training window has size 2. The `3,5` range selects the same
effective window used by the aligned FlowGRPO recipe.

## 4-GPU Variant

If you have 4 GPUs, use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run-diffusion-grpo-ocr-4gpu-flowgrpo-aligned.sh
```

The 4-GPU script doubles `--rollout-batch-size` from 16 to 32 while keeping the
per-rank training load at 64 samples per optimizer step.

## Expected Result

A successful launch should:

1. download `flowgrpo_ocr/**` if it is not already present;
2. start the colocated FSDP actor and sglang-diffusion rollout engine;
3. generate Qwen-Image OCR rollouts;
4. compute OCR rewards;
5. begin GRPO LoRA updates;
6. save checkpoints under the run-specific `logs/` directory.

If the run fails before training starts, first check GPU visibility, Hugging Face
access, the base environment, and the OCR task dependencies.
