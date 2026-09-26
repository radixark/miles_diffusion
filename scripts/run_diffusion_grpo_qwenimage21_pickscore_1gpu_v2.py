"""Qwen-Image 2.1 PickScore Flow-GRPO, 1-GPU recipe.

CFG 1.0, SDE 3–5, LoRA 64/128, 512² / 10 steps.

  16 prompts × 16 samples = 256 / rollout
  lr 1e-4
  200 rollouts default
  micro-batch 4, PickScore batch 8, sglang concurrency 4

Usage:
    python3 scripts/run_diffusion_grpo_qwenimage21_pickscore_1gpu_v2.py \\
        --hf-checkpoint /path/to/Qwen-Image-2.1 \\
        --prompt-data /path/to/flowgrpo_pickscore
"""

import os
from dataclasses import dataclass
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "Qwen/Qwen-Image-2.1"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_pickscore"
WANDB_PROJECT = "miles-diffusion-grpo"
SGLANG_21_PYTHON = os.environ.get("SGLANG_21_PYTHON", "")
DIFFUSERS_21_SRC = os.environ.get("DIFFUSERS_21_SRC", "")
PICKSCORE_PROCESSOR = os.environ.get("PICKSCORE_PROCESSOR_PATH", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
PICKSCORE_MODEL = os.environ.get("PICKSCORE_MODEL_PATH", "yuvalkirstain/PickScore_v1")


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 200
    hf_checkpoint: str = MODEL
    prompt_data: str = ""
    data_dir: str = "/root/datasets"
    eval_interval: int = 0
    extra_args: str = ""


def prepare(args: ScriptArgs) -> str:
    if args.prompt_data:
        return args.prompt_data
    local_dir = U.hf_download_dataset(DATASET, include=f"{DATASET_SUBSET}/**", data_dir=args.data_dir)
    return f"{local_dir}/{DATASET_SUBSET}"


def execute(args: ScriptArgs, data_dir: str) -> None:
    run_name = f"diffusion_grpo_qwenimage21_pickscore_1gpu_v2_{U.create_run_id()}"
    train_jsonl = f"{data_dir}/train.jsonl" if Path(data_dir).is_dir() else data_dir

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --diffusion-model-family qwen_image21 "
        f"--save {args.output_dir}/{run_name}/ckpt --save-interval 10 "
    )

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {train_jsonl} "
        "--input-key input "
        "--rollout-batch-size 16 "
        "--n-samples-per-prompt 16 "
        f"--num-rollout {args.num_rollout} "
        "--num-steps-per-rollout 2 "
        "--rollout-microgroup-size 8 "
        "--train-dp-split-mode stride "
        "--diffusion-train-iter-order sample_major "
        "--diffusion-num-steps 10 "
        "--diffusion-guidance-scale 1.0 "
        "--diffusion-true-cfg-scale 1.0 "
        "--diffusion-noise-level 1.2 "
        "--diffusion-height 512 "
        "--diffusion-width 512 "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window "
        "--diffusion-num-sde-steps 2 "
        "--diffusion-sde-window-range 3,5 "
        "--rollout-return-full-trajectory "
        "--rollout-patch-group qwen_image21 "
    )

    eval_args = "--skip-eval-before-train "
    if args.eval_interval:
        test_jsonl = f"{data_dir}/test.jsonl"
        eval_args = (
            f"--eval-prompt-data pickscore_test {test_jsonl} "
            f"--eval-interval {args.eval_interval} "
            "--diffusion-eval-num-steps 20 "
            "--skip-eval-before-train "
        )

    grpo_args = "--advantage-estimator grpo --globalize-reward-std --diffusion-clip-range 1e-4 "

    optimizer_args = "--lr 1e-4 --adam-beta2 0.999 --weight-decay 1e-4 "

    lora_args = (
        "--use-lora --lora-ipc-weight-sync --lora-rank 64 --lora-alpha 128 --lora-init-weights gaussian "
        "--sglang-lora-merge-mode dynamic "
    )

    reward_args = (
        "--rm-type pickscore "
        "--pickscore-reward-colocate "
        "--pickscore-num-workers 1 "
        "--pickscore-batch-size 8 "
        f"--pickscore-processor-path {PICKSCORE_PROCESSOR} "
        f"--pickscore-model-path {PICKSCORE_MODEL} "
    )

    wandb_args = U.get_default_wandb_args(
        __file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=8, wandb_log_image_interval=10
    )

    sglang_args = (
        "--use-miles-router "
        "--rollout-fetch-in-parser "
        "--rollout-parser-num-workers 4 "
        "--sglang-server-concurrency 4 "
        "--sglang-attention-backend torch_sdpa "
        "--sglang-model-id Qwen-Image-2.1 "
        "--update-weight-buffer-size 1073741824 "
    )

    train_backend_args = (
        "--train-backend fsdp --fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 --diffusion-forward-dtype bf16 "
    )

    perf_args = "--gradient-checkpointing --micro-batch-size-sample 4 --micro-batch-size-tstep 1 "

    misc_args = (
        "--actor-num-gpus-per-node 1 "
        "--rollout-num-gpus 1 "
        "--rollout-num-gpus-per-engine 1 "
        "--num-gpus-per-node 1 "
        "--colocate "
        "--deterministic-mode "
        "--diffusion-debug-mode "
    )

    extra_env_vars = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "offline",
    }
    pythonpath = ":".join(p for p in (SGLANG_21_PYTHON, DIFFUSERS_21_SRC, os.environ.get("PYTHONPATH", "")) if p)
    if pythonpath:
        extra_env_vars["PYTHONPATH"] = pythonpath
    if os.environ.get("HF_HOME"):
        extra_env_vars["HF_HOME"] = os.environ["HF_HOME"]
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        extra_env_vars["HUGGINGFACE_HUB_CACHE"] = os.environ["HUGGINGFACE_HUB_CACHE"]

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {eval_args} {grpo_args} {optimizer_args} "
            f"{lora_args} {reward_args} {wandb_args} {sglang_args} {train_backend_args} {perf_args} "
            f"{misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=1,
        config=args,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    data_dir = prepare(args)
    execute(args, data_dir)


if __name__ == "__main__":
    typer.run(main)
