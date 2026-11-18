"""Async Krea-2-Raw Flow-GRPO with score centering (experimental) and OCR on 4 training + 4 rollout GPUs.

Run through train_diffusion_async.py: each batch is sampled by engines holding the previous
step's actor weights while the current batch trains, and the LoRA adapters sync to the rollout
GPUs over NCCL. The engines run the DiT in fp8 and apply the adapters on the fly, since they
cannot be merged into fp8 weights; training stays bf16. The one-step-stale, fp8 rollout means
the trainer re-scores SDE transitions drawn around a mean it no longer owns;
--diffusion-score-centering removes the drift that puts into the PPO gradient (see
docs/advanced/sde-backend.md §5). Rollout tensors travel through the output store. 30 prompts
x 8 samples per step, two SDE steps per sample (steps 3-4 of 10), micro-batch 2, no gradient
checkpointing.

Usage:
    python3 scripts/run_diffusion_grpo_krea2_async.py
    python3 scripts/run_diffusion_grpo_krea2_async.py --no-score-centering   # plain async baseline
    python3 scripts/run_diffusion_grpo_krea2_async.py --four-gpu-ci
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "krea/Krea-2-Raw"
DATASET = "rockdu/miles-diffusion-datasets"
SUBSET = "flowgrpo_ocr"
WANDB_PROJECT = "miles-diffusion-grpo"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 300
    data_dir: str = "/root/datasets"
    extra_args: str = ""
    score_centering: bool = True
    # CI cap: 3 training + 1 rollout GPUs with 3 prompts per step
    four_gpu_ci: bool = False


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{SUBSET}/**", data_dir=args.data_dir)
    return f"{local_dir}/{SUBSET}"


def execute(args: ScriptArgs, data_dir: str) -> None:
    run_name = f"diffusion_grpo_krea2_ocr_async_{U.create_run_id()}"
    num_train_gpus, num_rollout_gpus = (3, 1) if args.four_gpu_ci else (4, 4)
    # 30 x 8 = 240 samples per step, the nearest to 32 x 8 that both 4 and 6 trainer ranks divide
    num_prompts = 3 if args.four_gpu_ci else 30
    num_gpus = num_train_gpus + num_rollout_gpus

    ckpt_args = (
        f"--hf-checkpoint {MODEL} --diffusion-task t2i --save {args.output_dir}/{run_name}/ckpt --save-interval 20 "
    )

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        f"--num-rollout {args.num_rollout} "
        "--num-steps-per-rollout 1 "
        "--diffusion-num-steps 10 "
        "--diffusion-guidance-scale 1.0 "
        "--diffusion-noise-level 0.7 "
        "--diffusion-sde-type sde "
        "--diffusion-height 1024 "
        "--diffusion-width 1024 "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window "
        "--diffusion-num-sde-steps 2 "
        "--diffusion-sde-window-range 3,5 "
        # score centering reads the rollout's per-step means and noise stds from the debug tensors
        "--diffusion-debug-mode "
        "--rollout-microgroup-size 1 "
        f"--rollout-batch-size {num_prompts} --n-samples-per-prompt 8 "
    )

    eval_args = (
        "--diffusion-eval-num-steps 52 --skip-eval-before-train "
        # the 1018-prompt OCR eval is slow, so it runs every 100 rollouts
        f"--eval-prompt-data ocr_test {data_dir}/test.jsonl --eval-interval 100 "
    )

    grpo_args = (
        "--loss-type policy_loss "
        "--advantage-estimator grpo "
        "--globalize-reward-std "
        "--diffusion-clip-range 1e-4 "
        "--diffusion-kl-beta 0.0 "
    ) + ("--diffusion-score-centering " if args.score_centering else "")

    optimizer_args = "--lr 3e-4 --adam-beta2 0.999 --weight-decay 1e-4 --clip-grad 1.0 "

    # without --colocate the adapters sync over NCCL
    lora_args = "--use-lora --lora-rank 32 --lora-alpha 64 --lora-init-weights gaussian "

    wandb_args = U.get_default_wandb_args(
        __file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=8, wandb_log_image_interval=10
    )

    # the launcher starts the Mooncake master on the --output-store-mooncake-master port
    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 8 "
        "--sglang-lora-merge-mode dynamic "
        "--sglang-quantization fp8 "
        "--sglang-dit-precision bf16 "
        "--sglang-vae-slicing "
        "--update-weight-buffer-size 2147483648 "
        "--use-output-store --output-store-mooncake-master 127.0.0.1:50070 "
    )

    train_backend_args = "--train-backend fsdp --diffusion-forward-dtype bf16 --micro-batch-size 2 "

    misc_args = (
        f"--actor-num-gpus-per-node {num_train_gpus} "
        f"--rollout-num-gpus {num_rollout_gpus} "
        "--rollout-num-gpus-per-engine 1 "
        f"--num-gpus-per-node {num_gpus} "
        "--deterministic-mode "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {eval_args} {grpo_args} "
            f"{optimizer_args} {lora_args} --rm-type ocr {wandb_args} {sglang_args} "
            f"{train_backend_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=num_gpus,
        train_script="train_diffusion_async.py",
        config=args,
        extra_env_vars={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    data_dir = prepare(args)
    execute(args, data_dir)


if __name__ == "__main__":
    typer.run(main)
