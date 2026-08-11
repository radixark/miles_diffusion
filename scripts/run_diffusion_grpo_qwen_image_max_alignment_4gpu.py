"""Maximally aligned Qwen-Image train<->rollout run: model_output_*_abs_diff in debug mode.

Every optimizer step is on-policy (num_steps_per_rollout=1) and the train micro-batch
reproduces one rollout microgroup exactly (micro_batch_size_sample == rollout_microgroup_size,
contiguous dp split). The remaining diff comes from the legacy whole-window cond pad and its
attention mask, which move the train forward off the rollout's mask-less SDPA flash kernel.

Layout: 4 GPUs, train + rollout + pickscore reward all colocated.

Usage:
    python3 scripts/run_diffusion_grpo_qwen_image_max_alignment_4gpu.py
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "Qwen/Qwen-Image"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_pickscore"
WANDB_PROJECT = "miles-diffusion-grpo"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    cuda_visible_devices: str = "4,5,6,7"
    num_rollout: int = 3
    data_dir: str = "/root/datasets"
    extra_args: str = ""


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{DATASET_SUBSET}/**", data_dir=args.data_dir)
    return f"{local_dir}/{DATASET_SUBSET}"


def execute(args: ScriptArgs, data_dir: str) -> None:
    run_name = f"diffusion_grpo_qwen_image_max_alignment_4gpu_{U.create_run_id()}"

    ckpt_args = f"--hf-checkpoint {MODEL} --save {args.output_dir}/{run_name}/ckpt --save-interval 10 "

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 16 "
        f"--num-rollout {args.num_rollout} "
        "--num-steps-per-rollout 1 "
        "--rollout-microgroup-size 8 "
        "--diffusion-train-iter-order sample_major "
        "--diffusion-num-steps 10 "
        "--diffusion-guidance-scale 4.0 "
        "--diffusion-true-cfg-scale 4.0 "
        "--diffusion-noise-level 1.2 "
        "--diffusion-height 512 "
        "--diffusion-width 512 "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window "
        "--diffusion-num-sde-steps 2 "
        "--diffusion-sde-window-range 3,5 "
        "--rollout-patch-group qwen_image "
    )

    grpo_args = "--advantage-estimator grpo --globalize-reward-std --diffusion-clip-range 1e-4 "

    optimizer_args = "--lr 3e-4 --adam-beta2 0.999 --weight-decay 1e-4 "

    lora_args = "--use-lora --lora-ipc-weight-sync --lora-rank 64 --lora-alpha 128 --lora-init-weights gaussian "

    reward_args = (
        "--rm-type pickscore "
        "--pickscore-num-workers 1 "
        "--pickscore-batch-size 8 "
        "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
        "--pickscore-model-path yuvalkirstain/PickScore_v1 "
    )

    wandb_args = U.get_default_wandb_args(
        __file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=8, wandb_log_image_interval=10
    )

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 4 "
        "--sglang-attention-backend torch_sdpa "
        "--update-weight-buffer-size 2147483648 "
    )

    train_backend_args = (
        "--train-backend fsdp --fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 --diffusion-forward-dtype bf16 "
    )

    perf_args = "--gradient-checkpointing --micro-batch-size-sample 8 --micro-batch-size-tstep 1 "

    misc_args = (
        "--actor-num-gpus-per-node 4 "
        "--rollout-num-gpus 4 "
        "--rollout-num-gpus-per-engine 1 "
        "--num-gpus-per-node 4 "
        "--colocate "
        "--colocate-reward "
        "--deterministic-mode "
        "--diffusion-debug-mode "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {grpo_args} {optimizer_args} "
            f"{lora_args} {reward_args} {wandb_args} {sglang_args} {train_backend_args} {perf_args} "
            f"{misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=4,
        config=args,
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    data_dir = prepare(args)
    execute(args, data_dir)


if __name__ == "__main__":
    typer.run(main)
