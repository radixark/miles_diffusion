"""Wan2.2-T2V-A14B dual-expert 5-frame video GRPO with PickScore, 4 train GPUs + 1 reward GPU.

resolution=480, num_frames=5, num_steps=10, eval_steps=28, flow_shift=3.0 overriding the
sgl-d serving default of 12.0, guidance 4.0 high-noise / 3.0 low-noise, Flow-SDE
noise_level=0.9, beta=0, per-prompt mean and std.

SDE schedule: epoch_global_random_choice draws ONE step per rollout, shared across the
batch, from candidate steps 1,2,3. At flow_shift=3.0 the dual-expert boundary is t=875, so
steps 1,2 train `transformer` (high-noise) and step 3 trains `transformer_2` (low-noise);
both experts get gradient stochastically and --update-weight-target-module syncs both.

Per rollout: 48 prompts x 16 samples = 768 samples; num_steps_per_rollout=2 gives 384 per
optimizer step over 4 train GPUs. micro-batch-size 2 keeps every micro-batch phase-pure
(one DiT, one CFG scale); 4 OOMs on H200.

Gradient checkpointing stays off: Wan2.2 under FSDP2 mixed precision hits a
torch.utils.checkpoint CheckpointError on the fp32 RoPE freq buffers. If you OOM, lower
--rollout-batch-size, --n-samples-per-prompt or --rollout-microgroup-size.

Usage:
    python3 scripts/run_diffusion_grpo_wan22_pickscore_5gpu.py
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_pickscore"
WANDB_PROJECT = "miles-diffusion-grpo"

# Wan2.2 DiT LoRA targets: self-attn (attn1), cross-attn (attn2), and FFN.
LORA_TARGET_MODULES = (
    "attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0 "
    "attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0 "
    "ffn.net.0.proj ffn.net.2"
)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 10000
    data_dir: str = "/root/datasets"
    extra_args: str = ""


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{DATASET_SUBSET}/**", data_dir=args.data_dir)
    return f"{local_dir}/{DATASET_SUBSET}"


def execute(args: ScriptArgs, data_dir: str) -> None:
    run_name = f"diffusion_grpo_wan22_pickscore_5gpu_{U.create_run_id()}"

    ckpt_args = f"--hf-checkpoint {MODEL} --save {args.output_dir}/{run_name}/ckpt --save-interval 10 "

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        "--rollout-batch-size 48 "
        "--n-samples-per-prompt 16 "
        f"--num-rollout {args.num_rollout} "
        "--num-steps-per-rollout 2 "
        "--rollout-microgroup-size 8 "
        "--diffusion-num-steps 10 "
        "--diffusion-output-num-frames 5 "
        "--diffusion-guidance-scale 4.0 "
        "--diffusion-guidance-scale-2 3.0 "
        "--diffusion-noise-level 0.9 "
        "--diffusion-height 480 "
        "--diffusion-width 480 "
        "--diffusion-flow-shift 3.0 "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_random_choice "
        "--diffusion-num-sde-steps 1 "
        "--diffusion-sde-candidate-steps 1,2,3 "
    )

    eval_args = (
        f"--eval-prompt-data pickscore_test {data_dir}/test.jsonl "
        "--eval-interval 30 "
        "--diffusion-eval-num-steps 28 "
        "--skip-eval-before-train "
    )

    grpo_args = "--advantage-estimator grpo --diffusion-clip-range 1e-4 "

    optimizer_args = "--lr 1e-4 --adam-beta2 0.999 --weight-decay 1e-4 "

    lora_args = (
        "--use-lora "
        "--lora-ipc-weight-sync "
        "--lora-rank 64 "
        "--lora-alpha 128 "
        f"--lora-target-modules {LORA_TARGET_MODULES} "
        "--lora-init-weights gaussian "
    )

    reward_args = (
        "--rm-type pickscore "
        "--pickscore-num-workers 1 "
        "--pickscore-num-gpus-per-worker 1.0 "
        "--pickscore-batch-size 8 "
        "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
        "--pickscore-model-path yuvalkirstain/PickScore_v1 "
    )

    wandb_args = U.get_default_wandb_args(
        __file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=8, wandb_log_image_interval=10
    )

    sglang_args = "--use-miles-router --sglang-server-concurrency 8 --update-weight-buffer-size 2147483648 "

    train_backend_args = (
        "--train-backend fsdp --fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 --diffusion-forward-dtype bf16 "
        "--update-weight-target-module transformer,transformer_2 "
    )

    perf_args = "--micro-batch-size 2 "

    misc_args = (
        "--actor-num-gpus-per-node 4 "
        "--rollout-num-gpus 4 "
        "--rollout-num-gpus-per-engine 1 "
        "--num-gpus-per-node 5 "
        "--colocate "
        "--diffusion-debug-mode "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {eval_args} {grpo_args} {optimizer_args} "
            f"{lora_args} {reward_args} {wandb_args} {sglang_args} {train_backend_args} {perf_args} "
            f"{misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=5,
        config=args,
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"},
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    data_dir = prepare(args)
    execute(args, data_dir)


if __name__ == "__main__":
    typer.run(main)
