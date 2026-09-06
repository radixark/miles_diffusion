"""Cosmos3-Nano T2I GRPO with PickScore, fully colocated on 4 GPUs.

pretrained = nvidia/Cosmos3-Nano (16B MoT: 8B UND tower frozen, 8B GEN tower
trained via LoRA r64), 832x480 single frame, num_steps=16, eval_steps=35,
guidance 4.0, Flow-SDE noise_level=0.7, no KL, per-prompt mean + global std.

Layout: train, rollout and PickScore reward all share the same 4 GPUs
(--colocate --colocate-reward, one PickScore worker per rollout engine).

SDE schedule: epoch_global_random_choice draws 2 steps per epoch from
candidates 8-11. The Cosmos3 checkpoint ships a Karras flow-sigma grid whose
head steps 1-7 sit at sigma>0.96 with |dt|<0.02 and train nothing; steps 8-11
are the true high-noise segment (sigma 0.94-0.80). Step numbers are NOT
transferable across sigma-grid families - re-derive candidates from |dt| when
changing model/grid.

Pacing: lr 1e-4 x 1 optimizer step per rollout (the whole rollout is one
batch). CFG amplifies per-step policy displacement, so training with
guidance > 1 needs this slower pacing than a comparable CFG-free recipe.

--diffusion-recompute-old-log-prob: the trainer recomputes old log-probs at
rollout ingestion so the PPO ratio is implementation-self-consistent (rollout
fa kernels vs train SDPA would otherwise leak into the ratio). With 1 step per
rollout this makes every optimizer step exactly on-policy.

Usage:
    python3 scripts/run_diffusion_grpo_cosmos3_pickscore_t2i_4gpu.py
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "nvidia/Cosmos3-Nano"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_pickscore"
WANDB_PROJECT = "miles-diffusion-grpo"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 10000
    data_dir: str = "/root/datasets"
    extra_args: str = ""


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{DATASET_SUBSET}/**", data_dir=args.data_dir)
    return f"{local_dir}/{DATASET_SUBSET}"


def execute(args: ScriptArgs, data_dir: str) -> None:
    run_name = f"diffusion_grpo_cosmos3_pickscore_t2i_4gpu_{U.create_run_id()}"

    ckpt_args = f"--hf-checkpoint {MODEL} --save {args.output_dir}/{run_name}/ckpt --save-interval 10 "

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        "--rollout-batch-size 48 "
        "--n-samples-per-prompt 16 "
        f"--num-rollout {args.num_rollout} "
        "--num-steps-per-rollout 1 "
        # The Cosmos3 transformer is a packed-sequence single-sample interface;
        # one request cannot batch multiple outputs.
        "--rollout-microgroup-size 1 "
        "--micro-batch-size 1 "
    )

    diffusion_args = (
        "--diffusion-num-steps 16 "
        "--diffusion-output-num-frames 1 "
        "--diffusion-guidance-scale 4.0 "
        "--diffusion-noise-level 0.7 "
        "--diffusion-height 480 "
        "--diffusion-width 832 "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_random_choice "
        "--diffusion-num-sde-steps 2 "
        "--diffusion-sde-candidate-steps 8,9,10,11 "
        "--diffusion-recompute-old-log-prob "
    )

    eval_args = (
        f"--eval-prompt-data pickscore_test {data_dir}/test.jsonl "
        "--eval-interval 30 "
        "--diffusion-eval-num-steps 35 "
        "--skip-eval-before-train "
    )

    grpo_args = "--advantage-estimator grpo --globalize-reward-std --diffusion-clip-range 1e-3 "

    optimizer_args = "--lr 1e-4 --adam-beta2 0.999 --weight-decay 1e-4 "

    # UND/GEN towers share layers and differ by parameter name (to_q vs
    # add_q_proj, mlp vs mlp_moe_gen); LoRA targeting defaults to the GEN
    # fragments in the cosmos3 train pipeline config.
    lora_args = "--use-lora --lora-ipc-weight-sync --lora-rank 64 --lora-alpha 128 --lora-init-weights gaussian "

    reward_args = (
        "--rm-type pickscore "
        "--colocate-reward "
        "--pickscore-num-workers 4 "
        "--pickscore-batch-size 8 "
        "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
        "--pickscore-model-path yuvalkirstain/PickScore_v1 "
    )

    wandb_args = U.get_default_wandb_args(
        __file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=8, wandb_log_image_interval=10
    )

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 8 "
        "--update-weight-buffer-size 2147483648 "
        "--update-weight-target-module transformer "
    )

    train_backend_args = (
        "--train-backend fsdp --fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 --diffusion-forward-dtype bf16 "
    )

    misc_args = (
        "--actor-num-gpus-per-node 4 "
        "--rollout-num-gpus 4 "
        "--rollout-num-gpus-per-engine 1 "
        "--num-gpus-per-node 4 "
        "--colocate "
    )

    debug_args = "--diffusion-debug-mode "

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {diffusion_args} {eval_args} {grpo_args} {optimizer_args} "
            f"{lora_args} {reward_args} {wandb_args} {sglang_args} {train_backend_args} {misc_args} "
            f"{debug_args} {args.extra_args}"
        ),
        num_gpus_per_node=4,
        config=args,
        extra_env_vars={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            # RL rollout scores raw samples; skip the serving-side guardrail models.
            "SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1",
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    data_dir = prepare(args)
    execute(args, data_dir)


if __name__ == "__main__":
    typer.run(main)
