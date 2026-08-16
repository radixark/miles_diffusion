"""17-GPU scale-down of the low-memory Wan2.2 high-noise GRPO recipe.

Derived from scripts/run_diffusion_grpo_wan22_pickscore_17gpu_multinode.py (PR #173):
same multi-node submit path (existing ray cluster, MILES_SCRIPT_EXTERNAL_RAY=1), same
2×8 colocate train+rollout layout plus one dedicated reward GPU shared by four
PickScore workers.

LoRA r64 for 40GB GPUs. Trainer: 2 FSDP hybrid replicas ×
Ulysses SP2 + gradient checkpointing (SP2 alone peaks at 39GB activations).
Engines are TP4 (not the sgl-d auto
tp1/sp2/cfg2 split, which leaves ~70GB of unsharded weights on every GPU).
T5 + VAE stay on CPU. The low-noise `transformer_2` uses component offload
during rollout so both experts do not reside on GPU together. Only `transformer`
is trained; SDE candidates 1,2. 13-frame 480P (480×832), no CFG, microgroup 1.

    24 prompts × 16 samples = 192 global_batch × 2 steps_per_rollout
    dp_size = 16/2 = 8 → 24 pairs/rank, micro-batch-size-sample 1 → 24 GA
    rollout microgroup 1

Usage (after bringing up the 16+1 GPU ray cluster):
    MILES_SCRIPT_EXTERNAL_RAY=1 python3 scripts/run_diffusion_grpo_wan22_pickscore_low_memory_17gpu.py
    MILES_SCRIPT_EXTERNAL_RAY=1 python3 scripts/run_diffusion_grpo_wan22_pickscore_low_memory_17gpu.py --num-rollout 1
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_pickscore"
WANDB_PROJECT = "miles-diffusion-grpo"

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
    assert U.get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY"), (
        "This recipe needs a 17-GPU ray cluster (16 train/rollout + 1 reward); "
        "bring it up first, then run with MILES_SCRIPT_EXTERNAL_RAY=1."
    )

    run_name = f"diffusion_grpo_wan22_pickscore_low_memory_17gpu_{U.create_run_id()}"

    ckpt_args = f"--hf-checkpoint {MODEL} --save {args.output_dir}/{run_name}/ckpt --save-interval 100 "

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        "--rollout-batch-size 24 "
        "--n-samples-per-prompt 16 "
        f"--num-rollout {args.num_rollout} "
        "--num-steps-per-rollout 2 "
        "--rollout-microgroup-size 1 "
        "--micro-batch-size-sample 1 "
        "--micro-batch-size-tstep 1 "
        "--diffusion-num-steps 10 "
        "--diffusion-output-num-frames 13 "
        "--diffusion-guidance-scale 1.0 "
        "--diffusion-guidance-scale-2 1.0 "
        "--diffusion-noise-level 0.9 "
        "--diffusion-height 480 "
        "--diffusion-width 832 "
        "--diffusion-flow-shift 3.0 "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_random_choice "
        "--diffusion-num-sde-steps 1 "
        "--diffusion-sde-candidate-steps 1,2 "
        "--diffusion-recompute-old-log-prob "
    )

    eval_args = (
        f"--eval-prompt-data pickscore_test {data_dir}/test.jsonl "
        "--eval-interval 100 "
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
        "--pickscore-num-workers 4 "
        "--pickscore-num-gpus-per-worker 0.25 "
        "--pickscore-batch-size 8 "
        "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
        "--pickscore-model-path yuvalkirstain/PickScore_v1 "
    )

    wandb_args = U.get_default_wandb_args(
        __file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=4, wandb_log_image_interval=5
    )

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 8 "
        "--miles-router-health-check-failure-threshold 30 "
        "--update-weight-buffer-size 2147483648 "
        "--sglang-tp-size 4 "
        "--sglang-sp-degree 1 "
        "--sglang-vae-cpu-offload "
        "--sglang-text-encoder-cpu-offload "
        "--sglang-cpu-offload-components transformer_2 "
    )

    train_backend_args = (
        "--train-backend fsdp --fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 --diffusion-forward-dtype bf16 "
        "--gradient-checkpointing "
        "--update-weight-target-module transformer "
    )

    topology_args = (
        "--actor-num-nodes 2 "
        "--actor-num-gpus-per-node 8 "
        "--num-gpus-per-node 8 "
        "--rollout-num-gpus 16 "
        "--rollout-num-gpus-per-engine 4 "
        "--dp-replicate-size 2 "
        "--sequence-parallel-size 2 "
        "--ulysses-degree 2 "
        "--colocate "
        "--diffusion-debug-mode "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {eval_args} {grpo_args} {optimizer_args} "
            f"{lora_args} {reward_args} {wandb_args} {sglang_args} {train_backend_args} "
            f"{topology_args} {args.extra_args}"
        ),
        num_gpus_per_node=8,
        config=args,
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"},
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    data_dir = prepare(args)
    execute(args, data_dir)


if __name__ == "__main__":
    typer.run(main)
