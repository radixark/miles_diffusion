"""LTX-2.3 video VideoAlign GRPO: 4-GPU FSDP train + sglang rollout colocate, 1 reward GPU.

57-frame 512x768 video at 24 fps, 24 denoising steps, CPS-SDE with 3 trainable steps drawn
per epoch from candidate steps 0-9. Everything runs bf16 end to end — master, reduce,
forward and the sgl-d engine — on the sdpa_math attention backend.

VideoAlign scores 8 uniformly sampled frames per clip. The optimized scalar is the sum
of normalized VQ, MQ and TA; each head is also logged separately. The default two
rollouts are an integration smoke test, not a verified training curve.

Video rollouts take minutes per request, so the health checker gets a far longer interval
and failure budget than the image recipes.

Usage:
    python3 scripts/run_diffusion_grpo_ltx23_videoalign_sglang.py
"""

from dataclasses import dataclass
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "Lightricks/LTX-2.3"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_pickscore"
WANDB_PROJECT = "miles-diffusion-grpo"

EVAL_SUBSET_SIZE = 64
EVAL_SUBSET_SEED = 0


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 2
    data_dir: str = "/root/datasets"
    extra_args: str = ""


def _write_eval_subset(subset_dir: str) -> str:
    """Sample a fixed eval subset from test.jsonl with a deterministic seed.

    The same held-out prompts are used for every eval, so reward curves stay
    comparable across checkpoints.
    """
    import random

    src = Path(subset_dir) / "test.jsonl"
    out = src.with_name(f"test_eval_{EVAL_SUBSET_SIZE}.jsonl")
    if out.exists():
        return str(out)
    lines = src.read_text(encoding="utf-8").splitlines()
    chosen = random.Random(EVAL_SUBSET_SEED).sample(lines, EVAL_SUBSET_SIZE)
    out.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    return str(out)


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{DATASET_SUBSET}/**", data_dir=args.data_dir)
    subset_dir = f"{local_dir}/{DATASET_SUBSET}"
    _write_eval_subset(subset_dir)
    return subset_dir


def execute(args: ScriptArgs, data_dir: str) -> None:
    run_name = f"diffusion_grpo_ltx23_videoalign_{U.create_run_id()}"

    ckpt_args = (
        f"--hf-checkpoint {MODEL} --diffusion-task t2v --save {args.output_dir}/{run_name}/ckpt --save-interval 10 "
    )

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        f"--num-rollout {args.num_rollout} "
        "--num-steps-per-rollout 2 "
        "--rollout-microgroup-size 1 "
        "--train-dp-split-mode stride "
        "--diffusion-train-iter-order sample_major "
        "--rollout-patch-group ltx "
        "--diffusion-num-steps 24 "
        "--diffusion-output-num-frames 57 "
        "--diffusion-fps 24 "
        "--diffusion-guidance-scale 1.0 "
        "--diffusion-noise-level 0.8 "
        "--diffusion-height 512 "
        "--diffusion-width 768 "
        "--diffusion-sde-type cps "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_random_choice "
        "--diffusion-num-sde-steps 3 "
        "--diffusion-sde-candidate-steps 0,1,2,3,4,5,6,7,8,9 "
    )

    grpo_args = (
        "--advantage-estimator grpo --globalize-reward-std --diffusion-clip-range 1e-5 --diffusion-kl-beta 0.0 "
    )

    optimizer_args = "--lr 2e-4 --adam-beta2 0.999 --weight-decay 1e-4 "

    lora_args = "--use-lora --lora-rank 64 --lora-alpha 128 --lora-init-weights gaussian "

    reward_args = (
        "--rm-type videoalign "
        "--reward-key overall "
        "--videoalign-num-frames 8 "
        "--videoalign-num-gpus-per-worker 1.0 "
        "--videoalign-num-workers 1 "
        "--videoalign-batch-size 4 "
    )

    eval_args = (
        "--eval-interval 20 "
        f"--eval-prompt-data test {data_dir}/test_eval_{EVAL_SUBSET_SIZE}.jsonl "
        "--eval-reward-key overall "
        "--diffusion-eval-guidance-scale 3.0 "
    )

    wandb_args = U.get_default_wandb_args(
        __file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=4, wandb_log_image_interval=10
    )

    sglang_args = (
        "--use-miles-router "
        "--rollout-fetch-in-parser "
        "--sglang-server-concurrency 4 "
        "--sglang-lora-merge-mode merge "
        "--sglang-attention-backend torch_sdpa "
        "--sglang-dit-precision bf16 "
        "--update-weight-buffer-size 2147483648 "
    )

    train_backend_args = (
        "--train-backend fsdp "
        "--fsdp-master-dtype bf16 "
        "--fsdp-reduce-dtype bf16 "
        "--diffusion-forward-dtype bf16 "
        "--fsdp-attention-backend sdpa_math "
    )

    perf_args = (
        "--gradient-checkpointing "
        "--micro-batch-size-sample 1 "
        "--micro-batch-size-tstep 1 "
        "--rollout-parser-num-workers 16 "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
        "--rollout-num-gpus 4 "
        "--rollout-num-gpus-per-engine 1 "
        "--num-gpus-per-node 4 "
        "--colocate "
        "--deterministic-mode "
        "--rollout-health-check-interval 120 "
        "--miles-router-health-check-failure-threshold 30 "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {grpo_args} {optimizer_args} {lora_args} "
            f"{reward_args} {eval_args} {wandb_args} {sglang_args} {train_backend_args} {perf_args} {misc_args} "
            f"{args.extra_args}"
        ),
        num_gpus_per_node=5,
        config=args,
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    data_dir = prepare(args)
    execute(args, data_dir)


if __name__ == "__main__":
    typer.run(main)
