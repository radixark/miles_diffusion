"""MiniMax H3 t2va PickScore Flow-GRPO: 2-GPU FSDP train + sglang rollout, reward colocated.

Ported from the verl-omni DanceGRPO task18 recipe. Two of its settings cannot be
reproduced here and are deliberately different:
  * Resolution/frames: sglang H3 pins short_edge=768 and duration >= 4s with 17n+5
    frame alignment at 24 fps, so the smallest legal spec is 1344x768 / 107 frames.
    verl's 480x864 / 56f (2.33s) is not a valid H3 request.
  * 2 GPUs instead of 8, so batch x group is smaller than verl's 8x16.

The SDE window starts at step 1, not 0: the flow-SDE diffusion factor carries a
1/(1-sigma) that is singular at the first step (sigma=1), where the rollout engine
silently emits NaN latents. Excluding that one step is what lets this run use sde
(matching verl) instead of falling back to cps.

Aligned with verl: LoRA 64/128, lr 1e-4, weight_decay 1e-4, adam_eps 1e-15,
noise_level 0.7, 10 inference steps, SDE window size 2, flow_shift 12.0 /
audio_flow_shift 3.0, PickScore over 8 frames.

Engine concurrency is 2, not 1: at 1 the engine idles through the encode and handoff
of a finished sample before the next denoise starts, and 2 overlaps them. Higher does not
help -- the extra requests queue inside the engine and only stretch per-sample latency.

The router health-check window is widened well past its 30s x 3 default: H3 denoising
blocks the engine's uvicorn event loop for ~43s per sample, so /health cannot answer
while a request runs. A 16-sample eval keeps the loop busy for ~800s straight, which
trips the default threshold and quarantines a perfectly healthy engine permanently
(router.py marks DEAD with no revive path).

Batch math: rollout_batch_size prompts x n_samples_per_prompt samples per rollout,
split into num_steps_per_rollout optimizer steps. The GRPO group is
n_samples_per_prompt, since advantages are standardized within a prompt's group, so a
small group leaves a weak, noisy signal. eval_interval and save_interval count rollout
iterations, not optimizer steps.

Usage:
    python3 scripts/run_diffusion_grpo_h3_t2va_2gpu.py
    python3 scripts/run_diffusion_grpo_h3_t2va_2gpu.py --num-rollout 5 --cuda-visible-devices 0,3

    # rollout-only smoke: sglang rollout + reward, no FSDP train
    python3 scripts/run_diffusion_grpo_h3_t2va_2gpu.py --num-rollout 1 \
        --n-samples-per-prompt 1 --eval-interval 0 --extra-args "--debug-rollout-only"

    # train/rollout alignment diagnostic: freeze the weights so log_prob_mean_abs_diff
    # measures pure train-vs-rollout deviation rather than parameter drift
    python3 scripts/run_diffusion_grpo_h3_t2va_2gpu.py --num-rollout 2 \
        --eval-interval 0 --extra-args "--debug-skip-optimizer-step"
"""

from dataclasses import dataclass
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "MiniMaxAI/MiniMax-H3"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_pickscore"
WANDB_PROJECT = "miles-diffusion-grpo"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 30
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 8
    num_steps_per_rollout: int = 2
    eval_interval: int = 10
    # miles has no eval-sample cap (verl used val_max_samples=64), so a fixed slice of
    # the test split stands in: all 2048 prompts would take >60h at 20 steps here.
    eval_size: int = 16
    save_interval: int = 10
    data_dir: str = "/root/datasets"
    extra_args: str = ""


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{DATASET_SUBSET}/**", data_dir=args.data_dir)
    return f"{local_dir}/{DATASET_SUBSET}"


def _eval_slice(prompt_dir: str, eval_size: int) -> str:
    eval_data = Path(prompt_dir) / f"val_{eval_size}.jsonl"
    if not eval_data.exists():
        with open(Path(prompt_dir) / "test.jsonl") as f:
            lines = [next(f) for _ in range(eval_size)]
        eval_data.write_text("".join(lines))
    return str(eval_data)


def execute(args: ScriptArgs, prompt_dir: str) -> None:
    run_name = f"diffusion_grpo_h3_t2va_{U.create_run_id()}"

    ckpt_args = (
        f"--hf-checkpoint {MODEL} "
        f"--save {args.output_dir}/{run_name}/ckpt "
        f"--save-interval {args.save_interval} "
    )

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {prompt_dir}/train.jsonl "
        "--input-key input "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--num-steps-per-rollout {args.num_steps_per_rollout} "
        f"--num-rollout {args.num_rollout} "
        "--rollout-microgroup-size 1 "
        "--micro-batch-size-sample 1 "
        "--micro-batch-size-tstep 1 "
        "--diffusion-train-iter-order sample_major "
    )

    diffusion_args = (
        "--diffusion-num-steps 10 "
        "--diffusion-guidance-scale 1.0 "
        "--diffusion-noise-level 0.7 "
        "--diffusion-sde-type sde "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window "
        "--diffusion-num-sde-steps 2 "
        "--diffusion-sde-window-range 1,4 "
        "--diffusion-h3-aspect-ratio 16:9 "
        "--diffusion-h3-duration-seconds 4 "
        "--diffusion-audio-flow-shift 3.0 "
        "--diffusion-flow-shift 12.0 "
    )

    eval_args = ""
    if args.eval_interval:
        eval_args = (
            f"--eval-interval {args.eval_interval} "
            f"--eval-prompt-data pickscore_val {_eval_slice(prompt_dir, args.eval_size)} "
            "--n-samples-per-eval-prompt 1 "
            "--diffusion-eval-num-steps 20 "
        )

    grpo_args = (
        "--advantage-estimator grpo --globalize-reward-std --diffusion-clip-range 1e-4 --diffusion-kl-beta 0.0 "
    )

    optimizer_args = "--lr 1e-4 --weight-decay 1e-4 --adam-eps 1e-15 "

    # H3's rollout DiT renames modules and fuses Q/K/V, so weights only reach the engine
    # through the LoRA IPC path's layer grouper; the family rejects any other sync mode.
    lora_args = "--use-lora --lora-ipc-weight-sync --lora-rank 64 --lora-alpha 128 "

    reward_args = (
        "--rm-type pickscore "
        "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
        "--pickscore-model-path yuvalkirstain/PickScore_v1 "
        "--pickscore-num-frames 8 "
        "--pickscore-num-workers 1 "
        "--pickscore-num-gpus-per-worker 0 "
        "--pickscore-batch-size 4 "
        "--rollout-parser-num-workers 2 "
    )

    wandb_args = U.get_default_wandb_args(__file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=4)

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 2 "
        "--sglang-tp-size 2 "
        "--sglang-sp-degree 1 "
        "--sglang-ulysses-degree 1 "
        "--sglang-ring-degree 1 "
        "--sglang-dit-precision bf16 "
        "--update-weight-buffer-size 2147483648 "
    )

    train_backend_args = (
        "--train-backend fsdp "
        "--fsdp-master-dtype bf16 "
        "--fsdp-reduce-dtype bf16 "
        "--diffusion-forward-dtype bf16 "
    )

    perf_args = "--gradient-checkpointing "

    misc_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 2 "
        "--rollout-num-gpus 2 "
        "--rollout-num-gpus-per-engine 2 "
        "--num-gpus-per-node 2 "
        "--colocate "
        "--colocate-reward "
        "--rollout-health-check-interval 60 "
        "--miles-router-health-check-failure-threshold 30 "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {diffusion_args} {eval_args} {grpo_args} {optimizer_args} "
            f"{lora_args} {reward_args} {wandb_args} {sglang_args} {train_backend_args} {perf_args} "
            f"{misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=2,
        config=args,
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    prompt_dir = prepare(args)
    execute(args, prompt_dir)


if __name__ == "__main__":
    typer.run(main)
