"""4-GPU Wan2.2-T2V-A14B dual-expert LoRA SFT on a (video, prompt) jsonl dataset.

No sglang engines: the sft_rollout plugin lazily encodes each round's cache misses through
a colocated encoder actor pool, writing one content-addressed file per sample into
.sft_cache/ next to the jsonl. Epoch 2 onward is all cache hits.

Dataset rows: {"prompt": "...", "metadata": {"video": "/abs/path.mp4"}}

Per rollout step: 64 samples, num_steps_per_rollout=4, so 16 samples per optimizer step
over 4 dp ranks is 4 samples per rank at mbs=1.

Usage:
    MILES_SCRIPT_DATA_JSONL=/abs/data.jsonl python3 scripts/run_diffusion_sft_wan22.py
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
WANDB_PROJECT = "miles-diffusion-sft"

# Wan2.2 DiT LoRA targets: self-attn (attn1), cross-attn (attn2), and FFN.
LORA_TARGET_MODULES = (
    "attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0 "
    "attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0 "
    "ffn.net.0.proj ffn.net.2"
)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    cuda_visible_devices: str = "0,1,2,3"
    data_jsonl: str = ""
    resume_ckpt: str = ""
    start_rollout: int = -1
    extra_args: str = ""


def execute(args: ScriptArgs) -> None:
    if not args.data_jsonl:
        raise SystemExit("set --data-jsonl (or MILES_SCRIPT_DATA_JSONL) to a jsonl with prompt + metadata.video")
    run_name = f"diffusion_sft_wan22_{U.create_run_id()}"

    ckpt_args = (
        f"--hf-checkpoint {MODEL} --sft-encoder-checkpoint {MODEL} "
        f"--save {args.output_dir}/{run_name}/ckpt --save-interval 20 "
    )
    if args.resume_ckpt:
        ckpt_args += f"--load {args.resume_ckpt} "
        if args.start_rollout >= 0:
            ckpt_args += f"--start-rollout-id {args.start_rollout} "

    rollout_args = (
        "--rollout-function-path miles.rollout.sft_rollout.generate_rollout "
        f"--prompt-data {args.data_jsonl} "
        "--input-key prompt "
        "--rollout-batch-size 64 "
        "--num-epoch 3 "
        "--num-steps-per-rollout 4 "
        "--diffusion-height 480 "
        "--diffusion-width 832 "
        "--diffusion-output-num-frames 81 "
    )

    sft_args = (
        "--loss-type sft_loss "
        "--train-only "
        "--custom-convert-samples-to-train-data-path miles.rollout.sft_rollout.convert_samples_to_train_data "
        "--custom-rollout-log-function-path miles.rollout.sft_rollout.log_rollout_data "
        "--custom-prepare-train-batch-path miles.backends.fsdp_utils.loss_hub.sft.prepare_sft_batch "
        "--custom-loss-function-path miles.backends.fsdp_utils.loss_hub.sft.sft_loss_formula "
        "--sft-frame-stride 2 "
    )

    optimizer_args = "--lr 1e-4 --adam-beta2 0.999 --weight-decay 1e-4 "

    lora_args = (
        "--use-lora "
        "--lora-rank 64 "
        "--lora-alpha 128 "
        f"--lora-target-modules {LORA_TARGET_MODULES} "
        "--lora-init-weights gaussian "
    )

    wandb_args = U.get_default_wandb_args(__file__, run_id=run_name, project=WANDB_PROJECT)

    train_backend_args = (
        "--train-backend fsdp "
        "--fsdp-master-dtype fp32 "
        "--fsdp-reduce-dtype fp32 "
        "--diffusion-forward-dtype bf16 "
        "--fsdp-flow-shift 3.0 "
        "--update-weight-target-module transformer,transformer_2 "
    )

    perf_args = "--micro-batch-size 1 "

    misc_args = "--actor-num-gpus-per-node 4 --num-gpus-per-node 4 "

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {sft_args} {optimizer_args} {lora_args} "
            f"{wandb_args} {train_backend_args} {perf_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=4,
        config=args,
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    execute(args)


if __name__ == "__main__":
    typer.run(main)
