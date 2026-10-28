"""FLUX.3 Image LoRA SFT on a (image, prompt) jsonl, encoded by an sglang-diffusion --disagg-encode-only server.

Dataset rows: {"prompt": "...", "target": {"visual": "/abs/path.png"}}

Usage:
    python3 scripts/run_diffusion_sft_flux3_image.py --data-jsonl /abs/data.jsonl --encode-url http://127.0.0.1:30000
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

WANDB_PROJECT = "miles-diffusion-sft"
# The package's sampler.alpha, so training sigmas follow the inference schedule.
FLOW_SHIFT = 6.93


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    data_jsonl: str = ""
    encode_url: str = ""
    model_path: str = "black-forest-labs/flux-3-image-klein-base"
    num_gpus: int = 4
    extra_args: str = ""


def execute(args: ScriptArgs) -> None:
    if not args.data_jsonl or not args.encode_url:
        raise SystemExit("set --data-jsonl and --encode-url")
    run_name = f"diffusion_sft_flux3_image_{U.create_run_id()}"

    ckpt_args = (
        f"--hf-checkpoint {args.model_path} --diffusion-task t2i --sft-encode-url {args.encode_url} "
        f"--save {args.output_dir}/{run_name}/ckpt --save-interval 20 "
    )

    rollout_args = (
        "--rollout-function-path miles.rollout.sft_rollout.generate_rollout "
        f"--prompt-data {args.data_jsonl} "
        "--input-key prompt "
        "--rollout-batch-size 16 "
        "--num-epoch 3 "
        "--num-steps-per-rollout 1 "
        "--diffusion-height 1024 "
        "--diffusion-width 1024 "
    )

    sft_args = (
        "--loss-type sft_loss "
        "--train-only "
        "--custom-convert-samples-to-train-data-path miles.rollout.sft_rollout.convert_samples_to_train_data "
        "--custom-rollout-log-function-path miles.rollout.sft_rollout.log_rollout_data "
        "--custom-prepare-train-batch-path miles.backends.fsdp_utils.loss_hub.sft.prepare_sft_batch "
        "--custom-loss-function-path miles.backends.fsdp_utils.loss_hub.sft.sft_loss_formula "
    )

    optimizer_args = "--lr 1e-4 --adam-beta2 0.999 --weight-decay 1e-4 --clip-grad 1.0 "

    lora_args = "--use-lora --lora-rank 32 --lora-alpha 64 --lora-init-weights gaussian "

    wandb_args = U.get_default_wandb_args(__file__, run_id=run_name, project=WANDB_PROJECT)

    train_backend_args = (
        "--train-backend fsdp "
        "--fsdp-master-dtype fp32 "
        "--fsdp-reduce-dtype fp32 "
        "--diffusion-forward-dtype bf16 "
        "--fsdp-attention-backend _flash_3 "
        f"--fsdp-flow-shift {FLOW_SHIFT} "
    )

    perf_args = "--gradient-checkpointing --micro-batch-size 1 "

    misc_args = f"--actor-num-gpus-per-node {args.num_gpus} --num-gpus-per-node {args.num_gpus} "

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {sft_args} {optimizer_args} {lora_args} "
            f"{wandb_args} {train_backend_args} {perf_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=args.num_gpus,
        config=args,
        train_script="train_sft.py",
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    execute(args)


if __name__ == "__main__":
    typer.run(main)
