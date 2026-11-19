"""Launch π₀.₅ LIBERO supervised fine-tuning through miles' existing Ray/FSDP SFT entry point."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import miles


def train_command(args, extra=()):
    command = [
        sys.executable,
        str(Path(miles.__file__).resolve().parents[1] / "train_sft.py"),
        "--hf-checkpoint",
        str(args.bundle.resolve()),
        "--diffusion-model-family",
        "pi05",
        "--diffusion-task",
        "vla",
        "--loss-type",
        "sft_loss",
        "--train-only",
        "--train-backend",
        "fsdp",
        "--update-weight-target-module",
        "policy",
        "--fsdp-master-dtype",
        "fp32",
        "--fsdp-reduce-dtype",
        "fp32",
        "--diffusion-forward-dtype",
        "fp32",
        "--fsdp-supervised-streams",
        "action",
        "--diffusion-guidance-scale",
        "1",
        "--rollout-function-path",
        "pi05_tools.libero_rollout.generate_rollout",
        "--custom-prepare-train-batch-path",
        "miles.backends.fsdp_utils.models.pi05.loss.prepare_batch",
        "--custom-loss-function-path",
        "miles.backends.fsdp_utils.loss_hub.sft.sft_loss_formula",
        "--custom-convert-samples-to-train-data-path",
        "miles.rollout.sft_rollout.convert_samples_to_train_data",
        "--custom-rollout-log-function-path",
        "miles.rollout.sft_rollout.log_rollout_data",
        "--prompt-data",
        str(args.manifest.resolve()),
        "--input-key",
        "prompt",
        "--rollout-shuffle",
        "--rollout-batch-size",
        str(args.batch_size),
        "--global-batch-size",
        str(args.batch_size),
        "--n-samples-per-prompt",
        "1",
        "--num-rollout",
        str(args.steps),
        "--micro-batch-size",
        "1",
        "--gradient-checkpointing",
        "--lr",
        "5e-5",
        "--weight-decay",
        "0.0",
        "--adam-beta2",
        "0.95",
        "--actor-num-nodes",
        str(args.nodes),
        "--actor-num-gpus-per-node",
        str(args.gpus),
        "--num-gpus-per-node",
        str(args.gpus),
        "--save",
        str(args.save.resolve()),
        "--save-interval",
        str(args.save_interval),
        "--log-loss-sigma-bucket",
        "0",
        *extra,
    ]
    return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument(
        "--nodes",
        "--actor-num-nodes",
        type=int,
        default=1,
        help="Number of training nodes",
    )
    parser.add_argument("--gpus", type=int, default=4, help="Training GPUs per node")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Cumulative optimizer-step target, including resumed steps",
    )
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if min(args.nodes, args.gpus, args.batch_size, args.steps, args.save_interval) <= 0:
        parser.error("Node, GPU, batch, step and save-interval counts must be positive")
    if args.batch_size % (args.nodes * args.gpus):
        parser.error("Global batch size must be divisible by total training GPU count (nodes * gpus)")
    command = train_command(args, extra)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        from .provenance import verify_base_initialization

        verify_base_initialization(args.bundle)
        with args.manifest.open() as stream:
            count = sum(bool(line.strip()) for line in stream)
        if count < args.batch_size:
            parser.error("The manifest must contain at least one full rollout batch")
        subprocess.run(command, check=True)
