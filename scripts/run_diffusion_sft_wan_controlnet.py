"""Launch native Miles SFT for a new Wan2.2-TI2V-5B pose-control branch.

This launcher starts its own local Ray instance without stopping other jobs.
One additional GPU runs frozen text/VAE encoding; no SGLang rollout is needed.
Credentials are inherited from the environment and never put in command args.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shlex
import sys
import tempfile
import time
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    result.add_argument("--data-jsonl", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--train-gpus", type=int, default=4)
    result.add_argument(
        "--cuda-visible-devices", help="Exactly train-gpus + 1 devices; omit to inherit CUDA visibility"
    )
    result.add_argument("--num-cpus", type=int, default=min(os.cpu_count() or 8, 40))
    result.add_argument("--steps", type=int, default=200, help="Optimizer steps, divisible by steps-per-rollout")
    result.add_argument("--steps-per-rollout", type=int, default=2)
    result.add_argument("--control-blocks", type=int, default=4)
    result.add_argument(
        "--initial-controlnet", type=Path, help="Optional branch warm start; omitted means newly initialized"
    )
    result.add_argument(
        "--resume-checkpoint", type=Path, help="Miles checkpoint root, including latest iteration tracker"
    )
    result.add_argument("--lr", type=float, default=2e-5)
    result.add_argument("--height", type=int, default=480)
    result.add_argument("--width", type=int, default=832)
    result.add_argument("--frames", type=int, default=49)
    result.add_argument("--flow-shift", type=float, default=5.0)
    result.add_argument("--warmup-steps", type=int, default=20)
    result.add_argument(
        "--samples-per-gpu", type=int, default=1, help="Samples accumulated per optimizer step per GPU"
    )
    result.add_argument("--frame-stride", type=int, default=1)
    result.add_argument(
        "--save-interval", type=int, default=20, help="Save every this many SFT rollouts, not optimizer steps"
    )
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--wandb", action="store_true", help="Use WANDB_API_KEY/WANDB_PROJECT/WANDB_ENTITY/WANDB_MODE")
    result.add_argument("--run-name", default="wan-pose-control")
    result.add_argument(
        "--dry-run", action="store_true", help="Print native training arguments without initializing Ray"
    )
    return result


def build_train_args(args):
    if args.train_gpus < 1 or args.steps < 1 or args.steps_per_rollout < 1:
        raise ValueError("train-gpus, steps, and steps-per-rollout must be positive")
    if args.steps % args.steps_per_rollout:
        raise ValueError("steps must be divisible by steps-per-rollout")
    if args.height % 32 or args.width % 32 or args.height < 32 or args.width < 32:
        raise ValueError("height and width must be positive multiples of 32")
    if args.frames < 1 or (args.frames - 1) % 4 or args.frame_stride < 1:
        raise ValueError("frames must be 4k+1 and frame-stride must be positive")
    if not 1 <= args.control_blocks <= 30:
        raise ValueError("control-blocks must be between 1 and 30 for Wan2.2-TI2V-5B")
    if args.flow_shift <= 0 or args.warmup_steps < 0 or args.samples_per_gpu < 1:
        raise ValueError("flow-shift and samples-per-gpu must be positive; warmup-steps must be nonnegative")
    if args.lr <= 0 or args.save_interval < 1:
        raise ValueError("lr and save-interval must be positive")
    if not args.data_jsonl.is_file():
        raise FileNotFoundError(args.data_jsonl)
    if args.initial_controlnet and args.resume_checkpoint:
        raise ValueError("choose an initial branch OR a full Miles resume checkpoint")
    visible = args.cuda_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and len([device for device in visible.split(",") if device.strip()]) != args.train_gpus + 1:
        raise ValueError("this recipe uses train-gpus + 1 visible GPUs (one dedicated encoder GPU)")
    local_base = Path(args.base_model).expanduser()
    base_model = str(local_base.resolve()) if local_base.is_dir() else args.base_model
    values = [
        "--hf-checkpoint",
        base_model,
        "--sft-encoder-checkpoint",
        base_model,
        "--diffusion-model-family",
        "wan_controlnet",
        "--wan-controlnet-num-blocks",
        str(args.control_blocks),
        "--save",
        str(args.output_dir.resolve() / "checkpoints"),
        "--save-interval",
        str(args.save_interval),
        "--rollout-function-path",
        "miles.rollout.sft_rollout.generate_rollout",
        "--prompt-data",
        str(args.data_jsonl.resolve()),
        "--input-key",
        "prompt",
        "--rollout-batch-size",
        str(args.train_gpus * args.steps_per_rollout * args.samples_per_gpu),
        "--num-rollout",
        str(args.steps // args.steps_per_rollout),
        "--num-steps-per-rollout",
        str(args.steps_per_rollout),
        "--diffusion-height",
        str(args.height),
        "--diffusion-width",
        str(args.width),
        "--diffusion-output-num-frames",
        str(args.frames),
        "--loss-type",
        "sft_loss",
        "--train-only",
        "--custom-convert-samples-to-train-data-path",
        "miles.rollout.sft_rollout.convert_samples_to_train_data",
        "--custom-rollout-log-function-path",
        "miles.rollout.sft_rollout.log_rollout_data",
        "--custom-prepare-train-batch-path",
        "miles.backends.fsdp_utils.loss_hub.sft.prepare_sft_batch",
        "--custom-loss-function-path",
        "miles.backends.fsdp_utils.loss_hub.sft.sft_loss_formula",
        "--sft-frame-stride",
        str(args.frame_stride),
        "--lr",
        str(args.lr),
        "--lr-warmup-iters",
        str(args.warmup_steps),
        "--adam-beta2",
        "0.999",
        "--weight-decay",
        "0.01",
        "--train-backend",
        "fsdp",
        "--fsdp-master-dtype",
        "fp32",
        "--fsdp-reduce-dtype",
        "fp32",
        "--diffusion-forward-dtype",
        "bf16",
        "--fsdp-flow-shift",
        str(args.flow_shift),
        "--update-weight-target-module",
        "transformer",
        "--sequence-parallel-size",
        "1",
        "--gradient-checkpointing",
        "--micro-batch-size",
        "1",
        "--actor-num-gpus-per-node",
        str(args.train_gpus),
        "--num-gpus-per-node",
        str(args.train_gpus + 1),
        "--rollout-num-gpus",
        "1",
        "--seed",
        str(args.seed),
    ]
    if args.initial_controlnet:
        values.extend(["--wan-controlnet-checkpoint", str(args.initial_controlnet.resolve())])
    if args.resume_checkpoint:
        values.extend(["--load", str(args.resume_checkpoint.resolve())])
    if args.wandb:
        values.extend(
            [
                "--use-wandb",
                "--wandb-project",
                os.environ.get("WANDB_PROJECT", "wan-pose-control"),
                "--wandb-group",
                args.run_name,
                "--disable-wandb-random-suffix",
                "--wandb-mode",
                os.environ.get("WANDB_MODE", "online"),
                "--wandb-dir",
                str(args.output_dir.resolve() / "wandb"),
            ]
        )
        if os.environ.get("WANDB_ENTITY"):
            values.extend(["--wandb-team", os.environ["WANDB_ENTITY"]])
    return values


def write_launch_manifest(args, train_args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "launch.json"
    if manifest.exists():
        if not args.resume_checkpoint:
            raise FileExistsError(f"{manifest} exists; use a new output directory or an explicit resume")
        # Preserve the initial run configuration when appending a resume event.
        manifest = args.output_dir / f"resume_launch_{time.time_ns()}.json"
    manifest.write_text(json.dumps({"train_entrypoint": "train_sft.py", "train_args": train_args}, indent=2) + "\n")
    return manifest


def main():
    args = parser().parse_args()
    train_args = build_train_args(args)
    if args.dry_run:
        print(shlex.join([sys.executable, "train_sft.py", *train_args]))
        return
    write_launch_manifest(args, train_args)
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import ray

    ray.init(
        address="local",
        num_gpus=args.train_gpus + 1,
        num_cpus=args.num_cpus,
        include_dashboard=False,
        object_store_memory=4 * 1024**3,
        _temp_dir=tempfile.mkdtemp(prefix="miles-wan-pose-"),
    )
    try:
        sys.argv = [str(repo / "train_sft.py"), *train_args]
        runpy.run_path(str(repo / "train_sft.py"), run_name="__main__")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
