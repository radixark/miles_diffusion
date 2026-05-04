import argparse
import json
import logging
import os
from typing import Any

import yaml
from transformers import AutoConfig

from miles.backends.sglang_diffusion_utils.arguments import add_sglang_diffusion_arguments
from miles.backends.sglang_diffusion_utils.arguments import validate_args as sglang_validate_args
from miles.utils.eval_config import EvalDatasetConfig, build_eval_dataset_configs, ensure_dataset_list
from miles.utils.logging_utils import configure_logger

logger = logging.getLogger(__name__)


def reset_arg(parser, name, **kwargs):
    """
    Reset the default value of a Megatron argument.
    :param parser: The argument parser.
    :param name: The name of the argument to reset.
    :param default: The new default value.
    """
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)


def get_miles_extra_args_provider(add_custom_arguments=None):
    def add_miles_arguments(parser):
        # Ray
        def add_cluster_arguments(parser):
            parser.add_argument("--actor-num-nodes", type=int, default=1, help="Number of nodes for training actor")
            parser.add_argument(
                "--actor-num-gpus-per-node", type=int, default=8, help="Number of gpus per node for training actor"
            )
            parser.add_argument(
                "--rollout-num-gpus",
                type=int,
                default=None,
                help=(
                    "Number of GPUs for inference. Note that when using --colocate, "
                    "i.e. the training and the inference engines are on the same gpus, this param will be ignored and will be set as "
                    "actor_num_gpus_per_node * actor_num_nodes."
                ),
            )
            parser.add_argument(
                "--rollout-num-gpus-per-engine",
                type=int,
                default=1,
                help="Number of GPUs per inference engine, just like the tp_size in sglang.",
            )
            parser.add_argument(
                "--num-gpus-per-node",
                type=int,
                default=8,
                help=(
                    "Number of gpus per node for rollout."
                    "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
                ),
            )
            parser.add_argument(
                "--colocate",
                action="store_true",
                default=False,
                help=(
                    "Whether to colocate the inference engines and the actor. "
                    "Turning this on will also set --offload to true."
                ),
            )
            parser.add_argument(
                "--offload",
                action="store_true",
                default=False,
                help=("Equivalent to --offload-train + --offload-rollout. "),
            )
            parser.add_argument(
                "--offload-train",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the training actor to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )
            parser.add_argument(
                "--offload-rollout",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the rollout generator to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )

            reset_arg(parser, "--distributed-backend", type=str, default="nccl")
            reset_arg(parser, "--distributed-timeout-minutes", type=int, default=10)

            return parser

        def add_train_arguments(parser):
            parser.add_argument(
                "--train-backend",
                type=str,
                choices=["fsdp"],
                default="fsdp",
                help="The backend for training.",
            )
            # Diffusion GRPO training knobs (used by DiffusionFSDPTrainRayActor).
            #
            # Per-optim-step the train loop sees an (M, T_sde) grid — M samples
            # in this optim window × T_sde SDE timesteps per sample. The grid is
            # processed as tiles of size (sample_microbatch, tstep_microbatch),
            # gradients accumulating across tiles, optimizer steps once at the
            # end of the window. Two extreme presets:
            #
            #   sample_microbatch = M, tstep_microbatch = 1, iter_order = sample_major
            #     → outer loop over T_sde, inner forward = (M, 1, ...)
            #     → memory peak ∝ M (the current default).
            #
            #   sample_microbatch = 1, tstep_microbatch = T_sde, iter_order = timestep_major
            #     → outer loop over samples, inner forward = (1, T_sde, ...)
            #     → memory peak ∝ T_sde (lower when M is the limit on 2-GPU runs).
            #
            # Loss scaling is uniform across plans: each tile's mean PPO loss is
            # divided by total tile count, so net gradient = mean over (M, T_sde).
            parser.add_argument(
                "--micro-batch-size-sample",
                type=int,
                default=None,
                help="Samples per DiT forward in train (sample-axis tile size). None = full window (= num_samples_in_window).",
            )
            parser.add_argument(
                "--micro-batch-size-tstep",
                type=int,
                default=1,
                help="SDE timesteps per DiT forward in train (tstep-axis tile size). Default 1.",
            )
            parser.add_argument(
                "--diffusion-train-iter-order",
                type=str,
                choices=["sample_major", "timestep_major"],
                default="sample_major",
                help=(
                    "Outer-loop axis when iterating tiles. sample_major: outer "
                    "loop over timestep tiles (low memory when sample_microbatch "
                    "is large). timestep_major: outer loop over sample tiles "
                    "(low memory when tstep_microbatch is large)."
                ),
            )
            parser.add_argument(
                "--diffusion-clip-range",
                type=float,
                default=1e-4,
                help="Clip range for diffusion GRPO ratio.",
            )
            parser.add_argument(
                "--diffusion-adv-clip-max",
                type=float,
                default=5.0,
                help="Max absolute value for advantage clipping in diffusion training.",
            )
            parser.add_argument(
                "--fsdp-cfg-batching",
                action=argparse.BooleanOptionalAction,
                default=False,
                help=(
                    "Batch positive and negative CFG branches into a single DiT "
                    "forward (concat along batch dim, one forward, then chunk). "
                    "Default False = two separate forwards. Set True to bit-exact "
                    "match sgl-d models that join CFG into one batched forward "
                    "(e.g. Flux variants); leave False for split-CFG models like "
                    "Qwen-Image."
                ),
            )
            parser.add_argument(
                "--fsdp-cpu-offload",
                action=argparse.BooleanOptionalAction,
                default=False,
                help=(
                    "Enable FSDP CPU offload for parameters and gradients. "
                    "Default False (keep everything on GPU)."
                ),
            )
            parser.add_argument(
                "--fsdp-cpu-backend",
                type=str,
                default="gloo",
                help=(
                    "CPU collective backend for FSDP CPU offload (e.g. 'gloo'). "
                    "Used together with --fsdp-cpu-offload to set up the hybrid "
                    "cpu+cuda process group. Set to empty to disable."
                ),
            )
            parser.add_argument(
                "--fsdp-master-dtype",
                type=str,
                default="fp32",
                choices=["fp16", "bf16", "fp32"],
                help=(
                    "dtype for the FSDP-wrapped master copy of the model. "
                    "Loaded at this dtype, sharded at this dtype, optimizer state "
                    "lives at this precision. fp32 (default) gives proper "
                    "mixed-precision training when paired with a lower "
                    "--diffusion-forward-dtype."
                ),
            )
            parser.add_argument(
                "--fsdp-reduce-dtype",
                type=str,
                default="fp32",
                choices=["fp16", "bf16", "fp32"],
                help=(
                    "dtype for FSDP MixedPrecisionPolicy.reduce_dtype "
                    "(grad reduce-scatter precision). fp32 (default) keeps "
                    "multi-rank gradient sums numerically stable; bf16 matches "
                    "flow_grpo's all-bf16 mixed-precision policy at the cost "
                    "of bf16 add-non-associativity noise across ranks."
                ),
            )
            parser.add_argument(
                "--diffusion-forward-dtype",
                type=str,
                default="bf16",
                choices=["fp16", "bf16", "fp32"],
                help=(
                    "dtype for the DiT forward compute. Used in three places "
                    "with the same value: sglang-d rollout engine, FSDP "
                    "MixedPrecisionPolicy.param_dtype on the training side, "
                    "and the training-side input cast that matches rollout "
                    "for log-prob alignment."
                ),
            )
            parser.add_argument(
                "--train-env-vars",
                type=json.loads,
                default="{}",
                help="Extra environment variables for training process, e.g. PyTorch memory management ones.",
            )
            parser.add_argument(
                "--train-memory-margin-bytes",
                type=int,
                default=1024**3,
                help="Add margin for train memory allocation. By default we will reserve 1GB as margin.",
            )
            parser.add_argument(
                "--recompute-loss-function",
                action="store_true",
                help="Whether to disable recompute loss function to save memory during training.",
            )
            parser.add_argument(
                "--log-probs-chunk-size", type=int, default=-1, help="Chunk size to compute log probs to save memory"
            )

            return parser

        # rollout
        def add_rollout_arguments(parser):
            parser.add_argument(
                "--hf-checkpoint",
                type=str,
                default=None,
                help=(
                    "The huggingface checkpoint of the trained model. "
                    "This is used to initialize sglang and also provide the tokenizer. "
                    "Note that, we will always update the parameters in sglang with that of megatron before training, "
                    "so you only need to provide a huggingface checkpoint that has the same architecture as the model you want to train. "
                    "It doesn't necessary need to contain the most up-to-date parameters."
                ),
            )
            parser.add_argument(
                "--model-name",
                type=str,
                default=None,
                help=(
                    "The name of the model, this is used to convert the megatron weights into huggingface format. "
                    "If not set, we will use `type(AutoConfig.from_pretrained(args.hf_checkpoint)).__name__.lower()` as model_name. "
                    "Also, sometimes this will help alleviate the bug that transformers cannot find certain model."
                ),
            )
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default="miles.rollout.sglang_rollout.generate_rollout",
                help=(
                    "Path to the rollout generation function."
                    "You should use this model to create your own custom rollout function, "
                    "and then set this to the path of your custom rollout function. "
                    "The signature of the function should be "
                    "`def generate_rollout(args, rollout_id, *, evaluation=False) -> list[list[Sample]]`"
                    "and within the output sample, you should at least set `tokens`, `response_length`, `reward` "
                    "and `truncated`."
                ),
            )
            parser.add_argument(
                "--diffusion-model",
                type=str,
                default="stabilityai/stable-diffusion-3.5-medium",
                help="HuggingFace model id for diffusion rollout.",
            )
            parser.add_argument(
                "--diffusion-device",
                type=str,
                default=None,
                help="Device for diffusion rollout, e.g. cuda or cpu. Defaults to auto.",
            )
            parser.add_argument(
                "--diffusion-num-steps",
                type=int,
                default=10,
                help="Number of diffusion inference steps for rollout.",
            )
            parser.add_argument(
                "--diffusion-microgroup-size",
                type=int,
                default=1,
                help="Diffusion rollout microgroup size (sub-batch of samples per prompt). Defaults to 1.",
            )
            parser.add_argument(
                "--diffusion-eval-num-steps",
                type=int,
                default=None,
                help="Number of diffusion inference steps for eval rollout. Defaults to diffusion-num-steps.",
            )
            parser.add_argument(
                "--diffusion-guidance-scale",
                type=float,
                default=4.0,
                help="Guidance scale for diffusion rollout.",
            )
            parser.add_argument(
                "--diffusion-noise-level",
                type=float,
                default=0.7,
                help="SDE noise level for diffusion rollout (matches flow_grpo sample.noise_level; sent as rollout_noise_level on POST /rollout/generate).",
            )
            parser.add_argument(
                "--diffusion-height",
                type=int,
                default=512,
                help="Output image height for diffusion rollout.",
            )
            parser.add_argument(
                "--diffusion-width",
                type=int,
                default=512,
                help="Output image width for diffusion rollout.",
            )
            parser.add_argument(
                "--diffusion-negative-prompt",
                type=str,
                default=None,
                help="Negative prompt for sglang-diffusion POST /rollout/generate.",
            )
            parser.add_argument(
                "--diffusion-true-cfg-scale",
                type=float,
                default=None,
                help="Optional true_cfg_scale for sglang-diffusion POST /rollout/generate.",
            )
            parser.add_argument(
                "--diffusion-generator-device",
                type=str,
                default="cuda",
                help="generator_device field for POST /rollout/generate.",
            )
            parser.add_argument(
                "--diffusion-sde-type",
                type=str,
                default="sde",
                help="rollout_sde_type for POST /rollout/generate.",
            )
            parser.add_argument(
                "--diffusion-sde-window-size",
                type=int,
                default=0,
                help="flow_grpo-style random SDE window; 0 disables. Steps outside "
                     "the window run ODE and are not returned for training.",
            )
            parser.add_argument(
                "--diffusion-sde-window-range",
                type=str,
                default=None,
                help="'lo,hi' bounds for the SDE window start (inclusive, exclusive). "
                     "Defaults to [0, num_inference_steps].",
            )
            parser.add_argument(
                "--diffusion-step-strategy-path",
                type=str,
                default=None,
                help="Dotted path to a factory(args) -> StepStrategy callable. "
                     "Overrides --diffusion-sde-window-size.",
            )
            parser.add_argument(
                "--diffusion-log-prob-no-const",
                action="store_true",
                default=False,
                help="Set rollout_log_prob_no_const=true on POST /rollout/generate.",
            )
            parser.add_argument(
                "--apply-sgld-monkey-patches",
                action="store_true",
                default=False,
                help=(
                    "Apply miles.backends.sglang_diffusion_utils.monkey_patches at "
                    "sglang-d startup so its DiT forward is bit-exact with diffusers' "
                    "implementation. Makes rollout (sglang-d path) and training-side "
                    "log-prob agree on noise_pred down to bf16 ULPs. Small perf hit on "
                    "the rollout engine."
                ),
            )
            parser.add_argument(
                "--diffusion-debug-mode",
                action="store_true",
                default=False,
                help="Set rollout_debug_mode=true on POST /rollout/generate.",
            )
            parser.add_argument(
                "--diffusion-return-prev-latents-mean",
                action="store_true",
                help="Whether to store prev_latents_mean for KL regularization.",
            )
            parser.add_argument(
                "--diffusion-reward",
                type=str,
                default="pickscore",
                help="Reward function name for diffusion rollout.",
            )
            parser.add_argument(
                "--diffusion-reward-device",
                type=str,
                default=None,
                help="Device for diffusion reward model, defaults to diffusion-device.",
            )
            parser.add_argument(
                "--diffusion-log-images",
                type=int,
                default=0,
                help="Number of diffusion images to log to W&B per rollout (0 disables).",
            )
            parser.add_argument(
                "--diffusion-log-image-interval",
                type=int,
                default=1,
                help="Log diffusion images every N rollouts. Only used when diffusion-log-images > 0.",
            )
            parser.add_argument(
                "--rollout-seed",
                type=int,
                default=42,
                help=(
                    "The seed for the random number generator during rollout. "
                    "This is used to shuffle the prompts and also for the random sampling of the prompts."
                ),
            )

            # sampling
            parser.add_argument(
                "--over-sampling-batch-size",
                type=int,
                default=None,
                help=(
                    "This defines the granularity of the sampling batch in the rollout function. "
                    "When the number of available samples falls below the target, a sampling "
                    "operation of size over_sampling_batch_size will be triggered."
                    "Regardless of whether partial rollout is used or filters are applied, "
                    "the sampling granularity is always determined by this value. "
                    "If this value is None, rollout_batch_size will be used as the default over_sampling_batch_size."
                ),
            )
            parser.add_argument(
                "--dynamic-sampling-filter-path",
                type=str,
                default=None,
                help=(
                    "This is the filter function for dynamic sampling. "
                    "It should be able to judge whether the result of a prompt should be selected or not."
                    "We will do dynamic filter for sampling as in DAPO. e.g. not all correct or all wrong samples."
                    "You could use `miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
                ),
            )

            parser.add_argument(
                "--buffer-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the buffer filter function. "
                    "It should be able to select the samples in the buffer. "
                    "The function should take list[list[Sample]] and return list[list[Sample]]."
                ),
            )

            # Customization extension hooks (load_function dispatch).
            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Substitute the inner `def generate(args, sample, sampling_params)` call inside "
                    "the diffusion rollout. Useful for multi-turn refinement / custom CFG schedules / "
                    "specialised sampling pipelines. The function may optionally accept `evaluation=...`."
                ),
            )
            parser.add_argument(
                "--custom-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "Custom function for logging train rollout data. Signature: "
                    "`def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool`. "
                    "Truthy return skips the default logging."
                ),
            )
            parser.add_argument(
                "--custom-eval-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "Custom function for logging eval rollout data. Signature: "
                    "`def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool`. "
                    "Truthy return skips the default logging."
                ),
            )
            parser.add_argument(
                "--custom-convert-samples-to-train-data-path",
                type=str,
                default=None,
                help=(
                    "Replace `RolloutManager._convert_samples_to_train_data`. Signature: "
                    "`def convert_samples_to_train_data(args, samples) -> dict`."
                ),
            )
            parser.add_argument(
                "--rollout-sample-filter-path",
                type=str,
                default=None,
                help=(
                    "Per-sample loss-mask filter run after dynamic_sampling_filter. Signature: "
                    "`def filter(args, data: list[list[Sample]]) -> None`. The function should set "
                    "`sample.remove_sample = True` on samples that should be excluded from loss "
                    "computation (they still participate in advantage normalisation)."
                ),
            )
            # update weight
            parser.add_argument(
                "--update-weight-buffer-size",
                type=int,
                default=512 * 1024**2,
                help=(
                    "buffer size for update weight, in bytes. "
                    "This is used for updating weights by chunk and should be useful for MoE models."
                ),
            )
            parser.add_argument(
                "--update-weights-interval",
                type=int,
                default=1,
                help="Interval for updating the weights",
            )
            return parser

        def add_fault_tolerance_arguments(parser):
            parser.add_argument(
                "--use-fault-tolerance",
                action="store_true",
                default=False,
                help="Whether to enable the fault tolerance function during rollout.",
            )
            parser.add_argument(
                "--rollout-health-check-interval",
                type=float,
                default=30.0,
                help="Interval in seconds between rollout engine /health_generate checks during generate/eval.",
            )
            parser.add_argument(
                "--rollout-health-check-timeout",
                type=float,
                default=30.0,
                help="Timeout in seconds to wait for a rollout engine /health_generate response before killing it.",
            )
            parser.add_argument(
                "--rollout-health-check-first-wait",
                type=float,
                default=0,
                help="Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm.",
            )
            return parser

        # data
        def add_data_arguments(parser):
            # dataset
            # TODO: maybe add an num_epoch and calculate the num_rollout from buffer
            parser.add_argument(
                "--num-rollout",
                type=int,
                default=None,
                help="Number of rollout steps. If not set, we will calculate the number of rollout steps from the dataset size.",
            )
            parser.add_argument(
                "--num-epoch",
                type=int,
                default=None,
                help=(
                    "Number of epochs for the training. "
                    "This is used to calculate the number of rollout steps from the dataset size. "
                    "If set, we will calculate the number of rollout steps as `num_rollout = num_epoch * dataset_size // rollout_batch_size`."
                    "If both `--num-epoch` and `--num-rollout` are set, `--num-epoch` will be ignored."
                ),
            )

            parser.add_argument(
                "--disable-rollout-global-dataset",
                action="store_false",
                dest="rollout_global_dataset",
                help=(
                    "Whether to use a global dataset for rollout. "
                    "If set, the rollout will use the `--prompt-data` as the prompt dataset, "
                    "and the prompts for rollout will be sampled from the dataset. "
                    "If not set, you need to manage the data by your self."
                ),
            )

            parser.add_argument(
                "--data-source-path",
                type=str,
                default="miles.rollout.data_source.RolloutDataSourceWithBuffer",
                help="The data source class for rollout data.",
            )
            parser.add_argument(
                "--prompt-data",
                type=str,
                default=None,
                help=(
                    "The path to the prompt data. "
                    "Currently we only support jsonl format, and each line should contains --input-key and --label-key, "
                    "which will be used as the prompt and the label respectively. "
                    "If you want to use a custom template, you can set --apply-chat-template to true, in that case, "
                    "the input should be the same structure as an openai message, e.g. [{'role': 'user', 'content': 'blabla'}]. "
                ),
            )
            parser.add_argument("--input-key", type=str, default="input", help="JSON dataset key")
            parser.add_argument("--metadata-key", type=str, default="metadata", help="JSON dataset key")

            parser.add_argument(
                "--start-rollout-id",
                type=int,
                default=0,
                help="The starting rollout step.",
            )

            # batch sizes
            parser.add_argument(
                "--rollout-batch-size",
                type=int,
                required=True,
                help=(
                    "The number of prompts in each rollout step. "
                    "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
                ),
            )
            parser.add_argument(
                "--n-samples-per-prompt", type=int, default=1, help="Number of responses for each prompt in generation"
            )

            # gbs of the training, note that the gbs is of sample, not of prompts,
            # so if you hope to train 1 step for each rollout, the global_bach_size should be set as
            # `rollout_batch_size * n_samples_per_prompt`.
            reset_arg(parser, "--global-batch-size", type=int, default=None)
            parser.add_argument(
                "--num-steps-per-rollout",
                type=int,
                default=None,
                help=(
                    "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
                    "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
                ),
            )
            reset_arg(parser, "--micro-batch-size", type=int, default=1)
            return parser

        def add_eval_arguments(parser):
            parser.add_argument(
                "--eval-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the eval generation function."
                    "If not set, we will use rollout_function_path as the default. "
                ),
            )

            # change the default value of eval_interval from Megatron to None
            reset_arg(parser, "--eval-interval", type=int, default=None)

            parser.add_argument(
                "--eval-prompt-data",
                type=str,
                default=None,
                nargs="+",
                help=(
                    "Path to the evaluation prompt data, "
                    "should first input the name of the eval dataset and then the path, e.g. "
                    "aime /path/to/aime.jsonl"
                ),
            )
            parser.add_argument(
                "--eval-config",
                type=str,
                default=None,
                help=(
                    "Path to an OmegaConf YAML/JSON file describing evaluation datasets. "
                    "When provided, this overrides --eval-prompt-data."
                ),
            )
            parser.add_argument(
                "--skip-eval-before-train",
                action="store_true",
                default=False,
                help="Whether to skip evaluation before training.",
            )

            # The following keys are used to override the rollout version during eval.
            parser.add_argument("--eval-input-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--n-samples-per-eval-prompt",
                type=int,
                default=1,
                help="number of responses for each prompt in generation",
            )

            return parser

        def add_algo_arguments(parser):
            reset_arg(parser, "--load", type=str, default=None)
            reset_arg(parser, "--save", type=str, default=None)
            reset_arg(parser, "--save-interval", type=int, default=None)
            reset_arg(parser, "--async-save", action="store_true")
            reset_arg(
                parser,
                "--no-save-optim",
                action="store_true",
                default=False,
                help=(
                    "If set, do not save the optimizer state when saving checkpoints. "
                    "This reduces checkpoint size but disables training resumption from the saved checkpoint."
                ),
            )
            reset_arg(parser, "--seed", type=int, default=1234)
            reset_arg(parser, "--clip-grad", type=float, default=1.0)
            reset_arg(parser, "--calculate-per-token-loss", action="store_true")
            reset_arg(parser, "--lr", type=float, default=1e-6)

            parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip range")
            parser.add_argument("--eps-clip-high", type=float, default=None, help="PPO clip upper range")
            parser.add_argument(
                "--loss-type",
                type=str,
                choices=["policy_loss", "sft_loss", "custom_loss"],
                default="policy_loss",
                help=(
                    "Choose loss type, currently support ppo policy_loss or sft_loss, "
                    "if custom_loss is set, we will use the function path from `--custom-loss-function-path`."
                ),
            )
            parser.add_argument(
                "--advantage-estimator",
                type=str,
                choices=[
                    "grpo",
                ],
                default="grpo",
            )
            parser.add_argument(
                "--disable-compute-advantages-and-returns",
                action="store_false",
                dest="compute_advantages_and_returns",
                help=(
                    "Whether to disable computing advantages and returns. "
                    "If set, we will not compute the advantages and returns, "
                    "This is useful for sft or custom loss function."
                ),
            )
            parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy loss coef")
            parser.add_argument("--normalize-advantages", action="store_true", default=False)
            parser.add_argument(
                "--disable-grpo-std-normalization",
                action="store_false",
                dest="grpo_std_normalization",
                help="from Dr.GRPO https://arxiv.org/pdf/2503.20783",
            )
            parser.add_argument(
                "--globalize-reward-mean",
                action="store_true",
                default=False,
                help=(
                    "Use batch-wide mean instead of per-prompt mean for GRPO advantage. "
                    "flow_grpo's PerPromptStatTracker uses per-prompt mean, so leave this OFF "
                    "for flow_grpo parity."
                ),
            )
            parser.add_argument(
                "--globalize-reward-std",
                action="store_true",
                default=False,
                help=(
                    "Use batch-wide std instead of per-group std for GRPO advantage. "
                    "flow_grpo's pickscore recipe sets global_std=True, so enable this for parity."
                ),
            )
            parser.add_argument(
                "--use-rollout-entropy",
                action="store_true",
                default=False,
                help=(
                    "Whether to calculate the entropy when calculating the logprobs from actor and reference model. "
                    "This is useful for doing special loss mask."
                ),
            )
            return parser

        def add_router_arguments(parser):
            parser.add_argument(
                "--use-miles-router",
                action="store_true",
                default=False,
                help="Whether to use MilesRouter for text-based routing instead of SGLang token-based routing",
            )
            parser.add_argument(
                "--miles-router-timeout",
                type=float,
                default=None,
                help="Timeout for MilesRouter HTTP requests in seconds.",
            )
            parser.add_argument(
                "--miles-router-max-connections",
                type=int,
                default=None,
                help="Max connections for MilesRouter HTTP client.",
            )
            parser.add_argument(
                "--miles-router-health-check-failure-threshold",
                type=int,
                default=3,
                help="Number of consecutive failures before marking a worker as unhealthy.",
            )
            return parser

        # wandb
        def add_wandb_arguments(parser):
            # wandb parameters
            parser.add_argument("--use-wandb", action="store_true", default=False)
            parser.add_argument(
                "--wandb-mode",
                type=str,
                default=None,
                choices=["online", "offline", "disabled"],
                help="W&B mode: online (default), offline (local only), or disabled. Overrides WANDB_MODE env var.",
            )
            parser.add_argument(
                "--wandb-dir",
                type=str,
                default=None,
                help="Directory to store wandb logs. Default is ./wandb in current directory.",
            )
            parser.add_argument("--wandb-key", type=str, default=None)
            parser.add_argument("--wandb-host", type=str, default=None)
            parser.add_argument("--wandb-team", type=str, default=None)
            parser.add_argument("--wandb-group", type=str, default=None)
            reset_arg(parser, "--wandb-project", type=str, default=None)
            parser.add_argument(
                "--disable-wandb-random-suffix",
                action="store_false",
                dest="wandb_random_suffix",
                default=True,
                help=(
                    "Whether to add a random suffix to the wandb run name. "
                    "By default, we will add a random 6 length string with characters to the run name."
                ),
            )
            parser.add_argument("--wandb-run-id", type=str, default=None)
            return parser


        # debug
        def add_debug_arguments(parser):
            parser.add_argument(
                "--save-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Save the rollout data to this path for debugging. "
                    "The file will be saved to `save_debug_rollout_data.format(rollout_id)`."
                ),
            )
            parser.add_argument(
                "--load-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Load the rollout data from this path for debugging. "
                    "The file will be loaded from `load_debug_rollout_data.format(rollout_id)`. "
                    "When this is enabled, miles will not instantiate sglang servers."
                ),
            )
            parser.add_argument(
                "--load-debug-rollout-data-subsample",
                type=float,
                default=None,
                help="Subsample a portion of the debug rollout data for faster debugging.",
            )
            parser.add_argument(
                "--debug-rollout-only",
                action="store_true",
                default=False,
                help=(
                    "Whether to only run the rollout generation without training. "
                    "This is useful for debugging the rollout generation function."
                ),
            )
            parser.add_argument(
                "--debug-train-only",
                action="store_true",
                default=False,
                help=(
                    "Whether to only run the training without sglang servers. "
                    "This is useful for debugging the rollout generation function."
                ),
            )
            parser.add_argument(
                "--debug-skip-optimizer-step",
                action="store_true",
                default=False,
                help=(
                    "Skip loss.backward() and optimizer.step() so trainer weights "
                    "never drift. Used with --debug-disable-weight-sync to measure "
                    "pure forward-path divergence from the rollout engine."
                ),
            )
            # LoRA
            parser.add_argument("--diffusion-ignore-last", type=int, default=0,
                help="Skip last N denoising steps for training (avoids small-sigma numerical issues). FlowGRPO/DanceGRPO use 1.")
            parser.add_argument("--use-lora", action="store_true", default=False,
                help="Use LoRA adapters instead of full finetune.")
            parser.add_argument("--lora-rank", type=int, default=64)
            parser.add_argument("--lora-alpha", type=int, default=64)
            parser.add_argument("--lora-target-modules", type=str, nargs="+", default=None,
                help="Override LoRA target modules. Default: per-model from TrainPipelineConfig.")
            parser.add_argument(
                "--diffusion-init-lora-weight",
                type=str,
                default="gaussian",
                help=(
                    "PEFT LoraConfig.init_lora_weights. flow_grpo's Qwen-Image uses 'gaussian' "
                    "(N(0, 1/r) for lora_A, 0 for lora_B). 'kaiming-uniform' maps to PEFT's "
                    "default Kaiming-uniform init. Other PEFT schemes ('olora', 'pissa', "
                    "'pissa_niter_N', 'loftq', ...) pass through unchanged."
                ),
            )

            parser.add_argument(
                "--save-debug-train-data",
                type=str,
                default=None,
                help=(
                    "Save the train data to this path for debugging. "
                    "The file will be saved to `save_debug_train_data.format(rollout_id)`."
                ),
            )
            parser.add_argument(
                "--dump-details",
                type=str,
                default=None,
                help=("Dump all details of training for post-hoc analysis and visualization."),
            )
            return parser

        def add_network_arguments(parser):
            parser.add_argument("--use-distributed-post", action="store_true", default=False)
            return parser

        def add_reward_model_arguments(parser):
            parser.add_argument(
                "--rm-type",
                type=str,
                default=None,
                help="Type of the reward model",
            )
            parser.add_argument(
                "--reward-key",
                type=str,
                default=None,
                help=(
                    "Some reward model may return a dict instead of a value, "
                    "this is the key to extract the reward value from the dict. "
                ),
            )
            parser.add_argument(
                "--eval-reward-key",
                type=str,
                default=None,
                help="The eval variant for --reward-key",
            )
            parser.add_argument(
                "--group-rm", action="store_true", default=False, help="Whether to do rm on a whole group."
            )
            parser.add_argument(
                "--rm-url",
                type=str,
                default=None,
                help="URL for the reward model service for --rm-type remote_rm, e.g. http://localhost:8000",
            )
            parser.add_argument(
                "--ocr-num-workers",
                type=int,
                default=4,
                help="Number of Ray OCR actors used when --rm-type ocr.",
            )
            parser.add_argument(
                "--pickscore-num-workers",
                type=int,
                default=1,
                help="Number of Ray PickScore actors used when --rm-type pickscore.",
            )
            parser.add_argument(
                "--pickscore-num-gpus-per-worker",
                type=float,
                default=1.0,
                help="GPU resources per PickScore actor. Use 1.0 for a dedicated GPU smoke test.",
            )
            parser.add_argument(
                "--pickscore-batch-size",
                type=int,
                default=8,
                help="Batch size per PickScore actor call.",
            )
            parser.add_argument(
                "--pickscore-processor-path",
                type=str,
                default=None,
                help="Hugging Face processor path for PickScore. Required when --rm-type pickscore.",
            )
            parser.add_argument(
                "--pickscore-model-path",
                type=str,
                default=None,
                help="Hugging Face model path for PickScore. Required when --rm-type pickscore.",
            )

            # Customization extension hooks for reward computation.
            parser.add_argument(
                "--custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Replace the built-in reward dispatch with a user-supplied batched function. "
                    "Signature: `async def custom_rm(args, samples: list[Sample], **kwargs) -> list[float]`. "
                    "Wired in batched_async_rm only — per-sample async_rm dispatch was deliberately "
                    "removed to avoid the (args, sample) vs (args, list) signature ambiguity. "
                    "If you want per-sample routing, do it inside your batched function."
                ),
            )
            parser.add_argument(
                "--custom-reward-post-process-path",
                type=str,
                default=None,
                help=(
                    "Replace `RolloutManager._post_process_rewards` (advantage normalisation). "
                    "Signature: `def post_process(args, samples) -> tuple[list[float], list[float]]` "
                    "returning (raw_rewards, normalised_rewards)."
                ),
            )
            return parser

        def add_ci_arguments(parser):
            parser.add_argument(
                "--ci-test",
                action="store_true",
            )
            parser.add_argument(
                "--ci-metric-checker-key",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--ci-metric-checker-threshold",
                type=float,
                default=None,
            )
            parser.add_argument(
                "--ci-save-grad-norm",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--ci-load-grad-norm",
                type=str,
                default=None,
            )
            return parser

        def add_sglang_tp_size():
            temp_parser = argparse.ArgumentParser(add_help=False)
            temp_parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
            temp_args, _ = temp_parser.parse_known_args()
            sglang_tp_size = temp_args.rollout_num_gpus_per_engine
            return sglang_tp_size

        # Add custom arguments in front to prevent overwritten some miles arguments.
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)

        parser = add_cluster_arguments(parser)
        parser = add_train_arguments(parser)
        parser = add_rollout_arguments(parser)
        parser = add_fault_tolerance_arguments(parser)
        parser = add_data_arguments(parser)
        parser = add_eval_arguments(parser)
        parser = add_algo_arguments(parser)
        parser = add_wandb_arguments(parser)
        parser = add_router_arguments(parser)
        parser = add_debug_arguments(parser)
        parser = add_sglang_diffusion_arguments(parser)
        parser = add_network_arguments(parser)
        parser = add_reward_model_arguments(parser)
        parser = add_ci_arguments(parser)
        reset_arg(
            parser,
            "--custom-config-path",
            type=str,
            default=None,
            help="Path to the YAML config for custom function arguments.",
        )

        parser.set_defaults(sglang_tensor_parallel_size=add_sglang_tp_size())
        return parser

    return add_miles_arguments


def parse_args(add_custom_arguments=None):
    # Users may call `parse_args` very early, thus we ensure logger is configured here
    configure_logger()

    # TODO: Diffusion FSDP
    add_miles_arguments = get_miles_extra_args_provider(add_custom_arguments)

    backend = parse_args_train_backend()
    from miles.backends.fsdp_utils.arguments import load_fsdp_args
    args = load_fsdp_args(extra_args_provider=add_miles_arguments)
    args.rank = 0  # Primary process rank for wandb initialization
    args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    assert args.context_parallel_size == 1, "Context parallelism is not supported for FSDP backend."

    miles_validate_args(args)
    sglang_validate_args(args)

    return args


def parse_args_train_backend():
    if os.environ.get("MILES_BACKEND") is not None:
        raise Exception("`MILES_BACKEND` is deprecated, please use --train-backend directly.")

    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args_partial, _ = parser.parse_known_args()
    return args_partial.train_backend


def _resolve_eval_datasets(args) -> list[EvalDatasetConfig]:
    """
    Build evaluation dataset configurations from either --eval-config or --eval-prompt-data.
    """
    datasets_config = []
    defaults: dict[str, Any] = {}

    if args.eval_config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.eval_config)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise ValueError("--eval-config must contain a mapping at the root.")

        eval_cfg = cfg_dict.get("eval", cfg_dict)
        if not isinstance(eval_cfg, dict):
            raise ValueError("--eval-config must define an `eval` mapping or be a mapping itself.")

        defaults = dict(eval_cfg.get("defaults") or {})
        datasets_config = ensure_dataset_list(eval_cfg.get("datasets"))
        if not datasets_config:
            raise ValueError("--eval-config does not define any datasets under `eval.datasets`.")
    elif args.eval_prompt_data:
        values = list(args.eval_prompt_data)
        if len(values) % 2 != 0:
            raise ValueError("eval prompt data must be provided as name/path pairs.")
        datasets_config = [{"name": values[i], "path": values[i + 1]} for i in range(0, len(values), 2)]
    else:
        datasets_config = []

    eval_datasets = build_eval_dataset_configs(args, datasets_config, defaults)
    if eval_datasets:
        args.eval_prompt_data = [item for dataset in eval_datasets for item in (dataset.name, dataset.path)]
    else:
        args.eval_prompt_data = None

    return eval_datasets


def miles_validate_args(args):
    args.eval_datasets = _resolve_eval_datasets(args)

    if args.eval_interval is not None:
        assert args.eval_datasets, "Evaluation datasets must be configured when eval_interval is set."

    if args.save_interval is not None:
        assert args.save is not None, "'--save' is required when save_interval is set."

    if args.advantage_estimator in ["reinforce_plus_plus", "reinforce_plus_plus_baseline"]:
        assert args.normalize_advantages, (
            "The 'reinforce_plus_plus' and 'reinforce_plus_plus_baseline' advantage estimators "
            "require advantage normalization. Please add `--normalize-advantages` to your command."
        )

    if args.eps_clip_high is None:
        args.eps_clip_high = args.eps_clip

    if args.eval_reward_key is None:
        args.eval_reward_key = args.reward_key

    if args.dump_details is not None:
        args.save_debug_rollout_data = f"{args.dump_details}/rollout_data/{{rollout_id}}.pt"
        args.save_debug_train_data = f"{args.dump_details}/train_data/{{rollout_id}}_{{rank}}.pt"

    if args.load_debug_rollout_data is not None:
        logger.info(
            f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
            "will not instantiate sglang servers and will only run the training process."
        )
        args.debug_train_only = True

    if args.offload:
        args.offload_train = True
        args.offload_rollout = True
    del args.offload

    if args.debug_rollout_only:
        if args.colocate and (not args.rollout_num_gpus):
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        else:
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        args.colocate = False
        args.offload_train = args.offload_rollout = False
        if args.train_memory_margin_bytes > 0:
            logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
            args.train_memory_margin_bytes = 0

    assert not (args.debug_rollout_only and args.debug_train_only), (
        "debug_rollout_only and debug_train_only cannot be set at the same time, " "please set only one of them."
    )

    # always true on offload for colocate at the moment.
    if args.colocate:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        if args.rollout_num_gpus != args.actor_num_gpus_per_node * args.actor_num_nodes:
            logger.info(
                f"rollout_num_gpus {args.rollout_num_gpus} != actor_num_gpus_per_node {args.actor_num_gpus_per_node} "
                f"* actor_num_nodes {args.actor_num_nodes}, overriding rollout_num_gpus to match actor_num_gpus_per_node * actor_num_nodes."
            )
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes

    if args.offload_train is None:
        args.offload_train = False
    if args.offload_rollout is None:
        args.offload_rollout = False

    if args.eval_function_path is None:
        args.eval_function_path = args.rollout_function_path

    if args.num_steps_per_rollout is not None:
        samples_per_rollout = args.rollout_batch_size * args.n_samples_per_prompt
        derived_gbs = samples_per_rollout // args.num_steps_per_rollout
        if args.global_batch_size is not None and args.global_batch_size != derived_gbs:
            raise ValueError(
                f"global_batch_size={args.global_batch_size} contradicts "
                f"rollout_batch_size×n_samples_per_prompt÷num_steps_per_rollout={derived_gbs}; "
                f"do not pass both."
            )
        args.global_batch_size = derived_gbs

    dp_size = args.actor_num_gpus_per_node * args.actor_num_nodes
    if args.global_batch_size is not None:
        assert args.global_batch_size % dp_size == 0, (
            f"global_batch_size {args.global_batch_size} is not divisible by dp_size {dp_size}"
        )
    else:
        args.global_batch_size = dp_size

    if args.n_samples_per_prompt == 1:
        args.grpo_std_normalization = False
        logger.info("n_samples_per_prompt is set to 1, grpo_std_normalization will be set to False.")

    if args.over_sampling_batch_size is None:
        args.over_sampling_batch_size = args.rollout_batch_size

    assert args.over_sampling_batch_size >= args.rollout_batch_size, (
        f"over_sampling_batch_size {args.over_sampling_batch_size} should be greater than or equal to "
        f"rollout_batch_size {args.rollout_batch_size}"
    )

    if args.num_epoch is not None:
        if args.num_rollout is not None:
            logger.info("Both num_epoch and num_rollout are set, num_epoch will be ignored.")
        else:
            assert args.rollout_global_dataset, (
                "num_epoch is set, but rollout_global_dataset is not set, "
                "please remove --disable-rollout-global-dataset to use num_epoch"
            )
    else:
        # if num_epoch is not set, we should set num_rollout
        assert args.num_rollout is not None, (
            "num_epoch is not set, but num_rollout is not set, " "please set --num-rollout or --num-epoch"
        )

    if args.custom_config_path:
        with open(args.custom_config_path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(args, k):
                logger.info(f"Warning: Argument {k} is already set to {getattr(args, k)}, will override with {v}.")
            setattr(args, k, v)

def hf_validate_args(args, hf_config):
    def equal(x, y):
        return x == y

    errors = []

    # multimodal models have different config structure
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config

    for hf_config_name, megatron_config_name, compare_fn in [
        ("hidden_size", "hidden_size", equal),
        ("num_attention_heads", "num_attention_heads", equal),
        ("num_hidden_layers", "num_layers", equal),
        ("intermediate_size", "ffn_hidden_size", equal),
        ("tie_word_embeddings", "untie_embeddings_and_output_weights", lambda x, y: not x == y),
        ("rms_norm_eps", "norm_epsilon", equal),
        ("rope_theta", "rotary_base", equal),
    ]:
        if hasattr(hf_config, hf_config_name):
            if not compare_fn(getattr(hf_config, hf_config_name), getattr(args, megatron_config_name)):
                errors.append(
                    f"{hf_config_name} in hf config {getattr(hf_config, hf_config_name)} is not equal to "
                    f"{megatron_config_name} {getattr(args, megatron_config_name)}, please check the config."
                )

    if len(errors) > 0:
        raise AssertionError("hf_validate_args failed: " + "; ".join(errors))
