"""One representative command line per --loss-type parses, takes its defaults and passes validation.

    policy_loss   Flow-GRPO SDE rollout
    nft           ODE rollout with --use-ema --rollout-weights ema
    sft_loss      --train-only with an external producer; the SFT prepare / loss hooks need no path
    dmd2          --teacher-load, ODE rollout stopped at a random step; the critic group's and the actor's hooks
                  come from the loss hub by role, so the trainer args carry no hook path

The arguments pass through load_fsdp_args, set_default_diffusion_args and miles_validate_args exactly as parse_args
runs them, so a validator that reads a value the defaults no longer fill fails here, not on the launch node.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import shlex
import sys
from argparse import Namespace

import pytest

from miles.backends.fsdp_utils.arguments import load_fsdp_args
from miles.backends.fsdp_utils.loss_hub import LOSS_HOOKS, get_loss_hooks
from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args, set_default_diffusion_args
from miles.utils.trainer_config import compute_trainer_args, compute_trainer_configs

COMMON_ARGV = (
    "test --hf-checkpoint checkpoint --diffusion-model-family sd3 --diffusion-task t2i --prompt-data prompts.jsonl "
    "--rollout-batch-size 4 --n-samples-per-prompt 2 --global-batch-size 8 --num-rollout 1 --num-steps-per-rollout 1 "
    "--actor-num-gpus-per-node 2 --rollout-num-gpus 2 --rollout-num-gpus-per-engine 1 --micro-batch-size 1 "
    "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout --rm-type pickscore "
)
ARGV_BY_LOSS_TYPE = {
    "policy_loss": COMMON_ARGV + "--loss-type policy_loss --diffusion-sde-type sde",
    "nft": COMMON_ARGV
    + "--loss-type nft --diffusion-sde-type ode --diffusion-noise-level 0.0 --use-ema --rollout-weights ema",
    "sft_loss": (
        "test --hf-checkpoint checkpoint --diffusion-model-family wan2_2 --diffusion-task t2v --train-only "
        "--loss-type sft_loss --prompt-data dataset.yaml --rollout-function-path external.generate "
        "--fsdp-flow-shift 5 --rollout-batch-size 1 --global-batch-size 1 --num-rollout 1 "
        "--actor-num-gpus-per-node 1 --micro-batch-size 1 "
        "--custom-convert-samples-to-train-data-path external.convert "
        "--custom-rollout-log-function-path external.log"
    ),
    "dmd2": COMMON_ARGV
    + "--loss-type dmd2 --teacher-load teacher --dmd2-teacher-guidance-scale 1.0 --dmd2-critic-steps-per-rollout 2 "
    "--fsdp-flow-shift visual=3.0 --diffusion-sde-type ode --diffusion-guidance-scale 1.0 --rm-type none "
    "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.stop_at_random_step",
}


@pytest.fixture
def parse(monkeypatch: pytest.MonkeyPatch):
    from miles.backends.fsdp_utils.configs import train_pipeline_config

    class _FakePipelineConfig:
        model_backend_path = "external.Backend"
        supports_cfg_training = True
        tasks = ("t2i", "t2v")

        @classmethod
        def validate_args(cls, args: Namespace) -> None:
            pass

    monkeypatch.setattr(train_pipeline_config, "get_train_pipeline_config_cls", lambda family: _FakePipelineConfig)

    def run(argv: str) -> Namespace:
        monkeypatch.setattr(sys, "argv", shlex.split(argv))
        args = load_fsdp_args(extra_args_provider=get_miles_extra_args_provider())
        set_default_diffusion_args(args)
        miles_validate_args(args)
        return args

    return run


@pytest.mark.parametrize("loss_type", list(ARGV_BY_LOSS_TYPE))
def test_each_loss_type_recipe_parses_and_validates(parse, loss_type):
    args = parse(ARGV_BY_LOSS_TYPE[loss_type])
    assert args.loss_type == loss_type


def test_dmd2_trainer_groups_get_their_hooks_by_role(parse):
    args = parse(ARGV_BY_LOSS_TYPE["dmd2"])

    critic_config, actor_config = compute_trainer_configs(args)
    critic = compute_trainer_args(args, critic_config)
    actor = compute_trainer_args(args, actor_config)

    assert get_loss_hooks(critic, critic_config.role) is LOSS_HOOKS[("dmd2", "critic")]
    assert get_loss_hooks(actor, actor_config.role) is LOSS_HOOKS[("dmd2", "actor")]
    # No hook path in the parsed args: the overrides are the whole critic / actor difference.
    assert vars(critic) == {
        **vars(args),
        "hf_checkpoint": "teacher",
        "num_steps_per_rollout": 2,
        "global_batch_size": 4,
    }
    assert vars(actor) == {**vars(args), "teacher_load": None}


def test_a_hook_path_outside_custom_loss_fails_validation(parse):
    with pytest.raises(ValueError, match="read only under --loss-type custom_loss"):
        parse(ARGV_BY_LOSS_TYPE["nft"] + " --teacher-load teacher --custom-loss-function-path my.formula")
