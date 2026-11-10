"""One representative command line per --loss-type parses, takes its defaults and passes validation.

    policy_loss   Flow-GRPO SDE rollout
    nft           ODE rollout with --use-ema --rollout-weights ema

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
from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args, set_default_diffusion_args

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
