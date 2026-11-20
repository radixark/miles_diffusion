"""One representative command line per --loss-type parses, takes its defaults and passes validation.

    policy_loss   Flow-GRPO SDE rollout
    nft           ODE rollout with --use-ema --rollout-weights ema
    sft_loss      --train-only with an external producer; the SFT prepare / loss hooks need no path
    dmd2          --teacher-load, ODE rollout stopped at a random step; the critic group's and the actor's hooks
                  come from the loss hub by role, so the trainer args carry no hook path

Per-stream SDE settings: one value for every stream, or one for each stream of the run, each named once.

--rollout-extra-streams on top of a recipe:

    a stream the model trains, under policy_loss or custom_loss        --> accepted
    a stream it does not train / named twice / under nft                --> rejected

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
        primary_stream = "main"
        trainable_extra_streams = ("echo",)
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


def test_the_models_primary_stream_is_resolved_into_args(parse):
    """The rollout side keys train pairs by ``args.primary_stream``; left unresolved, it fails on the first GPU rollout."""
    assert parse(ARGV_BY_LOSS_TYPE["policy_loss"]).primary_stream == "main"


@pytest.mark.parametrize("loss_type", ["policy_loss", "custom_loss"])
def test_extra_rollout_streams_the_model_trains_are_accepted(parse, loss_type):
    """Rejecting custom_loss would lock out the custom converters the flag's help tells to read the streams."""
    argv = ARGV_BY_LOSS_TYPE["policy_loss"].replace("--loss-type policy_loss", f"--loss-type {loss_type}")
    assert parse(argv + " --rollout-extra-streams echo").rollout_extra_streams == ["echo"]
    assert parse(argv).rollout_extra_streams == []


def test_every_stream_of_the_run_gets_the_sde_settings(parse):
    """A stream missing from these maps raises a KeyError on the first rollout request or training step."""
    args = parse(ARGV_BY_LOSS_TYPE["policy_loss"] + " --diffusion-noise-level 0.4 --rollout-extra-streams echo")
    assert args.diffusion_sde_type == {"main": "sde", "echo": "sde"}
    assert args.diffusion_noise_level == {"main": 0.4, "echo": 0.4}
    assert args.diffusion_log_prob_no_const == {"main": False, "echo": False}
    backend = "miles.backends.fsdp_utils.sde_step_backend.FlowSdeStepBackend"
    assert args.sde_step_backend_path == {"main": backend, "echo": backend}
    # The echo stream inherits the primary's settings, so the request sends it none of its own.
    assert args.rollout_extra_stream_sde_args == {}


def test_an_extra_stream_differing_only_in_noise_level_is_sent_its_own(parse):
    """Comparing only the dynamics would send no override: the engine would roll the stream out at the primary's noise
    level while the loss scores it at its own, and with no override nothing checks the echo."""
    args = parse(
        ARGV_BY_LOSS_TYPE["policy_loss"] + " --rollout-extra-streams echo --diffusion-noise-level main=0.7,echo=0.3"
    )
    assert args.rollout_extra_stream_sde_args == {
        "echo": {"rollout_sde_type": "sde", "rollout_noise_level": 0.3, "rollout_log_prob_no_const": False}
    }


def test_the_sde_defaults_hold_for_every_stream(parse):
    """The defaults are strings the per-stream parsers turn into values; a changed default moves every recipe."""
    args = parse(ARGV_BY_LOSS_TYPE["policy_loss"].replace("--diffusion-sde-type sde", ""))
    assert args.diffusion_sde_type == {"main": "sde"}
    assert args.diffusion_noise_level == {"main": 0.7}


def test_each_stream_takes_its_own_sde_settings(parse):
    """The request must carry the extra stream's own kernel, and an ode stream gets no backend to score it with."""
    args = parse(
        ARGV_BY_LOSS_TYPE["policy_loss"]
        + " --rollout-extra-streams echo --diffusion-sde-type main=sde,echo=ode --diffusion-noise-level main=0.7,echo=0.2"
    )

    assert args.diffusion_sde_type == {"main": "sde", "echo": "ode"}
    assert args.diffusion_log_prob_no_const == {"main": False, "echo": True}
    assert args.sde_step_backend_path == {"main": "miles.backends.fsdp_utils.sde_step_backend.FlowSdeStepBackend"}
    assert args.rollout_extra_stream_sde_args == {
        "echo": {"rollout_sde_type": "ode", "rollout_noise_level": 0.2, "rollout_log_prob_no_const": True}
    }


@pytest.mark.parametrize(
    "sde_type",
    [
        pytest.param("main=sde", id="a stream left out"),
        pytest.param("main=sde,echo=sde,other=sde", id="an unknown stream"),
    ],
)
def test_per_stream_sde_settings_must_name_every_stream(parse, sde_type):
    """Filling a missing stream from a default would hide a typo in a stream name."""
    with pytest.raises(ValueError, match="give one value, or one for each of"):
        parse(ARGV_BY_LOSS_TYPE["policy_loss"] + f" --rollout-extra-streams echo --diffusion-sde-type {sde_type}")


@pytest.mark.parametrize(
    ("loss_type", "extra_streams", "match"),
    [
        pytest.param("policy_loss", "audio", "trains the extra streams", id="a stream the model cannot train"),
        pytest.param("policy_loss", "echo,echo", "twice", id="a stream named twice"),
        pytest.param("nft", "echo", "reads the primary stream alone", id="a loss whose converter drops the stream"),
    ],
)
def test_extra_rollout_streams_that_would_not_be_trained_are_rejected(parse, loss_type, extra_streams, match):
    """Each of these would otherwise fail after the engine loaded, or roll a stream out that nothing trains on."""
    with pytest.raises(ValueError, match=match):
        parse(ARGV_BY_LOSS_TYPE[loss_type] + f" --rollout-extra-streams {extra_streams}")


def test_a_stream_named_twice_in_a_per_stream_flag_is_rejected(parse, capsys):
    """A dict keeps the last of the two values, so the repeat would pass the check against the run's streams."""
    with pytest.raises(SystemExit):
        parse(
            ARGV_BY_LOSS_TYPE["policy_loss"]
            + " --rollout-extra-streams echo --diffusion-noise-level main=0.7,echo=0.3,echo=0.5"
        )
    assert "a stream is named twice" in capsys.readouterr().err


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
