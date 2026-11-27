"""One representative command line per --loss-type parses, takes its defaults and passes validation.

    policy_loss   Flow-GRPO SDE rollout
    nft           ODE rollout with --use-ema --rollout-weights ema
    sft_loss      --train-only with an external producer; the SFT prepare / loss hooks need no path
    dmd2          --teacher-load, ODE rollout stopped at a random step; the critic group's and the actor's hooks
                  come from the loss hub by role, so the trainer args carry no hook path

Per-stream SDE settings and KL betas: one value for every stream, or one for each stream of the run, each named
once. One KL beta > 0 over ode and scored streams is rejected: their KL terms differ in scale. Any stream's beta > 0
turns on the lora_base reference by default.
The log-prob form the request asks for follows the SDE type; a custom step backend goes to the scored streams.
One rejected command line per validator shows the parser still calls it.

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
        lora_target_modules = ("to_q",)

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


@pytest.mark.parametrize(("teacher_scale", "fetch_scale"), [("1.0", None), ("4.5", 4.5)])
def test_a_dmd2_teacher_that_runs_cfg_has_the_rollout_fetch_the_neg_cond(parse, teacher_scale, fetch_scale):
    """The student's unguided rollouts return no negative cond; left unset, the teacher's first CFG forward fails."""
    argv = ARGV_BY_LOSS_TYPE["dmd2"].replace(
        "--dmd2-teacher-guidance-scale 1.0", f"--dmd2-teacher-guidance-scale {teacher_scale}"
    )
    assert parse(argv).rollout_neg_cond_guidance_scale == fetch_scale
    other_loss = ARGV_BY_LOSS_TYPE["policy_loss"] + f" --dmd2-teacher-guidance-scale {teacher_scale}"
    assert parse(other_loss).rollout_neg_cond_guidance_scale is None


def test_a_dmd2_teacher_that_runs_cfg_is_rejected_on_a_model_that_trains_unguided(parse, monkeypatch):
    """Such a model's forward drops the negative branch, so the teacher would score unguided without any error."""
    from miles.backends.fsdp_utils.configs import train_pipeline_config

    monkeypatch.setattr(train_pipeline_config.get_train_pipeline_config_cls("sd3"), "supports_cfg_training", False)
    with pytest.raises(ValueError, match="trains unguided"):
        parse(
            ARGV_BY_LOSS_TYPE["dmd2"].replace("--dmd2-teacher-guidance-scale 1.0", "--dmd2-teacher-guidance-scale 4.5")
        )


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


def test_no_kl_beta_means_no_kl_on_any_stream_and_no_reference(parse):
    """A default that left a stream unresolved, or turned the reference on, would change every recipe without KL."""
    args = parse(ARGV_BY_LOSS_TYPE["policy_loss"] + " --rollout-extra-streams echo")
    assert args.diffusion_kl_beta == {"main": 0.0, "echo": 0.0}
    assert args.ref_mode == "none"


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


def test_one_kl_beta_covers_every_scored_stream(parse):
    """The recipes pass one positive beta; a rule against one beta over mixed forms must not catch scored streams."""
    args = parse(
        ARGV_BY_LOSS_TYPE["policy_loss"] + " --rollout-extra-streams echo --diffusion-kl-beta 0.04 --use-lora"
    )
    assert args.diffusion_kl_beta == {"main": 0.04, "echo": 0.04}
    assert args.ref_mode == "lora_base"


def test_a_kl_beta_on_the_ode_stream_alone_turns_on_the_lora_base_reference(parse):
    """Reading one beta for the reference default would leave a run with only an extra stream's KL without one."""
    args = parse(
        ARGV_BY_LOSS_TYPE["policy_loss"]
        + " --rollout-extra-streams echo --diffusion-sde-type main=sde,echo=ode --diffusion-kl-beta main=0,echo=0.04"
        + " --use-lora"
    )
    assert args.diffusion_kl_beta == {"main": 0.0, "echo": 0.04}
    assert args.ref_mode == "lora_base"


@pytest.mark.parametrize(
    ("kl_beta", "loss", "raises"),
    [
        pytest.param("0.04", "--loss-type policy_loss", True, id="one beta over both"),
        pytest.param("0", "--loss-type policy_loss", False, id="no KL"),
        pytest.param("main=0.04,echo=0.04", "--loss-type policy_loss", False, id="a beta per stream"),
        pytest.param(
            "0.04",
            "--loss-type custom_loss --custom-loss-function-path miles.backends.fsdp_utils.loss_hub.nft.nft_loss_formula",
            False,
            id="a custom formula with its own KL",
        ),
    ],
)
def test_one_kl_beta_over_ode_and_scored_streams_is_rejected(parse, kl_beta, loss, raises):
    """One beta would hold the ode stream's velocity MSE 60-1000x harder per element than the scored stream's KL."""
    argv = (
        ARGV_BY_LOSS_TYPE["policy_loss"].replace("--loss-type policy_loss", loss)
        + f" --rollout-extra-streams echo --diffusion-sde-type main=sde,echo=ode --diffusion-kl-beta {kl_beta}"
        + " --use-lora"
    )
    if raises:
        with pytest.raises(ValueError, match="differ in scale; give one per stream"):
            parse(argv)
    else:
        parse(argv)


@pytest.mark.parametrize(
    ("loss_type", "extra_streams", "match"),
    [
        pytest.param("policy_loss", "audio", "trains the extra streams", id="a stream the model cannot train"),
        pytest.param("policy_loss", "echo,echo", "twice", id="a stream named twice"),
        pytest.param("nft", "echo", "reads the primary stream alone", id="nft: its converter drops the stream"),
        pytest.param("dmd2", "echo", "reads the primary stream alone", id="dmd2: its converter drops the stream"),
        pytest.param("sft_loss", "echo", "reads the primary stream alone", id="sft: its converter drops the stream"),
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


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        pytest.param("--diffusion-sde-type cps", True, id="cps drops the constants"),
        pytest.param(
            "--diffusion-sde-type sde --diffusion-log-prob-no-const --sde-step-backend-path my_pkg.NoConstBackend",
            True,
            id="the flag drops them for sde",
        ),
        pytest.param("--diffusion-sde-type sde", False, id="sde keeps them"),
    ],
)
def test_the_request_asks_for_the_log_prob_form_the_trainer_scores(parse, flags, expected):
    """A cps rollout that kept the constants would be scored by a backend that drops them: every ratio would be off."""
    argv = ARGV_BY_LOSS_TYPE["policy_loss"].replace("--diffusion-sde-type sde", flags)
    assert parse(argv).diffusion_log_prob_no_const == {"main": expected}


def test_a_custom_step_backend_is_given_to_the_scored_streams_only(parse):
    """An ode stream with a backend would be scored: a zero-noise transition in the joint log-prob."""
    args = parse(
        ARGV_BY_LOSS_TYPE["policy_loss"]
        + " --sde-step-backend-path my_pkg.Backend --rollout-extra-streams echo --diffusion-sde-type main=sde,echo=ode"
    )
    assert args.sde_step_backend_path == {"main": "my_pkg.Backend"}


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        pytest.param(
            "--diffusion-sde-type cps --diffusion-noise-level 0", "must be positive", id="the SDE scoring rules"
        ),
        pytest.param(
            "--diffusion-score-centering --diffusion-debug-mode --rollout-extra-streams echo",
            "drop --rollout-extra-streams",
            id="the score-centering rules",
        ),
    ],
)
def test_the_validators_run_on_a_parsed_command_line(parse, flags, match):
    """Each rule is tested on a hand-built namespace; a validator the parser no longer calls would reject nothing."""
    with pytest.raises(ValueError, match=match):
        parse(ARGV_BY_LOSS_TYPE["policy_loss"].replace("--diffusion-sde-type sde", "") + " " + flags)


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
