"""--custom-config-path is applied right after parsing, so a YAML value is parsed, checked and derived from exactly as
the same value given on the command line."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import shlex
import sys
from argparse import Namespace

import pytest

from miles.backends.fsdp_utils.arguments import load_fsdp_args
from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args, set_default_diffusion_args

BASE_ARGV = (
    "test --hf-checkpoint checkpoint --diffusion-model-family sd3 --diffusion-task t2i --prompt-data prompts.jsonl "
    "--rollout-batch-size 4 --n-samples-per-prompt 2 --num-rollout 1 --actor-num-gpus-per-node 2 "
    "--rollout-num-gpus 2 --rollout-num-gpus-per-engine 1 --micro-batch-size 1 "
    "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout --rm-type pickscore "
    "--loss-type policy_loss --diffusion-sde-type sde "
)


@pytest.fixture
def parse(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from miles.backends.fsdp_utils.configs import train_pipeline_config

    class _FakePipelineConfig:
        model_backend_path = "external.Backend"
        supports_cfg_training = True
        tasks = ("t2i",)
        lora_target_modules = ["to_q"]

        @classmethod
        def validate_args(cls, args: Namespace) -> None:
            pass

    monkeypatch.setattr(train_pipeline_config, "get_train_pipeline_config_cls", lambda family: _FakePipelineConfig)

    def run(flags: str = "", yaml_text: str | None = None) -> Namespace:
        argv = BASE_ARGV + flags
        if yaml_text is not None:
            path = tmp_path / "custom.yaml"
            path.write_text(yaml_text)
            argv += f" --custom-config-path {path}"
        monkeypatch.setattr(sys, "argv", shlex.split(argv))
        args = load_fsdp_args(extra_args_provider=get_miles_extra_args_provider())
        set_default_diffusion_args(args)
        miles_validate_args(args)
        return args

    return run


def _comparable(args: Namespace) -> dict:
    return {key: value for key, value in vars(args).items() if key != "custom_config_path"}


# Each case: the flag on the command line, and the same value in the YAML. A YAML applied after validation would miss
# the derivations (ref_mode from the KL beta, num_steps_per_rollout from global_batch_size); one not parsed like the
# command line would keep a string 1e-4, a raw dict or list, or skip a choice.
@pytest.mark.parametrize(
    ("flags", "yaml_text"),
    [
        pytest.param("--lr 1e-4", "lr: 1e-4", id="a float PyYAML reads as a string"),
        pytest.param("--global-batch-size 4", "global_batch_size: 4", id="num_steps_per_rollout is re-derived"),
        pytest.param(
            "--use-lora --diffusion-kl-beta 0.04",
            "use_lora: true\ndiffusion_kl_beta: 0.04",
            id="ref_mode is derived from the KL beta",
        ),
        pytest.param("--train-dp-split-mode stride", "train_dp_split_mode: stride", id="a choice"),
        pytest.param(
            "--dmd2-score-sigma-range 0.1 0.9", "dmd2_score_sigma_range: [0.1, 0.9]", id="a two-value argument"
        ),
        pytest.param("--disable-rollout-global-dataset", "rollout_global_dataset: false", id="a store_false dest"),
        pytest.param("--fsdp-flow-shift visual=3.0", "fsdp_flow_shift: {visual: 3.0}", id="a per-stream mapping"),
    ],
)
def test_a_yaml_value_is_the_command_line_value(parse, flags, yaml_text):
    assert _comparable(parse(yaml_text=yaml_text)) == _comparable(parse(flags))


# Each case: a YAML value the command line would reject, or a key that names no argument; set late or unparsed, it
# would be taken as given.
@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        pytest.param("train_dp_split_mode: strided", "is not one of", id="a value outside the choices"),
        pytest.param("use_lora: 'false'", "takes a YAML boolean", id="a string for an on/off flag"),
        pytest.param("lr: fast", "is not a valid --lr", id="a value the type rejects"),
        pytest.param("dmd2_score_sigma_range: [0.1]", "takes 2 values", id="the wrong number of values"),
        pytest.param("not_an_argument: 1", "no argument's dest", id="a key that is no argument"),
        pytest.param("global-batch-size: 4", "no argument's dest", id="a flag name instead of its dest"),
        pytest.param("disable_rollout_global_dataset: true", "no argument's dest", id="a flag whose dest differs"),
    ],
)
def test_a_yaml_value_the_command_line_would_reject_is_rejected(parse, yaml_text, message):
    with pytest.raises(ValueError, match=message):
        parse(yaml_text=yaml_text)
