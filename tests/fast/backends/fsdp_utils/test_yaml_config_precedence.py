"""YAML --config precedence: explicit CLI > YAML > dataclass defaults.

Regression test for load_fsdp_args silently ignoring YAML values: the old
`if not hasattr(args, k)` guard rejected every registered option because
argparse pre-fills all dests with defaults. CPU-only, no GPU needed.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import sys
from contextlib import contextmanager

import yaml

from miles.backends.fsdp_utils.arguments import load_fsdp_args


@contextmanager
def _argv(*argv):
    old = sys.argv
    sys.argv = ["train.py", *argv]
    try:
        yield
    finally:
        sys.argv = old


def _write_config(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_yaml_overrides_default(tmp_path):
    config = _write_config(tmp_path, {"attn_implementation": "sdpa", "lr": 1e-4})
    with _argv("--config", config):
        args = load_fsdp_args()
    assert args.attn_implementation == "sdpa"
    assert args.lr == 1e-4


def test_explicit_cli_beats_yaml(tmp_path):
    config = _write_config(tmp_path, {"attn_implementation": "sdpa"})
    with _argv("--config", config, "--attn-implementation", "eager"):
        args = load_fsdp_args()
    assert args.attn_implementation == "eager"


def test_explicit_cli_beats_yaml_even_at_default_value(tmp_path):
    config = _write_config(tmp_path, {"attn_implementation": "sdpa"})
    with _argv("--config", config, "--attn-implementation", "flash_attention_2"):
        args = load_fsdp_args()
    assert args.attn_implementation == "flash_attention_2"


def test_yaml_sets_bool_flag(tmp_path):
    config = _write_config(tmp_path, {"gradient_checkpointing": True})
    with _argv("--config", config):
        args = load_fsdp_args()
    assert args.gradient_checkpointing is True


def test_cli_bool_flag_beats_yaml_false(tmp_path):
    config = _write_config(tmp_path, {"fp16": False})
    with _argv("--config", config, "--fp16"):
        args = load_fsdp_args()
    assert args.fp16 is True


def test_unknown_yaml_key_still_attached(tmp_path):
    config = _write_config(tmp_path, {"custom_reward_arg": 42})
    with _argv("--config", config):
        args = load_fsdp_args()
    assert args.custom_reward_arg == 42


def test_extra_args_provider_yaml_and_set_defaults(tmp_path):
    def provider(parser):
        parser.add_argument("--extra-knob", type=str, default="a")
        parser.set_defaults(extra_knob_two="x")
        return parser

    config = _write_config(tmp_path, {"extra_knob": "b", "extra_knob_two": "y"})
    with _argv("--config", config):
        args = load_fsdp_args(extra_args_provider=provider)
    assert args.extra_knob == "b"
    assert args.extra_knob_two == "y"

    with _argv("--config", config, "--extra-knob", "c"):
        args = load_fsdp_args(extra_args_provider=provider)
    assert args.extra_knob == "c"


def test_no_config_unchanged():
    with _argv("--attn-implementation", "eager"):
        args = load_fsdp_args()
    assert args.attn_implementation == "eager"
    assert args.config is None


if __name__ == "__main__":
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        test_yaml_overrides_default(tmp)
        test_explicit_cli_beats_yaml(tmp)
        test_explicit_cli_beats_yaml_even_at_default_value(tmp)
        test_yaml_sets_bool_flag(tmp)
        test_cli_bool_flag_beats_yaml_false(tmp)
        test_unknown_yaml_key_still_attached(tmp)
        test_extra_args_provider_yaml_and_set_defaults(tmp)
        test_no_config_unchanged()
    print("PASS: YAML --config precedence (CLI > YAML > defaults)")
