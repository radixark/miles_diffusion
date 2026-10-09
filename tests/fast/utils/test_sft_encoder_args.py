"""SFT argument validation: encoder requirements apply only to the built-in producer; flow shifts are per stream."""

import shlex
import sys
from argparse import Namespace

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from miles.backends.fsdp_utils.arguments import load_fsdp_args
from miles.utils.arguments import (
    get_miles_extra_args_provider,
    miles_validate_args,
    parse_stream_flow_shifts,
    set_default_diffusion_args,
)


@pytest.fixture
def args(monkeypatch: pytest.MonkeyPatch) -> Namespace:
    from miles.backends.fsdp_utils.configs import train_pipeline_config

    class Config:
        model_backend_path = "external.Backend"
        supports_cfg_training = True

        @classmethod
        def validate_args(cls, args: Namespace) -> None:
            pass

    config = Config
    monkeypatch.setattr(train_pipeline_config, "get_train_pipeline_config_cls", lambda family: config)
    monkeypatch.setattr(
        sys,
        "argv",
        shlex.split(
            "test --hf-checkpoint external-checkpoint --diffusion-model-family wan2_2 "
            "--train-only --loss-type sft_loss --prompt-data dataset.yaml "
            "--rollout-function-path external.generate --fsdp-flow-shift 5 "
            "--rollout-batch-size 1 --global-batch-size 1 --num-rollout 1 "
            "--actor-num-gpus-per-node 1 --micro-batch-size 1 "
            "--custom-convert-samples-to-train-data-path external.convert "
            "--custom-rollout-log-function-path external.log "
            "--custom-prepare-train-batch-path external.prepare --custom-loss-function-path external.loss"
        ),
    )
    result = load_fsdp_args(extra_args_provider=get_miles_extra_args_provider())
    set_default_diffusion_args(result)
    return result


def test_flow_shift_is_keyed_by_stream(args: Namespace) -> None:
    # --fsdp-flow-shift 5 is the visual shift; name=value terms key each stream.
    assert args.fsdp_flow_shift == {"visual": 5.0}
    assert parse_stream_flow_shifts("visual=12,audio=3") == {"visual": 12.0, "audio": 3.0}


@pytest.mark.parametrize(
    "flow_shift,message", [({"audio": 3.0}, "visual"), ({"visual": 5.0, "audio": float("nan")}, "positive")]
)
def test_flow_shift_rejects_missing_visual_and_bad_values(args: Namespace, flow_shift, message) -> None:
    args.fsdp_flow_shift = flow_shift
    with pytest.raises(ValueError, match=message):
        miles_validate_args(args)


def test_external_producer_needs_no_builtin_encoder(args: Namespace) -> None:
    args.diffusion_model_family = "external"
    assert args.sft_encoder_checkpoint is None
    miles_validate_args(args)


def test_builtin_producer_still_requires_checkpoint(args: Namespace) -> None:
    args.rollout_function_path = "miles.rollout.sft_rollout.generate_rollout"
    with pytest.raises(ValueError, match="sft-encoder-checkpoint"):
        miles_validate_args(args)


def test_external_producer_still_requires_sft_hooks(args: Namespace) -> None:
    args.custom_prepare_train_batch_path = None
    with pytest.raises(ValueError, match="custom-prepare-train-batch-path"):
        miles_validate_args(args)
