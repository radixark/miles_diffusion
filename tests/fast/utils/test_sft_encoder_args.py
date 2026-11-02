"""SFT argument validation: encoder requirements apply only to the built-in producer; flow shifts and supervised
streams are named per stream; a distilled CFG scale needs a config with CFG distilled in.

    --fsdp-flow-shift          keyed by stream; needs a finite positive visual    flow_shift_is_keyed / rejects_missing
    --diffusion-task           inside the config's tasks                          task_outside_config_tasks
    --fsdp-supervised-streams  subset of {visual, audio}, includes visual         supervised_streams_*
    rollout function           external: no encoder, still the four SFT hooks     external_producer_*
                               built-in: --sft-encoder-checkpoint                 builtin_producer_still_requires_*
    --sft-distilled-cfg-scale  config with cfg_distilled, positive                distilled_cfg_scale_*
"""

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
    parse_stream_names,
    set_default_diffusion_args,
)


@pytest.fixture
def args(monkeypatch: pytest.MonkeyPatch) -> Namespace:
    from miles.backends.fsdp_utils.configs import train_pipeline_config

    class Config:
        model_backend_path = "external.Backend"
        supports_cfg_training = True
        cfg_distilled = False
        tasks = ("t2v",)

        @classmethod
        def validate_args(cls, args: Namespace) -> None:
            pass

    config = Config
    monkeypatch.setattr(train_pipeline_config, "get_train_pipeline_config_cls", lambda family: config)
    monkeypatch.setattr(
        sys,
        "argv",
        shlex.split(
            "test --hf-checkpoint external-checkpoint --diffusion-model-family wan2_2 --diffusion-task t2v "
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


def test_task_outside_config_tasks_is_rejected(args: Namespace) -> None:
    args.diffusion_task = "t2i"
    with pytest.raises(ValueError, match="trains --diffusion-task"):
        miles_validate_args(args)


def test_supervised_streams_are_comma_separated(args: Namespace) -> None:
    # Both streams by default; "visual,audio" names them like --fsdp-flow-shift's terms.
    assert args.fsdp_supervised_streams == ["visual", "audio"]
    assert parse_stream_names("visual,audio") == ["visual", "audio"]


@pytest.mark.parametrize("streams,message", [(["audio"], "include visual"), (["visual", "audoi"], "visual and audio")])
def test_supervised_streams_reject_missing_visual_and_unknown_names(args: Namespace, streams, message) -> None:
    args.fsdp_supervised_streams = streams
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


def test_distilled_cfg_scale_needs_a_cfg_distilled_config(args: Namespace) -> None:
    args.sft_distilled_cfg_scale = 4.0
    with pytest.raises(ValueError, match="cfg_distilled=False"):
        miles_validate_args(args)


@pytest.mark.parametrize("scale", [0.0, -2.0, float("nan")], ids=["zero", "negative", "nan"])
def test_distilled_cfg_scale_must_be_positive(args: Namespace, monkeypatch: pytest.MonkeyPatch, scale) -> None:
    from miles.backends.fsdp_utils.configs import train_pipeline_config

    # The fixture's lookup returns its Config for any family.
    monkeypatch.setattr(train_pipeline_config.get_train_pipeline_config_cls("wan2_2"), "cfg_distilled", True)
    args.sft_distilled_cfg_scale = scale
    with pytest.raises(ValueError, match="must be positive"):
        miles_validate_args(args)
