"""The --custom-*-path hook flags are read only under --loss-type custom_loss, so any other loss type rejects them."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.utils.arguments import validate_loss_hook_paths

HOOK_PATH_NAMES = (
    "custom_prepare_train_batch_path",
    "custom_loss_function_path",
    "custom_train_input_postprocess_path",
    "custom_train_output_function_path",
)


def _args(loss_type, **paths):
    return Namespace(loss_type=loss_type, **{**dict.fromkeys(HOOK_PATH_NAMES), **paths})


@pytest.mark.parametrize("loss_type", ["policy_loss", "nft", "sft_loss", "dmd2"])
@pytest.mark.parametrize("name", HOOK_PATH_NAMES)
def test_hook_paths_outside_custom_loss_are_rejected(loss_type, name):
    with pytest.raises(ValueError, match=f"--{name.replace('_', '-')} is read only under --loss-type custom_loss"):
        validate_loss_hook_paths(_args(loss_type, **{name: "my.hook"}))
    validate_loss_hook_paths(_args(loss_type))


def test_custom_loss_accepts_any_subset_of_hook_paths():
    validate_loss_hook_paths(_args("custom_loss", custom_loss_function_path="my.formula"))
    validate_loss_hook_paths(_args("custom_loss"))
