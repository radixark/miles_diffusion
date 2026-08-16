"""`_cvd_export` turns the recipe's device list into a `ray start` prefix:

    unset        -> ""                                     inherit the environment
    "4,5,2", 3   -> "export CUDA_VISIBLE_DEVICES=4,5,2 && " pin the raylet
    "0,1",   5   -> AssertionError                          ray would hand out unknown ids
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest

from miles.utils.external_utils.command_utils import ExecuteTrainConfig, _cvd_export


def test_unset_inherits_the_environment():
    assert _cvd_export(ExecuteTrainConfig(), num_gpus_per_node=4) == ""


def test_set_exports_for_ray_start():
    config = ExecuteTrainConfig(cuda_visible_devices="4,5,2")
    assert _cvd_export(config, num_gpus_per_node=3) == "export CUDA_VISIBLE_DEVICES=4,5,2 && "


def test_count_mismatch_is_rejected():
    config = ExecuteTrainConfig(cuda_visible_devices="0,1")
    with pytest.raises(AssertionError, match="lists 2 GPU"):
        _cvd_export(config, num_gpus_per_node=5)
