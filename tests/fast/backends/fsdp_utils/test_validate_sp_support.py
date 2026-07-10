from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest
import torch

from miles.backends.fsdp_utils.configs.qwen_image import QwenImageTrainPipelineConfig
from miles.backends.fsdp_utils.configs.wan2_2 import Wan2_2TrainPipelineConfig
from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend, ModelBackend, validate_sp_support

DIFFUSERS_BACKEND_PATH = "miles.backends.fsdp_utils.model_backend.DiffusersModelBackend"


def _args(**overrides):
    defaults = dict(fsdp_attention_backend=None, model_backend_path=DIFFUSERS_BACKEND_PATH)
    defaults.update(overrides)
    return Namespace(**defaults)


class TestValidateSpSupport:
    def test_wan_family_passes(self):
        validate_sp_support(_args(), Wan2_2TrainPipelineConfig)

    # Under SP the attention installer replaces every processor, so an explicit
    # backend selection can never take effect.
    def test_explicit_attention_backend_rejected(self):
        with pytest.raises(ValueError, match="fsdp-attention-backend"):
            validate_sp_support(_args(fsdp_attention_backend="flash"), Wan2_2TrainPipelineConfig)

    def test_family_without_sp_attention_rejected(self):
        with pytest.raises(ValueError, match="apply_sp_attention"):
            validate_sp_support(_args(), QwenImageTrainPipelineConfig)

    def test_backend_without_plan_rejected(self):
        path = f"{__name__}._PlanlessBackend"
        with pytest.raises(ValueError, match="does not support sequence parallelism"):
            validate_sp_support(_args(model_backend_path=path), QwenImageTrainPipelineConfig)

    # A native backend owning its whole plan is accepted without a config hook.
    def test_backend_with_own_plan_passes(self):
        path = f"{__name__}._OwnPlanBackend"
        validate_sp_support(_args(model_backend_path=path), QwenImageTrainPipelineConfig)


class _PlanlessBackend(ModelBackend):
    def load_models_and_scheduler(self, args, *, master_dtype):
        raise NotImplementedError


class _OwnPlanBackend(ModelBackend):
    def load_models_and_scheduler(self, args, *, master_dtype):
        raise NotImplementedError

    def sequence_parallel_plan(self, model):
        raise NotImplementedError


class TestPlanConstruction:
    def test_wildcard_boundaries_rejected(self):
        class _WildcardModel(torch.nn.Module):
            _cp_plan = {"blocks.*": None}

        with pytest.raises(ValueError, match="wildcard"):
            DiffusersModelBackend(Wan2_2TrainPipelineConfig()).sequence_parallel_plan(_WildcardModel())

    def test_missing_cp_plan_rejected(self):
        with pytest.raises(ValueError, match="_cp_plan"):
            DiffusersModelBackend(Wan2_2TrainPipelineConfig()).sequence_parallel_plan(torch.nn.Linear(2, 2))
