"""Backend contracts:

role checkpoint -> component loader
model family    -> lifecycle hooks and FSDP parallel plan
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import json

import torch

from miles.backends.fsdp_utils.model_backend import BaseModelBackend, DiffusersModelBackend, MilesModelBackend
from miles.backends.fsdp_utils.models.diffusers import load_fsdp_parallel_plan


class _RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.selected = None
        self.gradient_checkpointing_enabled = False
        self._no_split_modules = ["TransformerBlock"]

    def set_attention_backend(self, backend):  # diffusers protocol method
        self.selected = backend

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing_enabled = True


class _CheckpointComponent(torch.nn.Module):
    @classmethod
    def from_pretrained(cls, checkpoint_path, **kwargs):
        model = cls()
        model.checkpoint_path = checkpoint_path
        model.load_kwargs = kwargs
        return model


class TestBackendHierarchy:
    def test_concrete_backends_are_sibling_implementations(self):
        assert issubclass(MilesModelBackend, BaseModelBackend)
        assert issubclass(DiffusersModelBackend, BaseModelBackend)
        assert not issubclass(DiffusersModelBackend, MilesModelBackend)

    def test_diffusers_implements_model_lifecycle_hooks(self):
        backend = DiffusersModelBackend(None)
        model = _RecordingModel()

        backend.enable_gradient_checkpointing(model)
        backend.set_attention_backend(model, "flash")
        plan = backend.fsdp_parallel_plan(model)

        assert model.gradient_checkpointing_enabled
        assert model.selected == "flash"
        assert plan.no_split_modules == ("TransformerBlock",)
        assert plan.param_dtype_patterns == {}

    def test_all_diffusers_model_plans_load(self):
        assert load_fsdp_parallel_plan("sd3").param_dtype_patterns == {}
        assert load_fsdp_parallel_plan("qwen_image").param_dtype_patterns == {}
        assert load_fsdp_parallel_plan("wan2_2").param_dtype_patterns

    def test_component_loads_from_its_own_role_checkpoint(self, tmp_path):
        reference_checkpoint = tmp_path / "reference"
        reference_checkpoint.mkdir()
        (reference_checkpoint / "model_index.json").write_text(
            json.dumps({"_class_name": "DiffusionPipeline", "transformer": [__name__, "_CheckpointComponent"]})
        )

        model = DiffusersModelBackend(None).load_component(
            "transformer",
            checkpoint_path=str(reference_checkpoint),
            master_dtype=torch.float32,
            materialize_weights=True,
        )

        assert model.checkpoint_path == str(reference_checkpoint)
        assert model.load_kwargs == {
            "subfolder": "transformer",
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
        }
