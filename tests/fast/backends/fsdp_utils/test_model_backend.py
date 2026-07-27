from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import torch

from miles.backends.fsdp_utils.model_backend import BaseModelBackend, DiffusersModelBackend, MilesModelBackend


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

        assert model.gradient_checkpointing_enabled
        assert model.selected == "flash"
        assert backend.fsdp_no_split_modules(model) == ["TransformerBlock"]
