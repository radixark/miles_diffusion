from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import torch

from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend, ModelBackend


class _RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.selected = None

    def set_attention_backend(self, backend):  # diffusers protocol method
        self.selected = backend


class TestSetAttentionBackend:
    # Default hook delegates to the model's own diffusers protocol method.
    def test_diffusers_default_delegates_to_model(self):
        model = _RecordingModel()
        DiffusersModelBackend(None).set_attention_backend(model, "flash")
        assert model.selected == "flash"

    # A custom backend overrides the hook (non-diffusers models select attention a
    # different way, or opt out) — the actor no longer calls model.set_attention_backend
    # directly, so a model without that method never crashes.
    def test_subclass_can_override(self):
        seen = {}

        class _CustomBackend(ModelBackend):
            def load_models_and_scheduler(self, args, *, master_dtype):
                raise NotImplementedError

            def set_attention_backend(self, model, backend):
                seen["backend"] = backend

        model = _RecordingModel()
        _CustomBackend(None).set_attention_backend(model, "fa3")
        assert seen["backend"] == "fa3"
        assert model.selected is None  # diffusers default path not taken

    # The concrete crash the refactor removes: a model without set_attention_backend
    # (e.g. ltx_core transformers) must not raise when a backend that opts out of the
    # diffusers protocol handles it — the actor never calls the model method directly.
    def test_model_without_method_does_not_crash(self):
        class _NoAttnModel(torch.nn.Module):
            pass  # no set_attention_backend, like a native ltx_core transformer

        class _OptOutBackend(ModelBackend):
            def load_models_and_scheduler(self, args, *, master_dtype):
                raise NotImplementedError

            def set_attention_backend(self, model, backend):
                pass  # this family selects attention its own way

        _OptOutBackend(None).set_attention_backend(_NoAttnModel(), "fa3")  # no AttributeError
