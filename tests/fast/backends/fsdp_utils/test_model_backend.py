"""Backend contracts:

role checkpoint -> component loader
model family    -> lifecycle hooks and FSDP parallel plan
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import json
import logging

import diffusers
import httpx
import huggingface_hub
import pytest
import torch
from huggingface_hub.errors import GatedRepoError, RemoteEntryNotFoundError

from miles.backends.fsdp_utils import model_backend
from miles.backends.fsdp_utils.model_backend import BaseModelBackend, DiffusersModelBackend, MilesModelBackend
from miles.backends.fsdp_utils.models.diffusers import load_fsdp_parallel_plan

# MiniMaxAI/MiniMax-H3-Image at 8ced6006: transformer/ holds the engine's native layout
_H3_IMAGE_INDEX = {"_class_name": "MiniMaxH3ImagePipeline", "transformer": ["diffusers", "MiniMaxH3DiTModel"]}
_H3_IMAGE_MODULAR_INDEX = {
    "_class_name": "MiniMaxH3ImageModularPipeline",
    "transformer": [
        "diffusers",
        "MiniMaxH3ImageTransformer2DModel",
        {
            "type_hint": ["diffusers", "MiniMaxH3ImageTransformer2DModel"],
            "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3-Image",
            "subfolder": "transformer_diffusers",
            "variant": None,
            "revision": None,
        },
    ],
}
# MiniMaxAI/MiniMax-H3 at 42ed227e: both index files carry these entries
_H3_INDEX = {
    "_class_name": "MiniMaxH3ModularPipeline",
    "transformer_ref": [
        "diffusers",
        "MiniMaxH3Transformer3DModel",
        {
            "type_hint": ["diffusers", "MiniMaxH3Transformer3DModel"],
            "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
            "subfolder": "transformer_ref",
            "variant": None,
            "revision": None,
        },
    ],
}


def _write_checkpoint(path, index, modular_index=None):
    (path / "model_index.json").write_text(json.dumps(index))
    if modular_index is not None:
        (path / "modular_model_index.json").write_text(json.dumps(modular_index))
    return str(path)


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

    @pytest.mark.parametrize(
        ("index", "modular_index", "component", "expected"),
        [
            (
                _H3_IMAGE_INDEX,
                _H3_IMAGE_MODULAR_INDEX,
                "transformer",
                (diffusers.MiniMaxH3ImageTransformer2DModel, "transformer_diffusers"),
            ),
            (_H3_INDEX, _H3_INDEX, "transformer_ref", (diffusers.MiniMaxH3Transformer3DModel, "transformer_ref")),
        ],
        ids=["h3-image", "h3"],
    )
    def test_modular_index_names_the_class_and_the_folder(self, tmp_path, index, modular_index, component, expected):
        """MiniMax-H3-Image's root index names classes diffusers does not have, in a folder of another layout."""
        checkpoint = _write_checkpoint(tmp_path, index, modular_index)

        assert DiffusersModelBackend._resolve_component(checkpoint, component) == expected

    def test_component_loads_from_the_checkpoint_not_the_repository_its_spec_names(self, tmp_path):
        """A local mirror or a fine-tuned copy keeps the release's index; following it would train the release."""
        spec = {"pretrained_model_name_or_path": "org/release", "subfolder": "transformer_diffusers"}
        checkpoint = _write_checkpoint(
            tmp_path, _H3_IMAGE_INDEX, {"transformer": [__name__, "_CheckpointComponent", spec]}
        )

        model = DiffusersModelBackend(None).load_component(
            "transformer", checkpoint_path=checkpoint, master_dtype=torch.float32, materialize_weights=True
        )

        assert model.checkpoint_path == checkpoint
        assert model.load_kwargs["subfolder"] == "transformer_diffusers"

    def test_loading_spec_without_a_subfolder_is_rejected(self, tmp_path):
        checkpoint = _write_checkpoint(
            tmp_path, _H3_IMAGE_INDEX, {"transformer": [__name__, "_CheckpointComponent", {"subfolder": None}]}
        )

        with pytest.raises(ValueError, match="names no subfolder"):
            DiffusersModelBackend._resolve_component(checkpoint, "transformer")

    def test_loading_spec_with_a_weight_variant_is_rejected(self, tmp_path):
        """Loading without the variant could read the repository's other weights of the same component."""
        spec = [__name__, "_CheckpointComponent", {"subfolder": "transformer", "variant": "fp16"}]
        checkpoint = _write_checkpoint(tmp_path, _H3_IMAGE_INDEX, {"transformer": spec})

        with pytest.raises(ValueError, match="weight variant 'fp16'"):
            DiffusersModelBackend._resolve_component(checkpoint, "transformer")

    def test_hub_checkpoint_without_a_modular_index_reads_model_index(self, monkeypatch, caplog):
        caplog.set_level(logging.INFO, logger=model_backend.__name__)

        def raise_entry_not_found(repo_id, filename):
            response = httpx.Response(404, request=httpx.Request("HEAD", f"https://huggingface.co/{repo_id}"))
            raise RemoteEntryNotFoundError(f"{repo_id} has no {filename}", response=response)

        monkeypatch.setattr(model_backend, "hf_hub_download", raise_entry_not_found)
        monkeypatch.setattr(
            model_backend.DiffusionPipeline,
            "load_config",
            lambda checkpoint_path: {"transformer": [__name__, "_CheckpointComponent"]},
        )

        assert DiffusersModelBackend._resolve_component("org/classic", "transformer") == (
            _CheckpointComponent,
            "transformer",
        )
        # every classic checkpoint answers 404: a line per component load on each rank would say nothing
        assert not caplog.records

    def test_cached_classic_checkpoint_resolves_while_the_hub_is_offline(self, monkeypatch, tmp_path, caplog):
        """A classic checkpoint's cache never holds a modular index, so the probe has no copy to fall back to."""
        caplog.set_level(logging.INFO, logger=model_backend.__name__)
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setattr(
            model_backend.DiffusionPipeline,
            "load_config",
            lambda checkpoint_path: {"transformer": [__name__, "_CheckpointComponent"]},
        )

        assert DiffusersModelBackend._resolve_component("org/classic", "transformer") == (
            _CheckpointComponent,
            "transformer",
        )
        # the log names why the Hub was not asked, not the client's "check your connection" wrapper
        [record] = caplog.records
        assert "offline mode is enabled" in record.getMessage()

    def test_cached_modular_checkpoint_resolves_from_its_modular_index_while_the_hub_is_offline(
        self, monkeypatch, tmp_path
    ):
        """Falling through to the root index here would name the engine's native classes for MiniMax-H3-Image."""
        snapshot = tmp_path / "models--org--modular" / "snapshots" / ("0" * 40)
        snapshot.mkdir(parents=True)
        (snapshot / "modular_model_index.json").write_text(json.dumps(_H3_IMAGE_MODULAR_INDEX))
        refs = tmp_path / "models--org--modular" / "refs"
        refs.mkdir()
        (refs / "main").write_text("0" * 40)
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))

        assert DiffusersModelBackend._resolve_component("org/modular", "transformer") == (
            diffusers.MiniMaxH3ImageTransformer2DModel,
            "transformer_diffusers",
        )

    def test_cached_classic_checkpoint_resolves_when_the_hub_refuses_the_probe(self, monkeypatch):
        """A gated repo answers 401 without a token, also for a file it does not have."""

        def raise_gated(repo_id, filename):
            response = httpx.Response(401, request=httpx.Request("HEAD", f"https://huggingface.co/{repo_id}"))
            raise GatedRepoError(f"{repo_id} is gated", response=response)

        monkeypatch.setattr(model_backend, "hf_hub_download", raise_gated)
        monkeypatch.setattr(
            model_backend.DiffusionPipeline,
            "load_config",
            lambda checkpoint_path: {"transformer": [__name__, "_CheckpointComponent"]},
        )

        assert DiffusersModelBackend._resolve_component("org/gated", "transformer") == (
            _CheckpointComponent,
            "transformer",
        )

    def test_missing_local_checkpoint_is_reported_as_a_missing_directory(self, tmp_path):
        """A mistyped path is not a repo id; the probe must not turn it into a repo-id error."""
        with pytest.raises(OSError, match="not the path to a directory containing a model_index.json"):
            DiffusersModelBackend._resolve_component(str(tmp_path / "missing"), "transformer")

    def test_unresolvable_class_names_the_index_it_was_read_from(self, tmp_path):
        """With a partial cache the root index is read; its native class must not look like the modular one's."""
        checkpoint = _write_checkpoint(tmp_path, {"transformer": ["diffusers", "NoSuchNativeModel"]})

        with pytest.raises(ValueError, match="from model_index.json of"):
            DiffusersModelBackend._resolve_component(checkpoint, "transformer")
