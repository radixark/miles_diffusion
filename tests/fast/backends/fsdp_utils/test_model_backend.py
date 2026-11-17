"""Backend contracts:

role checkpoint -> component loader
model family    -> lifecycle hooks and FSDP parallel plan
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import json
from itertools import chain

import pytest
import torch
from diffusers import ConfigMixin, ModelMixin, SD3Transformer2DModel, WanTransformer3DModel
from diffusers.configuration_utils import register_to_config

from miles.backends.fsdp_utils.model_backend import BaseModelBackend, DiffusersModelBackend, MilesModelBackend
from miles.backends.fsdp_utils.model_loader import model_init_context
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


class _PinnedComponent(torch.nn.Module):
    _keep_in_fp32_modules = ["norm"]

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.norm = torch.nn.LayerNorm(2)

    @classmethod
    def from_pretrained(cls, checkpoint_path, *, torch_dtype, **kwargs):
        assert cls._keep_in_fp32_modules is None
        return cls().to(torch_dtype)


class _PinnedBufferModel(ModelMixin, ConfigMixin):
    """Persistent buffers (BatchNorm's running statistics) under a module the class pins to fp32."""

    _keep_in_fp32_modules = ["norm"]

    @register_to_config
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.norm = torch.nn.BatchNorm1d(2)


def _tiny_sd3():
    # Its pos_embed is a persistent buffer the constructor makes in fp32 whatever the default dtype.
    return SD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        caption_projection_dim=16,
        pooled_projection_dim=16,
        out_channels=4,
        pos_embed_max_size=8,
    )


def _tiny_wan():
    # Pins modules to fp32 and has buffers outside the state dict (the rope tables).
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=12,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=1,
        cross_attn_norm=True,
        rope_max_seq_len=32,
    )


@pytest.mark.parametrize(
    "make, class_spec",
    [
        pytest.param(_tiny_sd3, ["diffusers", "SD3Transformer2DModel"], id="a persistent buffer made in fp32"),
        pytest.param(_tiny_wan, ["diffusers", "WanTransformer3DModel"], id="modules pinned to fp32"),
        pytest.param(_PinnedBufferModel, [__name__, "_PinnedBufferModel"], id="a persistent buffer under the pin"),
    ],
)
@pytest.mark.parametrize("master_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_a_component_built_without_weights_has_the_dtypes_a_load_with_weights_gives(
    tmp_path, make, class_spec, master_dtype
):
    # Every rank may build on meta and read its shards from the files, so no loaded model is there to copy from.
    make().save_pretrained(tmp_path / "transformer")
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "DiffusionPipeline", "transformer": class_spec})
    )
    dtypes = {}
    for materialize_weights in (True, False):
        with model_init_context(materialize_weights=materialize_weights):
            model = DiffusersModelBackend(None).load_component(
                "transformer",
                checkpoint_path=str(tmp_path),
                master_dtype=master_dtype,
                materialize_weights=materialize_weights,
            )
        tensors = chain(model.named_parameters(), model.named_buffers())
        dtypes[materialize_weights] = {name: tensor.dtype for name, tensor in tensors}
    assert dtypes[False] == dtypes[True]


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

    def test_a_component_built_without_weights_keeps_its_fp32_modules(self, tmp_path):
        # Every rank may build on meta, so rank 0's loaded dtypes are not there to copy the fp32 pin from.
        (tmp_path / "model_index.json").write_text(
            json.dumps({"_class_name": "DiffusionPipeline", "transformer": [__name__, "_PinnedComponent"]})
        )

        model = DiffusersModelBackend(None).load_component(
            "transformer", checkpoint_path=str(tmp_path), master_dtype=torch.bfloat16, materialize_weights=False
        )

        assert {name: parameter.dtype for name, parameter in model.named_parameters()} == {
            "proj.weight": torch.bfloat16,
            "proj.bias": torch.bfloat16,
            "norm.weight": torch.float32,
            "norm.bias": torch.float32,
        }
        assert _PinnedComponent._keep_in_fp32_modules == ["norm"]

    def test_a_hub_checkpoint_resolves_to_the_weight_files_of_its_snapshot(self, tmp_path, monkeypatch):
        folder = tmp_path / "transformer"
        folder.mkdir()
        (folder / "diffusion_pytorch_model.safetensors").touch()
        (folder / "diffusion_pytorch_model.fp16.safetensors").touch()
        downloads = []

        def snapshot_download(repo_id, *, allow_patterns):
            downloads.append((repo_id, allow_patterns))
            return str(tmp_path)

        monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

        files = DiffusersModelBackend(None).safetensors_files("transformer", checkpoint_path="org/model")

        assert files == [folder / "diffusion_pytorch_model.safetensors"]
        assert downloads == [
            (
                "org/model",
                [
                    "transformer/diffusion_pytorch_model.safetensors*",
                    "transformer/diffusion_pytorch_model-*.safetensors",
                ],
            )
        ]
