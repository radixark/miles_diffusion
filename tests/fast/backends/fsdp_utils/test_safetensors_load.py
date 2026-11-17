"""Every rank reads its own shards of a diffusers safetensors checkpoint.

    the index's files, or the single file, hold the state dict's names and shapes --> every rank on meta --> dcp.load
    a variant file beside them                                                    --> not read
    listed file missing / other name / other shape / a rank that disagrees / LoRA --> rank 0 loads and broadcasts

FSDP2 is real on a one-rank CPU mesh and the broadcast is recorded; the CUDA worker splits the shards over two ranks.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=["fsdp"])

import json
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from tests.fast.backends.fsdp_utils.test_reference_models import (
    TinyBackend,
    TinyPipelineConfig,
    make_model_loader_args,
)
from torch.distributed.device_mesh import init_device_mesh

from miles.backends.fsdp_utils import model_loader
from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend

BROADCAST_VALUE = 0.75


class FileBackend(TinyBackend):
    safetensors_files = DiffusersModelBackend.safetensors_files

    def __init__(self):
        super().__init__()
        self.materialize_requests = []

    def load_component(self, component, *, checkpoint_path, master_dtype, materialize_weights):
        self.materialize_requests.append(materialize_weights)
        model = super().load_component(
            component, checkpoint_path="actor", master_dtype=master_dtype, materialize_weights=False
        )
        if materialize_weights:
            with torch.no_grad():
                model.block.proj.weight.fill_(BROADCAST_VALUE)
        return model


def saved_weight(component):
    return torch.arange(12, dtype=torch.bfloat16).reshape(4, 3) + (100 if component == "transformer_2" else 0)


def save_checkpoint(path, *, name="block.proj.weight", shape=(4, 3), sharded=True):
    for component in make_model_loader_args().update_weight_target_modules:
        folder = path / component
        folder.mkdir(parents=True)
        if not sharded:
            save_file({name: saved_weight(component).reshape(shape)}, folder / "diffusion_pytorch_model.safetensors")
            continue
        file = "diffusion_pytorch_model-00001-of-00001.safetensors"
        save_file({name: saved_weight(component).reshape(shape)}, folder / file)
        (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: file}})
        )


def load(monkeypatch, path, *, agree=lambda flag: flag, **kwargs):
    broadcasts = []
    monkeypatch.setattr(model_loader.checkpoint, "all_ranks_agree", agree)
    monkeypatch.setattr(model_loader.checkpoint, "sync_model_dtypes", lambda model: None)
    monkeypatch.setattr(
        model_loader.checkpoint, "broadcast_full_state_to_fsdp", lambda model, state, **_: broadcasts.append(state)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        parallel = SimpleNamespace(get_mesh=lambda name: mesh, get_optional_mesh=lambda name: None)
        args, backend = make_model_loader_args(), FileBackend()
        args.use_lora = False
        models = model_loader.load_fsdp_models(
            args, backend, TinyPipelineConfig(), parallel, checkpoint_path=str(path), **kwargs
        )
        return models, backend.materialize_requests, broadcasts
    finally:
        dist.destroy_process_group()


def build_with(load_component):
    """build_component with the backend's load_component replaced."""
    build_component = model_loader.build_component

    def build(args, model_backend, *rest, **kwargs):
        model_backend.load_component = load_component.__get__(model_backend)
        return build_component(args, model_backend, *rest, **kwargs)

    return build


def add_variant_file(path, *, sharded):
    save_checkpoint(path, sharded=sharded)
    for component in make_model_loader_args().update_weight_target_modules:
        save_file(
            {"block.proj.weight": torch.zeros(4, 3)}, path / component / "diffusion_pytorch_model.fp16.safetensors"
        )


@pytest.mark.parametrize(
    "save",
    [
        pytest.param(save_checkpoint, id="the files the index lists"),
        pytest.param(lambda path: save_checkpoint(path, sharded=False), id="the single file"),
        pytest.param(lambda path: add_variant_file(path, sharded=True), id="a variant beside the listed files"),
        pytest.param(lambda path: add_variant_file(path, sharded=False), id="a variant beside the single file"),
    ],
)
def test_every_rank_reads_its_shards_from_the_files(tmp_path, monkeypatch, save):
    save(tmp_path)
    models, materialize_requests, broadcasts = load(monkeypatch, tmp_path)
    assert materialize_requests == [False, False] and broadcasts == []
    for component, model in models.items():
        weight = model.block.proj.weight
        assert weight.dtype == torch.float32
        assert torch.equal(weight.to_local(), saved_weight(component).float())
        assert model.forward_count.item() == 0
        torch.testing.assert_close(
            model(torch.ones(2, 3)), (torch.ones(2, 3) @ saved_weight(component).float().T).tanh()
        )
        assert model.forward_count.item() == 1


def remove_a_listed_file(path):
    save_checkpoint(path)
    (path / "transformer" / "diffusion_pytorch_model-00001-of-00001.safetensors").unlink()


@pytest.mark.parametrize(
    "save, rebuilt",
    [
        pytest.param(remove_a_listed_file, False, id="a listed file is missing"),
        pytest.param(lambda path: save_checkpoint(path, name="block.proj.kernel"), True, id="another name"),
        pytest.param(lambda path: save_checkpoint(path, shape=(3, 4)), True, id="another shape"),
    ],
)
def test_files_that_are_not_the_state_dict_fall_back_to_the_broadcast(tmp_path, monkeypatch, save, rebuilt):
    save(tmp_path)
    models, materialize_requests, broadcasts = load(monkeypatch, tmp_path)
    assert materialize_requests[:2] == ([False, True] if rebuilt else [True, False])
    assert [set(state) for state in broadcasts[:1]] == [{"block.proj.weight"}]
    assert torch.all(broadcasts[0]["block.proj.weight"] == BROADCAST_VALUE)


def test_rank_0_built_twice_leaves_the_generator_where_one_build_does(tmp_path, monkeypatch):
    # Some diffusers constructors draw a parameter with torch.randn, on meta as with weights.
    def drawing_load_component(self, component, **kwargs):
        torch.randn(1)
        return FileBackend.load_component(self, component, **kwargs)

    save_checkpoint(tmp_path / "renamed", name="block.proj.kernel")
    (tmp_path / "no files").mkdir()
    states = {}
    for path in ("renamed", "no files"):
        monkeypatch.setattr(model_loader, "build_component", build_with(drawing_load_component))
        torch.manual_seed(0)
        _, materialize_requests, _ = load(monkeypatch, tmp_path / path)
        states[path] = (materialize_requests, torch.get_rng_state())
    assert states["renamed"][0] == [False, True, False, True] and states["no files"][0] == [True, True]
    assert torch.equal(states["renamed"][1], states["no files"][1])


def test_one_rank_that_cannot_read_the_files_sends_every_rank_to_the_broadcast(tmp_path, monkeypatch):
    save_checkpoint(tmp_path)
    flags = []
    _, materialize_requests, broadcasts = load(monkeypatch, tmp_path, agree=lambda flag: flags.append(flag) or False)
    assert flags == [True, True] and materialize_requests == [False, True, False, True] and len(broadcasts) == 2


def test_lora_models_are_broadcast(tmp_path, monkeypatch):
    save_checkpoint(tmp_path)
    monkeypatch.setattr(model_loader, "load_lora_adapter", lambda model, path, *, trainable: model)
    agree = lambda flag: pytest.fail("LoRA loads are never read from the files")  # noqa: E731
    _, materialize_requests, broadcasts = load(monkeypatch, tmp_path, agree=agree, lora_adapter_paths=["a", "b"])
    assert materialize_requests == [True, True] and len(broadcasts) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
