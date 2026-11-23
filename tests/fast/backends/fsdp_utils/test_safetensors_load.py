"""Every rank reads its own shards of a diffusers safetensors checkpoint.

    the index's files, or the single file                 --> every rank on meta --> dcp.load of its own shards
    a variant file beside them, a tensor the model lacks  --> not read
    listed file missing / LoRA                            --> rank 0 loads and broadcasts, decided before building
    files only some ranks see, files that lack a tensor   --> ValueError on every rank (never a silent fallback)
    or hold another shape, a buffer outside the state
    dict left on meta

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
    TinyComponent,
    TinyPipelineConfig,
    make_model_loader_args,
)
from torch.distributed.device_mesh import init_device_mesh

from miles.backends.fsdp_utils import checkpoint, model_loader
from miles.backends.fsdp_utils.model_backend import diffusers_weight_files

BROADCAST_VALUE = 0.75


class FileBackend(TinyBackend):
    def safetensors_files(self, component, *, checkpoint_path):
        return diffusers_weight_files(checkpoint_path, component)

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


def save_checkpoint(path, *, name="block.proj.weight", shape=(4, 3), sharded=True, extra=()):
    for component in make_model_loader_args().update_weight_target_modules:
        folder = path / component
        folder.mkdir(parents=True)
        tensors = {
            name: saved_weight(component).reshape(shape),
            **{extra_name: torch.zeros(2) for extra_name in extra},
        }
        if not sharded:
            save_file(tensors, folder / "diffusion_pytorch_model.safetensors")
            continue
        file = "diffusion_pytorch_model-00001-of-00001.safetensors"
        save_file(tensors, folder / file)
        (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {tensor_name: file for tensor_name in tensors}})
        )


def load(monkeypatch, path, *, holds=lambda condition, **_: condition, backend_cls=None, use_lora=False, **kwargs):
    broadcasts = []
    monkeypatch.setattr(model_loader.checkpoint, "holds_on_every_rank", holds)
    monkeypatch.setattr(model_loader.checkpoint, "sync_model_dtypes", lambda model: None)
    monkeypatch.setattr(
        model_loader.checkpoint, "broadcast_full_state_to_fsdp", lambda model, state, **_: broadcasts.append(state)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        parallel = SimpleNamespace(get_mesh=lambda name: mesh, get_optional_mesh=lambda name: None)
        args, backend = make_model_loader_args(), (backend_cls or FileBackend)()
        args.use_lora = use_lora
        models = model_loader.load_fsdp_models(
            args, backend, TinyPipelineConfig(), parallel, checkpoint_path=str(path), **kwargs
        )
        return models, backend.materialize_requests, broadcasts
    finally:
        dist.destroy_process_group()


def add_variant_file(path, *, sharded):
    save_checkpoint(path, sharded=sharded)
    for component in make_model_loader_args().update_weight_target_modules:
        # Another shape: a reader that took the variant fails whichever file its folder lists last.
        save_file(
            {"block.proj.weight": torch.zeros(8, 3)}, path / component / "diffusion_pytorch_model.fp16.safetensors"
        )


@pytest.mark.parametrize(
    "save",
    [
        pytest.param(save_checkpoint, id="the files the index lists"),
        pytest.param(lambda path: save_checkpoint(path, sharded=False), id="the single file"),
        pytest.param(lambda path: add_variant_file(path, sharded=True), id="a variant beside the listed files"),
        pytest.param(lambda path: add_variant_file(path, sharded=False), id="a variant beside the single file"),
        pytest.param(
            lambda path: save_checkpoint(path, extra=("block.unused",)), id="a tensor the model lacks in the files"
        ),
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


def test_a_checkpoint_with_a_listed_file_missing_is_broadcast(tmp_path, monkeypatch):
    # No files to read for the first component: the choice is per component and made before building, so rank 0
    # builds that one with weights once; the second still reads its shards.
    remove_a_listed_file(tmp_path)
    _, materialize_requests, broadcasts = load(monkeypatch, tmp_path)
    assert materialize_requests == [True, False] and len(broadcasts) == 1
    assert set(broadcasts[0]) == {"block.proj.weight"}
    assert torch.all(broadcasts[0]["block.proj.weight"] == BROADCAST_VALUE)


@pytest.mark.parametrize(
    "save",
    [
        pytest.param(lambda path: save_checkpoint(path, name="block.proj.kernel"), id="another name"),
        pytest.param(lambda path: save_checkpoint(path, shape=(3, 4)), id="another shape"),
    ],
)
def test_files_that_do_not_hold_the_state_dict_are_an_error(tmp_path, monkeypatch, save):
    # A checkpoint this path does not understand must not silently take the slow path.
    save(tmp_path)
    with pytest.raises(ValueError, match="do not hold the model's state dict"):
        load(monkeypatch, tmp_path)


class DerivedBufferComponent(TinyComponent):
    """A buffer the constructor derives from a parameter: on meta when the component is built without weights."""

    persistent = True

    def __init__(self):
        super().__init__()
        self.register_buffer("row_sums", self.block.proj.weight.sum(dim=1), persistent=self.persistent)


class TransientDerivedBufferComponent(DerivedBufferComponent):
    persistent = False


class SharedBufferComponent(TinyComponent):
    """One tensor registered as a buffer of two modules."""

    def __init__(self):
        super().__init__()
        table = torch.arange(3.0)
        self.block.register_buffer("table", table, persistent=False)
        self.register_buffer("table", table, persistent=False)


def backend_of(component_cls):
    class Backend(FileBackend):
        def load_component(self, component, *, checkpoint_path, master_dtype, materialize_weights):
            self.materialize_requests.append(materialize_weights)
            model = component_cls().to(dtype=master_dtype)
            if materialize_weights:
                with torch.no_grad():
                    model.block.proj.weight.fill_(BROADCAST_VALUE)
            return model

    return Backend


def save_row_sums(path):
    """The weight in one file and the buffer in another: the state dict is the union of what the index lists."""
    for component in make_model_loader_args().update_weight_target_modules:
        folder = path / component
        folder.mkdir(parents=True)
        files = {
            "block.proj.weight": "diffusion_pytorch_model-00001-of-00002.safetensors",
            "row_sums": "diffusion_pytorch_model-00002-of-00002.safetensors",
        }
        save_file({"block.proj.weight": saved_weight(component)}, folder / files["block.proj.weight"])
        save_file({"row_sums": saved_weight(component).sum(dim=1)}, folder / files["row_sums"])
        (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": files}))


def test_a_state_dict_buffer_the_constructor_left_on_meta_is_read_from_the_files(tmp_path, monkeypatch):
    save_row_sums(tmp_path)
    models, materialize_requests, broadcasts = load(
        monkeypatch, tmp_path, backend_cls=backend_of(DerivedBufferComponent)
    )
    assert materialize_requests == [False, False] and broadcasts == []
    for component, model in models.items():
        assert torch.equal(model.row_sums, saved_weight(component).sum(dim=1).float())
        assert model.forward_count.item() == 0


def test_a_buffer_on_meta_that_no_file_holds_is_an_error(tmp_path, monkeypatch):
    save_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="buffers outside the state dict are on meta"):
        load(monkeypatch, tmp_path, backend_cls=backend_of(TransientDerivedBufferComponent))


def test_a_buffer_two_modules_share_keeps_its_value_on_both(tmp_path, monkeypatch):
    save_checkpoint(tmp_path)
    models, materialize_requests, broadcasts = load(
        monkeypatch, tmp_path, backend_cls=backend_of(SharedBufferComponent)
    )
    assert materialize_requests == [False, False] and broadcasts == []
    for model in models.values():
        assert torch.equal(model.table, torch.arange(3.0)) and torch.equal(model.block.table, torch.arange(3.0))


@pytest.mark.parametrize(
    "answers, holds",
    [([True, True], True), ([False, False], False), pytest.param([True, False], None, id="ranks that disagree")],
)
def test_the_files_are_seen_on_every_rank_or_none_or_the_ranks_raise_together(monkeypatch, answers, holds):
    # No rank may take the read while another takes the broadcast; a copy only some ranks see is an error for all.
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda: len(answers))
    monkeypatch.setattr(
        checkpoint.dist, "all_gather_object", lambda out, answer: out.__setitem__(slice(None), answers)
    )
    if holds is None:
        with pytest.raises(ValueError, match=r"the files are seen on ranks \[0\] only"):
            checkpoint.holds_on_every_rank(answers[0], what="the files are seen")
    else:
        assert checkpoint.holds_on_every_rank(answers[0], what="the files are seen") is holds


def test_a_frozen_model_reads_its_shards_in_a_run_that_trains_a_lora(tmp_path, monkeypatch):
    # --use-lora describes the trainable model; a frozen role without adapter paths has no LoRA.
    save_checkpoint(tmp_path)
    _, materialize_requests, broadcasts = load(monkeypatch, tmp_path, use_lora=True, trainable=False)
    assert materialize_requests == [False, False] and broadcasts == []


def test_lora_models_are_broadcast(tmp_path, monkeypatch):
    save_checkpoint(tmp_path)
    monkeypatch.setattr(model_loader, "load_lora_adapter", lambda model, path, *, trainable: model)
    holds = lambda condition, **_: pytest.fail("LoRA loads are never read from the files")  # noqa: E731
    _, materialize_requests, broadcasts = load(monkeypatch, tmp_path, holds=holds, lora_adapter_paths=["a", "b"])
    assert materialize_requests == [True, True] and len(broadcasts) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
