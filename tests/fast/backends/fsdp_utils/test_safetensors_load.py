"""Every rank reads its own shards of a diffusers safetensors checkpoint.

    the index's files, or the single file, hold the state dict's names and shapes --> every rank on meta --> dcp.load
    a variant file beside them                                                    --> not read
    listed file missing / other name / other shape / a rank that disagrees        --> rank 0 loads and broadcasts
    LoRA, saved or new --> every rank builds the adapter as rank 0 does, on a base without weights, and reads the base
    LoRA whose initialization decomposes the base weights (PiSSA)                  --> rank 0 loads and broadcasts

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
from peft import AdaLoraConfig, LoHaConfig, LoraConfig, get_peft_model
from safetensors.torch import save_file
from tests.fast.backends.fsdp_utils.test_reference_models import (
    TinyBackend,
    TinyComponent,
    TinyPipelineConfig,
    make_model_loader_args,
    save_component_lora_adapters,
)
from torch.distributed.device_mesh import init_device_mesh

from miles.backends.fsdp_utils import model_loader
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
        # from_pretrained initializes no weight: with weights it draws what the meta build draws.
        rng_state = torch.get_rng_state()
        model = super().load_component(
            component, checkpoint_path="actor", master_dtype=master_dtype, materialize_weights=False
        )
        if materialize_weights:
            torch.set_rng_state(rng_state)
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


def load(monkeypatch, path, *, holds=lambda condition: condition, lora_init=None, backend_cls=None, **kwargs):
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
        args.use_lora = lora_init is not None
        args.lora_rank, args.lora_alpha, args.lora_target_modules, args.lora_init_weights = 2, 4, ["proj"], lora_init
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
    for component in make_model_loader_args().update_weight_target_modules:
        file = path / component / "diffusion_pytorch_model.safetensors"
        folder_state = {"block.proj.weight": saved_weight(component), "row_sums": saved_weight(component).sum(dim=1)}
        file.parent.mkdir(parents=True)
        save_file(folder_state, file)


def test_a_state_dict_buffer_the_constructor_left_on_meta_is_read_from_the_files(tmp_path, monkeypatch):
    save_row_sums(tmp_path)
    models, materialize_requests, broadcasts = load(
        monkeypatch, tmp_path, backend_cls=backend_of(DerivedBufferComponent)
    )
    assert materialize_requests == [False, False] and broadcasts == []
    for component, model in models.items():
        assert torch.equal(model.row_sums, saved_weight(component).sum(dim=1).float())
        assert model.forward_count.item() == 0


def test_a_buffer_on_meta_that_no_file_holds_sends_every_rank_to_the_broadcast(tmp_path, monkeypatch):
    save_checkpoint(tmp_path)
    _, materialize_requests, broadcasts = load(
        monkeypatch, tmp_path, backend_cls=backend_of(TransientDerivedBufferComponent)
    )
    assert materialize_requests == [False, True, False, True] and len(broadcasts) == 2


def test_a_buffer_two_modules_share_keeps_its_value_on_both(tmp_path, monkeypatch):
    save_checkpoint(tmp_path)
    models, materialize_requests, broadcasts = load(
        monkeypatch, tmp_path, backend_cls=backend_of(SharedBufferComponent)
    )
    assert materialize_requests == [False, False] and broadcasts == []
    for model in models.values():
        assert torch.equal(model.table, torch.arange(3.0)) and torch.equal(model.block.table, torch.arange(3.0))


def test_one_rank_that_cannot_see_the_files_sends_every_rank_to_the_broadcast(tmp_path, monkeypatch):
    save_checkpoint(tmp_path)
    conditions = []
    _, materialize_requests, broadcasts = load(
        monkeypatch, tmp_path, holds=lambda condition: conditions.append(condition) or False
    )
    assert conditions == [True, True] and materialize_requests == [True, True] and len(broadcasts) == 2


def holds_for_the_files_only():
    """Every rank sees the files; then one of them finds another state dict in its copy."""
    answers = []
    return lambda condition: answers.append(condition) or len(answers) % 2 == 1


def test_one_rank_whose_files_hold_another_state_sends_every_rank_to_the_broadcast(tmp_path, monkeypatch):
    save_checkpoint(tmp_path)
    _, materialize_requests, broadcasts = load(monkeypatch, tmp_path, holds=holds_for_the_files_only())
    assert materialize_requests == [False, True, False, True] and len(broadcasts) == 2


def test_a_model_without_an_adapter_skips_the_adapter_step(tmp_path, monkeypatch):
    # The step gives every base parameter storage and runs a collective: neither is for a model with no adapter.
    save_checkpoint(tmp_path)
    monkeypatch.setattr(
        model_loader.checkpoint, "rank0_rng_state", lambda: pytest.fail("no adapter is built on any rank")
    )
    monkeypatch.setattr(
        model_loader, "replace_base_parameters", lambda model, data_of: pytest.fail("no parameter gets storage")
    )
    _, materialize_requests, broadcasts = load(monkeypatch, tmp_path)
    assert materialize_requests == [False, False] and broadcasts == []


def test_the_base_takes_no_memory_while_the_adapter_is_built(monkeypatch):
    # A full torch.empty is written through under --deterministic-mode: a copy of the model on every rank.
    monkeypatch.setattr(model_loader.checkpoint, "rank0_rng_state", torch.get_rng_state)
    monkeypatch.setattr(model_loader.dist, "get_rank", lambda: 0)
    with model_loader.model_init_context(materialize_weights=False):
        model = TinyComponent()
    with model_loader.base_storage_for_lora(model, enabled=True):
        weight = model.block.proj.weight
        assert weight.device.type == "cpu" and weight.shape == (4, 3)
        assert weight.untyped_storage().nbytes() == weight.element_size()
    assert model.block.proj.weight.is_meta


class BiasedBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 4)
        self.norm = torch.nn.LayerNorm(4)


def save_adapter(path, config, *, init_lora_weights=None):
    get_peft_model(BiasedBlock(), config).save_pretrained(path)
    if init_lora_weights is not None:
        adapter_config = json.loads((path / "adapter_config.json").read_text())
        (path / "adapter_config.json").write_text(
            json.dumps(adapter_config | {"init_lora_weights": init_lora_weights})
        )
    return str(path)


@pytest.mark.parametrize(
    "config, init_lora_weights, builds_on_every_rank",
    [
        pytest.param(LoraConfig(r=2, target_modules=["proj"]), None, True, id="adapter tensors only"),
        pytest.param(LoraConfig(r=2, target_modules=["proj"], bias="all"), None, False, id="a bias of the base"),
        pytest.param(LoraConfig(r=2, target_modules=["proj"], modules_to_save=["norm"]), None, False, id="a module"),
        pytest.param(LoraConfig(r=2, target_modules=["proj"]), "pissa", False, id="an init from the base weights"),
        pytest.param(LoraConfig(r=2, target_modules=["proj"], use_dora=True), None, False, id="DoRA"),
        pytest.param(LoHaConfig(r=2, target_modules=["proj"]), None, False, id="not a LoRA"),
        pytest.param(
            AdaLoraConfig(init_r=2, target_r=1, total_step=10, target_modules=["proj"]), None, False, id="AdaLoRA"
        ),
    ],
)
def test_a_saved_adapter_is_built_on_every_rank_only_when_it_holds_lora_tensors_alone(
    tmp_path, config, init_lora_weights, builds_on_every_rank
):
    # PEFT loads a bias or a saved module into the base, and the read of the base checkpoint would replace it;
    # DoRA's magnitude is the norm of the base weights, which no rank holds while it builds the adapter.
    adapter_path = save_adapter(tmp_path, config, init_lora_weights=init_lora_weights)
    assert model_loader.lora_needs_no_base_weights(make_model_loader_args(), adapter_path) is builds_on_every_rank


def test_a_base_built_with_weights_is_refused_before_the_adapter_step_would_hide_it(tmp_path, monkeypatch):
    class RealBaseBackend(FileBackend):
        def load_component(self, component, **kwargs):
            model = super().load_component(component, **kwargs)
            # Past the ambient init_empty_weights, which would put a parameter registered here on meta.
            model.block.proj._parameters["weight"] = torch.nn.Parameter(torch.zeros(4, 3))
            return model

    save_checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="did not honor meta initialization"):
        load(monkeypatch, tmp_path, lora_init="gaussian", trainable=True, backend_cls=RealBaseBackend)


def lora_weights(state):
    return {name: tensor for name, tensor in state.items() if "lora_" in name}


def test_saved_adapters_are_built_on_every_rank_and_the_base_is_read_from_the_files(tmp_path, monkeypatch):
    save_checkpoint(tmp_path)
    adapter_paths = save_component_lora_adapters(tmp_path / "adapters")
    models, materialize_requests, broadcasts = load(monkeypatch, tmp_path, lora_adapter_paths=adapter_paths)
    assert materialize_requests == [False, False] and broadcasts == []
    for index, (component, model) in enumerate(models.items()):
        proj = model.base_model.model.block.proj
        assert torch.equal(proj.base_layer.weight.to_local(), saved_weight(component).float())
        assert torch.all(proj.lora_A.default.weight.to_local() == 0.15 + 0.1 * index)
        assert torch.all(proj.lora_B.default.weight.to_local() == 0.25 + 0.1 * index)
        assert not any(parameter.requires_grad for parameter in model.parameters())
        assert model.base_model.model.forward_count.item() == 0


@pytest.mark.parametrize("lora_init", ["gaussian", "kaiming-uniform"])
def test_a_new_adapter_gets_the_values_and_leaves_the_generator_as_rank_0_does_with_weights(
    tmp_path, monkeypatch, lora_init
):
    save_checkpoint(tmp_path)
    torch.manual_seed(0)
    models, materialize_requests, broadcasts = load(monkeypatch, tmp_path, lora_init=lora_init, trainable=True)
    generator = torch.get_rng_state()
    assert materialize_requests == [False, False] and broadcasts == []

    torch.manual_seed(0)
    _, materialize_requests, with_weights = load(
        monkeypatch, tmp_path, lora_init=lora_init, trainable=True, holds=holds_for_the_files_only()
    )
    assert materialize_requests == [False, True, False, True] and torch.equal(torch.get_rng_state(), generator)
    for (component, model), state in zip(models.items(), with_weights, strict=True):
        adapter = lora_weights({name: parameter.to_local() for name, parameter in model.named_parameters()})
        assert len(adapter) == 2 and adapter.keys() == lora_weights(state).keys()
        assert all(torch.equal(adapter[name], state[name]) and adapter[name].requires_grad for name in adapter)
        assert adapter["base_model.model.block.proj.lora_A.default.weight"].abs().sum() > 0
        base = model.base_model.model.block.proj.base_layer.weight
        assert torch.equal(base.to_local(), saved_weight(component).float()) and not base.requires_grad


def test_an_adapter_initialized_from_the_base_weights_is_built_by_rank_0_and_broadcast(tmp_path, monkeypatch):
    class NeverAsked(FileBackend):
        def safetensors_files(self, component, *, checkpoint_path):
            pytest.fail("PiSSA decomposes the base weights, which only rank 0 holds")

    save_checkpoint(tmp_path)
    conditions = []
    _, materialize_requests, broadcasts = load(
        monkeypatch,
        tmp_path,
        lora_init="pissa",
        trainable=True,
        holds=lambda condition: conditions.append(condition) or condition,
        backend_cls=NeverAsked,
    )
    # Every rank still says that it reads no files, so no rank waits in a collective the others skip.
    assert conditions == [False, False] and materialize_requests == [True, True] and len(broadcasts) == 2
    assert all(len(lora_weights(state)) == 2 for state in broadcasts)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
