"""Independent model roles keep actor parameters and gradients isolated.

    full / new / saved LoRA -> actor (trainable) ---> loss ---> gradients + optimizer step
                           +-> reference (full or LoRA) --no_grad--/
                           +-> teacher (LoRA) -----------no_grad--/
    two components         ---> select the matching role component
    LoRA base reference    ---> disable adapter ---> adapters and their gradients back on the actor
    FSDP2 kept-gathered    ---> lora_base forward ---> next step still trains the adapters
    FSDP2 mixed precision  ---> lazy init at load ---> fp32 adapter gradients whatever runs first
    NFT                    ---> old_pred = actor with EMA weights (pi_old), ref_pred = --ref-mode (KL)
    loaded adapter r/alpha ---> IPC publication metadata (ignores CLI init defaults)
    FSDP boundary          ---> TinyBlock containing the complete LoRA projection
    checkpoint_path        ---> backend loads base; loader applies the role's LoRA

The CPU tests exercise real PEFT checkpoint loading and the production actor forward.
Loader tests replace FSDP collectives; the CUDA worker uses the real sharded loader.
The IPC test captures transport metadata without launching a rollout engine.
Dense model copies provide an independent output and gradient oracle.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=["fsdp"])

import ast
import copy
import logging
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.testing._internal.distributed.fake_pg  # noqa: F401
from peft import LoraConfig, PeftModel, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from miles.backends.fsdp_utils import model_loader
from miles.backends.fsdp_utils.ema import EMAOptimizer, reshard_model
from miles.backends.fsdp_utils.input_dtype_policy import apply_input_dtype_policy
from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 4, bias=False)

    def forward(self, inputs):
        return self.proj(inputs).tanh()


class TinyComponent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = TinyBlock()
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.int64), persistent=False)

    def forward(self, inputs):
        self.forward_count.add_(1)
        return self.block(inputs)


class TinyBackend:
    def __init__(self):
        self.component_load_requests = []

    def load_component(self, component, *, checkpoint_path, master_dtype, materialize_weights):
        self.component_load_requests.append((component, checkpoint_path))
        model = TinyComponent().to(dtype=master_dtype)
        if materialize_weights:
            value = {"actor": 0.1, "reference": 0.2, "teacher": 0.3}[checkpoint_path]
            value += 0.05 if component == "transformer_2" else 0.0
            with torch.no_grad():
                model.block.proj.weight.fill_(value)
        return model

    def fsdp_parallel_plan(self, model):
        return FSDPParallelPlan(no_split_modules=("TinyBlock",))


class TinyPipelineConfig:
    input_dtype_policy = {}

    def postprocess_model_after_materialize(self, model):
        pass

    def compute_noise_pred(self, *, model, latents_input, **kwargs):
        return model(latents_input)


def make_model_loader_args():
    return Namespace(
        hf_checkpoint="actor",
        use_lora=True,
        update_weight_target_modules=["transformer", "transformer_2"],
        fsdp_master_dtype="fp32",
        fsdp_reduce_dtype="fp32",
        diffusion_forward_dtype="fp32",
        fsdp_attention_backend=None,
        gradient_checkpointing=False,
        offload_train=False,
    )


def component_adapter_paths(path):
    """One plain PEFT adapter directory per component, in --update-weight-target-module order."""
    return [str(path / component) for component in make_model_loader_args().update_weight_target_modules]


def save_component_lora_adapters(path):
    adapter_paths = component_adapter_paths(path)
    for index, adapter_path in enumerate(adapter_paths):
        model = get_peft_model(TinyComponent(), LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]))
        with torch.no_grad():
            model.base_model.model.block.proj.lora_A.default.weight.fill_(0.15 + 0.1 * index)
            model.base_model.model.block.proj.lora_B.default.weight.fill_(0.25 + 0.1 * index)
        model.save_pretrained(adapter_path)
    return adapter_paths


def load_actor_forward_method():
    path = Path(__file__).resolve().parents[4] / "miles/backends/fsdp_utils/actor.py"
    source = ast.parse(path.read_text())
    actor = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "FSDPTrainRayActor")
    method = next(
        node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_train_pair_batch"
    )
    namespace = dict(
        torch=torch,
        reshard_model=reshard_model,
        apply_input_dtype_policy=apply_input_dtype_policy,
        DiffusionLossContext=object,
        MetricBuffer=object,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


def make_actor_forward_harness(
    models, reference_models, component, inputs, ref_mode, *, loss_type="policy_loss", ema_optimizer=None
):
    observed = {}
    prepared = SimpleNamespace(
        model=models[component],
        component_name=component,
        latents=inputs,
        timesteps_for_model=torch.ones(inputs.shape[0], device=inputs.device),
        pos_cond=None,
        neg_cond=None,
        joint_cond=None,
        use_cfg=False,
        cfg_batching=False,
        guidance_scale=1.0,
        true_cfg_scale=None,
    )

    def loss_formula(ctx, batch, prepared, *, new_pred, old_pred, ref_pred, **kwargs):
        observed.update(new_pred=new_pred, old_pred=old_pred, ref_pred=ref_pred)
        return (new_pred - ref_pred).square().mean()

    harness = SimpleNamespace(
        args=Namespace(ref_mode=ref_mode, loss_type=loss_type, train_async=False),
        models=models,
        ema_optimizer=ema_optimizer,
        model=torch.nn.ModuleDict(models),
        reference_models=reference_models,
        train_pipeline_config=TinyPipelineConfig(),
        _forward_dtype=torch.float32,
        custom_prepare_train_batch_func=lambda *args, **kwargs: prepared,
        custom_loss_formula_func=loss_formula,
    )
    return harness, observed


@pytest.fixture
def unsharded_model_loader(monkeypatch):
    calls = []
    monkeypatch.setattr(model_loader.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(model_loader.checkpoint, "sync_model_dtypes", lambda model: None)
    monkeypatch.setattr(model_loader.checkpoint, "broadcast_full_state_to_fsdp", lambda *args, **kwargs: None)

    def record_fsdp(model, plan, *, mesh, cpu_offload, args):
        calls.append((mesh, cpu_offload))
        return model

    monkeypatch.setattr(model_loader, "apply_fsdp2", record_fsdp)
    monkeypatch.setattr(model_loader, "finish_fsdp_lazy_init", lambda model: None)
    parallel = SimpleNamespace(get_mesh=lambda name: "shared-fsdp-mesh", get_optional_mesh=lambda name: None)
    return parallel, calls


@pytest.mark.parametrize("actor_mode", ["full", "new_lora", "saved_lora"])
@pytest.mark.parametrize("reference_has_adapter", [False, True])
def test_loaded_reference_and_teacher_are_independent_and_frozen(
    tmp_path, unsharded_model_loader, actor_mode, reference_has_adapter
):
    parallel, calls = unsharded_model_loader
    adapter_paths = save_component_lora_adapters(tmp_path)
    args, backend, config = make_model_loader_args(), TinyBackend(), TinyPipelineConfig()
    actor_has_adapter = actor_mode != "full"
    args.use_lora = actor_has_adapter
    args.lora_rank = 1
    args.lora_alpha = 2
    args.lora_target_modules = ["proj"]
    args.lora_init_weights = "kaiming-uniform"
    actor = model_loader.load_fsdp_models(
        args,
        backend,
        config,
        parallel,
        checkpoint_path="actor",
        lora_adapter_paths=adapter_paths if actor_mode == "saved_lora" else None,
        trainable=True,
    )
    reference = model_loader.load_fsdp_models(
        args,
        backend,
        config,
        parallel,
        checkpoint_path="reference",
        lora_adapter_paths=adapter_paths if reference_has_adapter else None,
        cpu_offload=True,
    )
    teacher = model_loader.load_fsdp_models(
        args,
        backend,
        config,
        parallel,
        checkpoint_path="teacher",
        lora_adapter_paths=adapter_paths,
    )
    assert args.hf_checkpoint == "actor" and args.use_lora is actor_has_adapter
    assert calls == [("shared-fsdp-mesh", flag) for flag in (False, False, True, True, False, False)]
    assert backend.component_load_requests == [
        (component, checkpoint_path)
        for checkpoint_path in ("actor", "reference", "teacher")
        for component in args.update_weight_target_modules
    ]
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10
    for component in args.update_weight_target_modules:
        assert isinstance(actor[component], PeftModel) is actor_has_adapter
        if actor_has_adapter:
            adapter_config = actor[component].peft_config["default"]
            assert adapter_config.r == (2 if actor_mode == "saved_lora" else args.lora_rank)
            assert adapter_config.lora_alpha == (4 if actor_mode == "saved_lora" else args.lora_alpha)
        assert isinstance(reference[component], PeftModel) is reference_has_adapter
        oracle = copy.deepcopy(actor[component])
        frozen_parameters_before_update = {
            (role, name): value.detach().clone()
            for role, model in (("ref", reference[component]), ("teacher", teacher[component]))
            for name, value in model.named_parameters()
        }
        with torch.no_grad():
            target = reference[component](inputs)
            teacher_output = teacher[component](inputs)
        assert not torch.equal(target, teacher_output)
        harness, observed = make_actor_forward_harness(actor, reference, component, inputs, "ref")
        load_actor_forward_method()(harness, None, [], metrics=None).backward()
        oracle_output = oracle(inputs)
        (oracle_output - target).square().mean().backward()
        torch.testing.assert_close(observed["new_pred"], oracle_output)
        torch.testing.assert_close(observed["ref_pred"], target)
        assert not observed["ref_pred"].requires_grad
        for actual, expected in zip(actor[component].parameters(), oracle.parameters(), strict=True):
            if actual.requires_grad:
                torch.testing.assert_close(actual.grad, expected.grad)
        torch.optim.AdamW(actor[component].parameters(), lr=0.01).step()
        torch.optim.AdamW(oracle.parameters(), lr=0.01).step()
        for actual, expected in zip(actor[component].parameters(), oracle.parameters(), strict=True):
            torch.testing.assert_close(actual, expected)
        for role, model in (("ref", reference[component]), ("teacher", teacher[component])):
            assert not model.training
            for name, parameter in model.named_parameters():
                assert not parameter.requires_grad and parameter.grad is None
                torch.testing.assert_close(parameter, frozen_parameters_before_update[role, name])


def test_lora_base_forward_restores_adapters_and_actor_gradients(tmp_path, unsharded_model_loader):
    parallel, _ = unsharded_model_loader
    adapter_paths = save_component_lora_adapters(tmp_path)
    actor = model_loader.load_fsdp_models(
        make_model_loader_args(),
        TinyBackend(),
        TinyPipelineConfig(),
        parallel,
        checkpoint_path="actor",
        lora_adapter_paths=adapter_paths,
        trainable=True,
    )
    component = "transformer_2"
    model, inputs = actor[component], torch.ones(2, 3)
    oracle = copy.deepcopy(model)
    base = TinyComponent()
    base.block.proj.weight.data.copy_(model.base_model.model.block.proj.base_layer.weight)
    with torch.no_grad():
        target = base(inputs)
    harness, observed = make_actor_forward_harness(actor, {}, component, inputs, "lora_base")
    load_actor_forward_method()(harness, None, [], metrics=None).backward()
    oracle_output = oracle(inputs)
    (oracle_output - target).square().mean().backward()
    torch.testing.assert_close(observed["ref_pred"], target)
    torch.testing.assert_close(observed["new_pred"], oracle_output)
    assert not torch.equal(target, oracle_output)
    assert model.training and not model.base_model.model.block.proj.disable_adapters
    assert model.base_model.model.forward_count.item() == 2  # reference forward + actor forward
    assert actor["transformer"].base_model.model.forward_count.item() == 0
    for actual, expected in zip(model.parameters(), oracle.parameters(), strict=True):
        if actual.requires_grad:
            torch.testing.assert_close(actual.grad, expected.grad)


def test_first_forward_with_adapters_disabled_keeps_master_dtype_gradients():
    # FSDP2 fixes each group's gradient dtype at its lazy init from the parameters that require grad:
    #   load --finish_fsdp_lazy_init (adapters trainable)--> lora_base forward FIRST --> actor backward
    #   --> fp32 adapter gradients on the fp32 shards, whichever forward runs first
    dist.init_process_group("fake", store=dist.HashStore(), rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        model = get_peft_model(TinyComponent(), LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        fully_shard(model.base_model.model.block, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        model_loader.finish_fsdp_lazy_init(model)
        inputs = torch.ones(2, 3)
        with torch.no_grad(), model.disable_adapter():
            model(inputs)
        reshard_model(model)
        model(inputs).square().mean().backward()
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                assert parameter.grad is not None and parameter.grad.dtype == torch.float32, name
    finally:
        dist.destroy_process_group()


def test_lora_base_reference_keeps_adapter_gradients_on_fsdp_shards():
    # FSDP2 with reshard_after_forward=False keeps the gathered parameters registered after forward:
    #   lora_base forward --PEFT re-enables adapter gradients on exit--> must land on the shards
    #   --> every following training step still produces adapter gradients
    dist.init_process_group("fake", store=dist.HashStore(), rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        model = get_peft_model(TinyComponent(), LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]))
        fully_shard(model.base_model.model.block, mesh=mesh, reshard_after_forward=False)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
        adapter_parameters = {name: parameter for name, parameter in model.named_parameters() if "lora_" in name}
        forward = load_actor_forward_method()
        for step in range(2):
            harness, _ = make_actor_forward_harness(
                {"transformer": model}, {}, "transformer", torch.ones(2, 3), "lora_base"
            )
            forward(harness, None, [], metrics=None).backward()
            for name, parameter in adapter_parameters.items():
                assert parameter.requires_grad and parameter.grad is not None, (step, name)
            model.zero_grad(set_to_none=True)
    finally:
        dist.destroy_process_group()


def test_nft_gets_ema_as_old_and_lora_base_as_kl_reference(tmp_path, unsharded_model_loader):
    # actor adapters moved away from their EMA copy:
    #   old_pred = actor with EMA weights (pi_old), ref_pred = actor with adapters disabled, all three differ
    parallel, _ = unsharded_model_loader
    adapter_paths = save_component_lora_adapters(tmp_path)
    actor = model_loader.load_fsdp_models(
        make_model_loader_args(),
        TinyBackend(),
        TinyPipelineConfig(),
        parallel,
        checkpoint_path="actor",
        lora_adapter_paths=adapter_paths,
        trainable=True,
    )
    component = "transformer"
    model, inputs = actor[component], torch.ones(2, 3)
    ema_optimizer = EMAOptimizer(model)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.5)
        expected_new = model(inputs)
        with ema_optimizer.use_weights(model):
            expected_old = model(inputs)
        with model.disable_adapter():
            expected_ref = model(inputs)
    harness, observed = make_actor_forward_harness(
        actor, {}, component, inputs, "lora_base", loss_type="nft", ema_optimizer=ema_optimizer
    )
    load_actor_forward_method()(harness, None, [], metrics=None)
    torch.testing.assert_close(observed["new_pred"], expected_new)
    torch.testing.assert_close(observed["old_pred"], expected_old)
    torch.testing.assert_close(observed["ref_pred"], expected_ref)
    assert not torch.equal(expected_old, expected_ref) and not torch.equal(expected_old, expected_new)


def test_ipc_publication_uses_loaded_adapter_config(tmp_path, unsharded_model_loader):
    parallel, _ = unsharded_model_loader
    adapter_paths = save_component_lora_adapters(tmp_path)
    models = model_loader.load_fsdp_models(
        make_model_loader_args(),
        TinyBackend(),
        TinyPipelineConfig(),
        parallel,
        checkpoint_path="actor",
        lora_adapter_paths=adapter_paths,
        trainable=True,
    )
    path = Path(__file__).resolve().parents[4] / "miles/backends/fsdp_utils/diffusion_update_weight_utils.py"
    source = ast.parse(path.read_text())
    updater = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "DiffusionUpdateWeightFromTensor"
    )
    method = next(
        node for node in updater.body if isinstance(node, ast.FunctionDef) and node.name == "update_bucket_weights"
    )
    sent = []

    def gather_object(*, obj, object_gather_list, **kwargs):
        object_gather_list[0] = obj

    namespace = dict(
        logger=logging.getLogger(__name__),
        monkey_patch_torch_reductions=lambda: None,
        FlattenedTensorBucket=lambda named_tensors: SimpleNamespace(
            get_metadata=lambda: [],
            get_flattened_tensor=lambda: named_tensors[0][1],
        ),
        MultiprocessingSerializer=SimpleNamespace(serialize=lambda *args, **kwargs: "payload"),
        dist=SimpleNamespace(get_rank=lambda: 0, get_world_size=lambda group: 1, gather_object=gather_object),
        get_physical_gpu_id=lambda: "gpu-0",
        ray=SimpleNamespace(get=lambda result: result),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    harness = SimpleNamespace(
        args=Namespace(lora_rank=64, lora_alpha=64),
        models=models,
        _ipc_gather_src=0,
        _ipc_gather_group=None,
        _ipc_engine=SimpleNamespace(
            update_weights_from_tensor=SimpleNamespace(remote=lambda **kwargs: sent.append(kwargs))
        ),
    )
    namespace[method.name](
        harness, [("block.proj.lora_A.weight", torch.ones(2, 3))], "transformer", weight_update_mode="lora_merge"
    )
    assert len(sent) == 1
    assert sent[0]["lora_rank"] == 2 and sent[0]["lora_alpha"] == 4
