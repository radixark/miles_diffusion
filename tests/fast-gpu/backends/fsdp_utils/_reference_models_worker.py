"""Distributed counterpart to the independent-reference CPU contract.

    real PEFT checkpoint --> rank 0 CPU / rank 1 meta --> same FSDP mesh
       |                         |                          |
       +--> dense oracle         +--> full / LoRA ref       +--> LoRA actor
                                        frozen                  gradients + AdamW
    role checkpoint_path --> base component --> optional pretrained adapter

Native CPU offload and GPU-resident references must produce the same outputs.
LoRA-base evaluation must leave adapter execution and adapter gradients intact.
"""

import itertools
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from peft import PeftModel
from tests.fast.backends.fsdp_utils.test_reference_models import (
    TinyBackend,
    TinyPipelineConfig,
    load_actor_forward_method,
    make_actor_forward_harness,
    make_model_loader_args,
    save_component_lora_adapters,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from miles.backends.fsdp_utils.model_loader import load_fsdp_models


def load_dense_model(checkpoint_path, component, adapter_path, *, trainable=False):
    model = TinyBackend().load_component(
        component, checkpoint_path=checkpoint_path, master_dtype=torch.float32, materialize_weights=True
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path, subfolder=component, is_trainable=trainable)
    model.train(trainable)
    return model.cuda()


def local_tensor(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def check_roles(mesh, adapter_path, reference_has_adapter, cpu_offload):
    args, backend, config = make_model_loader_args(), TinyBackend(), TinyPipelineConfig()
    parallel = SimpleNamespace(get_mesh=lambda name: mesh, get_optional_mesh=lambda name: None)
    actor = load_fsdp_models(
        args, backend, config, parallel, checkpoint_path="actor", lora_adapter_path=adapter_path, trainable=True
    )
    reference = load_fsdp_models(
        args,
        backend,
        config,
        parallel,
        checkpoint_path="reference",
        lora_adapter_path=adapter_path if reference_has_adapter else None,
        cpu_offload=cpu_offload,
    )
    teacher = load_fsdp_models(
        args,
        backend,
        config,
        parallel,
        checkpoint_path="teacher",
        lora_adapter_path=adapter_path,
        cpu_offload=cpu_offload,
    )
    inputs = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3) / 10
    for component in args.update_weight_target_modules:
        oracle = load_dense_model("actor", component, adapter_path, trainable=True)
        reference_oracle = load_dense_model("reference", component, adapter_path if reference_has_adapter else None)
        teacher_oracle = load_dense_model("teacher", component, adapter_path)
        optimizer = torch.optim.AdamW(actor[component].parameters(), lr=0.01)
        oracle_optimizer = torch.optim.AdamW(oracle.parameters(), lr=0.01)
        expected_device = "cpu" if cpu_offload else "cuda"
        frozen_parameters_before_update = {}
        for role, model in (("ref", reference[component]), ("teacher", teacher[component])):
            assert not model.training
            for name, parameter in model.named_parameters():
                assert not parameter.requires_grad and local_tensor(parameter).device.type == expected_device
                if cpu_offload:
                    assert local_tensor(parameter).is_pinned()
                frozen_parameters_before_update[role, name] = local_tensor(parameter).detach().clone()
        with torch.no_grad():
            target = reference_oracle(inputs)
            torch.testing.assert_close(teacher[component](inputs), teacher_oracle(inputs))
        harness, observed = make_actor_forward_harness(actor, reference, component, inputs, "ref")
        load_actor_forward_method()(harness, None, [], metrics=None).backward()
        expected_actor_output = oracle(inputs)
        (expected_actor_output - target).square().mean().backward()
        torch.testing.assert_close(observed["ref_pred"], target)
        torch.testing.assert_close(observed["new_pred"], expected_actor_output)
        for actual, expected in zip(actor[component].parameters(), oracle.parameters(), strict=True):
            if actual.requires_grad:
                torch.testing.assert_close(actual.grad.full_tensor(), expected.grad, atol=2e-6, rtol=2e-6)
        optimizer.step()
        oracle_optimizer.step()
        for actual, expected in zip(actor[component].parameters(), oracle.parameters(), strict=True):
            torch.testing.assert_close(actual.full_tensor(), expected, atol=2e-6, rtol=2e-6)
        for role, model in (("ref", reference[component]), ("teacher", teacher[component])):
            for name, parameter in model.named_parameters():
                assert parameter.grad is None
                torch.testing.assert_close(local_tensor(parameter), frozen_parameters_before_update[role, name])

        actor[component].zero_grad(set_to_none=True)
        oracle.zero_grad(set_to_none=True)
        base_oracle = load_dense_model("actor", component, None)
        with torch.no_grad():
            base_target = base_oracle(inputs)
        forward_count = actor[component].base_model.model.forward_count.item()
        harness, observed = make_actor_forward_harness(actor, {}, component, inputs, "lora_base")
        load_actor_forward_method()(harness, None, [], metrics=None).backward()
        expected_actor_output = oracle(inputs)
        (expected_actor_output - base_target).square().mean().backward()
        torch.testing.assert_close(observed["ref_pred"], base_target)
        assert actor[component].training
        assert not actor[component].base_model.model.block.proj.disable_adapters
        assert actor[component].base_model.model.forward_count.item() == forward_count + 2  # reference + actor
        for actual, expected in zip(actor[component].parameters(), oracle.parameters(), strict=True):
            if actual.requires_grad:
                torch.testing.assert_close(actual.grad.full_tensor(), expected.grad, atol=2e-6, rtol=2e-6)
        optimizer.step()
        oracle_optimizer.step()
        for actual, expected in zip(actor[component].parameters(), oracle.parameters(), strict=True):
            torch.testing.assert_close(actual.full_tensor(), expected, atol=2e-6, rtol=2e-6)


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    adapter_path = Path(sys.argv[1])
    if dist.get_rank() == 0:
        save_component_lora_adapters(adapter_path)
    dist.barrier()
    for reference_has_adapter, cpu_offload in itertools.product((False, True), repeat=2):
        check_roles(mesh, str(adapter_path), reference_has_adapter, cpu_offload)
    dist.barrier()
    if dist.get_rank() == 0:
        print("OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
