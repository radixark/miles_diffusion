"""EMA is optimizer state, independent of Adam and the active model.

    dense / LoRA parameters --step--> EMA state --DCP--> resumed EMA
    frozen base / buffers       --------> unchanged (never averaged)
    actor --use_weights--> EMA forward --finally--> original actor weights
    component A --use_weights--> EMA A; component B stays untouched

Dense and adapter-only checkpoints save EMA independently of Adam and restore its
schedule and update count. Older checkpoints seed EMA from the loaded actor weights.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=["fsdp"])

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

from miles.backends.fsdp_utils import checkpoint
from miles.backends.fsdp_utils.ema import EMAOptimizer


class Model(nn.Module):
    def __init__(self, *, frozen_base=False):
        super().__init__()
        self.base = nn.Parameter(torch.tensor([2.0]), requires_grad=not frozen_base)
        self.lora_A = nn.Parameter(torch.tensor([4.0]))
        self.register_buffer("counter", torch.tensor(5))
        self.register_buffer("floating_buffer", torch.tensor([8.0]))


@pytest.mark.parametrize("frozen_base", [False, True])
def test_update_averages_only_trainable_parameters(frozen_base):
    model = Model(frozen_base=frozen_base)
    ema = EMAOptimizer(model, decay=0.5, uprate=0.1, uphold=0.9, flat_steps=1)
    with torch.no_grad():
        model.base.fill_(6.0)
        model.lora_A.fill_(8.0)
    assert ema.step() == 0.5
    torch.testing.assert_close(ema.state[model.lora_A]["ema"], torch.tensor([6.0]))
    if frozen_base:
        assert model.base not in ema.state
    else:
        torch.testing.assert_close(ema.state[model.base]["ema"], torch.tensor([4.0]))
    assert ema.step() == 0.1
    assert ema.update_count == 2
    torch.testing.assert_close(ema.state[model.lora_A]["ema"], torch.tensor([7.8]))
    assert len(ema.state) == (1 if frozen_base else 2)
    assert model.counter.item() == 5
    assert model.floating_buffer.item() == 8.0


def test_use_weights_restores_actor_weights_after_failure():
    # actor lora_A=10, EMA lora_A=4 --use_weights--> 4 --exception--> 10 again, same Parameter object
    model = Model(frozen_base=True)
    original_parameter = model.lora_A
    ema = EMAOptimizer(model)
    with torch.no_grad():
        model.lora_A.fill_(10.0)
    with pytest.raises(RuntimeError, match="reference failed"), ema.use_weights(model):
        assert model.lora_A.item() == 4.0
        raise RuntimeError("reference failed")
    assert model.lora_A is original_parameter and model.lora_A.item() == 10.0
    torch.testing.assert_close(ema.state[model.lora_A]["ema"], torch.tensor([4.0]))
    assert model.lora_A.requires_grad and not model.base.requires_grad


def test_use_weights_scopes_parameters_to_selected_component():
    models = nn.ModuleDict({"transformer": Model(), "transformer_2": Model()})
    ema = EMAOptimizer(models)
    with torch.no_grad():
        for parameter in models.parameters():
            parameter.add_(10.0)
    first, second = models.values()
    second_parameters = dict(second.named_parameters())
    second_buffers = dict(second.named_buffers())

    with ema.use_weights(first):
        assert first.base.item() == 2.0 and first.lora_A.item() == 4.0
        assert second.base.item() == 12.0 and second.lora_A.item() == 14.0
        for name, parameter in second.named_parameters():
            assert parameter is second_parameters[name]
        for name, buffer in second.named_buffers():
            assert buffer is second_buffers[name]
    assert first.base.item() == second.base.item() == 12.0
    assert first.lora_A.item() == second.lora_A.item() == 14.0

    # Publishing the container still selects every component.
    with ema.use_weights(models):
        assert first.base.item() == second.base.item() == 2.0
        assert first.lora_A.item() == second.lora_A.item() == 4.0
    assert first.base.item() == second.base.item() == 12.0
    assert first.lora_A.item() == second.lora_A.item() == 14.0


@pytest.fixture
def cpu_checkpoint(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)


def make_actor(path, *, use_ema=True, use_lora=False):
    model = nn.ModuleDict({"transformer": Model(frozen_base=use_lora), "transformer_2": Model(frozen_base=True)})
    optimizer = torch.optim.AdamW(parameter for parameter in model.parameters() if parameter.requires_grad)
    return SimpleNamespace(
        model=model,
        optimizer=optimizer,
        ema_optimizer=EMAOptimizer(model, decay=0.4, uprate=0.2, uphold=0.8, flat_steps=2) if use_ema else None,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        global_step=3,
        micro_step=6,
        train_pipeline_config=SimpleNamespace(optimizer_state_allowed_missing=[]),
        args=SimpleNamespace(
            save=str(path), load=str(path), ckpt_step=None, use_lora=use_lora, no_save_optim=True, no_load_optim=True
        ),
    )


@pytest.mark.parametrize("use_lora", [False, True])
def test_checkpoint_restores_ema_without_adam_and_continues_schedule(tmp_path, cpu_checkpoint, use_lora):
    actor = make_actor(tmp_path, use_lora=use_lora)
    with torch.no_grad():
        for parameter in actor.model.parameters():
            if parameter.requires_grad:
                parameter.add_(2.0)
    actor.ema_optimizer.step()
    actor.ema_optimizer.step()
    expected = get_optimizer_state_dict(actor.model, actor.ema_optimizer)
    expected_names = {"transformer.lora_A", "transformer_2.lora_A"}
    if not use_lora:
        expected_names.add("transformer.base")
    assert set(expected["state"]) == expected_names
    checkpoint.save(actor, iteration=2)
    assert (tmp_path / "iter_0000003" / "ema" / ".metadata").exists()
    assert not (tmp_path / "iter_0000003" / "optimizer" / ".metadata").exists()

    resumed = make_actor(tmp_path, use_lora=use_lora)
    resumed.ema_optimizer.param_groups[0]["uprate"] = 0.9
    checkpoint.load(resumed)
    assert resumed.ema_optimizer.update_count == 2
    assert resumed.ema_optimizer.param_groups[0]["uprate"] == 0.2
    actual = get_optimizer_state_dict(resumed.model, resumed.ema_optimizer)
    for name, state in expected["state"].items():
        torch.testing.assert_close(actual["state"][name]["ema"], state["ema"])
    assert actor.ema_optimizer.step() == resumed.ema_optimizer.step() == 0.2
    actual = get_optimizer_state_dict(resumed.model, resumed.ema_optimizer)
    expected = get_optimizer_state_dict(actor.model, actor.ema_optimizer)
    for name, state in expected["state"].items():
        torch.testing.assert_close(actual["state"][name]["ema"], state["ema"])


def test_legacy_checkpoint_seeds_ema_from_loaded_actor(tmp_path, cpu_checkpoint, caplog):
    actor = make_actor(tmp_path, use_ema=False)
    with torch.no_grad():
        for parameter in actor.model.parameters():
            parameter.add_(9.0)
    checkpoint.save(actor, iteration=0)
    resumed = make_actor(tmp_path)
    resumed.ema_optimizer.step()
    with caplog.at_level("INFO"):
        checkpoint.load(resumed)
    assert resumed.ema_optimizer.update_count == 0
    assert "initialized EMA from the loaded model" in caplog.text
    for parameter, state in resumed.ema_optimizer.state.items():
        torch.testing.assert_close(state["ema"], parameter)
