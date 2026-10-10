"""EMA is optimizer state, independent of Adam and the active model.

    dense / LoRA parameters --step--> EMA state --DCP--> resumed EMA
    frozen base / buffers       --------> unchanged (never averaged)
    actor --use_weights--> EMA forward --finally--> original actor weights
    component A --use_weights--> EMA A; component B stays untouched
    actor.train --> one EMA update at the actor's optimizer step --> repeated publications (no further updates)
    keep_previous: EMA --step--> previous holds the pre-step EMA --DCP--> resumed previous EMA
                   (a checkpoint without one starts it at the loaded EMA)

Dense and adapter-only checkpoints save EMA independently of Adam; the decay schedule follows the
actor's optimizer step count, which the checkpoint metadata restores. Older checkpoints seed EMA
from the loaded actor weights.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=["fsdp"])

import ast
from contextlib import nullcontext
from pathlib import Path
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
    ema.step(optimizer_step=1)  # flat: decay 0.5
    torch.testing.assert_close(ema.state[model.lora_A]["ema"], torch.tensor([6.0]))
    if frozen_base:
        assert model.base not in ema.state
    else:
        torch.testing.assert_close(ema.state[model.base]["ema"], torch.tensor([4.0]))
    ema.step(optimizer_step=2)  # ramp: decay min((2 - 1) * 0.1, 0.9) = 0.1
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


def test_keep_previous_tracks_ema_before_latest_step():
    # EMA 4 --step(actor 8, decay 0.5)--> 6 (previous 4) --step--> 7 (previous 6)
    model = Model(frozen_base=True)
    ema = EMAOptimizer(model, decay=0.5, flat_steps=10, keep_previous=True)
    with torch.no_grad():
        model.lora_A.fill_(8.0)
    for previous, current in ((4.0, 6.0), (6.0, 7.0)):
        ema.step(optimizer_step=1)
        assert ema.state[model.lora_A]["previous_ema"].item() == previous
        assert ema.state[model.lora_A]["ema"].item() == current
    assert list(ema.state) == [model.lora_A]

    with ema.use_weights(model, previous=True):
        assert model.lora_A.item() == 6.0
    assert model.lora_A.item() == 8.0
    assert ema.state[model.lora_A]["previous_ema"].item() == 6.0
    assert ema.state[model.lora_A]["ema"].item() == 7.0


def test_previous_weights_need_keep_previous():
    model = Model()
    ema = EMAOptimizer(model)
    assert all("previous_ema" not in state for state in ema.state.values())
    with pytest.raises(RuntimeError, match="keep_previous"), ema.use_weights(model, previous=True):
        pass
    assert model.lora_A.item() == 4.0


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


def test_actor_train_updates_ema_once_and_publication_only_reads_it():
    path = Path(__file__).resolve().parents[4] / "miles/backends/fsdp_utils/actor.py"
    actor_class = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "FSDPTrainRayActor"
    )
    methods = [
        node
        for node in actor_class.body
        if isinstance(node, ast.FunctionDef) and node.name in ("train", "update_weights")
    ]
    for method in methods:
        method.decorator_list = []
    namespace = dict(
        ray=SimpleNamespace(get=lambda value: value),
        dist=SimpleNamespace(get_rank=lambda: 0),
        timer=lambda name: nullcontext(),
        inverse_timer=lambda name: nullcontext(),
        nullcontext=nullcontext,
        clear_memory=lambda: None,
        train_metric_utils=SimpleNamespace(log_perf_data_raw=lambda **kwargs: None),
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)

    model = Model()
    published_weights = []

    @torch.no_grad()
    def train_core(**kwargs):
        for parameter in model.parameters():
            parameter.add_(2.0)
        actor.global_step += 1

    actor = SimpleNamespace(
        args=SimpleNamespace(offload_train=False, debug_rollout_only=False, train_only=False, rollout_weights="ema"),
        model=model,
        global_step=0,
        ema_optimizer=EMAOptimizer(model, decay=0.5, flat_steps=10),
        _train_core=train_core,
        parallel_state=SimpleNamespace(get_mesh=lambda name: SimpleNamespace(get_local_rank=lambda: 0)),
        rollout_manager=SimpleNamespace(get_rollout_engines_and_lock=SimpleNamespace(remote=lambda: (None, None, 0))),
        weight_updater=SimpleNamespace(update_weights=lambda: published_weights.append(model.base.detach().clone())),
    )
    namespace["train"](actor, rollout_id=0, rollout_data_ref=[SimpleNamespace(inner={})])
    assert model.base.item() == 4.0
    assert actor.ema_optimizer.state[model.base]["ema"].item() == 3.0

    for _ in range(2):
        namespace["update_weights"](actor)
        assert published_weights[-1].item() == 3.0
        assert model.base.item() == 4.0
        assert actor.ema_optimizer.state[model.base]["ema"].item() == 3.0


@pytest.fixture
def cpu_checkpoint(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)


def make_actor(path, *, use_ema=True, use_lora=False, keep_previous=False):
    model = nn.ModuleDict({"transformer": Model(frozen_base=use_lora), "transformer_2": Model(frozen_base=True)})
    optimizer = torch.optim.AdamW(parameter for parameter in model.parameters() if parameter.requires_grad)
    return SimpleNamespace(
        model=model,
        optimizer=optimizer,
        ema_optimizer=(
            EMAOptimizer(model, decay=0.4, uprate=0.2, uphold=0.8, flat_steps=2, keep_previous=keep_previous)
            if use_ema
            else None
        ),
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
    actor.ema_optimizer.step(optimizer_step=1)
    actor.ema_optimizer.step(optimizer_step=2)
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
    assert resumed.ema_optimizer.param_groups[0]["uprate"] == 0.2
    actual = get_optimizer_state_dict(resumed.model, resumed.ema_optimizer)
    for name, state in expected["state"].items():
        torch.testing.assert_close(actual["state"][name]["ema"], state["ema"])
    actor.ema_optimizer.step(optimizer_step=3)
    resumed.ema_optimizer.step(optimizer_step=3)
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
    resumed.ema_optimizer.step(optimizer_step=1)
    with caplog.at_level("INFO"):
        checkpoint.load(resumed)
    assert "initialized EMA from the loaded model" in caplog.text
    for parameter, state in resumed.ema_optimizer.state.items():
        torch.testing.assert_close(state["ema"], parameter)


def test_checkpoint_restores_previous_ema(tmp_path, cpu_checkpoint):
    # Async resume resamples its first batch with the previous EMA, so the checkpoint keeps it exactly.
    actor = make_actor(tmp_path, keep_previous=True)
    for optimizer_step in (1, 2):
        with torch.no_grad():
            for parameter in actor.model.parameters():
                parameter.add_(2.0)
        actor.ema_optimizer.step(optimizer_step=optimizer_step)
    checkpoint.save(actor, iteration=2)
    assert checkpoint._has_previous_ema(tmp_path / "iter_0000003" / "ema")

    resumed = make_actor(tmp_path, keep_previous=True)
    checkpoint.load(resumed)
    for (name, parameter), original in zip(resumed.model.named_parameters(), actor.model.parameters(), strict=True):
        if parameter not in resumed.ema_optimizer.state:
            continue
        previous = resumed.ema_optimizer.state[parameter]["previous_ema"]
        torch.testing.assert_close(previous, actor.ema_optimizer.state[original]["previous_ema"], msg=name)
        assert not torch.equal(previous, resumed.ema_optimizer.state[parameter]["ema"])
    resumed.ema_optimizer.step(optimizer_step=3)
    actor.ema_optimizer.step(optimizer_step=3)
    pairs = zip(
        resumed.ema_optimizer.param_groups[0]["params"], actor.ema_optimizer.param_groups[0]["params"], strict=True
    )
    for parameter, original in pairs:
        torch.testing.assert_close(
            resumed.ema_optimizer.state[parameter]["previous_ema"], actor.ema_optimizer.state[original]["previous_ema"]
        )


def test_checkpoint_without_previous_ema_starts_it_at_the_loaded_ema(tmp_path, cpu_checkpoint):
    # A synchronous run's checkpoint has no previous EMA; the async resume then samples the loaded EMA.
    actor = make_actor(tmp_path)
    with torch.no_grad():
        for parameter in actor.model.parameters():
            parameter.add_(2.0)
    actor.ema_optimizer.step(optimizer_step=1)
    checkpoint.save(actor, iteration=2)

    resumed = make_actor(tmp_path, keep_previous=True)
    checkpoint.load(resumed)
    for parameter, state in resumed.ema_optimizer.state.items():
        previous = state["previous_ema"]
        torch.testing.assert_close(previous, state["ema"])
        assert previous.data_ptr() != state["ema"].data_ptr()
        assert not torch.equal(previous, parameter.detach())


def test_checkpoint_with_previous_ema_loads_without_keep_previous(tmp_path, cpu_checkpoint):
    # Resuming an --async-exact-resume checkpoint without the flag (or synchronously) loads only the EMA.
    actor = make_actor(tmp_path, keep_previous=True)
    with torch.no_grad():
        for parameter in actor.model.parameters():
            parameter.add_(2.0)
    actor.ema_optimizer.step(optimizer_step=1)
    checkpoint.save(actor, iteration=2)

    resumed = make_actor(tmp_path)
    checkpoint.load(resumed)
    pairs = zip(
        resumed.ema_optimizer.param_groups[0]["params"], actor.ema_optimizer.param_groups[0]["params"], strict=True
    )
    for parameter, original in pairs:
        assert list(resumed.ema_optimizer.state[parameter]) == ["ema"]
        torch.testing.assert_close(
            resumed.ema_optimizer.state[parameter]["ema"], actor.ema_optimizer.state[original]["ema"]
        )
