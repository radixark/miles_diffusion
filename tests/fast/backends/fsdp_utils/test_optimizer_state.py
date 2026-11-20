"""Every param the saving run stepped gets its optimizer state back; one it never stepped loads as never-stepped.

    saving run stepped             checkpoint holds            resumed optimizer
    attn1                          attn1                       attn1 restored; audio_attn1, audio_ff never-stepped
    attn1, audio_attn1, audio_ff   all three                   all three restored
    attn1, audio_attn1             attn1, audio_attn1          both restored; audio_ff never-stepped
    nothing (saved before a step)  attn1 (zero state)          --
    then attn1, saved again        attn1                       --   (the first save left the audio params stateless)

The config's ``optimizer_state_allowed_missing = ["audio"]`` only says which params may be absent; whether one is
absent is read from the checkpoint. Every case goes through ``checkpoint.save`` and ``checkpoint.load``.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=["fsdp"])

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from miles.backends.fsdp_utils import checkpoint

MODULES = ("attn1", "audio_attn1", "audio_ff")


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        for name in MODULES:
            setattr(self, name, nn.Linear(4, 4))


@pytest.fixture
def cpu_checkpoint(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)


def _actor(path, stepped=()):
    torch.manual_seed(0)
    model = _Block()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(3 if stepped else 0):
        optimizer.zero_grad()
        sum(getattr(model, name)(torch.ones(2, 4)).sum() for name in stepped).backward()
        optimizer.step()
    return SimpleNamespace(
        model=model,
        optimizer=optimizer,
        ema_optimizer=None,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        global_step=3,
        micro_step=3,
        train_pipeline_config=SimpleNamespace(optimizer_state_allowed_missing=["audio"]),
        args=SimpleNamespace(
            save=str(path), load=str(path), ckpt_step=None, use_lora=False, no_save_optim=False, no_load_optim=False
        ),
    )


@pytest.mark.parametrize(
    "stepped",
    [
        pytest.param(("attn1",), id="the audio branch is never stepped"),
        pytest.param(MODULES, id="the audio branch is trained"),
        pytest.param(("attn1", "audio_attn1"), id="part of the audio branch is trained"),
    ],
)
def test_stepped_params_resume_with_their_state_and_unstepped_ones_start_fresh(tmp_path, cpu_checkpoint, stepped):
    """Dropping state by name loses a trained audio branch's Adam moments; expecting state the run never made fails
    the whole optimizer load."""
    actor = _actor(tmp_path, stepped)
    checkpoint.save(actor, iteration=2)

    resumed = _actor(tmp_path)
    checkpoint.load(resumed)

    for name in MODULES:
        weight, resumed_weight = getattr(actor.model, name).weight, getattr(resumed.model, name).weight
        if name in stepped:
            assert resumed.optimizer.state[resumed_weight]["step"] == 3
            assert torch.equal(
                resumed.optimizer.state[resumed_weight]["exp_avg"], actor.optimizer.state[weight]["exp_avg"]
            )
        else:
            assert not resumed.optimizer.state.get(resumed_weight)


def test_a_save_before_the_first_step_gives_allowed_missing_params_no_state(tmp_path, cpu_checkpoint):
    """DCP gives an optimizer that never stepped zero state for every param and leaves it in the optimizer: this save
    and every later one would hand it to a resume as real state for params the run never trained."""
    actor = _actor(tmp_path)
    checkpoint.save(actor, iteration=0)
    for _ in range(3):
        actor.optimizer.zero_grad()
        actor.model.attn1(torch.ones(2, 4)).sum().backward()
        actor.optimizer.step()
    checkpoint.save(actor, iteration=1)

    before_a_step, after_three_steps = sorted(tmp_path.glob("iter_*/optimizer"))
    for optimizer_dir in (before_a_step, after_three_steps):
        saved = checkpoint.saved_optimizer_param_names(optimizer_dir, "optim_state")
        assert saved == {"attn1.weight", "attn1.bias"}, optimizer_dir
