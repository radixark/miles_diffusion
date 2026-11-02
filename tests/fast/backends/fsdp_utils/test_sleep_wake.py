"""wake_up moves a sleeping group's models and optimizers once; a second wake_up before sleep does nothing.

Consecutive phases of one group skip the sleep in between, and waking a frozen model twice would replace its host
copies with its GPU parameters, so the next sleep would leave it on the GPU.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=["fsdp"])

import ast
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace


def _load_sleep_and_wake_up(moves):
    path = Path(__file__).resolve().parents[4] / "miles/backends/fsdp_utils/actor.py"
    actor_class = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "FSDPTrainRayActor"
    )
    methods = [
        node for node in actor_class.body if isinstance(node, ast.FunctionDef) and node.name in ("sleep", "wake_up")
    ]
    namespace = dict(
        timer=lambda fn: fn,
        print_memory=lambda name: None,
        clear_memory=lambda: None,
        get_gloo_group=lambda: None,
        dist=SimpleNamespace(barrier=lambda group: None),
        move_model=lambda model, device: moves.append(("model", model, device)),
        sleep_frozen_model=lambda model, host_copies: moves.append(("frozen", model, "cpu")),
        wake_up_frozen_model=lambda model, host_copies: moves.append(("frozen", model, "cuda")),
        move_optimizer=lambda optimizer, device: moves.append(("optimizer", optimizer, device)),
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["sleep"], namespace["wake_up"]


def _actor():
    return SimpleNamespace(
        args=Namespace(offload_train=True, fsdp_cpu_offload=False),
        actor_models_to_sleep=["student"],
        frozen_models_to_sleep=["teacher"],
        frozen_parameter_host_copies_by_model={"teacher": {}},
        optimizer=SimpleNamespace(zero_grad=lambda set_to_none: None),
        ema_optimizer=None,
        asleep=False,
    )


def test_wake_up_moves_once_until_the_next_sleep():
    moves = []
    sleep, wake_up = _load_sleep_and_wake_up(moves)
    actor = _actor()

    sleep(actor)
    wake_up(actor)
    wake_up(actor)
    sleep(actor)

    to_cpu = [("model", "student", "cpu"), ("frozen", "teacher", "cpu"), ("optimizer", actor.optimizer, "cpu")]
    to_cuda = [("model", "student", "cuda"), ("frozen", "teacher", "cuda"), ("optimizer", actor.optimizer, "cuda")]
    assert moves == to_cpu + to_cuda + to_cpu
    assert actor.asleep


def test_wake_up_without_offload_does_nothing():
    moves = []
    _, wake_up = _load_sleep_and_wake_up(moves)
    actor = _actor()
    actor.args.offload_train = False
    actor.asleep = True

    wake_up(actor)

    assert moves == []
