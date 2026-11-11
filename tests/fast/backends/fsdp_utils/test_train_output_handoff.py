"""FSDPTrainRayActor.train hands its output to later groups through the object store, not through the driver."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from miles.utils.ray_utils import Box


def _load_methods(*names):
    """The named FSDPTrainRayActor methods, compiled with Ray, the timers and the perf log replaced."""
    path = Path(__file__).resolve().parents[4] / "miles/backends/fsdp_utils/actor.py"
    actor_class = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "FSDPTrainRayActor"
    )
    methods = [node for node in actor_class.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for method in methods:
        method.decorator_list = []
    store = {}

    def put(value):
        ref = f"ref{len(store)}"
        store[ref] = value
        return ref

    namespace = dict(
        Any=Any,
        Box=Box,
        ray=SimpleNamespace(get=store.__getitem__, put=put),
        dist=SimpleNamespace(get_rank=lambda: 0),
        timer=lambda name: nullcontext(),
        inverse_timer=lambda name: nullcontext(),
        train_metric_utils=SimpleNamespace(log_perf_data_raw=lambda **kwargs: None),
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, store, put


def test_train_unboxes_external_data_and_boxes_its_output():
    namespace, store, put = _load_methods("train")
    train = namespace["train"]
    seen = {}

    def train_core(rollout_id, rollout_data, external_data, optim_steps=None):
        seen.update(rollout_data=rollout_data, external_data=external_data)
        return {"scores": [rollout_id]}

    actor = SimpleNamespace(
        args=SimpleNamespace(offload_train=False, debug_rollout_only=False, use_output_store=False),
        ema_optimizer=None,
        _train_core=train_core,
        parallel_state=SimpleNamespace(get_mesh=lambda name: SimpleNamespace(get_local_rank=lambda: 0)),
    )
    shard = Box(put({"train_data": []}))
    critic_output = Box(put({"scores": [0]}))

    result = train(actor, rollout_id=3, rollout_data_ref=[shard], external_data={"critic": critic_output})

    assert seen == {"rollout_data": {"train_data": []}, "external_data": {"critic": {"scores": [0]}}}
    assert isinstance(result, Box)
    assert store[result.inner] == {"scores": [3]}
    assert result.inner not in (shard.inner, critic_output.inner)
