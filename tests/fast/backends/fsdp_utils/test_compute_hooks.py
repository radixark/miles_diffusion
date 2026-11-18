"""_compute_core runs a hook set forward-only over the pairs its train-input hook keeps, numbered past the rollout's
training micro-batches, and hands the train-output hook what the forwards kept; nothing trains.

    rollout pairs ──train_input──► kept pairs ──► one no-grad forward per micro-batch, ids after the training ones
                                              ──► reshard the models ──► train_output(ctx, kept pairs, the one list)
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=["fsdp"])

import ast
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from tests.fast.backends.fsdp_utils.test_train_input_hook import _actor

from miles.backends.fsdp_utils.loss_hub import LossHooks
from miles.utils.train_data_utils import build_microbatch_schedule


def _load_compute_core(resharded):
    path = Path(__file__).resolve().parents[4] / "miles/backends/fsdp_utils/actor.py"
    actor_class = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "FSDPTrainRayActor"
    )
    method = next(
        node for node in actor_class.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_core"
    )
    namespace = dict(
        Any=Any,
        LossHooks=LossHooks,
        torch=torch,
        build_microbatch_schedule=build_microbatch_schedule,
        timer=lambda name: nullcontext(),
        new_metric_buffer=lambda *args, **kwargs: "metrics",
        reshard_model=resharded.append,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_compute_core"]


def test_the_kept_pairs_forward_once_each_after_the_training_numbers():
    pairs = [{"sample_index": index} for index in range(6)]
    forwarded, resharded, hooks_seen = [], [], []
    actor = _actor(None, None, forwarded)
    actor.args.num_steps_per_rollout = 3
    actor.models = {"transformer": "critic"}
    actor.reference_models = {"transformer": "reference"}
    recorder = actor._forward_train_pair_batch
    actor._forward_train_pair_batch = lambda ctx, chunk, **kwargs: (
        hooks_seen.append(kwargs["hooks"]),
        recorder(ctx, chunk, **kwargs),
    )
    hooks = LossHooks(
        prepare=None,
        formula=None,
        train_input=lambda ctx, train_pairs: train_pairs[4:],
        train_output=lambda ctx, train_pairs, inputs: (ctx.external_data, train_pairs, inputs),
    )

    output = _load_compute_core(resharded)(
        actor, rollout_id=2, rollout_data={"train_data": pairs}, external_data={"scores": 1}, hooks=hooks
    )

    # Six training micro-batches (3 steps x 2) come first; the two kept pairs forward as 6 and 7, "at" the last step.
    assert forwarded == [(6, 2, False, [pairs[4]]), (7, 2, False, [pairs[5]])]
    assert hooks_seen == [hooks, hooks]
    assert output == ({"scores": 1}, pairs[4:], [6, 7])
    assert resharded == ["critic", "reference"]
    assert actor.global_step == 0


def test_without_hooks_nothing_is_selected_or_returned():
    pairs = [{"sample_index": index} for index in range(2)]
    forwarded, resharded = [], []
    actor = _actor(None, None, forwarded)
    actor.args.num_steps_per_rollout = 1

    output = _load_compute_core(resharded)(
        actor, rollout_id=0, rollout_data={"train_data": pairs}, external_data={}, hooks=LossHooks(None, None)
    )

    assert forwarded == [(2, 0, False, [pairs[0]]), (3, 0, False, [pairs[1]])]
    assert output is None
