"""The train-input hook preprocesses a rank's train pairs before they are split into micro-batches.

    train pairs (the converted rollout output) + ctx (external_data from earlier groups)
        --train-input hook (filter / reorder / rewrite)--> preprocessed pairs
    preprocessed pairs --> micro-batch schedule --> training forwards, in order
    preprocessed pairs --> train-output hook (the same ctx, the list the forwards filled)
    no hook            --> the converted train pairs, as before

The trunk numbers each forward: ``microbatch_id`` counts the rollout's micro-batches, ``optim_step_idx`` is the
window; the old-log-prob recompute of windows 1.. reuses their training forwards' numbers.

The production _train_core runs on CPU with the forward, metrics and rank checks replaced; the
optimizer step is skipped, so only the pairs, ctx and numbers each stage receives are observed.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=["fsdp"])

import ast
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext
from miles.utils.train_data_utils import build_microbatch_schedule


def _load_train_core():
    path = Path(__file__).resolve().parents[4] / "miles/backends/fsdp_utils/actor.py"
    actor_class = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "FSDPTrainRayActor"
    )
    method = next(
        node for node in actor_class.body if isinstance(node, ast.FunctionDef) and node.name == "_train_core"
    )
    namespace = dict(
        Any=Any,
        torch=torch,
        DiffusionLossContext=DiffusionLossContext,
        build_microbatch_schedule=build_microbatch_schedule,
        validate_same_microbatch_counts_across_train_ranks=lambda **kwargs: None,
        timer=lambda name: nullcontext(),
        new_metric_buffer=lambda *args, **kwargs: SimpleNamespace(reduce=lambda: {}),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_train_core"]


def _actor(train_input, train_output, forwarded, **args):
    lists = []  # the train_output_inputs object each forward received
    return SimpleNamespace(
        lists=lists,
        models={},
        reference_models={},
        teacher_models={},
        train_pipeline_config=None,
        sde_backend=None,
        args=Namespace(
            num_steps_per_rollout=2,
            micro_batch_size=1,
            loss_type="policy_loss",
            debug_skip_optimizer_step=True,
            log_loss_sigma_bucket=0,
            diffusion_num_steps=4,
            **{"diffusion_recompute_old_log_prob": False, **args},
        ),
        _forward_dtype=torch.float32,
        parallel_state=SimpleNamespace(dp_rank=0, dp_group=None),
        loss_hooks=SimpleNamespace(train_input=train_input, train_output=train_output),
        optimizer=SimpleNamespace(zero_grad=lambda set_to_none: None),
        global_step=0,
        _maybe_legacy_window_pad_len=lambda train_pairs, microbatch_ranges: None,
        _forward_train_pair_batch=lambda ctx, chunk, **kwargs: (
            forwarded.append(
                (kwargs["microbatch_id"], kwargs["optim_step_idx"], kwargs.get("write_old_log_prob", False), chunk)
            ),
            kwargs["train_output_inputs"].append(kwargs["microbatch_id"]),
            lists.append(kwargs["train_output_inputs"]),
        ),
        _log_metrics=lambda rollout_id, metrics, step: None,
    )


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


def test_later_stages_see_the_preprocessed_pairs():
    pairs = [{"sample_index": index} for index in range(6)]
    scores = {1: 0.1, 2: 0.2, 4: 0.4, 5: 0.5}
    seen_by_input = []

    def train_input(ctx, train_pairs):
        seen_by_input.append((ctx, list(train_pairs)))
        critic = ctx.external_data["critic"]
        return [
            {**pair, "score": critic[pair["sample_index"]]} for pair in train_pairs if pair["sample_index"] in critic
        ]

    forwarded = []
    actor = _actor(train_input, lambda ctx, train_pairs, inputs: (ctx, train_pairs, inputs), forwarded)
    ctx, output_pairs, train_output_inputs = _load_train_core()(
        actor, rollout_id=3, rollout_data={"train_data": pairs}, external_data={"critic": scores}
    )

    preprocessed = [{"sample_index": index, "score": scores[index]} for index in (1, 2, 4, 5)]
    assert seen_by_input == [(ctx, pairs)]
    assert (ctx.rollout_id, ctx.external_data) == (3, {"critic": scores})
    # Two optimizer steps of two micro-batches each: the id counts on, the step index follows the window.
    assert forwarded == [
        (0, 0, False, [preprocessed[0]]),
        (1, 0, False, [preprocessed[1]]),
        (2, 1, False, [preprocessed[2]]),
        (3, 1, False, [preprocessed[3]]),
    ]
    assert output_pairs == preprocessed
    # Every forward appended to the one list the train-output hook received.
    assert train_output_inputs == [0, 1, 2, 3]
    assert all(received is train_output_inputs for received in actor.lists)
    assert actor.global_step == 2


def test_without_the_hook_the_converted_pairs_train():
    pairs = [{"sample_index": index} for index in range(4)]
    forwarded = []
    actor = _actor(None, lambda ctx, train_pairs, inputs: train_pairs, forwarded)

    output_pairs = _load_train_core()(actor, rollout_id=0, rollout_data={"train_data": pairs}, external_data={})

    assert forwarded == [(index, index // 2, False, [pair]) for index, pair in enumerate(pairs)]
    assert output_pairs == pairs


def test_the_recompute_passes_the_numbers_of_the_training_forwards_of_windows_1_on():
    pairs = [{"sample_index": index} for index in range(4)]
    forwarded = []
    actor = _actor(None, lambda ctx, train_pairs, inputs: None, forwarded, diffusion_recompute_old_log_prob=True)

    _load_train_core()(actor, rollout_id=0, rollout_data={"train_data": pairs}, external_data={})

    recompute = [(2, 1, True, [pairs[2]]), (3, 1, True, [pairs[3]])]
    training = [(index, index // 2, False, [pair]) for index, pair in enumerate(pairs)]
    assert forwarded == recompute + training
