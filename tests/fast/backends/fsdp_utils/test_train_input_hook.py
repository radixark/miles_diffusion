"""The train-input hook preprocesses a rank's train pairs before they are split into micro-batches.

    train pairs (the converted rollout output) + ctx (external_data from earlier groups)
        --train-input hook (filter / reorder / rewrite)--> preprocessed pairs
    preprocessed pairs --> micro-batch schedule --> training forwards, in order
    preprocessed pairs --> train-output hook (the same ctx)
    no hook            --> the converted train pairs, as before

The production _train_core runs on CPU with the forward, metrics and rank checks replaced; the
optimizer step is skipped, so only the pairs and ctx each stage receives are observed.
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


def _actor(train_input, train_output, forwarded):
    return SimpleNamespace(
        models={},
        reference_models={},
        teacher_models={},
        train_pipeline_config=None,
        sde_backend=None,
        args=Namespace(
            num_steps_per_rollout=2,
            micro_batch_size=1,
            loss_type="policy_loss",
            diffusion_recompute_old_log_prob=False,
            debug_skip_optimizer_step=True,
            log_loss_sigma_bucket=0,
        ),
        _forward_dtype=torch.float32,
        parallel_state=SimpleNamespace(dp_rank=0, dp_group=None),
        custom_train_input_func=train_input,
        custom_train_output_func=train_output,
        optimizer=SimpleNamespace(zero_grad=lambda set_to_none: None),
        global_step=0,
        _maybe_legacy_window_pad_len=lambda train_pairs, microbatch_ranges: None,
        _forward_train_pair_batch=lambda ctx, chunk, **kwargs: forwarded.append((ctx.microbatch_id, chunk)),
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
    actor = _actor(train_input, lambda ctx, train_pairs: (ctx, train_pairs), forwarded)
    ctx, output_pairs = _load_train_core()(
        actor, rollout_id=3, rollout_data={"train_data": pairs}, external_data={"critic": scores}
    )

    preprocessed = [{"sample_index": index, "score": scores[index]} for index in (1, 2, 4, 5)]
    assert seen_by_input == [(ctx, pairs)]
    assert (ctx.rollout_id, ctx.external_data) == (3, {"critic": scores})
    assert forwarded == [(microbatch_id, [pair]) for microbatch_id, pair in enumerate(preprocessed)]
    assert output_pairs == preprocessed
    assert actor.global_step == 2


def test_without_the_hook_the_converted_pairs_train():
    pairs = [{"sample_index": index} for index in range(4)]
    forwarded = []
    actor = _actor(None, lambda ctx, train_pairs: train_pairs, forwarded)

    output_pairs = _load_train_core()(actor, rollout_id=0, rollout_data={"train_data": pairs}, external_data={})

    assert forwarded == [(index, [pair]) for index, pair in enumerate(pairs)]
    assert output_pairs == pairs
