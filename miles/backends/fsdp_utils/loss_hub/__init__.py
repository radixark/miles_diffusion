"""Diffusion batch preparation and loss hooks, selected by ``--loss-type`` and trainer role."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from miles.backends.fsdp_utils.loss_hub.dmd2 import (
    critic_loss_formula,
    dmd2_loss_formula,
    is_scored_before_training,
    prepare_critic_batch,
    prepare_generator_batch,
    prepare_scoring_batch,
    score_last_step_samples,
    scoring_formula,
    select_scored_pairs,
)
from miles.backends.fsdp_utils.loss_hub.flow_grpo import flow_grpo_loss_formula, prepare_flow_grpo_batch
from miles.backends.fsdp_utils.loss_hub.nft import nft_loss_formula, prepare_nft_batch
from miles.backends.fsdp_utils.loss_hub.sft import prepare_sft_batch, sft_loss_formula
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, PreparedBatch
from miles.utils.misc import load_function

__all__ = ["LOSS_HOOKS", "DiffusionLossContext", "ForwardOnlyHooks", "LossHooks", "PreparedBatch", "get_loss_hooks"]


@dataclass(frozen=True)
class ForwardOnlyHooks:
    """A no-grad pass over an optimizer step's micro-batches, run before the step trains them.

    It runs before the steps ``runs_before_step(ctx, optim_step_idx)`` names. ``prepare`` and ``formula`` are called
    as a ``LossHooks``' are, with micro-batch ids past the rollout's training ones; the formula returns no loss.
    """

    runs_before_step: Callable[[DiffusionLossContext, int], bool]
    prepare: Callable[..., PreparedBatch]
    formula: Callable[..., None]


@dataclass(frozen=True)
class LossHooks:
    """The hooks one trainer group runs on a rollout.

    ``prepare(ctx, batch, *, microbatch_id, optim_step_idx, pad_to_len=None)`` builds a micro-batch's DiT inputs,
    ``formula(ctx, batch, prepared, *, new_pred, old_pred, ref_pred, metrics, optim_step_idx, train_output_inputs,
    ...)`` turns the predictions into its loss, ``train_input`` preprocesses the rank's train pairs before
    micro-batching, and ``train_output(ctx, train_pairs, train_output_inputs)`` runs after the train steps and hands
    its return value to the groups that train later; ``train_output_inputs`` is the list the formulas filled during
    this train call, those of ``forward_only`` included.
    """

    prepare: Callable[..., PreparedBatch]
    formula: Callable[..., torch.Tensor | None]
    train_input: Callable[[DiffusionLossContext, list], list] | None = None
    train_output: Callable[[DiffusionLossContext, list, list], Any] | None = None
    forward_only: ForwardOnlyHooks | None = None


LOSS_HOOKS: dict[tuple[str, str], LossHooks] = {
    ("policy_loss", "actor"): LossHooks(prepare_flow_grpo_batch, flow_grpo_loss_formula),
    ("nft", "actor"): LossHooks(prepare_nft_batch, nft_loss_formula),
    ("sft_loss", "actor"): LossHooks(prepare_sft_batch, sft_loss_formula),
    ("dmd2", "actor"): LossHooks(prepare_generator_batch, dmd2_loss_formula, train_input=select_scored_pairs),
    ("dmd2", "critic"): LossHooks(
        prepare_critic_batch,
        critic_loss_formula,
        train_output=score_last_step_samples,
        forward_only=ForwardOnlyHooks(is_scored_before_training, prepare_scoring_batch, scoring_formula),
    ),
}


def get_loss_hooks(args: Namespace, role: str) -> LossHooks:
    """The hooks of ``--loss-type`` for the trainer ``role``; only ``custom_loss`` reads the ``--custom-*-path`` flags.

    Under ``custom_loss`` an unset prepare or formula path falls back to Flow-GRPO's.
    """
    if args.loss_type != "custom_loss":
        return LOSS_HOOKS[(args.loss_type, role)]
    flow_grpo_hooks = LOSS_HOOKS[("policy_loss", "actor")]
    prepare = load_function(args.custom_prepare_train_batch_path)
    formula = load_function(args.custom_loss_function_path)
    return LossHooks(
        prepare=flow_grpo_hooks.prepare if prepare is None else prepare,
        formula=flow_grpo_hooks.formula if formula is None else formula,
        train_input=load_function(args.custom_train_input_postprocess_path),
        train_output=load_function(args.custom_train_output_function_path),
    )
