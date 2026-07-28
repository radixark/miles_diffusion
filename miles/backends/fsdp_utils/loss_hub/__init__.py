"""Diffusion train hooks: prepare + loss formula (actor owns DiT forward).

Swap pieces via:
  ``--custom-prepare-train-batch-path``
  ``--custom-loss-function-path``  (formula only; receives new_pred / ref_pred)
  ``--custom-convert-samples-to-train-data-path``
"""

from miles.backends.fsdp_utils.loss_hub.advantages import grpo_normalize_rewards
from miles.backends.fsdp_utils.loss_hub.context import DiffusionLossContext, PreparedBatch
from miles.backends.fsdp_utils.loss_hub.losses import flow_grpo_loss_formula, resolve_loss_formula_fn
from miles.backends.fsdp_utils.loss_hub.prepare import prepare_flow_grpo_batch, prepare_nft_batch, resolve_prepare_fn

__all__ = [
    "DiffusionLossContext",
    "PreparedBatch",
    "flow_grpo_loss_formula",
    "grpo_normalize_rewards",
    "prepare_flow_grpo_batch",
    "prepare_nft_batch",
    "resolve_loss_formula_fn",
    "resolve_prepare_fn",
]
