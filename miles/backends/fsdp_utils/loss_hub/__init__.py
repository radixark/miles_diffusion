"""Diffusion train hooks: prepare + loss formula (actor owns DiT forward).

Swap pieces via:
  ``--custom-prepare-train-batch-path``
  ``--custom-loss-function-path``  (formula only; receives new_pred / ref_pred)
  ``--custom-reward-post-process-path``  (advantage normalisation)
  ``--loss-type nft`` selects the NFT prepare/loss hooks; its rollout converter
  lives in ``miles.ray.data_conversion_hub``.
"""

from miles.backends.fsdp_utils.loss_hub.flow_grpo import flow_grpo_loss_formula, prepare_flow_grpo_batch
from miles.backends.fsdp_utils.loss_hub.nft import prepare_nft_batch
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, PreparedBatch

__all__ = [
    "DiffusionLossContext",
    "PreparedBatch",
    "flow_grpo_loss_formula",
    "prepare_flow_grpo_batch",
    "prepare_nft_batch",
]
