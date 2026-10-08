"""Backend-neutral API for streaming training-side weights as HF-named tensors."""

from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Iterator

import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator.bucketing import (
    AtomicUpdateGroup,
    assemble_atomic_update_groups,
    pack_units_by_size,
)


class HfWeightIteratorBase(ABC):
    """Streams a training model's weights as HF-named tensors.

    Collective: every training rank must drive the iterators in lockstep.
    Yielded tensors are freshly allocated per bucket and stay valid while the
    caller holds a reference.
    """

    def __init__(self, args: Namespace, model: torch.nn.Module, *, diffusion_model_family: str | None) -> None:
        self.args = args
        self.model = model
        self.diffusion_model_family = diffusion_model_family

    def iter_hf_weights(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Model weights as size-bounded buckets of HF-named GPU tensors;
        atomic update groups are never split across buckets."""
        hf_param_units = assemble_atomic_update_groups(self._iter_hf_param_units(), self._hf_atomic_update_groups())
        yield from pack_units_by_size(hf_param_units, self.args.update_weight_buffer_size)

    def iter_hf_adapter_weights(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """The LoRA adapter alone, bucketed like ``iter_hf_weights`` under bare HF names."""
        hf_adapter_units = assemble_atomic_update_groups(
            self._iter_hf_adapter_units(), self._hf_atomic_update_groups()
        )
        yield from pack_units_by_size(hf_adapter_units, self.args.update_weight_buffer_size)

    @abstractmethod
    def _iter_hf_param_units(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Backend hook: one unit per training-side parameter, holding every HF
        tensor it converted into. Collectives must run lockstep on every rank."""

    def _hf_atomic_update_groups(self) -> list[AtomicUpdateGroup]:
        """Backend hook: HF-namespace atomic groups for this model, applied to both entry points. Default none."""
        return []

    @abstractmethod
    def _iter_hf_adapter_units(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Backend hook: the adapter as units under bare HF names. Collectives
        must run lockstep on every rank."""
