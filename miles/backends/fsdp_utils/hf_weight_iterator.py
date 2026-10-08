import re
from collections import deque
from collections.abc import Iterator

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor

from miles.backends.fsdp_utils.adaptations import get_param_transform
from miles.backends.fsdp_utils.dtensor import gather_full_param
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups

# PEFT appends the adapter name and ".weight" after lora_A / lora_B
_PEFT_LORA_SUFFIX = re.compile(r"\.(lora_[AB])(?:\.[^.]+)?(?:\.weight)?$")


def to_hf_name(train_name: str) -> str:
    """Strip PEFT wrappers off a training state-dict key, leaving the Diffusers checkpoint name.

    base_model.model.blocks.0.attn.to_q.base_layer.weight      -> blocks.0.attn.to_q.weight
    base_model.model.blocks.0.attn.to_q.lora_A.default.weight  -> blocks.0.attn.to_q.lora_A
    blocks.0.norm1.weight                                      -> blocks.0.norm1.weight
    """
    name = train_name.removeprefix("base_model.model.").replace(".base_layer.", ".")
    return _PEFT_LORA_SUFFIX.sub(r".\1", name)


class FSDPHfWeightIterator(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # base weight -> (lora_A, lora_B, scaling) of the PEFT LoRA wrapping it; its delta is merged in at sync
        self._lora_by_base_weight: dict[str, tuple] = {
            f"{name}.base_layer.weight": (module.lora_A[adapter], module.lora_B[adapter], module.scaling[adapter])
            for name, module in self.model.named_modules()
            if hasattr(module, "lora_A") and hasattr(module, "lora_B")
            for adapter in module.lora_A
        }

    def _iter_hf_param_units(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        weights = ((name, param) for name, param in self.model.state_dict().items() if "lora_" not in name)
        for name, full in self._iter_full_params(weights):
            if name in self._lora_by_base_weight:
                lora_A, lora_B, scaling = self._lora_by_base_weight[name]
                # one delta resident at a time: all of Qwen-Image's deltas at once take tens of GB
                delta = gather_full_param(lora_B.weight) @ gather_full_param(lora_A.weight) * scaling
                full = full + delta.to(full.dtype)
            hf_name = to_hf_name(name)
            expand = get_param_transform(hf_name, full, self.diffusion_model_family)
            yield [(hf_name, full)] if expand is None else list(expand(hf_name, full, self.model))

    def _iter_hf_adapter_units(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """One unit per LoRA module: sglang-d's lora_merge applies its lora_A and lora_B as a pair."""
        halves_by_module: dict[str, list[tuple[str, torch.Tensor]]] = {}
        adapters = (
            (name, param) for name, param in self.model.state_dict().items() if ".lora_A" in name or ".lora_B" in name
        )
        for name, full in self._iter_full_params(adapters):
            hf_name = to_hf_name(name)
            module = hf_name.rsplit(".lora_", 1)[0]
            halves = halves_by_module.setdefault(module, [])
            halves.append((hf_name, full))
            if len(halves) == 2:
                del halves_by_module[module]
                yield halves

    def _hf_atomic_update_groups(self):
        return get_hf_atomic_update_groups(self.diffusion_model_family)

    def _iter_full_params(self, named_params) -> Iterator[tuple[str, torch.Tensor]]:
        """Params as full CUDA tensors in order, with all-gathers issued up to one buffer ahead."""
        pending: deque[tuple[str, torch.Tensor]] = deque()
        pending_bytes = 0
        for name, param in named_params:
            pending.append((name, gather_full_param(param, async_op=True)))
            pending_bytes += param.numel() * param.element_size()
            if pending_bytes >= self.args.update_weight_buffer_size:
                yield from self._drain(pending)
                pending_bytes = 0
        yield from self._drain(pending)

    def _drain(self, pending: deque[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        while pending:
            name, full = pending.popleft()
            yield name, full.wait() if isinstance(full, AsyncCollectiveTensor) else full
