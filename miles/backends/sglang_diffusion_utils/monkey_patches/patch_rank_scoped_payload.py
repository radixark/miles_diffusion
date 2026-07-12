"""SP-aware payload routing for update_weights_from_tensor.

sgl-d's ``GPUWorkerPostTrainingMixin._select_rank_scoped_payload`` only
accepts ``len(payloads) in (1, tp_size)``. miles' colocate weight sync sends
one CUDA-IPC payload per training rank in the engine's GPU span so each
scheduler worker imports memory exported by its co-located training rank
(same-GPU affinity, no cross-GPU traffic). With rollout sequence parallelism
(tp_size=1, sp_degree>1) that list has ``world_size`` entries, which the
stock selector rejects — teach it to route by world rank in that case.
"""

from sglang.multimodal_gen.runtime.distributed import get_world_group
from sglang.multimodal_gen.runtime.post_training.gpu_worker_post_training_mixin import (
    GPUWorkerPostTrainingMixin,
)


def _select_rank_scoped_payload(self, payloads, field_name):
    if not isinstance(payloads, list):
        return None, f"{field_name} must be a list"
    if not payloads:
        return None, f"{field_name} is required"

    if len(payloads) == 1:
        return payloads[0], None

    world_group = get_world_group()
    if len(payloads) == world_group.world_size:
        return payloads[world_group.rank_in_group], None

    return None, (
        f"{field_name} size must be 1 or engine world_size ({world_group.world_size}), got {len(payloads)}"
    )


def apply() -> None:
    GPUWorkerPostTrainingMixin._select_rank_scoped_payload = _select_rank_scoped_payload
