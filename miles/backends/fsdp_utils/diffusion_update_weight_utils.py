import abc
import logging
import os
from argparse import Namespace
from collections.abc import Iterator, Sequence

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]

from sglang.srt.utils import MultiprocessingSerializer

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

try:
    from sglang.multimodal_gen.runtime.loader.weight_utils import compute_weights_checksum  # type: ignore[import]

    _checksum_import_error: ImportError | None = None
except ImportError as _e:
    compute_weights_checksum = None
    _checksum_import_error = _e

from miles.backends.fsdp_utils.hf_weight_iterator import FSDPHfWeightIterator
from miles.ray.utils import get_physical_gpu_id


logger = logging.getLogger(__name__)

LORA_IPC_WEIGHT_UPDATE_MODE = "lora_merge"


class DiffusionUpdateWeight(abc.ABC):
    """Base updater used by diffusion training actors: sends each component's weight buckets to the rollout."""

    # sent to sglang-d as weight_update_mode with every bucket; None replaces full weights
    sgl_d_weight_update_mode: str | None = None

    def __init__(self, args: Namespace, models: dict[str, torch.nn.Module]) -> None:
        self.args = args
        self.models = models
        self.weight_version = 0
        self.hf_weight_iterators = {
            component: FSDPHfWeightIterator(args, model, diffusion_model_family=args.diffusion_model_family)
            for component, model in models.items()
        }

    @abc.abstractmethod
    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
    ) -> None:
        pass

    def update_weights(self) -> None:
        self.weight_version += 1
        for target_module in self.models:
            self._update_component_weights(target_module)

    def _update_component_weights(self, target_module: str) -> None:
        for bucket in self._iter_buckets(target_module):
            self.update_bucket_weights(bucket, target_module)

    def _iter_buckets(self, target_module: str) -> Iterator[list[tuple[str, torch.Tensor]]]:
        return self.hf_weight_iterators[target_module].iter_hf_weights()

    @abc.abstractmethod
    def update_bucket_weights(self, named_tensors: list[tuple[str, torch.Tensor]], target_module: str) -> None:
        pass


class DiffusionUpdateWeightFromTensor(DiffusionUpdateWeight):
    """Tensor-based updater for diffusion rollout engines."""

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
    ) -> None:
        self.rollout_engines = rollout_engines

        # Here we assume the gpu id of rollout engines and train actors are the same.
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(
                ranks=group_ranks,
                backend="gloo",
            )
            if dist.get_rank() in group_ranks:
                self._ipc_gather_src = start_rank
                self._ipc_gather_group = new_group
                self._ipc_engine = engine
                # Calculate TP rank within this SGLang engine group.
                self.tp_rank = dist.get_rank() - start_rank

    def update_bucket_weights(self, named_tensors: list[tuple[str, torch.Tensor]], target_module: str) -> None:
        monkey_patch_torch_reductions()
        logger.info("Using flattened tensor bucket (diffusion updater, module=%s)", target_module)
        named_tensors_by_dtypes = {}
        for name, tensor in named_tensors:
            dtype = tensor.dtype
            if dtype not in named_tensors_by_dtypes:
                named_tensors_by_dtypes[dtype] = []
            named_tensors_by_dtypes[dtype].append((name, tensor))

        serialized_tensors = []
        for _dtype, named_tensors in named_tensors_by_dtypes.items():
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            metadata = flattened_tensor_bucket.get_metadata()
            # sglang-d WeightsUpdater expects per-module keyed dicts when
            # load_format="flattened_bucket".
            # Uses CUDA IPC for cross-process transfer; actor all-gathers FSDP
            # shards into buckets before the inference engine copies them in.
            # Requires --colocate (shared GPU visibility).
            flattened_tensor_data = {
                target_module: {
                    "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                    "metadata": metadata,
                }
            }
            serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

        if self._ipc_gather_src == dist.get_rank():
            gathered_batches = [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
        else:
            gathered_batches = None

        dist.gather_object(
            obj=(get_physical_gpu_id(), serialized_tensors),
            object_gather_list=gathered_batches,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        if dist.get_rank() == self._ipc_gather_src:
            payload_gpu_uuids = [gpu_uuid for gpu_uuid, _ in gathered_batches]
            gathered_serialized_batches = [tensors for _, tensors in gathered_batches]
            # TODO: here we assume all ranks have the same number of dtypes.
            num_dtypes = len(gathered_serialized_batches[0])
            assert num_dtypes > 0
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in gathered_serialized_batches],
                    "payload_gpu_uuids": payload_gpu_uuids,
                    "load_format": "flattened_bucket",
                    "target_modules": [target_module],
                    "weight_version": str(self.weight_version),
                }
                if self.sgl_d_weight_update_mode is not None:
                    model = self.models[target_module]
                    adapter_config = model.peft_config[model.active_adapter]
                    kwargs["weight_update_mode"] = self.sgl_d_weight_update_mode
                    kwargs["lora_alpha"] = adapter_config.lora_alpha
                    kwargs["lora_rank"] = adapter_config.r
                ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
                ray.get(ref)


# TODO: update weights only for sgl-d LoRA params
class DiffusionUpdateWeightFromTensorLoRA(DiffusionUpdateWeightFromTensor):
    """LoRA-aware updater: pushes base weights with the adapters merged in.

    The rollout engine has no LoRA layers -- it receives standard weight keys
    like ``transformer_blocks.0.attn.to_q.weight``; the iterator computes
    ``W_base + αBA/r`` on the fly (no in-place mutation of the FSDP model).
    """

    def _update_component_weights(self, target_module: str) -> None:
        verify = os.environ.get("MILES_VERIFY_WEIGHT_SYNC", "").lower() in ("1", "true", "yes")
        pushed: list[tuple[str, torch.Tensor]] = []
        for bucket in self._iter_buckets(target_module):
            self.update_bucket_weights(bucket, target_module)
            if verify:
                # CPU snapshots, so the hash covers exactly the bytes the rollout engine stored
                pushed.extend((name, tensor.detach().cpu().contiguous()) for name, tensor in bucket)
        if verify:
            self._verify_weight_sync(pushed, target_module)

    def _verify_weight_sync(self, pairs: list[tuple[str, torch.Tensor]], target_module: str) -> None:
        """Compare our expected merged-transformer SHA-256 against the live
        rollout engine's checksum. Both sides run sgl-d's own
        ``compute_weights_checksum``, so the algorithms cannot drift apart."""
        if dist.get_rank() != self._ipc_gather_src:
            return

        if compute_weights_checksum is None:
            logger.warning(
                "[weight_sync verify] installed sglang does not expose "
                "compute_weights_checksum (%s); skipping checksum verification",
                _checksum_import_error,
            )
            return

        expected = compute_weights_checksum(pairs)

        try:
            remote = ray.get(self._ipc_engine.get_weights_checksum.remote([target_module]))
        except Exception as e:
            logger.error(f"[weight_sync verify] failed to fetch remote checksum: {e}")
            return

        actual = (remote or {}).get(target_module)
        match = expected == actual
        logger.warning(
            f"[weight_sync verify v{self.weight_version}] rank={dist.get_rank()} "
            f"paired_engine_match={match} "
            f"expected={expected[:16] if expected else None} "
            f"actual={(actual or '')[:16] if isinstance(actual, str) else actual}"
        )

        # Cross-engine comparison: only rank 0 does this so we don't spam.
        # Queries ALL engines' checksums and prints them side by side — the
        # rank-specific noise_pred drift we've seen is consistent with
        # engines diverging silently, so this pins it down.
        if dist.get_rank() != 0:
            return
        try:
            per_engine = ray.get([e.get_weights_checksum.remote([target_module]) for e in self.rollout_engines])
        except Exception as e:
            logger.error(f"[weight_sync verify cross-engine] failed: {e}")
            return
        engine_sums = [(idx, (r or {}).get(target_module)) for idx, r in enumerate(per_engine)]
        first_sum = engine_sums[0][1]
        all_equal = all(s == first_sum for _, s in engine_sums)
        pretty = "  ".join(f"eng{idx}={s[:16] if isinstance(s, str) else s}" for idx, s in engine_sums)
        logger.warning(f"[weight_sync verify v{self.weight_version} cross-engine] " f"all_equal={all_equal}  {pretty}")


class DiffusionUpdateWeightFromTensorLoRAIPC(DiffusionUpdateWeightFromTensor):
    """Push only lora_A/lora_B tensors; rollout merges locally via weight_update_mode=lora_merge."""

    sgl_d_weight_update_mode = LORA_IPC_WEIGHT_UPDATE_MODE

    def _iter_buckets(self, target_module: str) -> Iterator[list[tuple[str, torch.Tensor]]]:
        return self.hf_weight_iterators[target_module].iter_hf_adapter_weights()

    def _update_component_weights(self, target_module: str) -> None:
        num_lora_tensors = num_buckets = 0
        for bucket in self._iter_buckets(target_module):
            self.update_bucket_weights(bucket, target_module)
            num_lora_tensors += len(bucket)
            num_buckets += 1
        if self.weight_version <= 2 and dist.is_initialized() and dist.get_rank() == 0:
            logger.info(
                "LoRA IPC weight sync v%s [%s]: pushed %d lora tensors in %d buckets",
                self.weight_version,
                target_module,
                num_lora_tensors,
                num_buckets,
            )
            if num_lora_tensors == 0:
                logger.error("LoRA IPC [%s]: no lora tensors found in training state_dict", target_module)
