import abc
import logging
import os
import socket
from argparse import Namespace
from collections.abc import Iterator, Sequence
from datetime import timedelta

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]

from sglang.srt.utils import MultiprocessingSerializer, init_custom_process_group
from sglang.srt.utils.network import NetworkAddress

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

LORA_WEIGHT_UPDATE_MODE = "lora_merge"


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


class DiffusionUpdateWeightLoRA(DiffusionUpdateWeight):
    """Push only lora_A/lora_B tensors; rollout merges locally via weight_update_mode=lora_merge."""

    sgl_d_weight_update_mode = LORA_WEIGHT_UPDATE_MODE

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
                "LoRA weight sync v%s [%s]: pushed %d lora tensors in %d buckets",
                self.weight_version,
                target_module,
                num_lora_tensors,
                num_buckets,
            )
            if num_lora_tensors == 0:
                logger.error("LoRA weight sync [%s]: no lora tensors found in training state_dict", target_module)


class DiffusionUpdateWeightFromTensorLoRAIPC(DiffusionUpdateWeightLoRA, DiffusionUpdateWeightFromTensor):
    pass


def connect_rollout_engines_from_distributed(rollout_engines, gpus_per_engine, group_name, timeout):
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    num_trainer_ranks = 1
    world_size = num_trainer_ranks + len(rollout_engines) * gpus_per_engine
    engine_joins = [
        engine.init_weights_update_group.remote(
            master_address=master_address,
            master_port=master_port,
            rank_offset=num_trainer_ranks + i * gpus_per_engine,
            world_size=world_size,
            group_name=group_name,
            backend="nccl",
        )
        for i, engine in enumerate(rollout_engines)
    ]
    options = dist.ProcessGroupNCCL.Options()
    group = init_custom_process_group(
        backend="nccl",
        init_method=NetworkAddress(master_address, master_port).to_tcp(),
        world_size=world_size,
        rank=0,
        group_name=group_name,
        timeout=timeout,
        pg_options=options,
    )
    # This group spans another world, so it must not split the default communicator. torch sets split_from
    # on these options when the default group is device-bound, and NCCL only reads it at the first
    # collective, so clearing it here still applies. Drop once sglang's init_custom_process_group does
    # this itself (sgl-project/sglang#42668).
    options.split_from = None
    ray.get(engine_joins)
    return group


def broadcast_bucket(rollout_engines, group, group_name, named_tensors, target_module, **kwargs):
    engine_receives = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in named_tensors],
            dtypes=[str(tensor.dtype).removeprefix("torch.") for _, tensor in named_tensors],
            shapes=[list(tensor.shape) for _, tensor in named_tensors],
            group_name=group_name,
            target_modules=[target_module],
            **kwargs,
        )
        for engine in rollout_engines
    ]
    tensors = [tensor.contiguous() for _, tensor in named_tensors]
    broadcasts = [dist.broadcast(tensor, src=0, group=group, async_op=True) for tensor in tensors]
    for broadcast in broadcasts:
        broadcast.wait()
    ray.get(engine_receives)


class DiffusionUpdateWeightFromDistributed(DiffusionUpdateWeight):
    def __init__(self, args, models):
        super().__init__(args, models)
        self.rollout_engines = []
        self._model_update_group = None
        self._group_name = "diffusion-weight-update"

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        if dist.get_rank() != 0:
            return
        # Engines reject a group name they already hold, which a partially failed init can leave on some of them,
        # and leaving an unknown group is a no-op there; so every connect first clears the group on every engine.
        engine_leaves = [engine.destroy_weights_update_group.remote(self._group_name) for engine in rollout_engines]
        if self._model_update_group is not None:
            dist.destroy_process_group(self._model_update_group)
            self._model_update_group = None
        ray.get(engine_leaves)
        self.rollout_engines = rollout_engines
        self._model_update_group = connect_rollout_engines_from_distributed(
            rollout_engines=rollout_engines,
            gpus_per_engine=self.args.rollout_num_gpus_per_engine,
            group_name=self._group_name,
            timeout=timedelta(minutes=self.args.distributed_timeout_minutes),
        )

    def update_bucket_weights(self, named_tensors: list[tuple[str, torch.Tensor]], target_module: str) -> None:
        if dist.get_rank() != 0:
            return
        lora_kwargs = {}
        if self.sgl_d_weight_update_mode is not None:
            model = self.models[target_module]
            adapter_config = model.peft_config[model.active_adapter]
            lora_kwargs = dict(
                weight_update_mode=self.sgl_d_weight_update_mode,
                lora_alpha=adapter_config.lora_alpha,
                lora_rank=adapter_config.r,
            )
        broadcast_bucket(
            rollout_engines=self.rollout_engines,
            group=self._model_update_group,
            group_name=self._group_name,
            named_tensors=named_tensors,
            target_module=target_module,
            **lora_kwargs,
        )


class DiffusionUpdateWeightLoRADistributed(DiffusionUpdateWeightLoRA, DiffusionUpdateWeightFromDistributed):
    pass
