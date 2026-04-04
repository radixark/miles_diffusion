import abc
import logging
from argparse import Namespace
from collections.abc import Sequence

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from torch.distributed.tensor import DTensor, Replicate

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]

from sglang.srt.utils import MultiprocessingSerializer

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]


logger = logging.getLogger(__name__)


class DiffusionUpdateWeight(abc.ABC):
    """Base updater used by diffusion training actors."""

    def __init__(self, args: Namespace, model: torch.nn.Module) -> None:
        self.args = args
        self.model = model
        self.weight_version = 0

    @abc.abstractmethod
    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
    ) -> None:
        pass

    def update_weights(self) -> None:
        self.weight_version += 1
        bucket = []
        bucket_size = 0
        for name, param in self.model.state_dict().items():
            param_size = param.numel() * param.element_size()
            if bucket and bucket_size + param_size >= self.args.update_weight_buffer_size:
                self.wait_and_update_bucket_weights(bucket)
                del bucket
                bucket = []
                bucket_size = 0

            param = param.cuda()
            if isinstance(param, DTensor):
                # async version of param.full_tensor
                param = param.redistribute(
                    placements=[Replicate()] * param.device_mesh.ndim,
                    async_op=True,
                ).to_local()
            bucket.append((name, param))
            bucket_size += param_size

        if bucket:
            self.wait_and_update_bucket_weights(bucket)
            del bucket

    def wait_and_update_bucket_weights(self, bucket):
        bucket = [(name, param.wait()) if hasattr(param, "wait") else (name, param) for name, param in bucket]
        self.update_bucket_weights(bucket, weight_version=self.weight_version)

    @abc.abstractmethod
    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
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

    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        monkey_patch_torch_reductions()
        logger.info("Using flattened tensor bucket (diffusion updater)")
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
            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                "metadata": metadata,
            }
            serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

        if self._ipc_gather_src == dist.get_rank():
            gathered_serialized_batches = [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
        else:
            gathered_serialized_batches = None

        dist.gather_object(
            obj=serialized_tensors,
            object_gather_list=gathered_serialized_batches,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        if dist.get_rank() == self._ipc_gather_src:
            # TODO: here we assume all ranks have the same number of dtypes.
            num_dtypes = len(gathered_serialized_batches[0])
            assert num_dtypes > 0
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in gathered_serialized_batches],
                    "load_format": "flattened_bucket",
                    "weight_version": str(weight_version),
                }
                ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
                ray.get(ref)