import abc
import hashlib
import logging
import os
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
        # Name of the sglang-d pipeline module to target. Defaults to "transformer",
        # which is the DiT component for diffusers-based pipelines.
        self.target_module = getattr(args, "diffusion_target_module", "transformer")

    @abc.abstractmethod
    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
    ) -> None:
        pass

    def update_weights(self) -> None:
        self.weight_version += 1
        state_dict = self.model.state_dict()
        if self.weight_version <= 2 and dist.get_rank() == 0:
            keys = list(state_dict.keys())
            print(
                f"[weight_sync v{self.weight_version}] total={len(keys)} keys, "
                f"first5={keys[:5]}, last3={keys[-3:]}",
                flush=True,
            )

        verify = os.environ.get("MILES_VERIFY_WEIGHT_SYNC", "").lower() in ("1", "true", "yes")
        verify_pairs: list[tuple[str, torch.Tensor]] | None = [] if verify else None

        bucket = []
        bucket_size = 0
        for name, param in state_dict.items():
            param_size = param.numel() * param.element_size()
            if bucket and bucket_size + param_size >= self.args.update_weight_buffer_size:
                self.wait_and_update_bucket_weights(bucket)
                del bucket
                bucket = []
                bucket_size = 0

            param = param.cuda()
            if isinstance(param, DTensor):
                # async version of param.full_tensor. Under Option B (FSDP shards
                # only the dp dim, SP replicates) device_mesh is 1D, so this
                # all-gathers across dp into the full tensor; every rank — dp and
                # sp alike — ends up holding identical full params.
                param = param.redistribute(
                    placements=[Replicate()] * param.device_mesh.ndim,
                    async_op=True,
                ).to_local()
            bucket.append((name, param))
            bucket_size += param_size
            if verify_pairs is not None:
                t = param.wait() if hasattr(param, "wait") else param
                verify_pairs.append((name, t.detach().cpu().contiguous()))

        if bucket:
            self.wait_and_update_bucket_weights(bucket)
            del bucket

        if verify_pairs is not None:
            self._verify_weight_sync(verify_pairs)

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

        # An engine spans the same per-engine span as rollout.init_rollout_engines
        # (min(per_engine_tp, gpus_per_node)). Assert the train world is fully
        # covered so a topology misconfig fails loudly here instead of as a
        # cryptic AttributeError when an orphaned rank reaches update_bucket_weights.
        num_gpu_per_engine = min(self.args.rollout_num_gpus_per_engine, self.args.num_gpus_per_node)
        assert dist.get_world_size() == len(rollout_engines) * num_gpu_per_engine, (
            f"train world {dist.get_world_size()} != {len(rollout_engines)} engines × "
            f"{num_gpu_per_engine} per-engine — rollout grouping would orphan ranks"
        )

        # Here we assume the gpu id of rollout engines and train actors are the same.
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * num_gpu_per_engine
            end_rank = (i + 1) * num_gpu_per_engine
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
        target_module = self.target_module
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
            # load_format="flattened_bucket"; wrap each bucket under the
            # target module name (default "transformer").
            flattened_tensor_data = {
                target_module: {
                    "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                    "metadata": metadata,
                }
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
                    "target_modules": [self.target_module],
                    "weight_version": str(weight_version),
                }
                ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
                ray.get(ref)

    def _verify_weight_sync(self, pairs: list[tuple[str, torch.Tensor]]) -> None:
        """Compare our expected transformer SHA-256 against the live rollout
        engine's checksum. Must match exactly — same algorithm as sglang-d's
        ``compute_weights_checksum`` (sorted by name, name+dtype+shape+bytes).

        Under SP every train rank holds identical full params after the dp-dim
        redistribute, so each gather group's representative rank
        (``_ipc_gather_src``) is a sound source — there is no per-replica
        divergence to reconcile. The cross-engine pass below is what would
        surface any silent divergence between rollout replicas.
        """
        if dist.get_rank() != self._ipc_gather_src:
            return

        # `pairs` are full reconstructed tensors. The rollout DiT is TP-sharded
        # (Column/RowParallelLinear) when an engine spans >1 GPU, so its
        # materialized weights — and thus get_weights_checksum — are per-shard.
        # A full-vs-shard comparison would report a stable mismatch even on a
        # correct sync, so the paired check is only meaningful at tp=1 (rollout
        # is non-SP / data-parallel this phase). The cross-engine check below
        # still detects replica divergence regardless of TP.
        tp_size = min(self.args.rollout_num_gpus_per_engine, self.args.num_gpus_per_node)
        if tp_size == 1:
            expected = self._sha256_named_tensors(pairs)
            try:
                remote = ray.get(
                    self._ipc_engine.get_weights_checksum.remote([self.target_module])
                )
            except Exception as e:
                logger.error(f"[weight_sync verify] failed to fetch remote checksum: {e}")
                return

            actual = (remote or {}).get(self.target_module)
            match = expected == actual
            logger.warning(
                f"[weight_sync verify v{self.weight_version}] rank={dist.get_rank()} "
                f"paired_engine_match={match} "
                f"expected={expected[:16] if expected else None} "
                f"actual={(actual or '')[:16] if isinstance(actual, str) else actual}"
            )
        else:
            logger.warning(
                f"[weight_sync verify v{self.weight_version}] paired checksum skipped: "
                f"rollout tp={tp_size} materializes per-shard weights, not comparable to full tensors"
            )

        # Cross-engine comparison: only rank 0 does this so we don't spam.
        # Queries ALL engines' checksums and prints them side by side, pinning
        # down any silent divergence between rollout replicas.
        if dist.get_rank() != 0:
            return
        try:
            per_engine = ray.get([
                e.get_weights_checksum.remote([self.target_module])
                for e in self.rollout_engines
            ])
        except Exception as e:
            logger.error(f"[weight_sync verify cross-engine] failed: {e}")
            return
        engine_sums = [
            (idx, (r or {}).get(self.target_module))
            for idx, r in enumerate(per_engine)
        ]
        first_sum = engine_sums[0][1]
        all_equal = all(s == first_sum for _, s in engine_sums)
        pretty = "  ".join(
            f"eng{idx}={s[:16] if isinstance(s, str) else s}"
            for idx, s in engine_sums
        )
        logger.warning(
            f"[weight_sync verify v{self.weight_version} cross-engine] "
            f"all_equal={all_equal}  {pretty}"
        )

    @staticmethod
    def _sha256_named_tensors(pairs: list[tuple[str, torch.Tensor]]) -> str:
        """Mirror ``sglang.multimodal_gen.runtime.loader.weight_utils.compute_weights_checksum``.

        Hashes name + dtype + shape + raw bytes so a transpose/reshape that
        keeps the same bytes still changes the checksum (covers shard/replica
        semantics, not only bytes). Must stay byte-for-byte in sync with the
        sglang-d side or verification always reports a mismatch.
        """
        hasher = hashlib.sha256()
        for name, tensor in sorted(pairs, key=lambda x: x[0]):
            hasher.update(name.encode())
            t = tensor.detach()
            if isinstance(t, DTensor):
                t = t._local_tensor
            hasher.update(str(t.dtype).encode())
            hasher.update(str(tuple(t.shape)).encode())
            hasher.update(t.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().data)
        return hasher.hexdigest()


# TODO: update weights only for sgl-d LoRA params
class DiffusionUpdateWeightFromTensorLoRA(DiffusionUpdateWeightFromTensor):
    """LoRA-aware updater: merges adapters into base before pushing to rollout.

    The rollout engine has no LoRA layers — it receives standard weight keys
    like ``transformer_blocks.0.attn.to_q.weight``.  We compute ``W_base + αBA/r``
    on the fly during sync (no in-place mutation of the FSDP model).
    """

    def __init__(self, args, model):
        super().__init__(args, model)
        self._lora_index: dict[str, tuple] = {}
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                for adapter in module.lora_A:
                    self._lora_index[name + ".base_layer.weight"] = (
                        module.lora_A[adapter],
                        module.lora_B[adapter],
                        module.scaling[adapter],
                    )
        logger.info(f"LoRA weight sync: {len(self._lora_index)} mergeable layers")

    def _gather_full(self, t: torch.Tensor) -> torch.Tensor:
        t = t.cuda()
        if isinstance(t, DTensor):
            return t.redistribute(placements=[Replicate()] * t.device_mesh.ndim).to_local()
        return t

    def update_weights(self):
        self.weight_version += 1

        verify = os.environ.get("MILES_VERIFY_WEIGHT_SYNC", "").lower() in ("1", "true", "yes")
        verify_pairs: list[tuple[str, torch.Tensor]] = [] if verify else None

        bucket, bucket_size = [], 0
        for name, param in self.model.state_dict().items():
            if "lora_" in name:
                continue

            param = param.cuda()
            if isinstance(param, DTensor):
                param = param.redistribute(
                    placements=[Replicate()] * param.device_mesh.ndim,
                    async_op=True,
                ).to_local()

            if name in self._lora_index:
                # Merge LoRA for this layer on the fly instead of pre-computing
                # all 720 deltas up front: Qwen-Image's MLP + attn deltas total
                # tens of GB at peak — here only one delta is resident at a time.
                A, B, s = self._lora_index[name]
                delta = (self._gather_full(B.weight) @ self._gather_full(A.weight)) * s
                param = param.wait() if hasattr(param, "wait") else param
                param = param + delta.to(param.device, param.dtype)
                del delta

            # Strip PEFT's two wrapping layers so the name matches sglang-d's
            # un-wrapped DiT state_dict (WeightsUpdater.load_weights_into_model
            # silently drops any name not in ``module.named_parameters()``):
            #
            #   LoRA target  in: base_model.model.transformer_blocks.0.attn.to_q.base_layer.weight
            #                out: transformer_blocks.0.attn.to_q.weight
            #   non-target   in: base_model.model.transformer_blocks.0.norm1.weight
            #                out: transformer_blocks.0.norm1.weight
            #
            # ``.base_layer`` is the inner wrapper (lora.Linear.base_layer);
            # ``base_model.model.`` is PeftModel.base_model (=LoraModel) .model.
            sglang_d_param_name = name.replace(".base_layer", "")
            if sglang_d_param_name.startswith("base_model.model."):
                sglang_d_param_name = sglang_d_param_name[len("base_model.model."):]


            sz = param.numel() * param.element_size()
            if bucket and bucket_size + sz >= self.args.update_weight_buffer_size:
                self.wait_and_update_bucket_weights(bucket)
                bucket, bucket_size = [], 0
            bucket.append((sglang_d_param_name, param))
            bucket_size += sz
            if verify_pairs is not None:
                # Wait on async redistribute handle, snapshot CPU copy so the
                # hash matches what the rollout engine stored (bytes-identical).
                t = param.wait() if hasattr(param, "wait") else param
                verify_pairs.append((sglang_d_param_name, t.detach().cpu().contiguous()))

        if bucket:
            self.wait_and_update_bucket_weights(bucket)

        if verify_pairs is not None:
            self._verify_weight_sync(verify_pairs)