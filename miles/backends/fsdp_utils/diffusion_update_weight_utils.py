import abc
import logging
import os
import re
from argparse import Namespace
from collections.abc import Mapping, Sequence

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

try:
    from sglang.multimodal_gen.runtime.loader.weight_utils import compute_weights_checksum  # type: ignore[import]

    _checksum_import_error: ImportError | None = None
except ImportError as _e:
    compute_weights_checksum = None
    _checksum_import_error = _e


logger = logging.getLogger(__name__)

LORA_IPC_WEIGHT_UPDATE_MODE = "lora_merge"


class PeftLoRAKeyMapper:
    """Map PEFT LoRA state-dict keys to sglang-d tensor names for IPC sync."""

    _LORA_KEY_RE = re.compile(r"\.lora_([AB])(?:\.[^.]+)?(?:\.weight)?$")
    _PEFT_PREFIX = "base_model.model."

    @classmethod
    def is_lora_key(cls, name: str) -> bool:
        return ".lora_A" in name or ".lora_B" in name

    @classmethod
    def to_sgld_name(cls, name: str) -> str | None:
        """Map a PEFT state-dict key to sglang-d LoRA tensor name."""
        if not cls.is_lora_key(name):
            return None

        stripped = name
        if stripped.startswith(cls._PEFT_PREFIX):
            stripped = stripped[len(cls._PEFT_PREFIX) :]

        match = cls._LORA_KEY_RE.search(stripped)
        if match is None:
            return None

        layer_prefix = stripped[: match.start()]
        ab = match.group(1)
        return f"{layer_prefix}.lora_{ab}"

    @classmethod
    def collect_sgld_names(cls, state_dict: Mapping[str, torch.Tensor]) -> set[str]:
        names: set[str] = set()
        for key in state_dict:
            sgld_name = cls.to_sgld_name(key)
            if sgld_name is not None:
                names.add(sgld_name)
        return names

    @classmethod
    def collect_layer_prefixes(cls, state_dict: Mapping[str, torch.Tensor]) -> set[str]:
        return {name.rsplit(".lora_", 1)[0] for name in cls.collect_sgld_names(state_dict)}

    @classmethod
    def summarize_mapping(
        cls,
        state_dict: Mapping[str, torch.Tensor],
    ) -> tuple[int, int, list[str], list[str]]:
        """Return (num_tensors, num_layers, sample_layer_prefixes, unmapped_peft_keys)."""
        sgld_names: list[str] = []
        unmapped: list[str] = []
        for key in state_dict:
            if not cls.is_lora_key(key):
                continue
            sgld_name = cls.to_sgld_name(key)
            if sgld_name is None:
                unmapped.append(key)
            else:
                sgld_names.append(sgld_name)
        layer_prefixes = {name.rsplit(".lora_", 1)[0] for name in sgld_names}
        sample = sorted(layer_prefixes)[:5]
        return len(sgld_names), len(layer_prefixes), sample, unmapped


class DiffusionUpdateWeight(abc.ABC):
    """Base updater used by diffusion training actors."""

    def __init__(self, args: Namespace, models: dict[str, torch.nn.Module]) -> None:
        self.args = args
        self.models = models
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
        for target_module, model in self.models.items():
            self._update_component_weights(target_module, model)

    def _update_component_weights(self, target_module: str, model: torch.nn.Module) -> None:
        state_dict = model.state_dict()
        if self.weight_version <= 2 and dist.get_rank() == 0:
            keys = list(state_dict.keys())
            print(
                f"[weight_sync v{self.weight_version} {target_module}] total={len(keys)} keys, "
                f"first5={keys[:5]}, last3={keys[-3:]}",
                flush=True,
            )
        bucket = []
        bucket_size = 0
        for name, param in state_dict.items():
            param_size = param.numel() * param.element_size()
            if bucket and bucket_size + param_size >= self.args.update_weight_buffer_size:
                self.wait_and_update_bucket_weights(bucket, target_module)
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
            self.wait_and_update_bucket_weights(bucket, target_module)
            del bucket

    def wait_and_update_bucket_weights(self, bucket, target_module: str, weight_update_mode=None):
        bucket = [(name, param.wait()) if hasattr(param, "wait") else (name, param) for name, param in bucket]
        self.update_bucket_weights(
            bucket,
            target_module,
            weight_version=self.weight_version,
            weight_update_mode=weight_update_mode,
        )

    @abc.abstractmethod
    def update_bucket_weights(
        self,
        named_tensors,
        target_module: str,
        weight_version=None,
        weight_update_mode: str | None = None,
    ) -> None:
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

    def update_bucket_weights(
        self,
        named_tensors,
        target_module: str,
        weight_version=None,
        weight_update_mode: str | None = None,
    ) -> None:
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
                    "target_modules": [target_module],
                    "weight_version": str(weight_version),
                }
                if weight_update_mode is not None:
                    kwargs["weight_update_mode"] = weight_update_mode
                    kwargs["lora_alpha"] = self.args.lora_alpha
                    kwargs["lora_rank"] = self.args.lora_rank
                ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
                ray.get(ref)


# TODO: update weights only for sgl-d LoRA params
class DiffusionUpdateWeightFromTensorLoRA(DiffusionUpdateWeightFromTensor):
    """LoRA-aware updater: merges adapters into base before pushing to rollout.

    The rollout engine has no LoRA layers — it receives standard weight keys
    like ``transformer_blocks.0.attn.to_q.weight``.  We compute ``W_base + αBA/r``
    on the fly during sync (no in-place mutation of the FSDP model).
    """

    def __init__(self, args, models):
        super().__init__(args, models)
        # Per-component LoRA index: component -> {param name -> (A, B, scaling)}.
        self._lora_index: dict[str, dict[str, tuple]] = {}
        for component, model in self.models.items():
            index: dict[str, tuple] = {}
            for name, module in model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    for adapter in module.lora_A:
                        index[name + ".base_layer.weight"] = (
                            module.lora_A[adapter],
                            module.lora_B[adapter],
                            module.scaling[adapter],
                        )
            self._lora_index[component] = index
            logger.info(f"LoRA weight sync [{component}]: {len(index)} mergeable layers")

    def _gather_full(self, t: torch.Tensor) -> torch.Tensor:
        t = t.cuda()
        if isinstance(t, DTensor):
            return t.redistribute(placements=[Replicate()] * t.device_mesh.ndim).to_local()
        return t

    def update_weights(self):
        self.weight_version += 1
        for target_module, model in self.models.items():
            self._update_component_weights(target_module, model)

    def _update_component_weights(self, target_module: str, model: torch.nn.Module) -> None:
        verify = os.environ.get("MILES_VERIFY_WEIGHT_SYNC", "").lower() in ("1", "true", "yes")
        verify_pairs: list[tuple[str, torch.Tensor]] = [] if verify else None
        lora_index = self._lora_index[target_module]

        bucket, bucket_size = [], 0
        for name, param in model.state_dict().items():
            if "lora_" in name:
                continue

            param = param.cuda()
            if isinstance(param, DTensor):
                param = param.redistribute(
                    placements=[Replicate()] * param.device_mesh.ndim,
                    async_op=True,
                ).to_local()

            if name in lora_index:
                # Merge LoRA for this layer on the fly instead of pre-computing
                # all 720 deltas up front: Qwen-Image's MLP + attn deltas total
                # tens of GB at peak — here only one delta is resident at a time.
                A, B, s = lora_index[name]
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
                sglang_d_param_name = sglang_d_param_name[len("base_model.model.") :]

            sz = param.numel() * param.element_size()
            if bucket and bucket_size + sz >= self.args.update_weight_buffer_size:
                self.wait_and_update_bucket_weights(bucket, target_module)
                bucket, bucket_size = [], 0
            bucket.append((sglang_d_param_name, param))
            bucket_size += sz
            if verify_pairs is not None:
                # Wait on async redistribute handle, snapshot CPU copy so the
                # hash matches what the rollout engine stored (bytes-identical).
                t = param.wait() if hasattr(param, "wait") else param
                verify_pairs.append((sglang_d_param_name, t.detach().cpu().contiguous()))

        if bucket:
            self.wait_and_update_bucket_weights(bucket, target_module)

        if verify_pairs is not None:
            self._verify_weight_sync(verify_pairs, target_module)

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

    def update_weights(self) -> None:
        self.weight_version += 1
        for target_module, model in self.models.items():
            bucket: list[tuple[str, torch.Tensor]] = []
            bucket_size = 0
            num_lora_keys = 0
            unmapped_keys: list[str] = []

            for name, param in model.state_dict().items():
                if not PeftLoRAKeyMapper.is_lora_key(name):
                    continue
                sgld_name = PeftLoRAKeyMapper.to_sgld_name(name)
                if sgld_name is None:
                    unmapped_keys.append(name)
                    continue

                param = param.cuda()
                if isinstance(param, DTensor):
                    param = param.redistribute(
                        placements=[Replicate()] * param.device_mesh.ndim,
                        async_op=True,
                    ).to_local()

                sz = param.numel() * param.element_size()
                if bucket and bucket_size + sz >= self.args.update_weight_buffer_size:
                    self.wait_and_update_bucket_weights(
                        bucket,
                        target_module,
                        weight_update_mode=LORA_IPC_WEIGHT_UPDATE_MODE,
                    )
                    bucket, bucket_size = [], 0

                bucket.append((sgld_name, param))
                bucket_size += sz
                num_lora_keys += 1

            if bucket:
                self.wait_and_update_bucket_weights(
                    bucket,
                    target_module,
                    weight_update_mode=LORA_IPC_WEIGHT_UPDATE_MODE,
                )

            if self.weight_version <= 2 and dist.get_rank() == 0:
                _, num_layers, sample_layers, _ = PeftLoRAKeyMapper.summarize_mapping(model.state_dict())
                logger.info(
                    "LoRA IPC weight sync v%s [%s]: pushed %d lora tensors, " "%d layer prefixes (unmapped=%d)",
                    self.weight_version,
                    target_module,
                    num_lora_keys,
                    num_layers,
                    len(unmapped_keys),
                )
                if sample_layers:
                    logger.info(
                        "LoRA IPC [%s] sample layer prefixes: %s",
                        target_module,
                        sample_layers,
                    )
                if unmapped_keys:
                    logger.warning(
                        "LoRA IPC unmapped PEFT keys [%s] (first 5): %s",
                        target_module,
                        unmapped_keys[:5],
                    )
                if num_lora_keys == 0:
                    logger.error(
                        "LoRA IPC [%s]: no lora tensors found in training state_dict",
                        target_module,
                    )
