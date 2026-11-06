"""Source-guarded FSDP param-dtype-map monkeypatch for PyTorch 2.14."""

from __future__ import annotations

import hashlib
import inspect
import math
import types
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy as TorchMixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard import _fsdp_collectives, _fsdp_init, _fsdp_param_group, _fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_api import OffloadPolicy, ReduceScatter
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    AllGather,
    AllGatherResult,
    _div_if_needed,
    _get_device_handle,
    _get_dim0_padded_size,
    _get_gradient_divide_factors,
    _raise_assert_with_print,
    _to_dtype_if_needed,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    DataParallelMeshInfo,
    FSDPMeshInfo,
    HSDPMeshInfo,
    ShardPlacementFnResult,
    resolve_shard_placement,
)
from torch.distributed.fsdp._fully_shard._fsdp_init import _get_post_forward_mesh_info
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    AllReduceState,
    DefaultAllGather,
    DefaultReduceScatter,
    FSDPCommContext,
    FSDPParamGroup,
    TrainingState,
    _get_param_module_infos,
    _ModuleToHandleDict,
)
from torch.distributed.tensor import DTensor

from miles.backends.fsdp_utils.monkey_patches.fsdp_param_dtype_patch import ParamDtypeMixedPrecisionPolicy

if TYPE_CHECKING:
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import foreach_reduce_scatter_copy_in
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

_PATCH_SENTINEL = "_miles_param_dtype_map_patch_applied"
_SOURCE_HASHES = {
    "_init_param_group": "d339dd132d10af36fd46cbc0e21c92c64bf51e0063c14f1469693cd23e3570e5",
    "FSDPParam.__init__": "ccfc41ad32cafabf6cc920b068deefa773e4db2ef35472df3b7ae3a28a7eecc8",
    "FSDPParam.init_dtype_attrs": "e2b28eb455dd6c580a3f2db45043d72d3c7186921bc919dea484d9974711b580",
    "FSDPParamGroup.__init__": "5904f00326c38f01633d01b567454a45d5498c17d4f951c73ba53979210d0315",
    "_get_param_all_gather_inputs": "c4d7773aaf03b40765add090ea0c34838fd444c14b6ffd2f5a07427e923dae2e",
    "foreach_reduce": "cc2a17ca7325d89e14b7d262c7900b1d27187f0f7321b6b7c6e6bb11f8ef17f9",
    "foreach_reduce_scatter_copy_in": "559a065467abbaa578f348bd6a8c6478cfc8ee516e602227b795bc3f6727eb22",
}

_ORIGINAL_INIT_PARAM_GROUP = _fsdp_init._init_param_group
_ORIGINAL_PARAM_INIT = FSDPParam.__init__
_ORIGINAL_PARAM_GROUP_INIT = FSDPParamGroup.__init__
_ORIGINAL_INIT_DTYPE_ATTRS = FSDPParam.init_dtype_attrs
_ORIGINAL_GET_PARAM_ALL_GATHER_INPUTS = _fsdp_collectives._get_param_all_gather_inputs
_ORIGINAL_FOREACH_REDUCE = _fsdp_collectives.foreach_reduce
_ORIGINAL_FOREACH_REDUCE_SCATTER_COPY_IN = _fsdp_collectives.foreach_reduce_scatter_copy_in


def _source_hash(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def _verify_source(name: str, fn) -> None:
    actual = _source_hash(fn)
    expected = _SOURCE_HASHES[name]
    if actual != expected:
        raise RuntimeError(
            f"Cannot apply the Miles FSDP patch: source hash for {name} changed "
            f"(expected {expected}, got {actual})"
        )


def _bind_to_collectives(fn, name: str, *, no_grad: bool):
    bound = types.FunctionType(
        fn.__code__,
        _fsdp_collectives.__dict__,
        name,
        fn.__defaults__,
        fn.__closure__,
    )
    bound.__kwdefaults__ = fn.__kwdefaults__
    bound.__annotations__ = fn.__annotations__
    bound.__module__ = _fsdp_collectives.__name__
    return torch.no_grad()(bound) if no_grad else bound


# =============================================================================
# MILES PATCH: Resolve exact FQNs to managed Parameter objects
# ------------------------------- UPSTREAM ------------------------------------
# No upstream counterpart.
# +++++++++++++++++++++++++++++ MILES ADDITION ++++++++++++++++++++++++++++++++
def _resolve_param_dtype_map(
    mp_policy: TorchMixedPrecisionPolicy,
    modules: tuple[nn.Module, ...],
    params: list[nn.Parameter],
) -> dict[nn.Parameter, torch.dtype]:
    param_dtype_map = mp_policy.param_dtype_map if isinstance(mp_policy, ParamDtypeMixedPrecisionPolicy) else None
    if not param_dtype_map:
        return {}
    managed_params = set(params)
    matched_fqns: set[str] = set()
    resolved_map: dict[nn.Parameter, torch.dtype] = {}
    resolved_fqns: dict[nn.Parameter, str] = {}
    for module in modules:
        for fqn, param in module.named_parameters(remove_duplicate=False):
            if param not in managed_params or fqn not in param_dtype_map:
                continue
            matched_fqns.add(fqn)
            dtype = param_dtype_map[fqn]
            if param in resolved_map and resolved_map[param] != dtype:
                previous_fqn = resolved_fqns[param]
                raise ValueError(
                    "param_dtype_map assigns conflicting dtypes to shared "
                    f"parameter aliases {previous_fqn!r} and {fqn!r}"
                )
            resolved_map[param] = dtype
            resolved_fqns.setdefault(param, fqn)
    unknown_fqns = sorted(set(param_dtype_map).difference(matched_fqns))
    if unknown_fqns:
        raise ValueError(
            "param_dtype_map contains FQNs that do not name a parameter managed "
            f"by this fully_shard call: {unknown_fqns}"
        )
    return resolved_map
    # ============================ END MILES PATCH ================================


# Copied from PyTorch v2.14.1 at 5c4886908584029761b579af026dcfb627c84070.
def _patched_init_param_group(
    state: FSDPState,
    params: list[nn.Parameter],
    modules: tuple[nn.Module, ...],
    mesh_info: DataParallelMeshInfo,
    post_forward_mesh_info: FSDPMeshInfo | None,
    device: torch.device,
    shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None,
    mp_policy: TorchMixedPrecisionPolicy,
    offload_policy: OffloadPolicy,
    reshard_after_forward: bool | int = True,
) -> None:
    """
    Initialize FSDP param groups for the given state.

    Params are grouped by their process group (derived from ``mesh_info`` via
    ``shard_placement_fn``). Each group becomes a separate ``FSDPParamGroup``.
    When ``shard_placement_fn`` is ``None`` or returns the same mesh for all
    params, this creates a single group.
    """
    # =============================================================================
    # MILES PATCH: Import at module top and resolve the public FQN map once for every group
    # ------------------------------- UPSTREAM ------------------------------------
    # # Import here to avoid circular imports
    # from ._fsdp_common import FSDPMeshInfo, resolve_shard_placement
    # from ._fsdp_param_group import FSDPParamGroup
    #
    # if not params:
    #     return
    # +++++++++++++++++++++++++++++++++ MILES +++++++++++++++++++++++++++++++++++++++
    if not params:
        return
    param_dtype_map = _resolve_param_dtype_map(mp_policy, modules, params)
    # ============================ END MILES PATCH ================================

    if shard_placement_fn is None:
        # No shard_placement_fn means all params use the same mesh_info,
        # so no grouping is needed. This also handles DDPMeshInfo from
        # replicate_with_fsdp, which doesn't have shard_process_group.
        state._fsdp_param_groups.append(
            FSDPParamGroup(
                params,
                modules,
                mesh_info,
                post_forward_mesh_info,
                device,
                shard_placement_fn,
                mp_policy,
                offload_policy,
                # =============================================================================
                # MILES PATCH: Pass resolved Parameter-to-dtype overrides to the group
                # ------------------------------- UPSTREAM ------------------------------------
                # No argument follows `offload_policy`.
                # +++++++++++++++++++++++++++++ MILES ADDITION ++++++++++++++++++++++++++++++++
                param_dtype_map,
                # ============================ END MILES PATCH ================================
            )
        )
        return

    # Group params by their process group to support per-param mesh,
    # e.g., expert params using ep_mesh vs regular params using dp_mesh.
    # For HSDP, also key by replicate_process_group to avoid grouping
    # FSDPMeshInfo params with HSDPMeshInfo params that share the same
    # shard_process_group but require different gradient reduction behavior.
    if not isinstance(mesh_info, FSDPMeshInfo):
        raise ValueError(
            "Per-param mesh via shard_placement_fn is not supported with "
            f"{type(mesh_info).__name__}; it requires FSDPMeshInfo (or subclass)"
        )
    pg_to_group: dict[
        tuple[dist.ProcessGroup, dist.ProcessGroup | None],
        tuple[FSDPMeshInfo, list[nn.Parameter]],
    ] = {}
    for param in params:
        param_mesh_info = resolve_shard_placement(
            shard_placement_fn(param),
            mesh_info,
        ).mesh_info
        shard_pg = param_mesh_info.shard_process_group
        replicate_pg: dist.ProcessGroup | None = None
        if isinstance(param_mesh_info, HSDPMeshInfo):
            replicate_pg = param_mesh_info.replicate_process_group
        key = (shard_pg, replicate_pg)
        if key not in pg_to_group:
            pg_to_group[key] = (param_mesh_info, [param])
        else:
            existing_mesh_info = pg_to_group[key][0]
            if existing_mesh_info is not param_mesh_info:
                raise ValueError(
                    f"Params sharing the same process group must use the same "
                    f"FSDPMeshInfo object, but got different objects: "
                    f"{existing_mesh_info} vs {param_mesh_info}"
                )
            pg_to_group[key][1].append(param)

    # Create a FSDPParamGroup per process group
    for group_mesh_info, group_params in pg_to_group.values():
        if group_mesh_info is not mesh_info:
            group_post_forward = _get_post_forward_mesh_info(reshard_after_forward, group_mesh_info)
        else:
            group_post_forward = post_forward_mesh_info
        state._fsdp_param_groups.append(
            FSDPParamGroup(
                group_params,
                modules,
                group_mesh_info,
                group_post_forward,
                device,
                shard_placement_fn,
                mp_policy,
                offload_policy,
                # =============================================================================
                # MILES PATCH: Pass resolved Parameter-to-dtype overrides to the group
                # ------------------------------- UPSTREAM ------------------------------------
                # No argument follows `offload_policy`.
                # +++++++++++++++++++++++++++++ MILES ADDITION ++++++++++++++++++++++++++++++++
                param_dtype_map,
                # ============================ END MILES PATCH ================================
            )
        )


# Copied from PyTorch v2.14.1 at 5c4886908584029761b579af026dcfb627c84070.
def _patched_param_group_init(
    self,
    params: list[nn.Parameter],
    modules: tuple[nn.Module, ...],
    mesh_info: DataParallelMeshInfo,
    post_forward_mesh_info: FSDPMeshInfo | None,
    device: torch.device,
    shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None,
    mp_policy: TorchMixedPrecisionPolicy,
    offload_policy: OffloadPolicy,
    # =============================================================================
    # MILES PATCH: Accept the resolved Parameter-to-dtype map
    # ------------------------------- UPSTREAM ------------------------------------
    # No argument follows `offload_policy`.
    # +++++++++++++++++++++++++++++ MILES ADDITION ++++++++++++++++++++++++++++++++
    param_dtype_map: dict[nn.Parameter, torch.dtype] | None = None,
    # ============================ END MILES PATCH ================================
) -> None:
    self.modules = modules  # permit ref cycle because 1:1 lifetime
    param_module_infos = _get_param_module_infos(params, modules)

    # =============================================================================
    # MILES PATCH: Validate the effective trainable parameter dtypes
    # ------------------------------- UPSTREAM ------------------------------------
    # No validation before constructing `self.fsdp_params`.
    # +++++++++++++++++++++++++++++++++ MILES +++++++++++++++++++++++++++++++++++++++
    param_dtype_map = param_dtype_map or {}
    if param_dtype_map:
        effective_dtypes = {
            param_dtype_map.get(param, mp_policy.param_dtype) or param.dtype for param in params if param.requires_grad
        }
        if len(effective_dtypes) > 1 and mp_policy.reduce_dtype is None:
            raise ValueError("Mixed parameter dtypes require an explicit reduce_dtype")
    # ============================ END MILES PATCH ================================

    # =============================================================================
    # MILES PATCH: Apply each override without changing FSDPParam.__init__
    # ------------------------------- UPSTREAM ------------------------------------
    # self.fsdp_params = [
    #     FSDPParam(
    #         param,
    #         module_info,
    #         mesh_info,
    #         post_forward_mesh_info,
    #         device,
    #         shard_placement_fn,
    #         mp_policy,
    #         offload_policy,
    #     )
    #     for param, module_info in zip(params, param_module_infos)
    # ]
    # +++++++++++++++++++++++++++++++++ MILES +++++++++++++++++++++++++++++++++++++++
    self.fsdp_params = []
    for param, module_info in zip(params, param_module_infos, strict=True):
        override = param_dtype_map.get(param)
        param_mp_policy = (
            replace(
                mp_policy,
                param_dtype=(override if override is not None else mp_policy.param_dtype),
                param_dtype_map=None,
            )
            if param_dtype_map
            else mp_policy
        )
        fsdp_param = FSDPParam(
            param,
            module_info,
            mesh_info,
            post_forward_mesh_info,
            device,
            shard_placement_fn,
            param_mp_policy,
            offload_policy,
        )
        self.fsdp_params.append(fsdp_param)
    # ============================ END MILES PATCH ================================
    self.mesh_info = mesh_info
    self.post_forward_mesh_info = post_forward_mesh_info
    self.device = device
    self.device_handle = _get_device_handle(device.type)
    self.mp_policy = mp_policy
    self.offload_policy = offload_policy
    self._training_state = TrainingState.IDLE
    # Group's sharded state always matches its parameters' sharded states
    self._sharded_state = ShardedState.SHARDED
    self._module_fqn: str | None = None  # prefixed from root module
    # Only consider resetting sharded parameters once in lazy init since it
    # can incur nontrivial overhead to reset them
    self._reset_sharded_params: bool = False

    # - Hook state
    self._module_to_pre_save_state_dict_hook_handle: _ModuleToHandleDict = {}
    self._module_to_pre_load_state_dict_hook_handle: _ModuleToHandleDict = {}
    self._all_reduce_hook: Callable[[torch.Tensor], None] | None = None
    self._all_gather_comm: AllGather = DefaultAllGather()
    self._all_gather_output = torch.empty(0, device=self.device)
    self._reduce_scatter_comm: ReduceScatter = DefaultReduceScatter()
    # Optional stream to run the user-defined all-reduce hook in
    # Saved here and not in the comm. context because we allow the user to
    # specify it, possibly at construction time before lazy init
    self._all_reduce_hook_stream: torch.cuda.Stream | None = None

    # - Communication and communication/computation overlap
    self.comm_ctx = FSDPCommContext()
    self._param_group_index: int = 0
    self._num_param_groups: int = 1
    # Group's indices in the shared post-forward order
    self._post_forward_indices: list[int] = []
    # Whether to reduce gradients at all (whether for FSDP or HSDP)
    self.reduce_grads: bool = True
    # Whether to all-reduce gradients for HSDP; only used if
    # `self.reduce_grads` is true, in which case setting this to false
    # means reduce-scatter but no all-reduce
    self.all_reduce_grads: bool = True
    # Whether to reshard parameters after backward (only useful for
    # gradient accumulation)
    self.reshard_after_backward: bool = True
    # Per-group input for the reduce-scatter copy-in (chunk_cat) buffer
    # cap-K, set via FSDPModule.set_reduce_scatter_max_input_buffers (see
    # its docstring). FSDP keeps 1 by default. These per-group values are
    # resolved into the single shared
    # comm_ctx.reduce_scatter_max_input_buffers that post_backward reads.
    self.reduce_scatter_max_input_buffers: int = 1
    # Optional custom factor for the gradient reduction op (e.g. to divide
    # by a factor other than the world size)
    self.gradient_divide_factor: float | None = None
    # Whether to include zero gradients for parameters that did not
    # receive a gradient in backward (e.g. due to conditional parameter
    # usage across ranks). This ensures all ranks participate in the same
    # reduce-scatter collectives, avoiding collective mismatch errors.
    self.reduce_scatter_unused_params: bool = False
    # Whether reduce-scatter and all-reduce should be issued using only
    # summations, potentially with separate pre-/post-scaling.
    self.force_sum_reduction_for_comms: bool = False
    # `async_op` arg used for pre-forward/pre-backward unshard; can be
    # overridden to only do explicit prefetching and avoid inter-stream
    # fragmentation from using separate unshard streams
    self.unshard_async_op: bool = False
    # Whether to unshard in backward: can be overridden by the user if the
    # parameters in this group are not needed for backward (e.g. embedding)
    self.unshard_in_backward: bool = True

    # - CUDA events for stream synchronization
    # Holds the all-gather output buffer, sync objects, and metadata
    self._all_gather_result: AllGatherResult | None = None
    # Holds the reduce-scatter/all-reduce view-out CUDA event that marks the end of
    # the group's post-backward (e.g. reduce-scatter, all-reduce and div), which
    # should be waited on at the end of backward
    self._post_reduce_event: torch.Event | None = None
    # Holds the reshard-after-forward CUDA event when resharding to a
    # different world size, which should be waited on in the next unshard
    self._reshard_after_forward_event: torch.Event | None = None

    # Only for HSDP, if accumulating gradients without all-reduce, save the
    # partial reduce output (only reduce-scattered but not all-reduced)
    self._partial_reduce_output: torch.Tensor | None = None
    # Holds the reduce-dtype AR buffer + completion event across
    # layers in HSDP+AR with reduce_dtype != orig_dtype (e.g., bf16
    # reduce + fp32 params). Structural invariant: the live Python
    # ref keeps the buffer off the caching allocator's free list,
    # preventing the next layer's RS from reusing the same physical
    # block while this layer's AR is still in flight. See
    # AllReduceState docstring and regression test PR #180900.
    self._all_reduce_state: AllReduceState | None = None


# Copied from PyTorch v2.14.1 at 5c4886908584029761b579af026dcfb627c84070.
def _patched_init_dtype_attrs(
    self: FSDPParam,
    mp_policy: TorchMixedPrecisionPolicy,
) -> None:
    # =============================================================================
    # MILES PATCH: Select this parameter's effective dtype
    # ------------------------------- UPSTREAM ------------------------------------
    # param_dtype, reduce_dtype = (mp_policy.param_dtype, mp_policy.reduce_dtype)
    # +++++++++++++++++++++++++++++++++ MILES +++++++++++++++++++++++++++++++++++++++
    has_param_dtype_map = isinstance(mp_policy, ParamDtypeMixedPrecisionPolicy) and bool(mp_policy.param_dtype_map)
    param_dtype, reduce_dtype = (
        self.mp_policy.param_dtype,
        self.mp_policy.reduce_dtype,
    )
    # ============================ END MILES PATCH ================================
    self.orig_dtype = self.sharded_param.dtype
    # Clamp `reduce_dtype` to `None` if no casting is required: since
    # gradients are computed in `param_dtype`, if `reduce_dtype` matches,
    # then we do not need extra casting
    # =============================================================================
    # MILES PATCH: Keep the common reduce dtype for a per-parameter dtype map
    # ------------------------------- UPSTREAM ------------------------------------
    # if reduce_dtype == param_dtype:
    #     reduce_dtype = None
    # +++++++++++++++++++++++++++++++++ MILES +++++++++++++++++++++++++++++++++++++++
    # Per-parameter mixed dtypes require one explicit group reduce dtype.
    if not has_param_dtype_map and reduce_dtype == param_dtype:
        reduce_dtype = None
    # ============================ END MILES PATCH ================================
    # Clamp `param_dtype` to `None` if no casting is required or if the
    # parameter is non-floating-point (mixed precision is only meaningful
    # for floating-point parameters)
    if param_dtype == self.orig_dtype or not self.orig_dtype.is_floating_point:
        param_dtype = None
    self.param_dtype = param_dtype
    self.reduce_dtype = reduce_dtype
    # None indicates that the mixed precision is not enabled


# Copied from PyTorch v2.14.1 at 5c4886908584029761b579af026dcfb627c84070.
def _patched_get_param_all_gather_inputs(
    fsdp_params: list[FSDPParam],
) -> list[list[torch.Tensor]]:
    # Intentionally try to run a fast-path that bypasses abstractions for the
    # common FSDP case of bf16/fp32 mixed precision in order to use foreach
    # copy for lower CPU overhead and more efficient copying in eager
    def use_foreach_copy(fsdp_param: FSDPParam) -> bool:
        return (
            fsdp_param.param_dtype is not None
            and not fsdp_param.offload_to_cpu
            and not hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
        )

    param_all_gather_inputs: list[list[torch.Tensor]] = [[] for _ in fsdp_params]
    # =============================================================================
    # MILES PATCH: Bucket foreach-copy metadata by destination dtype
    # ------------------------------- UPSTREAM ------------------------------------
    # foreach_copy_indices: list[int] = []
    # foreach_copy_inputs: list[torch.Tensor] = []
    # foreach_copy_input_numels: list[int] = []
    # +++++++++++++++++++++++++++++++++ MILES +++++++++++++++++++++++++++++++++++++++
    foreach_copy_infos: dict[torch.dtype, tuple[list[int], list[torch.Tensor], list[int]]] = {}
    # ============================ END MILES PATCH ================================

    # 1st pass: for foreach-copy parameters, get inputs and metadata for the
    # foreach copy, and for the others, actually get their all-gather inputs
    for i, fsdp_param in enumerate(fsdp_params):
        if use_foreach_copy(fsdp_param):
            # =============================================================================
            # MILES PATCH: Select the dtype bucket for this parameter
            # ------------------------------- UPSTREAM ------------------------------------
            # foreach_copy_indices.append(i)
            # +++++++++++++++++++++++++++++++++ MILES +++++++++++++++++++++++++++++++++++++++
            param_dtype = cast(torch.dtype, fsdp_param.param_dtype)
            indices, inputs, input_numels = foreach_copy_infos.setdefault(param_dtype, ([], [], []))
            indices.append(i)
            # ============================ END MILES PATCH ================================

            all_gather_input = (
                fsdp_param._sharded_param_data
                if fsdp_param.sharded_state == ShardedState.SHARDED
                else cast(torch.Tensor, fsdp_param._sharded_post_forward_param_data)
            )
            # =============================================================================
            # MILES PATCH: Record metadata in the selected dtype bucket
            # ------------------------------- UPSTREAM ------------------------------------
            # foreach_copy_inputs.append(all_gather_input)
            # foreach_copy_input_numels.append(all_gather_input.numel())
            # +++++++++++++++++++++++++++++++++ MILES +++++++++++++++++++++++++++++++++++++++
            inputs.append(all_gather_input)
            input_numels.append(all_gather_input.numel())
            # ============================ END MILES PATCH ================================
        else:
            param_all_gather_inputs[i] = fsdp_param.all_gather_inputs

    # =============================================================================
    # MILES PATCH: Allocate and cast one flat input per destination dtype
    # ------------------------------- UPSTREAM ------------------------------------
    # # 2nd pass: use foreach copy to compute the remaining all-gather inputs
    # if foreach_copy_inputs:
    #     fsdp_param_0 = fsdp_params[foreach_copy_indices[0]]
    #     param_dtype, device = fsdp_param_0.param_dtype, fsdp_param_0.device
    #     flat_foreach_copy_input = torch.empty(
    #         (sum(foreach_copy_input_numels),), device=device, dtype=param_dtype
    #     )
    #     splits = torch.split(flat_foreach_copy_input, foreach_copy_input_numels)
    #     torch._foreach_copy_(splits, foreach_copy_inputs)
    #     for i, split in zip(foreach_copy_indices, splits):
    #         param_all_gather_inputs[i] = [split]
    # +++++++++++++++++++++++++++++++++ MILES +++++++++++++++++++++++++++++++++++++++
    # 2nd pass: use foreach copy to compute the remaining all-gather inputs,
    # preserving one uniform source dtype per foreach invocation.
    for param_dtype, (indices, inputs, input_numels) in foreach_copy_infos.items():
        device = fsdp_params[indices[0]].device
        flat_foreach_copy_input = torch.empty((sum(input_numels),), device=device, dtype=param_dtype)
        splits = torch.split(flat_foreach_copy_input, input_numels)
        torch._foreach_copy_(splits, inputs)
        for i, split in zip(indices, splits, strict=True):
            param_all_gather_inputs[i] = [split]
    # ============================ END MILES PATCH ================================

    return param_all_gather_inputs


# Copied from PyTorch v2.14.1 at 5c4886908584029761b579af026dcfb627c84070.
def _patched_foreach_reduce(
    fsdp_params: list[FSDPParam],
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    reduce_scatter_comm: ReduceScatter,
    orig_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    device: torch.device,
    gradient_divide_factor: float | None,
    all_reduce_group: dist.ProcessGroup | None,  # not `None` iff HSDP
    all_reduce_stream: torch.Stream,
    all_reduce_grads: bool,
    partial_reduce_output: torch.Tensor | None,  # only used for HSDP
    all_reduce_hook: Callable[[torch.Tensor], None] | None,
    force_sum_reduction_for_comms: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Event,
    torch.Stream,
    torch.Event,
    torch.Tensor | None,
    torch.Event | None,
    torch.Tensor | None,
]:
    """
    ``unsharded_grads`` owns the references to the gradients computed by
    autograd, so clearing the list frees the gradients.
    """

    # =============================================================================
    # MILES PATCH: Permit mixed gradient dtypes when reduce_dtype is explicit
    # ------------------------------- UPSTREAM ------------------------------------
    # grad_dtypes = {grad.dtype for grad in unsharded_grads}
    # if len(grad_dtypes) != 1:
    #     # Check this at runtime since it could be a real runtime error if e.g.
    #     # fp8 weights do not produce the correct higher precision gradients
    #     _raise_assert_with_print(
    #         f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}"
    #     )
    # grad_dtype = unsharded_grads[0].dtype
    # reduce_dtype = reduce_dtype or grad_dtype
    # +++++++++++++++++++++++++++++++++ MILES +++++++++++++++++++++++++++++++++++++++
    grad_dtypes = {grad.dtype for grad in unsharded_grads}
    if reduce_dtype is None and len(grad_dtypes) != 1:
        _raise_assert_with_print(
            "FSDP reduce-scatter requires an explicit reduce dtype for mixed " f"gradient dtypes but got {grad_dtypes}"
        )
    reduce_dtype = reduce_dtype or unsharded_grads[0].dtype
    # ============================ END MILES PATCH ================================
    (predivide_factor, postdivide_factor, reduce_scatter_op, all_reduce_op) = _get_gradient_divide_factors(
        reduce_scatter_group,
        all_reduce_group,
        reduce_dtype,
        device.type,
        gradient_divide_factor,
        force_sum_reduction_for_comms,
    )

    if reduce_scatter_group is None:
        world_size = 1
    else:
        world_size = reduce_scatter_group.size()
    device_handle = _get_device_handle(device.type)
    current_stream = device_handle.current_stream()

    if world_size > 1:
        for i, (fsdp_param, unsharded_grad) in enumerate(zip(fsdp_params, unsharded_grads, strict=True)):
            if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
                continue
            if unsharded_grad.size(shard_dim) % world_size != 0:
                raise AssertionError(
                    f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} {world_size=}"
                )
            chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
            unsharded_grads[i] = torch.cat(chunks, dim=0)

    padded_unsharded_sizes = tuple(_get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads)
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
    reduce_scatter_input = reduce_scatter_comm.allocate(
        (reduce_scatter_input_numel,),
        dtype=reduce_dtype,
        device=device,
    )

    foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)

    # Only after the copy-in finishes can we free the gradients
    unsharded_grads.clear()
    reduce_scatter_stream.wait_stream(current_stream)
    all_reduce_input = None
    all_reduce_event = None

    with device_handle.stream(reduce_scatter_stream):
        reduce_output = reduce_scatter_comm.allocate(
            (reduce_scatter_output_numel,),
            dtype=reduce_dtype,
            device=device,
        )
        _div_if_needed(reduce_scatter_input, predivide_factor)
        if world_size > 1:
            reduce_scatter_comm(
                output_tensor=reduce_output,
                input_tensor=reduce_scatter_input,
                group=reduce_scatter_group,
                op=reduce_scatter_op,
            )
        else:
            # For single GPU, just copy the input to output (no actual reduce-scatter needed), and
            # account for a possible gradient_divide_factor.
            if gradient_divide_factor is not None:
                reduce_output.copy_(reduce_scatter_input / gradient_divide_factor)
            else:
                reduce_output.copy_(reduce_scatter_input)
        reduce_scatter_event = reduce_scatter_stream.record_event()
        post_reduce_stream = reduce_scatter_stream
        if all_reduce_group is not None:  # HSDP or DDP/replicate
            # Accumulations must run in the reduce-scatter stream
            if not all_reduce_grads:
                if partial_reduce_output is not None:
                    partial_reduce_output += reduce_output
                else:
                    partial_reduce_output = reduce_output
                return (
                    reduce_scatter_input,
                    reduce_scatter_event,
                    post_reduce_stream,
                    post_reduce_stream.record_event(),
                    all_reduce_input,
                    all_reduce_event,
                    partial_reduce_output,
                )
            if partial_reduce_output is not None:
                reduce_output += partial_reduce_output
            post_reduce_stream = all_reduce_stream
            if world_size >= 1:
                all_reduce_stream.wait_stream(reduce_scatter_stream)
            else:
                all_reduce_stream.wait_stream(current_stream)
            with device_handle.stream(all_reduce_stream):
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=all_reduce_op,
                )
                # Keep refs to the reduce-dtype AR buffer + completion
                # event so FSDPParamGroup._all_reduce_state can hold them
                # across layers. This keeps the buffer off the caching
                # allocator's free list; otherwise the next layer's
                # reduce-scatter can reuse the same physical block while
                # this layer's AR is still in flight, causing cross-layer
                # gradient aliasing under slow AR. See PR #140044,
                # regression test PR #180900.
                all_reduce_input = reduce_output
                all_reduce_event = all_reduce_stream.record_event()
    # -- END: ops in reduce_scatter stream

    if all_reduce_hook is not None:
        # Execute user-specified all reduce hook.
        # If native HSDP is used, this is executed after the HSDP all reduce.
        # If 1-d FSDP is used, this is executed post reduce-scatter.
        post_reduce_stream = all_reduce_stream
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        with device_handle.stream(all_reduce_stream):
            all_reduce_hook(reduce_output)
    # -- END: ops post reduce_scatter

    with device_handle.stream(post_reduce_stream):
        _div_if_needed(reduce_output, postdivide_factor)
        # Rebinds to a new orig_dtype tensor when reduce_dtype !=
        # orig_dtype. Do NOT rely on this stream-scoped rebind to manage
        # the old reduce-dtype buffer's lifetime: the rebind orders the
        # cast before the free-event on AR stream, but the freed block
        # lands on the caching allocator's free list and the next layer's
        # RS on RS stream can reuse it without waiting for this layer's
        # AR to finish. The reduce-dtype buffer is held across layers by
        # FSDPParamGroup._all_reduce_state (captured above) to prevent
        # this. See PR #140044, regression test PR #180900.
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        # View out and accumulate sharded gradients
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(padded_unsharded_sizes, fsdp_params, strict=True):
            # Assume even sharding for Shard(i), i > 0; otherwise would require
            # copy-out for contiguous strides
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            if fsdp_param.offload_to_cpu:
                # Only overlap the D2H copy (copying to pinned memory) when no
                # in-backward CPU consumer of the grad exists. Two such
                # consumers suppress the overlap:
                #   - Accumulating grads: the CPU add kernel depends on the
                #     copy result and we cannot run the add as a callback.
                #   - Post-accumulate-grad hooks: user code (e.g.
                #     optimizer-in-backward) reads ``param.grad`` on CPU
                #     synchronously. With ``non_blocking=True`` the hook would
                #     observe in-flight pinned memory — silently wrong
                #     optimizer updates.
                has_post_acc_grad_hook = bool(
                    getattr(
                        fsdp_param.sharded_param,
                        "_post_accumulate_grad_hooks",
                        None,
                    )
                )
                non_blocking = fsdp_param.pin_memory and not to_accumulate_grad and not has_post_acc_grad_hook
                # Since the GPU sharded gradient is allocated in the RS stream,
                # we can free it here by not keeping a ref without waiting for
                # the D2H copy since future RS-stream ops run after the copy
                new_sharded_grad = new_sharded_grad.to(torch.device("cpu"), non_blocking=non_blocking)
                if non_blocking:
                    # Record an event on which to block the CPU thread to
                    # ensure that the D2H copy finishes before the optimizer
                    fsdp_param.grad_offload_event = post_reduce_stream.record_event()
            if to_accumulate_grad:
                if not isinstance(fsdp_param.sharded_param.grad, DTensor):
                    raise AssertionError(
                        f"Expected fsdp_param.sharded_param.grad to be DTensor, got {type(fsdp_param.sharded_param.grad)}"
                    )
                fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
            else:
                new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            for hook in (getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {}) or {}).values():
                hook(fsdp_param.sharded_param)
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        post_reduce_event = post_reduce_stream.record_event()
    # The RS output is allocated in the RS stream and used in the default
    # stream (for optimizer). To ensure its memory is not reused for later
    # RSs, we do not need extra synchronization since the sharded parameters
    # hold refs through the end of backward.
    return (
        reduce_scatter_input,
        reduce_scatter_event,
        post_reduce_stream,
        post_reduce_event,
        all_reduce_input,
        all_reduce_event,
        None,
    )


# Copied from PyTorch v2.14.1 at 5c4886908584029761b579af026dcfb627c84070.
def _patched_foreach_reduce_scatter_copy_in(
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    world_size: int,
) -> None:
    reduce_scatter_input = reduce_scatter_input.view(world_size, -1)
    # =============================================================================
    # MILES PATCH: Preserve chunk_cat as the uniform-dtype fast path
    # ------------------------------- UPSTREAM ------------------------------------
    # torch.ops.fsdp.chunk_cat(
    #     unsharded_grads, dim=0, num_chunks=world_size, out=reduce_scatter_input
    # )
    # +++++++++++++++++++++++++++++++++ MILES +++++++++++++++++++++++++++++++++++++++
    if len({grad.dtype for grad in unsharded_grads}) == 1:
        torch.ops.fsdp.chunk_cat(
            unsharded_grads,
            dim=0,
            num_chunks=world_size,
            out=reduce_scatter_input,
        )
        return
    # ============================ END MILES PATCH ================================

    # =============================================================================
    # MILES PATCH: Pack mixed dtypes directly into the common reduce buffer
    # ------------------------------- UPSTREAM ------------------------------------
    # No mixed-dtype path; the unconditional chunk_cat above required one dtype.
    # +++++++++++++++++++++++++++++ MILES ADDITION ++++++++++++++++++++++++++++++++
    # Pack each parameter's padded rank chunks directly into the rank-major
    # reduce-scatter input, grouping by source dtype to batch cast-and-copy
    # without an intermediate buffer.
    copy_infos: dict[torch.dtype, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    padding_views: list[torch.Tensor] = []
    output_offset = 0
    for grad in unsharded_grads:
        chunk_size = math.ceil(grad.size(0) / world_size)
        trailing_numel = math.prod(grad.shape[1:])
        padded_chunk_numel = chunk_size * trailing_numel
        destinations, sources = copy_infos.setdefault(grad.dtype, ([], []))
        for rank in range(world_size):
            destination = reduce_scatter_input[rank].narrow(0, output_offset, padded_chunk_numel)
            start = rank * chunk_size
            length = min(chunk_size, max(grad.size(0) - start, 0))
            if length > 0:
                source = grad.narrow(0, start, length).reshape(-1)
                destinations.append(destination.narrow(0, 0, source.numel()))
                sources.append(source)
            if length < chunk_size:
                padding_start = length * trailing_numel
                padding_views.append(
                    destination.narrow(
                        0,
                        padding_start,
                        padded_chunk_numel - padding_start,
                    )
                )
        output_offset += padded_chunk_numel

    for destinations, sources in copy_infos.values():
        if destinations:
            torch._foreach_copy_(destinations, sources)
    if padding_views:
        torch._foreach_zero_(padding_views)
    # ============================ END MILES PATCH ================================


def apply_param_dtype_map_patch() -> None:
    if getattr(_fsdp_collectives, _PATCH_SENTINEL, False):
        return

    _verify_source("_init_param_group", _ORIGINAL_INIT_PARAM_GROUP)
    _verify_source("FSDPParam.__init__", _ORIGINAL_PARAM_INIT)
    _verify_source("FSDPParam.init_dtype_attrs", _ORIGINAL_INIT_DTYPE_ATTRS)
    _verify_source("FSDPParamGroup.__init__", _ORIGINAL_PARAM_GROUP_INIT)
    _verify_source(
        "_get_param_all_gather_inputs",
        _ORIGINAL_GET_PARAM_ALL_GATHER_INPUTS,
    )
    _verify_source("foreach_reduce", _ORIGINAL_FOREACH_REDUCE)
    _verify_source(
        "foreach_reduce_scatter_copy_in",
        _ORIGINAL_FOREACH_REDUCE_SCATTER_COPY_IN,
    )

    get_param_all_gather_inputs = _bind_to_collectives(
        _patched_get_param_all_gather_inputs,
        "_get_param_all_gather_inputs",
        no_grad=True,
    )
    foreach_reduce_scatter_copy_in = _bind_to_collectives(
        _patched_foreach_reduce_scatter_copy_in,
        "foreach_reduce_scatter_copy_in",
        no_grad=False,
    )
    foreach_reduce = _bind_to_collectives(
        _patched_foreach_reduce,
        "foreach_reduce",
        no_grad=True,
    )

    _fsdp_init._init_param_group = _patched_init_param_group
    _fully_shard._init_param_group = _patched_init_param_group
    FSDPParamGroup.__init__ = _patched_param_group_init
    FSDPParam.init_dtype_attrs = _patched_init_dtype_attrs
    _fsdp_collectives._get_param_all_gather_inputs = get_param_all_gather_inputs
    _fsdp_collectives.foreach_reduce_scatter_copy_in = foreach_reduce_scatter_copy_in
    _fsdp_collectives.foreach_reduce = foreach_reduce
    _fsdp_param_group.foreach_reduce = foreach_reduce
    setattr(_fsdp_collectives, _PATCH_SENTINEL, True)


__all__ = [
    "apply_param_dtype_map_patch",
]
