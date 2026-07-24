"""Pinned-memory CPU offload for the FSDP train actor's sleep/wake_up cycle.

The default ``model.cpu()`` / ``model.cuda()`` offload path lands weights in
*pageable* host memory. Every wake_up then pays two costs: CUDA first stages the
pageable buffer into an internal page-locked bounce buffer, then DMAs it to the
device — a synchronous, per-cycle tax that grows with model size.

``PinnedCPUOffload`` keeps one reusable *pinned* (page-locked) host buffer per
tensor. Sleep copies device->pinned (reusing the buffer, so pinning is paid once
for the whole run); wake_up is a single async host->device DMA off the pinned
buffer.

Two FSDP2 details shape the implementation:

* The model is moved with ``Module._apply`` — the same path ``model.cpu()`` uses.
  Rebinding params by hand (``swap_tensors`` on ``model.parameters()``) leaves
  FSDP2's internal shard (``FSDPParam._sharded_param_data``, which shares storage
  with the param) resident on the GPU; ``_apply`` fixes those internals up so the
  device memory is genuinely freed.
* ``DTensor`` has no ``pin_memory`` and ``from_local`` drags the shard onto the
  *mesh's* device, so the offloaded copy is a plain pinned host buffer wrapped in
  a CPU-mesh DTensor (metadata only); wake rebuilds the CUDA-mesh DTensor.

Optimizer state is not held by FSDP internals, so it is moved directly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor


def _local(t: torch.Tensor) -> torch.Tensor:
    """Local shard of a DTensor, or the tensor itself for a plain tensor."""
    return t._local_tensor if isinstance(t, DTensor) else t


# An optimizer "cell": a stable key, a getter, and a setter for one state tensor.
Cell = tuple[object, Callable[[], "torch.Tensor | None"], Callable[[torch.Tensor], None]]


def optimizer_state_cells(optimizer: torch.optim.Optimizer) -> Iterator[Cell]:
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    yield (
                        (id(p), key),
                        (lambda p=p, key=key: optimizer.state[p].get(key)),
                        (lambda t, p=p, key=key: optimizer.state[p].__setitem__(key, t)),
                    )


class PinnedCPUOffload:
    """Reusable pinned host buffers for a sleep/wake_up offload cycle."""

    def __init__(self) -> None:
        self._cpu_meshes: dict[object, object] = {}  # cuda mesh id -> cpu twin
        self._cuda_of_cpu: dict[object, object] = {}  # cpu mesh id -> original cuda mesh
        self._model_bufs: list[torch.Tensor] = []  # ordered pinned buffers for _apply
        self._opt_store: dict[object, dict] = {}  # optimizer key -> buffer + metadata
        self._opt_offloaded: set = set()  # keys parked on the last sleep
        self._cursor = 0
        self._wake_device: torch.device | int | None = None

    def _cpu_mesh(self, cuda_mesh):
        key = id(cuda_mesh)
        mesh = self._cpu_meshes.get(key)
        if mesh is None:
            mesh = init_device_mesh("cpu", tuple(cuda_mesh.mesh.shape), mesh_dim_names=cuda_mesh.mesh_dim_names)
            self._cpu_meshes[key] = mesh
            # Remember the original CUDA mesh so wake rebuilds on the *same* mesh
            # object FSDP2 holds — recreating one would spawn new process groups.
            self._cuda_of_cpu[id(mesh)] = cuda_mesh
        return mesh

    def _cuda_mesh_of(self, cpu_dtensor: DTensor):
        return self._cuda_of_cpu[id(cpu_dtensor.device_mesh)]

    def _wrap_cpu(self, pinned: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        if isinstance(src, DTensor):
            return DTensor.from_local(
                pinned,
                self._cpu_mesh(src.device_mesh),
                src.placements,
                run_check=False,
                shape=src.shape,
                stride=src.stride(),
            )
        return pinned

    # -- model: moved through Module._apply so FSDP2 shard internals are freed ----

    def _pin_fn(self, t: torch.Tensor | None):
        if t is None or not _local(t).is_cuda:
            return t
        local_src = _local(t)
        i = self._cursor
        self._cursor += 1
        buf = self._model_bufs[i] if i < len(self._model_bufs) else None
        if buf is None or buf.shape != local_src.shape or buf.dtype != local_src.dtype:
            buf = torch.empty(local_src.shape, dtype=local_src.dtype, device="cpu", pin_memory=True)
            if i < len(self._model_bufs):
                self._model_bufs[i] = buf
            else:
                self._model_bufs.append(buf)
        # Blocking D2H: _apply frees the source right after this returns, so the
        # copy must finish first. The pinned destination keeps it near DMA speed.
        buf.copy_(local_src)
        return self._wrap_cpu(buf, t)

    def _unpin_fn(self, t: torch.Tensor | None):
        if t is None or _local(t).is_cuda:
            return t
        # Pinned source is retained (in _model_bufs), so the async H2D is safe;
        # the caller syncs after _apply returns.
        gpu = _local(t).to(self._wake_device, non_blocking=True)
        if isinstance(t, DTensor):
            gpu = DTensor.from_local(
                gpu,
                self._cuda_mesh_of(t),
                t.placements,
                run_check=False,
                shape=t.shape,
                stride=t.stride(),
            )
        return gpu

    @torch.no_grad()
    def offload_model(self, model: torch.nn.Module) -> None:
        self._cursor = 0
        model._apply(self._pin_fn)
        torch.cuda.synchronize()

    @torch.no_grad()
    def reload_model(self, model: torch.nn.Module, device: torch.device | int) -> None:
        self._wake_device = device
        model._apply(self._unpin_fn)
        torch.cuda.synchronize()

    # -- optimizer state: not held by FSDP internals, moved directly -------------

    @torch.no_grad()
    def offload_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        pending = []
        self._opt_offloaded = set()
        for key, get, set_ in optimizer_state_cells(optimizer):
            src = get()
            if src is None or not _local(src).is_cuda:
                continue  # e.g. AdamW's CPU `step` scalar — leave it where it is
            local_src = _local(src)
            entry = self._opt_store.get(key)
            if entry is None or entry["pinned"].shape != local_src.shape or entry["pinned"].dtype != local_src.dtype:
                entry = {"pinned": torch.empty(local_src.shape, dtype=local_src.dtype, device="cpu", pin_memory=True)}
                self._opt_store[key] = entry
            entry["pinned"].copy_(local_src, non_blocking=True)
            self._opt_offloaded.add(key)
            pending.append((set_, entry["pinned"], src))
        torch.cuda.synchronize()
        for set_, pinned, src in pending:
            set_(self._wrap_cpu(pinned, src))

    @torch.no_grad()
    def reload_optimizer(self, optimizer: torch.optim.Optimizer, device: torch.device | int) -> None:
        pending = []
        offloaded = getattr(self, "_opt_offloaded", set())
        for key, get, set_ in optimizer_state_cells(optimizer):
            if key not in offloaded:  # only restore what we parked; don't move stray CPU state
                continue
            src = get()
            if src is None or _local(src).is_cuda:
                continue
            gpu = _local(src).to(device, non_blocking=True)
            pending.append((set_, gpu, src))
        torch.cuda.synchronize()
        for set_, gpu, src in pending:
            if isinstance(src, DTensor):
                gpu = DTensor.from_local(
                    gpu,
                    self._cuda_mesh_of(src),
                    src.placements,
                    run_check=False,
                    shape=src.shape,
                    stride=src.stride(),
                )
            set_(gpu)

    # -- top-level entry points --------------------------------------------------

    @torch.no_grad()
    def sleep(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.offload_model(model)
        self.offload_optimizer(optimizer)

    @torch.no_grad()
    def wake_up(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device | int) -> None:
        self.reload_model(model, device)
        self.reload_optimizer(optimizer, device)
