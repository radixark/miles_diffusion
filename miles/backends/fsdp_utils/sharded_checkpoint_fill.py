"""Fill FSDP2 shards straight from the safetensors checkpoint, each rank reading only its slice.

The default initial load builds the full model on rank 0 and broadcasts every tensor
(set_model_state_dict(broadcast_from_rank0=True)): rank 0 copies the whole model from host to
device and the network carries all of it. For parameters that exist unchanged in the checkpoint,
each rank can instead read the byte range of its own dim-0 shard and copy it to the device.
Anything that does not map cleanly (adapters, renamed or converted tensors, other placements) is
left to the broadcast.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import struct
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)

_MAX_INFLIGHT_BYTES = 2 << 30

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def enabled() -> bool:
    return os.environ.get("MILES_SHARDED_CHECKPOINT_FILL", "1") == "1"


def resolve_component_files(checkpoint_path: str, component: str) -> list[str]:
    """The safetensors files diffusers would load for ``component``, or [] if unsure."""
    if os.path.isdir(checkpoint_path):
        root = checkpoint_path
    else:
        from huggingface_hub import snapshot_download

        root = snapshot_download(checkpoint_path, allow_patterns=[f"{component}/*"], local_files_only=True)
    comp_dir = os.path.join(root, component)
    indexes = glob.glob(os.path.join(comp_dir, "*.safetensors.index.json"))
    if len(indexes) == 1:
        with open(indexes[0]) as f:
            names = sorted(set(json.load(f)["weight_map"].values()))
        files = [os.path.join(comp_dir, n) for n in names]
        return files if all(os.path.exists(f) for f in files) else []
    if indexes:
        return []  # several variants: leave the choice to diffusers
    files = sorted(glob.glob(os.path.join(comp_dir, "*.safetensors")))
    return files if len(files) == 1 else []


def index_files(files: list[str]) -> dict[str, tuple[str, int, torch.dtype, list[int]]]:
    """key -> (file, absolute byte offset of the tensor, dtype, shape); keys seen twice are dropped."""
    index, dup = {}, set()
    for path in files:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        for key, meta in header.items():
            if key == "__metadata__" or meta["dtype"] not in _DTYPES:
                continue
            if key in index:
                dup.add(key)
            index[key] = (path, 8 + n + meta["data_offsets"][0], _DTYPES[meta["dtype"]], meta["shape"])
    for key in dup:
        index.pop(key, None)
    return index


def _checkpoint_key(name: str) -> str:
    return name.replace(".base_layer.", ".")


def _local_rows(param: DTensor) -> tuple[int, int] | None:
    """(first row, row count) of this rank's shard if it is a full-width dim-0 slice, else None."""
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    shape, offset = compute_local_shape_and_global_offset(param.shape, param.device_mesh, param.placements)
    if any(o != 0 for o in offset[1:]) or tuple(shape[1:]) != tuple(param.shape[1:]):
        return None
    return offset[0], shape[0]


def fill_from_checkpoint(model: torch.nn.Module, files: list[str], *, device) -> set[str]:
    """Copy every cleanly mapped parameter shard from ``files``; returns the state-dict keys filled."""
    index = index_files(files)
    plan = []
    for name, param in model.named_parameters():
        entry = index.get(_checkpoint_key(name))
        if entry is None or not isinstance(param, DTensor) or list(param.shape) != list(entry[3]):
            continue
        rows = _local_rows(param)
        if rows is None:
            continue
        path, base, dtype, shape = entry
        row_bytes = dtype.itemsize
        for dim in shape[1:]:
            row_bytes *= dim
        plan.append((name, param, path, base + rows[0] * row_bytes, rows[1] * row_bytes, dtype))

    fds = {}
    pool = ThreadPoolExecutor(8, thread_name_prefix="shard-fill")
    use_pinned = torch.cuda.is_available() and torch.device(device).type == "cuda"
    try:
        for path in {p[2] for p in plan}:
            fds[path] = os.open(path, os.O_RDONLY)

        def read(item):
            name, param, path, start, nbytes, dtype = item
            buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=use_pinned)
            view, got = memoryview(buf.numpy()), 0
            while got < nbytes:
                n = os.preadv(fds[path], [view[got:]], start + got)
                if n <= 0:
                    raise OSError(f"short read in {path}")
                got += n
            return buf

        filled = set()
        # Reads run ahead on the pool, bounded by bytes in flight; copies happen in order here.
        pending, inflight_bytes, todo = deque(), 0, iter(plan)

        def submit_more():
            nonlocal inflight_bytes
            while not pending or inflight_bytes < _MAX_INFLIGHT_BYTES:
                item = next(todo, None)
                if item is None:
                    return
                pending.append((item, pool.submit(read, item)))
                inflight_bytes += item[4]

        submit_more()
        while pending:
            item, future = pending.popleft()
            buf = future.result()
            inflight_bytes -= item[4]
            submit_more()
            name, param, _, _, _, dtype = item
            local = param.to_local()
            with torch.no_grad():
                src = buf.to(device, non_blocking=use_pinned).view(dtype).view(local.shape)
                local.copy_(src)
            filled.add(name)
        if use_pinned:
            torch.cuda.synchronize()
        logger.info(
            "Filled %d of %d parameters from the checkpoint files on every rank",
            len(filled),
            len(list(model.parameters())),
        )
        return filled
    finally:
        pool.shutdown()
        for fd in fds.values():
            os.close(fd)
