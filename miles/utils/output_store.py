"""Clients of the run's output store: sglang-d engines put rollout tensors in Mooncake, Miles reads and deletes them."""

from __future__ import annotations

import dataclasses
import json
from argparse import Namespace
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from miles.utils.types import Sample

if TYPE_CHECKING:
    from radix_transfer_plugins import OutputStore, StoreRef

BACKEND = "mooncake"
_SIZE_UNITS = {"kb": 1024, "mb": 1024**2, "gb": 1024**3, "k": 1024, "m": 1024**2, "g": 1024**3}


def parse_size(value: int | str) -> int:
    if isinstance(value, int):
        return value
    text = value.strip().lower()
    for suffix, multiplier in _SIZE_UNITS.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * multiplier)
    return int(text)


def key_prefix(args: Namespace) -> str:
    return f"miles-diffusion-output-store/{args.run_uuid}"


def output_store_backend_config(args: Namespace, *, local_hostname: str, puts: bool) -> dict[str, Any]:
    """The radix_transfer_plugins MooncakeConfig of one client; only a client that puts registers a local buffer."""
    return {
        "master_server_address": args.output_store_mooncake_master,
        "local_hostname": local_hostname,
        "local_buffer_size": args.output_store_local_buffer_size if puts else 0,
        "key_prefix": key_prefix(args),
        "protocol": args.output_store_protocol,
        "device_name": args.output_store_device_name,
    }


def sglang_output_store_server_args(args: Namespace, *, local_hostname: str) -> dict[str, str]:
    config = output_store_backend_config(args, local_hostname=local_hostname, puts=True)
    return {"output_store_backend": BACKEND, "output_store_backend_extra_config": json.dumps(config)}


def connect_output_store(args: Namespace) -> OutputStore:
    """A client that reads and deletes from this process; radix_transfer_plugins ships with sglang."""
    import ray
    from radix_transfer_plugins import OutputStore, create_backend

    config = output_store_backend_config(args, local_hostname=ray.util.get_node_ip_address(), puts=False)
    return OutputStore(backend=create_backend(name=BACKEND, extra_config=json.dumps(config)))


def mooncake_client(args: Namespace, *, global_segment_size: int | str, local_buffer_size: int | str) -> Any:
    """A raw Mooncake client of the run's store; a positive ``global_segment_size`` lends the store that memory."""
    import ray
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    error = store.setup(
        {
            "local_hostname": ray.util.get_node_ip_address(),
            "metadata_server": "P2PHANDSHAKE",
            "global_segment_size": parse_size(global_segment_size),
            "local_buffer_size": parse_size(local_buffer_size),
            "protocol": args.output_store_protocol,
            "rdma_devices": args.output_store_device_name,
            "master_server_addr": args.output_store_mooncake_master,
        }
    )
    if error:
        raise RuntimeError(f"Mooncake store setup failed: {error}")
    return store


def sweep_output_store(args: Namespace) -> int:
    """Remove every object of this run still in the store; only for the end of the run."""
    store = mooncake_client(args, global_segment_size=0, local_buffer_size=0)
    try:
        removed = store.remove_by_regex(f"^{key_prefix(args)}/", True)
    finally:
        store.close()
    if removed < 0:
        raise RuntimeError(f"Mooncake remove_by_regex failed: {removed}")
    return removed


@dataclass(frozen=True)
class StoredTensor:
    """One stored field; the trainer reads it back into its tensor."""

    ref: StoreRef
    field: str


def store_refs_of(samples: Iterable[Any]) -> dict[str, StoreRef]:
    """Every output store ref the samples hold, by key; groups may nest samples in lists."""
    refs = {}
    for item in samples:
        if isinstance(item, Sample):
            if item.rollout_store_ref is not None:
                refs[item.rollout_store_ref.key] = item.rollout_store_ref
        else:
            refs.update(store_refs_of(item))
    return refs


_dropped_refs: dict[str, StoreRef] = {}


def record_dropped_samples(samples: Iterable[Any]) -> None:
    """Hand the refs of samples that will never train to the rollout manager, which deletes them with the rollout."""
    _dropped_refs.update(store_refs_of(samples))


def take_dropped_refs() -> dict[str, StoreRef]:
    refs = dict(_dropped_refs)
    _dropped_refs.clear()
    return refs


def materialize_stored_tensors(train_pairs: list[dict[str, Any]], *, store: OutputStore) -> None:
    """Replace every StoredTensor in the pairs with its tensor; each object is read once."""
    from radix_transfer_plugins.stashes.tensor_stash import TensorStash

    wanted: dict[str, tuple[StoreRef, set[str]]] = {}
    for stored in _stored_tensors(train_pairs):
        wanted.setdefault(stored.ref.key, (stored.ref, set()))[1].add(stored.field)
    if not wanted:
        return
    requests = [(ref, sorted(fields)) for ref, fields in wanted.values()]
    stashes = store.read_many(requests, stash_type=TensorStash)
    tensors = {
        (ref.key, name): tensor
        for (ref, _), stash in zip(requests, stashes, strict=True)
        for name, tensor in stash.tensors.items()
    }
    _map_stored_tensors(train_pairs, lambda stored: tensors[(stored.ref.key, stored.field)])


def _stored_tensors(obj: Any) -> Iterator[StoredTensor]:
    if isinstance(obj, StoredTensor):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _stored_tensors(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _stored_tensors(value)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for value in vars(obj).values():
            yield from _stored_tensors(value)


def _map_stored_tensors(obj: Any, fn: Callable[[StoredTensor], Any]) -> Any:
    """Replace the stored tensors under ``obj`` in place; dicts, lists and dataclasses are shared by the pairs of a sample."""
    if isinstance(obj, StoredTensor):
        return fn(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _map_stored_tensors(value, fn)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = _map_stored_tensors(value, fn)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = vars(obj)
        for name, value in fields.items():
            fields[name] = _map_stored_tensors(value, fn)
    return obj
