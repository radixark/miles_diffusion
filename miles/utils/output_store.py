"""Clients of the run's output store: sglang-d engines put rollout tensors in Mooncake, Miles reads and deletes them."""

from __future__ import annotations

import json
from argparse import Namespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radix_transfer_plugins import OutputStore

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
