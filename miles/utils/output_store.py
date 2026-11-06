"""Clients of the run's output store: sglang-d engines put rollout tensors in Mooncake, Miles reads and deletes them."""

from __future__ import annotations

import json
from argparse import Namespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radix_transfer_plugins import OutputStore

BACKEND = "mooncake"


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
