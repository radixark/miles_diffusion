"""--use-output-store: its flags, their validation, and the config each client gets."""

import json
import shlex
import sys
import types
from argparse import Namespace

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from miles.backends.fsdp_utils.arguments import load_fsdp_args
from miles.utils.arguments import get_miles_extra_args_provider, validate_output_store_args
from miles.utils.output_store import mooncake_client, output_store_backend_config, sglang_output_store_server_args


def _args(**overrides) -> Namespace:
    args = Namespace(
        use_output_store=True,
        output_store_mooncake_master="10.0.0.1:50051",
        output_store_local_buffer_size="2gb",
        output_store_global_segment_size="16gb",
        output_store_protocol="rdma",
        output_store_device_name="",
        save_debug_rollout_data=None,
        load_debug_rollout_data=None,
        run_uuid="abc",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_flags_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        shlex.split(
            "test --hf-checkpoint ckpt --rollout-batch-size 1 --global-batch-size 1 --num-rollout 1 "
            "--actor-num-gpus-per-node 1 --micro-batch-size 1 --prompt-data prompts.jsonl "
            "--use-output-store --output-store-mooncake-master 10.0.0.1:50051 --output-store-protocol tcp"
        ),
    )
    args = load_fsdp_args(extra_args_provider=get_miles_extra_args_provider())
    assert args.use_output_store
    assert args.output_store_mooncake_master == "10.0.0.1:50051"
    assert args.output_store_local_buffer_size == "2gb"
    assert args.output_store_global_segment_size == "16gb"
    assert args.output_store_protocol == "tcp"
    assert args.output_store_device_name == ""


def test_switch_off_passes_and_rejects_a_master() -> None:
    validate_output_store_args(_args(use_output_store=False, output_store_mooncake_master=None))
    with pytest.raises(ValueError, match="requires --use-output-store"):
        validate_output_store_args(_args(use_output_store=False))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"output_store_mooncake_master": None}, "requires --output-store-mooncake-master"),
        ({"save_debug_rollout_data": "x/{rollout_id}.pt"}, "--save-debug-rollout-data"),
        ({"load_debug_rollout_data": "x/{rollout_id}.pt"}, "--load-debug-rollout-data"),
        ({"output_store_global_segment_size": "lots"}, "not a size"),
    ],
)
def test_switch_on_rejects(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_output_store_args(_args(**overrides))


def test_engine_server_args_name_the_run_and_the_engine_host() -> None:
    server_args = sglang_output_store_server_args(_args(), local_hostname="10.0.0.7")
    assert server_args["output_store_backend"] == "mooncake"
    assert json.loads(server_args["output_store_backend_extra_config"]) == {
        "master_server_address": "10.0.0.1:50051",
        "local_hostname": "10.0.0.7",
        "local_buffer_size": "2gb",
        "key_prefix": "miles-diffusion-output-store/abc",
        "protocol": "rdma",
        "device_name": "",
    }


def test_a_reader_registers_no_local_buffer() -> None:
    config = output_store_backend_config(_args(), local_hostname="10.0.0.7", puts=False)
    assert config["local_buffer_size"] == 0


def test_a_segment_client_lends_the_store_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStore:
        def setup(self, config):
            self.config = config
            return 0

    mooncake = types.ModuleType("mooncake")
    mooncake.store = types.ModuleType("mooncake.store")
    mooncake.store.MooncakeDistributedStore = FakeStore
    monkeypatch.setitem(sys.modules, "mooncake", mooncake)
    monkeypatch.setitem(sys.modules, "mooncake.store", mooncake.store)
    import ray

    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "10.0.0.7")
    store = mooncake_client(_args(), global_segment_size="16gb", local_buffer_size=0)
    assert store.config == {
        "local_hostname": "10.0.0.7",
        "metadata_server": "P2PHANDSHAKE",
        "global_segment_size": 16 * 1024**3,
        "local_buffer_size": 0,
        "protocol": "rdma",
        "rdma_devices": "",
        "master_server_addr": "10.0.0.1:50051",
    }
