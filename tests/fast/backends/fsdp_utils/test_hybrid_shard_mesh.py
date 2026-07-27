"""Hybrid-shard argument and mesh validation."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="stage-a-cpu", labels=["fsdp"])

import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from miles.backends.fsdp_utils.arguments import validate_hybrid_shard_args

_WORKER = Path(__file__).with_name("_hybrid_shard_mesh_worker.py")


def hybrid_shard_args(world_size, dp_replicate_size, sp_size):
    return Namespace(
        actor_num_gpus_per_node=world_size,
        actor_num_nodes=1,
        sequence_parallel_size=sp_size,
        ulysses_degree=0,
        dp_replicate_size=dp_replicate_size,
    )


@pytest.mark.parametrize(
    "world_size, dp_replicate_size, sp_size",
    [
        (8, 3, 1),  # world not divisible by the replica count
        (8, 4, 4),  # each factor divides the world but their product does not
        (8, 0, 1),
        (8, -1, 1),
    ],
)
def test_validate_hybrid_shard_args_rejects(world_size, dp_replicate_size, sp_size):
    with pytest.raises(ValueError):
        validate_hybrid_shard_args(hybrid_shard_args(world_size, dp_replicate_size, sp_size))


@pytest.mark.parametrize(
    "world_size, dp_replicate_size, sp_size",
    [
        (8, 1, 1),  # default: one copy over the whole world (flat FSDP)
        (8, 1, 2),  # sp eats into the shard axis, not the replicate one
        (16, 2, 2),
        (16, 4, 4),  # every rank holds a full copy
    ],
)
def test_validate_hybrid_shard_args_accepts(world_size, dp_replicate_size, sp_size):
    validate_hybrid_shard_args(hybrid_shard_args(world_size, dp_replicate_size, sp_size))


def test_mesh_axes():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=4",
            str(_WORKER),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
