"""Startup-time helpers: no sglang import in the light processes, train rank creation, scheduler prespawn,
and the sglang CLI argument cache."""

import argparse
import subprocess
import sys

import pytest


def test_picklable_args_drops_only_unpicklable_values():
    import argparse
    import pickle

    from miles.backends.sglang_diffusion_utils.sglang_diffusion_engine import _picklable_args

    args = argparse.Namespace(hf_checkpoint="m", sglang_tp_size=2, hook=lambda: None)
    copy = _picklable_args(args)
    assert vars(copy) == {"hf_checkpoint": "m", "sglang_tp_size": 2}
    pickle.dumps(copy)


def test_engine_module_does_not_import_sglang():
    code = (
        "import sys; import miles.backends.sglang_diffusion_utils.sglang_diffusion_engine, miles.ray.rollout; "
        "print('sglang' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_train_ranks_are_created_together_then_get_rank0_address(monkeypatch):
    import argparse
    from types import SimpleNamespace

    from miles.ray import actor_group

    events = []

    class FakeActor:
        def __init__(self, rank):
            self.rank = rank
            self.get_master_addr_and_port = SimpleNamespace(
                remote=lambda: events.append(("get_addr", self.rank)) or ("10.0.0.1", 29500)
            )
            self.set_master_addr_and_port = SimpleNamespace(
                remote=lambda addr, port: events.append(("set_addr", self.rank, addr, port))
            )

    class FakeRemoteClass:
        def options(self, **kwargs):
            return SimpleNamespace(
                remote=lambda world_size, rank, addr, port: events.append(("create", rank, addr)) or FakeActor(rank)
            )

    monkeypatch.setattr(actor_group.ray, "remote", lambda **kwargs: (lambda cls: FakeRemoteClass()))
    monkeypatch.setattr(actor_group.ray, "get", lambda x: x)
    monkeypatch.setattr(actor_group, "PlacementGroupSchedulingStrategy", lambda **kwargs: None)
    args = argparse.Namespace(train_env_vars={}, colocate=False, deterministic_mode=False, train_backend="fsdp")

    actor_group.RayTrainGroup(args, num_nodes=1, num_gpus_per_node=3, pg=(object(), [0, 1, 2], [0, 1, 2]))

    assert events == [
        ("create", 0, None),
        ("create", 1, None),
        ("create", 2, None),
        ("get_addr", 0),
        ("set_addr", 1, "10.0.0.1", 29500),
        ("set_addr", 2, "10.0.0.1", 29500),
    ]


def test_prespawned_workers_get_args_pipes_and_env_at_start(monkeypatch):
    import multiprocessing as mp

    from miles.backends.sglang_diffusion_utils.scheduler_prespawn import prespawn

    workers = prespawn(2, target="tests.fast.utils.prespawn_fake_target:run")
    monkeypatch.setenv("PRESPAWN_TEST_ENV", "set-after-prespawn")
    readers, procs = [], []
    for rank, worker in enumerate(workers):
        reader, writer = mp.Pipe(duplex=False)
        proc = worker(args=(rank, writer, {"rank": rank}), name=f"w{rank}")
        proc.start()
        writer.close()
        readers.append(reader)
        procs.append(proc)
    got = sorted(reader.recv() for reader in readers)
    for proc in procs:
        proc.join(30)
    assert got == [(0, {"rank": 0}, "set-after-prespawn"), (1, {"rank": 1}, "set-after-prespawn")]
    assert [proc.exitcode for proc in procs] == [0, 0]


def test_cancelled_prespawned_worker_exits_cleanly():
    from miles.backends.sglang_diffusion_utils.scheduler_prespawn import prespawn

    (worker,) = prespawn(1, target="tests.fast.utils.prespawn_fake_target:run")
    worker.cancel()
    worker._process.join(30)
    assert worker._process.exitcode == 0


def _register(parser):
    from miles.backends.sglang_diffusion_utils.cli_cache import StoreBoolean, parse_int_list

    parser.add_argument("--sglang-tp-size", type=int, default=1)
    parser.add_argument("--sglang-name", type=str, default=None, help="h")
    parser.add_argument("--sglang-precisions", type=str, nargs="+", default=("fp32",))
    parser.add_argument("--sglang-steps", type=parse_int_list, default=[])
    parser.add_argument("--sglang-mode", type=str, choices=["a", "b"], default="a")
    parser.add_argument("--sglang-flag", action=StoreBoolean, default=False)
    parser.add_argument("--sglang-on", action="store_true")


def _parse(argv, monkeypatch, tmp_path, register):
    from miles.backends.sglang_diffusion_utils import cli_cache

    monkeypatch.setattr(cli_cache, "_cache_path", lambda: str(tmp_path / "args.json"))
    parser = argparse.ArgumentParser()
    cli_cache.add_cached(parser, register)
    return vars(parser.parse_args(argv))


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--sglang-tp-size", "2", "--sglang-steps", "1, 2,3", "--sglang-flag", "--sglang-mode", "b"],
        ["--sglang-precisions", "bf16", "fp16", "--sglang-flag", "false", "--sglang-on"],
    ],
)
def test_cli_cache_hit_parses_like_the_live_registration(argv, monkeypatch, tmp_path):
    live = _parse(argv, monkeypatch, tmp_path, _register)  # miss: registers and writes the cache
    assert (tmp_path / "args.json").exists()

    def must_not_register(parser):
        raise AssertionError("cache hit should not register")

    cached = _parse(argv, monkeypatch, tmp_path, must_not_register)
    assert cached == live
    assert type(cached["sglang_precisions"]) is type(live["sglang_precisions"])


def test_cli_cache_falls_back_when_the_cache_is_broken(monkeypatch, tmp_path):
    (tmp_path / "args.json").write_text('[{"kind": "bogus"}]')
    assert _parse(["--sglang-tp-size", "4"], monkeypatch, tmp_path, _register)["sglang_tp_size"] == 4


def test_cli_cache_is_skipped_for_unknown_types(monkeypatch, tmp_path):
    def register(parser):
        parser.add_argument("--sglang-x", type=lambda v: v.upper(), default="a")

    assert _parse(["--sglang-x", "q"], monkeypatch, tmp_path, register)["sglang_x"] == "Q"
    assert not (tmp_path / "args.json").exists()
