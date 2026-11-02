"""Startup-time helpers: no sglang import in the light processes, the sglang CLI argument cache, the driver's
startup plan and the sharded checkpoint fill's key mapping."""

import argparse
import subprocess
import sys

import pytest


def test_engine_module_does_not_import_sglang():
    code = (
        "import sys; import miles.backends.sglang_diffusion_utils.sglang_diffusion_engine, miles.ray.rollout; "
        "print('sglang' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


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


def _plan_args(**overrides):
    from types import SimpleNamespace

    base = dict(
        startup_overlap="off",
        startup_keep_engines_resident=False,
        train_only=False,
        debug_rollout_only=False,
        num_rollout=5,
        colocate=True,
        offload_rollout=True,
    )
    return SimpleNamespace(**{**base, **overrides})


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, ("off", False)),
        ({"startup_keep_engines_resident": True}, ("off", True)),
        ({"startup_overlap": "spawn"}, ("spawn", False)),
        ({"startup_overlap": "init", "startup_keep_engines_resident": True}, ("init", True)),
        # init needs the engines resident when colocated, and --num-rollout
        ({"startup_overlap": "init"}, ("spawn", False)),
        ({"startup_overlap": "init", "startup_keep_engines_resident": True, "num_rollout": None}, ("spawn", True)),
        # nothing offloaded: nothing to hold back
        ({"startup_overlap": "init", "colocate": False, "offload_rollout": False}, ("init", False)),
        ({"startup_overlap": "spawn", "train_only": True}, ("off", False)),
        ({"startup_overlap": "init", "debug_rollout_only": True}, ("off", False)),
    ],
)
def test_resolve_startup_plan(overrides, expected):
    from miles.ray.placement_group import resolve_startup_plan

    assert resolve_startup_plan(_plan_args(**overrides)) == expected


def test_finish_rollout_manager_offloads_unless_resident(monkeypatch):
    from miles.ray import placement_group

    calls = []

    class Method:
        def __init__(self, name):
            self.name = name

        def remote(self):
            calls.append(self.name)
            return self.name

    class Manager:
        offload = Method("offload")
        get_num_rollout_per_epoch = Method("per_epoch")
        get_rollout_engines_and_lock = Method("ready")

    monkeypatch.setattr(placement_group.ray, "get", lambda ref: 7 if ref == "per_epoch" else ref)

    args = _plan_args(num_rollout=None, num_epoch=2)
    assert placement_group.finish_rollout_manager(args, Manager) == 7
    assert (args.num_rollout, calls) == (14, ["per_epoch", "offload"])

    calls.clear()
    args = _plan_args()
    assert placement_group.finish_rollout_manager(args, Manager, offload=False, wait=True) is None
    assert calls == ["ready"]


@pytest.mark.parametrize(
    "train_name, key",
    [
        ("blocks.0.attn.to_q.weight", "blocks.0.attn.to_q.weight"),
        ("base_model.model.blocks.0.attn.to_q.base_layer.weight", "blocks.0.attn.to_q.weight"),
        ("base_model.model.blocks.0.norm.weight", "blocks.0.norm.weight"),
        ("blocks.0.attn.to_q.base_layer.weight", "blocks.0.attn.to_q.weight"),
    ],
)
def test_checkpoint_key_strips_peft_wrapping(train_name, key):
    from miles.backends.fsdp_utils.sharded_checkpoint_fill import _checkpoint_key

    assert _checkpoint_key(train_name) == key


def test_engine_server_kwargs_need_no_sglang():
    code = (
        "import sys; from types import SimpleNamespace as N; "
        "from miles.backends.sglang_diffusion_utils.sglang_diffusion_engine import _base_server_kwargs, _sglang_attrs; "
        "a = N(hf_checkpoint='m', rollout_num_gpus_per_engine=2, sglang_tp_size=2, sglang_sp_degree=1, "
        "sglang_enable_cfg_parallel=False, sglang_dit_precision='bf16', diffusion_flow_shift=12.0, use_lora=False, "
        "train_only=False, lora_ipc_weight_sync=False, colocate=True); "
        "k = _base_server_kwargs(a, '[::1]', 15000, 15001); "
        "assert k['host'] == '[::1]' and k['master_port'] == 25001 and k['flow_shift'] == 12.0 and k['num_gpus'] == 2, k; "
        "assert _sglang_attrs(a) == {'tp_size': 2, 'sp_degree': 1, 'enable_cfg_parallel': False, 'dit_precision': 'bf16'}; "
        "print('sglang' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_rollout_weights_tag_matches_sglang():
    constants = pytest.importorskip("sglang.srt.constants")
    from miles.ray.rollout import GPU_MEMORY_TYPE_WEIGHTS

    assert GPU_MEMORY_TYPE_WEIGHTS == constants.GPU_MEMORY_TYPE_WEIGHTS


@pytest.mark.parametrize(
    "overrides, order",
    [
        ({}, ["start", "finish", "alloc", "init", "wire"]),
        ({"startup_overlap": "spawn"}, ["start", "alloc", "finish", "init", "wire"]),
        (
            {"startup_overlap": "init", "startup_keep_engines_resident": True},
            ["start", "alloc", "init", "finish", "wire"],
        ),
    ],
)
def test_start_rollout_and_train_orders_the_steps_by_mode(monkeypatch, overrides, order):
    from miles.ray import placement_group as pg

    events = []

    class Group:
        def async_init(self, args, role, with_ref=False):
            events.append("init")
            return ["ref"]

    monkeypatch.setattr(pg, "start_rollout_manager", lambda args, p: events.append("start") or "rm")
    monkeypatch.setattr(
        pg, "finish_rollout_manager", lambda args, rm, **kw: events.append("finish") or (kw.get("offload"), 3)[1]
    )
    monkeypatch.setattr(pg, "allocate_train_group", lambda **kw: events.append("alloc") or Group())
    monkeypatch.setattr(pg, "init_training_models", lambda args, group, refs, rm: events.append("wire") or group)

    args = _plan_args(
        actor_num_nodes=1, actor_num_gpus_per_node=2, **{k: v for k, v in overrides.items()}
    )
    rm, group, per_epoch, resident = pg.start_rollout_and_train(args, {"rollout": 0, "actor": 0})
    assert (rm, per_epoch) == ("rm", 3)
    assert resident == bool(overrides.get("startup_keep_engines_resident"))
    assert events == order
