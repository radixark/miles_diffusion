from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="stage-b-2-gpu-h200",
    labels=[],
)

import os
from types import SimpleNamespace

import requests

import miles.backends.sglang_diffusion_utils.sglang_diffusion_engine as engine_mod
from miles.backends.sglang_diffusion_utils.sglang_diffusion_engine import SGLangDiffusionEngine


def test_launch_target_binds_to_parent_before_launch(monkeypatch):
    calls = []
    monkeypatch.setattr(engine_mod, "bind_lifetime_to_parent", lambda pid: calls.append(("bind", pid)))
    monkeypatch.setenv(engine_mod.SGLD_SERVER_PID_ENV, "sentinel")

    import sglang.multimodal_gen.runtime.launch_server as ls_mod

    # Register the original so the run_scheduler_process rebinding is restored.
    monkeypatch.setattr(ls_mod, "run_scheduler_process", ls_mod.run_scheduler_process)
    monkeypatch.setattr(ls_mod, "launch_server", lambda server_args: calls.append(("launch", None)))

    engine_mod._launch_server_target(SimpleNamespace(attention_backend_config=None), 4242)

    assert calls == [("bind", 4242), ("launch", None)]
    assert ls_mod.run_scheduler_process is engine_mod._scheduler_process_entrypoint
    assert os.environ[engine_mod.SGLD_SERVER_PID_ENV] == str(os.getpid())


def test_scheduler_entrypoint_binds_to_server_before_running(monkeypatch):
    calls = []
    monkeypatch.setattr(engine_mod, "bind_lifetime_to_parent", lambda pid: calls.append(("bind", pid)))
    monkeypatch.setenv(engine_mod.SGLD_SERVER_PID_ENV, "12345")

    from miles.backends.sglang_diffusion_utils.monkey_patches import ROLLOUT_PATCH_GROUPS_ENV

    monkeypatch.delenv(ROLLOUT_PATCH_GROUPS_ENV, raising=False)

    import sglang.multimodal_gen.runtime.managers.gpu_worker as gw_mod

    monkeypatch.setattr(gw_mod, "run_scheduler_process", lambda *a, **k: calls.append(("run", None)))

    engine_mod._scheduler_process_entrypoint()
    assert calls == [("bind", 12345), ("run", None)]


def test_shutdown_kills_tree_even_if_router_removal_fails(monkeypatch):
    killed = []
    monkeypatch.setattr(engine_mod, "kill_process_tree", killed.append)

    def _post_raises(*args, **kwargs):
        raise requests.ConnectionError("router already gone")

    monkeypatch.setattr(engine_mod.requests, "post", _post_raises)

    engine = SGLangDiffusionEngine(args=SimpleNamespace(use_miles_router=True), rank=0)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 1234
    engine.router_ip = "127.0.0.1"
    engine.router_port = 5678
    engine.process = SimpleNamespace(pid=4242)

    engine.shutdown()
    assert killed == [4242]
