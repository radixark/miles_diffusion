from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import multiprocessing
import os
import signal
import time
from pathlib import Path

import psutil

from miles.utils import process_utils


def _bind_then_exit_zero(expected_parent_pid):
    process_utils.bind_lifetime_to_parent(expected_parent_pid)
    os._exit(0)


def test_bind_lifetime_suicides_if_parent_already_gone():
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_bind_then_exit_zero, args=(999_999_999,))
    p.start()
    p.join(timeout=60)
    assert p.exitcode == -signal.SIGKILL


def test_bind_lifetime_lets_child_live_while_parent_alive():
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_bind_then_exit_zero, args=(os.getpid(),))
    p.start()
    p.join(timeout=60)
    assert p.exitcode == 0


def _sleep_forever():
    time.sleep(300)


def _server_with_unprotected_worker(parent_pid, pid_file):
    process_utils.bind_lifetime_to_parent(parent_pid)
    ctx = multiprocessing.get_context("spawn")
    worker = ctx.Process(target=_sleep_forever, daemon=True)
    worker.start()
    Path(pid_file).write_text(str(worker.pid))
    time.sleep(300)


def _gone(pid) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_sigterm_reaps_unprotected_workers(tmp_path):
    pid_file = tmp_path / "worker_pid"
    ctx = multiprocessing.get_context("spawn")
    server = ctx.Process(target=_server_with_unprotected_worker, args=(os.getpid(), str(pid_file)))
    server.start()
    try:
        deadline = time.time() + 60
        while not pid_file.exists() and time.time() < deadline:
            time.sleep(0.5)
        assert pid_file.exists(), "server never spawned its worker"
        worker_pid = int(pid_file.read_text())

        os.kill(server.pid, signal.SIGTERM)
        server.join(timeout=30)
        assert server.exitcode is not None, "server did not exit on SIGTERM"

        deadline = time.time() + 15
        while not _gone(worker_pid) and time.time() < deadline:
            time.sleep(0.5)
        assert _gone(worker_pid), "unprotected worker survived server SIGTERM"
    finally:
        if server.is_alive():
            server.kill()
