"""Start sglang-d scheduler workers before the server process imports sglang.

A scheduler worker spends most of its start-up importing torch, transformers, diffusers and sglang,
and the server process imports the same stack before it starts the workers. Starting the workers
first lets both imports run in parallel. Each worker imports its target, then waits for the
arguments the server would have spawned it with.

Keep this module stdlib only: it runs before anything heavy is imported.
"""

import importlib
import multiprocessing as mp
import os

DEFAULT_TARGET = "sglang.multimodal_gen.runtime.managers.gpu_worker:run_scheduler_process"


def _worker_main(conn, target: str) -> None:
    module_name, func_name = target.split(":")
    func = getattr(importlib.import_module(module_name), func_name)
    msg = conn.recv()
    conn.close()
    if msg is None:
        return
    env, args = msg
    # Match the environment the server process had when it would have spawned this worker.
    os.environ.clear()
    os.environ.update(env)
    func(*args)


class PrespawnedProcess:
    """Stands in for ``mp.Process(target=..., args=args)``: ``start()`` hands the args to a waiting worker."""

    def __init__(self, process, conn, args, name):
        self._process = process
        self._conn = conn
        self._args = args
        self.name = name

    def start(self) -> None:
        self._conn.send((dict(os.environ), self._args))
        self._conn.close()

    def __getattr__(self, attr):
        return getattr(self._process, attr)


class WaitingWorker:
    """A started worker waiting for its args; call it like ``factory(args=..., name=...)`` or cancel it."""

    def __init__(self, process, conn):
        self._process = process
        self._conn = conn

    def __call__(self, args, name):
        return PrespawnedProcess(self._process, self._conn, args, name)

    def cancel(self) -> None:
        self._conn.send(None)
        self._conn.close()


def prespawn(num_workers: int, target: str = DEFAULT_TARGET) -> list[WaitingWorker]:
    ctx = mp.get_context("spawn")
    workers = []
    for _ in range(num_workers):
        reader, writer = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_worker_main, args=(reader, target), daemon=True)
        process.start()
        reader.close()
        workers.append(WaitingWorker(process, writer))
    return workers
