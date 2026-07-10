import ctypes
import logging
import os
import signal
import sys

import psutil

logger = logging.getLogger(__name__)


def kill_descendants_and_exit(signum=None, frame=None) -> None:
    try:
        children = psutil.Process(os.getpid()).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    os._exit(1)


def bind_lifetime_to_parent(expected_parent_pid: int) -> None:
    """Reap our own subtree and exit when the parent dies; the ppid check covers the pre-arm window."""
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)
    signal.signal(signal.SIGTERM, kill_descendants_and_exit)
    if sys.platform == "linux":
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
