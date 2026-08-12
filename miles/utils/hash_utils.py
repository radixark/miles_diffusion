"""Hashes that hold still across processes and runs."""

from __future__ import annotations

import hashlib


def stable_hash(*parts: object) -> int:
    """A 63-bit hash of ``parts``, identical in every process and every run.

    ``hash()`` is not: PYTHONHASHSEED randomises it per process, so anything derived from
    it -- an RNG seed, a shard assignment, a cache key compared across ranks -- silently
    stops agreeing. Callers name the parts that make a value distinct and get the same
    integer back forever.
    """
    payload = ":".join(str(part) for part in parts).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & (2**63 - 1)
