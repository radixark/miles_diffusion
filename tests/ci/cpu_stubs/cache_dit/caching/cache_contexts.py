"""Stub cache_dit.caching.cache_contexts — see cache_dit/__init__.py for rationale."""

from unittest.mock import MagicMock


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return MagicMock(name=f"cache_dit.caching.cache_contexts.{name}")
