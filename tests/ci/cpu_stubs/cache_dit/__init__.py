"""Stub cache_dit for CPU-only CI.

sglang.multimodal_gen.__init__ eagerly imports DiffGenerator, which loads
cache_dit_integration and does `import cache_dit` at module load. The real
cache-dit package is not in the CPU runner's --no-deps install. Any attribute
access on this stub returns a MagicMock so collection succeeds; tests that
actually *call* cache-dit will fail loudly, which is correct.
"""

from unittest.mock import MagicMock


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return MagicMock(name=f"cache_dit.{name}")
