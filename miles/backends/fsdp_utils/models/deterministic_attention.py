"""Generic deterministic flash-attention patching for native model packages."""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)

# Each entry: (label, holder object, attribute name on holder).
FlashAttentionEntrypoint = tuple[str, Any, str]


def patch_flash_entrypoints_deterministic(
    backend: str,
    entrypoints: Iterable[FlashAttentionEntrypoint],
    *,
    required_label: str | None = None,
    package_name: str = "model package",
) -> None:
    """Wrap flash kernel callables with ``deterministic=True`` when supported."""
    patched: list[str] = []
    for label, holder, attr in entrypoints:
        fn = getattr(holder, attr, None)
        if fn is None:
            continue
        if "deterministic" not in inspect.signature(fn).parameters:
            continue
        setattr(holder, attr, functools.partial(fn, deterministic=True))
        patched.append(label)

    if required_label is not None and required_label not in patched:
        raise RuntimeError(
            f"deterministic_mode: {package_name} backend {backend!r} maps to {required_label}, "
            f"but its kernel is unavailable or exposes no deterministic argument "
            f"(patched: {patched or None}). Use a native/math attention backend for a "
            f"deterministic backward."
        )
    if patched:
        logger.info(
            "Enabled deterministic flash attention backward for %s: %s",
            package_name,
            ", ".join(patched),
        )


def resolve_required_flash_kernel_label(
    modeling: Any,
    backend: str,
) -> str | None:
    resolver = getattr(modeling, "required_flash_kernel_label", None)
    if resolver is None:
        return None
    return resolver(backend)


def patch_modeling_flash_attention_deterministic(modeling: Any, backend: str) -> None:
    """Patch flash kernels declared by ``modeling.flash_attention_entrypoints``."""
    get_entrypoints: Callable[[str], Iterable[FlashAttentionEntrypoint]] | None = getattr(
        modeling, "flash_attention_entrypoints", None
    )
    if get_entrypoints is None:
        raise NotImplementedError(
            f"{modeling.__name__} exposes no flash_attention_entrypoints; "
            f"use a native/math attention backend under deterministic mode."
        )
    patch_flash_entrypoints_deterministic(
        backend,
        get_entrypoints(backend),
        required_label=resolve_required_flash_kernel_label(modeling, backend),
        package_name=getattr(modeling, "__name__", "modeling"),
    )
