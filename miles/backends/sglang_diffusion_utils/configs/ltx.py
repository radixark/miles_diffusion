"""LTX-2 sglang-d rollout engine config (re-exports model family helpers)."""

from __future__ import annotations

from miles.backends.model_families.ltx import (
    LTX_DEFAULT_HF_MODEL,
    LTX_DEFAULT_MODEL_ID,
    ensure_materialized_model,
    is_ltx_model,
    resolve_hf_model_id,
    resolve_materialized_model_dir,
    resolve_model_id,
    resolve_transformer_checkpoint,
    server_kwargs_extras,
)

__all__ = [
    "LTX_DEFAULT_HF_MODEL",
    "LTX_DEFAULT_MODEL_ID",
    "ensure_materialized_model",
    "is_ltx_model",
    "resolve_hf_model_id",
    "resolve_materialized_model_dir",
    "resolve_model_id",
    "resolve_transformer_checkpoint",
    "resolve_sglang_model_path",
    "resolve_ltx_model_id",
    "resolve_ltx_transformer_weights_path",
    "server_kwargs_extras",
]


def resolve_sglang_model_path(args) -> str:
    return resolve_hf_model_id(args)


def resolve_ltx_model_id(args) -> str:
    return resolve_model_id(args)


def resolve_ltx_transformer_weights_path(
    diffusion_model: str | None,
    *,
    explicit_path: str | None = None,
) -> str | None:
    try:
        return resolve_transformer_checkpoint(
            diffusion_model, explicit_path=explicit_path,
        )
    except FileNotFoundError:
        return None
