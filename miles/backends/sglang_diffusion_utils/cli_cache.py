"""Cache of the sglang-d server CLI arguments, so parsing miles args does not import sglang.

Registering ServerArgs' CLI arguments imports sglang.multimodal_gen, which pulls in torch,
transformers and diffusers (~25 s). The resulting argparse actions are plain data, so they are
recorded once per sglang build and replayed from a JSON file afterwards. Anything that does not
round-trip through JSON disables the cache and falls back to the import.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os

logger = logging.getLogger(__name__)

_CACHE_VERSION = 4


def parse_int_list(value: str) -> list[int]:
    """Same as sglang.multimodal_gen.configs.pipeline_configs.base.parse_int_list."""
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",")]


# argparse ``type`` callables the cache can restore without importing sglang, by name.
_TYPES = {None: None, "int": int, "str": str, "float": float, "parse_int_list": parse_int_list}


class StoreBoolean(argparse.Action):
    """Same behavior as sglang.multimodal_gen.runtime.utils.argparse.StoreBoolean."""

    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs="?",
            const=True,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, True)
        elif isinstance(values, str):
            if values.lower() == "true":
                setattr(namespace, self.dest, True)
            elif values.lower() == "false":
                setattr(namespace, self.dest, False)
            else:
                raise ValueError(f"Invalid boolean value: {values}. Expected 'true' or 'false'.")
        else:
            setattr(namespace, self.dest, bool(values))


_ACTIONS = {
    argparse._StoreAction: "store",
    argparse._StoreTrueAction: "store_true",
    argparse._StoreFalseAction: "store_false",
    argparse._StoreConstAction: "store_const",
    argparse._AppendAction: "append",
}


def _cache_path() -> str | None:
    spec = importlib.util.find_spec("sglang")  # top-level only: does not run sglang/__init__
    if spec is None or not spec.submodule_search_locations:
        return None
    root = os.path.join(list(spec.submodule_search_locations)[0], "multimodal_gen")
    if not os.path.isdir(root):
        return None
    # ServerArgs.add_cli_args pulls definitions from several modules (pipeline configs, ...), so key
    # on every source file of multimodal_gen rather than server_args.py alone.
    digest = hashlib.sha1(f"{_CACHE_VERSION}:{root}".encode())
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                st = os.stat(os.path.join(dirpath, name))
                digest.update(f"{dirpath}/{name}:{st.st_size}:{st.st_mtime_ns};".encode())
    cache_dir = os.environ.get("MILES_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "miles"))
    return os.path.join(cache_dir, f"sglang_cli_args_{digest.hexdigest()[:16]}.json")


def _enc(value):
    # JSON has no tuple; keep tuples (e.g. default=("fp32",)) distinguishable from lists.
    if isinstance(value, tuple):
        return {"__tuple__": [_enc(v) for v in value]}
    if isinstance(value, list):
        return [_enc(v) for v in value]
    return value


def _dec(value):
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_dec(v) for v in value["__tuple__"])
    if isinstance(value, list):
        return [_dec(v) for v in value]
    return value


def _encode(action: argparse.Action) -> dict:
    kind = "store_boolean" if type(action).__name__ == "StoreBoolean" else _ACTIONS.get(type(action))
    if kind is None:
        raise TypeError(f"unsupported action {type(action)!r}")
    type_name = None if action.type is None else getattr(action.type, "__name__", None)
    if type_name not in _TYPES:
        raise TypeError(f"unsupported type {action.type!r}")
    entry = {
        "kind": kind,
        "option_strings": action.option_strings,
        "dest": action.dest,
        "nargs": action.nargs,
        "const": _enc(action.const),
        "default": _enc(action.default),
        "type": type_name,
        "choices": list(action.choices) if action.choices is not None else None,
        "required": action.required,
        "help": action.help,
        "metavar": action.metavar,
    }
    json.dumps(entry)  # raises on anything that would not round-trip
    return entry


def _add(parser: argparse.ArgumentParser, e: dict) -> None:
    e = {**e, "default": _dec(e["default"]), "const": _dec(e["const"])}
    common = {"dest": e["dest"], "required": e["required"], "help": e["help"]}
    kind = e["kind"]
    if kind == "store_boolean":
        parser.add_argument(*e["option_strings"], action=StoreBoolean, default=e["default"], **common)
    elif kind in ("store_true", "store_false"):
        parser.add_argument(*e["option_strings"], action=kind, default=e["default"], **common)
    elif kind == "store_const":
        parser.add_argument(*e["option_strings"], action=kind, const=e["const"], default=e["default"], **common)
    else:
        kwargs = {"default": e["default"], "type": _TYPES[e["type"]], "metavar": e["metavar"], **common}
        if e["nargs"] is not None:
            kwargs["nargs"] = e["nargs"]
        if e["const"] is not None:
            kwargs["const"] = e["const"]
        if e["choices"] is not None:
            kwargs["choices"] = e["choices"]
        parser.add_argument(*e["option_strings"], action=kind, **kwargs)


def add_cached(parser: argparse.ArgumentParser, register) -> None:
    """Add the sglang args to ``parser``: from the cache if present, else via ``register(parser)`` and cache them."""
    path = None
    try:
        path = _cache_path()
        if path and os.path.exists(path):
            with open(path) as f:
                entries = json.load(f)
            # Replay into a scratch parser first so a broken cache cannot leave ``parser`` half filled.
            scratch = argparse.ArgumentParser(add_help=False)
            for e in entries:
                _add(scratch, e)
            for e in entries:
                _add(parser, e)
            return
    except Exception as exc:  # a stale or broken cache must never break argument parsing
        logger.warning("sglang CLI arg cache unusable (%s); importing sglang", exc)
        if path:
            try:
                os.remove(path)
            except OSError:
                pass

    before = len(parser._actions)
    register(parser)
    if not path:
        return
    try:
        entries = [_encode(a) for a in parser._actions[before:]]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(entries, f)
        os.replace(tmp, path)
    except Exception as exc:
        logger.info("not caching sglang CLI args: %s", exc)


def prime() -> str | None:
    """Write the cache for the installed sglang (e.g. at image build time); returns its path."""
    from miles.backends.sglang_diffusion_utils.arguments import add_sglang_diffusion_arguments

    add_sglang_diffusion_arguments(argparse.ArgumentParser(add_help=False))
    path = _cache_path()
    return path if path and os.path.exists(path) else None


if __name__ == "__main__":
    print(prime() or "sglang CLI args not cacheable")
