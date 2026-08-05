"""Fine-grained weight-precision control for FSDP2, at module granularity.

A family declares gather dtypes as PrecisionSpec rules on its TrainPipelineConfig:
a rule selects modules by FQN glob and/or class-name glob (both narrows to the
intersection) and pins the dtype their params are cast to for the FSDP
all-gather and the forward. The resident (master) dtype is a run-level knob,
``--fsdp-master-dtype``, because no model wants it to vary per module; compute
dtype is not managed here either — the trainer autocasts the DiT forward,
model-boundary input dtypes are family policy applied by
``apply_input_dtype_policy`` below, and op-level exceptions belong to the
monkey-patch registry.

``compile_precision`` lowers the rules onto what FSDP2 can express:

    PrecisionSpec rules
        |
    (1) per module, parent-first: inherit the parent's gather dtype, then apply
        the rules selecting this module in spec order (so a deeper rule always
        wins and order only breaks ties on one module)
        |
    (2) differs from what the parent already provides? -> the module becomes its
        own wrap unit at that dtype (paramless modules have nothing to gather and
        are skipped), which makes the units a minimal cover of the tree
        |
    (3) compiled.wrap_plan() merges the units with the block modules into one
        deepest-first order, so FSDP2 always nests child-before-parent — that is
        how gather="default" carves a module back out of a non-default ancestor.

Module granularity is the floor FSDP2 gives us: fully_shard wraps modules, so a
finer-grained selector could not be lowered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fnmatch import fnmatch

import torch

logger = logging.getLogger(__name__)

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_dtype(name: str) -> torch.dtype:
    return _DTYPES[name]


def _resolve_axis(axis: str | None, default_dtype: torch.dtype) -> torch.dtype | None:
    """Shared axis semantics: None -> untouched, "default" -> the run's default dtype, else a dtype name."""
    if axis is None:
        return None
    return default_dtype if axis == "default" else _DTYPES[axis]


# ---------------------------------------------------------------------------
# Spec: per-family declaration (see TrainPipelineConfig.precision_spec)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleSel:
    """Module selector; fqn and cls are globs over the module FQN and class name."""

    fqn: str | None = None
    cls: str | None = None

    def __post_init__(self) -> None:
        if self.fqn is None and self.cls is None:
            raise ValueError("ModuleSel needs fqn or cls; an empty selector silently matches every module")


@dataclass(frozen=True)
class Rule:
    """gather is a dtype name ("fp32"/"bf16"/"fp16") or "default", the run's default dtype."""

    select: ModuleSel
    gather: str


@dataclass(frozen=True)
class PrecisionSpec:
    rules: tuple[Rule, ...] = ()


# ---------------------------------------------------------------------------
# Compiler: spec -> FSDP2 lowering (per-module wrap units)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WrapUnit:
    """A module to fully_shard on its own with param_dtype=gather."""

    fqn: str
    module: torch.nn.Module
    param_dtype: torch.dtype


@dataclass
class CompiledPrecision:
    wrap_units: list[WrapUnit]
    # Effective gather dtype of every module, i.e. the dtype its innermost wrap unit provides.
    gather_dtypes: dict[str, torch.dtype]

    def wrap_plan(self, model: torch.nn.Module, block_modules: list[torch.nn.Module]) -> list[WrapUnit]:
        """One wrap order for FSDP2, deepest module first. Block modules are extra wraps that FSDP
        needs for sharding granularity, so they must carry their own effective dtype: wrapping one at
        the default inside an overridden region would be the innermost wrap and undo the override."""
        plan: dict[torch.nn.Module, WrapUnit] = {unit.module: unit for unit in self.wrap_units}
        depths, fqns = {}, {}
        for mod_fqn, module in model.named_modules():
            depths[module], fqns[module] = mod_fqn.count("."), mod_fqn
        for module in block_modules:
            fqn = fqns[module]
            plan.setdefault(module, WrapUnit(fqn, module, self.gather_dtypes[fqn.removeprefix("base_model.model.")]))
        return [plan[module] for module in sorted(plan, key=lambda module: -depths[module])]


def _selects(sel: ModuleSel, mod_fqn: str, module: torch.nn.Module) -> bool:
    if sel.fqn is not None and not fnmatch(mod_fqn, sel.fqn):
        return False
    return sel.cls is None or fnmatch(type(module).__name__, sel.cls)


def _parent_fqn(mod_fqn: str) -> str:
    """The root module's FQN is "", and it is its own parent."""
    return mod_fqn.rsplit(".", 1)[0] if "." in mod_fqn else ""


def compile_precision(
    model: torch.nn.Module,
    spec: PrecisionSpec,
    *,
    default_dtype: torch.dtype,
) -> CompiledPrecision:
    """Resolve the spec against the (pre-LoRA, pre-FSDP) model into FSDP2 wrap units.

    The rule is one line: **a module becomes its own wrap unit exactly when its gather dtype differs
    from its parent's.** Anything matching its parent is already covered by the parent's unit, so the
    emitted units are the minimal set of fully_shard calls that realises the spec.

    The traversal makes that cheap. ``named_modules`` yields parents before children, so the parent's
    dtype is already in ``gather_dtypes`` when we reach a module: inheritance is one dict lookup, and
    each rule only has to be tested against the module it names rather than against its ancestors.
    Within a module the rules apply in spec order, so a later rule wins, while rules on ancestors
    have already acted through the inherited dtype. A buffer-only module never needs a unit — FSDP
    gathers parameters, not buffers — whereas a container does, since ``parameters()`` recurses.
    """
    wrap_units: list[WrapUnit] = []
    gather_dtypes: dict[str, torch.dtype] = {"": default_dtype}
    hits = [0] * len(spec.rules)

    for mod_fqn, module in model.named_modules():
        parent_gather = gather_dtypes[_parent_fqn(mod_fqn)]
        gather = parent_gather
        for i, rule in enumerate(spec.rules):
            if not _selects(rule.select, mod_fqn, module):
                continue
            hits[i] += 1
            gather = _resolve_axis(rule.gather, default_dtype)

        needs_unit = gather != parent_gather and next(module.parameters(), None) is not None
        if needs_unit and mod_fqn == "":
            raise ValueError("cannot wrap the root module for a gather override")
        if needs_unit:
            wrap_units.append(WrapUnit(mod_fqn, module, gather))
        gather_dtypes[mod_fqn] = gather if needs_unit else parent_gather

    # A rule that selected nothing is a typo'd pattern or class name, not a silent no-op.
    for rule, hit in zip(spec.rules, hits, strict=True):
        if not hit:
            raise ValueError(f"precision rule matched no module: {rule}")
    return CompiledPrecision(wrap_units=wrap_units, gather_dtypes=gather_dtypes)


def log_precision_summary(component: str, compiled: CompiledPrecision, *, default_dtype: torch.dtype) -> None:
    logger.info(
        f"precision[{component}]: default gather dtype {default_dtype}, "
        f"{len(compiled.wrap_units)} extra wrap units"
    )
    for unit in compiled.wrap_units:
        logger.info(f"precision[{component}]: wrap {unit.fqn} @ {unit.param_dtype}")


# ---------------------------------------------------------------------------
# Boundary-input dtype policy (TrainPipelineConfig.input_dtype_policy)
# ---------------------------------------------------------------------------

INPUT_DTYPE_POLICY_KEYS = ("latents", "cond", "timestep")


def apply_input_dtype_policy(
    policy: dict,
    *,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    conds: tuple,
    default_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """Cast float boundary inputs per family policy ("default"/dtype name/None=passthrough);
    autocast alone would leave element-wise ops running at the raw input dtype."""
    unknown = set(policy) - set(INPUT_DTYPE_POLICY_KEYS)
    if unknown:
        raise ValueError(f"input_dtype_policy has unknown keys {sorted(unknown)}; known: {INPUT_DTYPE_POLICY_KEYS}")

    def _axis(key: str) -> torch.dtype | None:
        axis = policy.get(key)
        if axis is not None and axis != "default" and axis not in _DTYPES:
            raise ValueError(f"input_dtype_policy[{key!r}] has unknown dtype {axis!r}")
        return _resolve_axis(axis, default_dtype)

    def _cast(value, dtype: torch.dtype | None):
        if dtype is None or not torch.is_tensor(value) or not value.is_floating_point():
            return value
        return value.to(dtype)

    latents_dtype, timestep_dtype, cond_dtype = _axis("latents"), _axis("timestep"), _axis("cond")
    out_conds = tuple(
        None if cond is None else {key: _cast(value, cond_dtype) for key, value in cond.items()} for cond in conds
    )
    return _cast(latents, latents_dtype), _cast(timesteps, timestep_dtype), out_conds
