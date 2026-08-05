"""Compiling PrecisionSpec rules into FSDP2 wrap units.

Every test uses this model with default_dtype=bf16, and every docstring draws the
resulting gather dtype per node (`[U]` = the module becomes its own wrap unit).
`blocks.1` mirrors `blocks.0`, so most diagrams only draw block 0.

    Tiny                            classes and own float tensors
    └── blocks          ModuleList  -
        ├── 0           Block       -
        │   ├── linear  Linear      weight, bias
        │   ├── norm    LayerNorm   weight, bias
        │   ├── attn    Attn        -
        │   │   └── norm_q  LayerNorm  weight, bias
        │   └── rope    Rope        freqs (buffer only)
        └── 1           Block       (same)
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest
import torch
import torch.nn as nn

from miles.backends.fsdp_utils.precision import ModuleSel, PrecisionSpec, Rule, compile_precision


class Rope(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("freqs", torch.zeros(4))


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_q = nn.LayerNorm(8)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.attn = Attn()
        self.rope = Rope()


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


def _units(compiled):
    return {unit.fqn: unit.param_dtype for unit in compiled.wrap_units}


def _plan(model, compiled):
    return [(unit.fqn, unit.param_dtype) for unit in compiled.wrap_plan(model, list(model.blocks))]


NORM_FQNS = {f"blocks.{i}{suffix}" for i in range(2) for suffix in (".norm", ".attn.norm_q")}


def test_empty_spec_compiles_to_nothing():
    """No rules, so every node keeps the default and nothing is emitted.

    blocks.0            bf16
    ├── linear          bf16
    ├── norm            bf16
    ├── attn            bf16
    │   └── norm_q      bf16
    └── rope            bf16
    """
    compiled = compile_precision(Tiny(), PrecisionSpec(), default_dtype=torch.bfloat16)
    assert compiled.wrap_units == []


def test_fqn_glob_selects_norms_across_depths():
    """`*norm*` crosses dots, so one rule catches both norm depths and skips the Linear siblings.

    Rule(fqn="*norm*", gather=fp32)

    blocks.0            bf16
    ├── linear          bf16
    ├── norm      [U]   fp32
    ├── attn            bf16
    │   └── norm_q [U]  fp32
    └── rope            bf16
    """
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="*norm*"), gather="fp32"),))
    compiled = compile_precision(Tiny(), spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == dict.fromkeys(NORM_FQNS, torch.float32)


def test_cls_glob_selects_by_class():
    """Selecting by class name reaches the same two norms without naming any path.

    Rule(cls="*LayerNorm", gather=fp32)

    blocks.0            bf16
    ├── norm      [U]   fp32
    └── attn
        └── norm_q [U]  fp32
    """
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="*LayerNorm"), gather="fp32"),))
    compiled = compile_precision(Tiny(), spec, default_dtype=torch.bfloat16)
    assert set(_units(compiled)) == NORM_FQNS


def test_rule_covers_the_matched_subtree():
    """A rule on a container hands its dtype to everything below, so one unit covers the subtree.

    Rule(fqn="blocks.1", gather=fp32)

    blocks.0            bf16      blocks.1        [U] fp32
    ├── linear          bf16      ├── linear          fp32  (inherits, no unit)
    ├── norm            bf16      ├── norm            fp32  (inherits, no unit)
    └── attn            bf16      └── attn            fp32  (inherits, no unit)
        └── norm_q      bf16          └── norm_q      fp32  (inherits, no unit)
    """
    model = Tiny()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="blocks.1"), gather="fp32"),))
    compiled = compile_precision(model, spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == {"blocks.1": torch.float32}
    assert compiled.gather_dtypes["blocks.1.attn.norm_q"] is torch.float32
    assert compiled.gather_dtypes["blocks.0.attn.norm_q"] is torch.bfloat16


def test_later_rule_overrides_earlier_selection():
    """Both rules select block 0's norms; the later one wins there while block 1 keeps the first.

    Rule(cls="LayerNorm",        gather=fp32)   # rule 1
    Rule(fqn="blocks.0.*norm*",  gather=fp16)   # rule 2, wins where they overlap

    blocks.0                      blocks.1
    ├── norm      [U]   fp16      ├── norm      [U]   fp32
    └── attn                      └── attn
        └── norm_q [U]  fp16          └── norm_q [U]  fp32
    """
    spec = PrecisionSpec(
        rules=(
            Rule(ModuleSel(cls="LayerNorm"), gather="fp32"),
            Rule(ModuleSel(fqn="blocks.0.*norm*"), gather="fp16"),
        )
    )
    compiled = compile_precision(Tiny(), spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == {
        "blocks.0.norm": torch.float16,
        "blocks.0.attn.norm_q": torch.float16,
        "blocks.1.norm": torch.float32,
        "blocks.1.attn.norm_q": torch.float32,
    }


def test_empty_module_sel_rejected():
    """A selector with neither fqn nor cls would silently match every module."""
    with pytest.raises(ValueError, match="needs fqn or cls"):
        ModuleSel()


def test_every_node_of_a_nested_chain_wraps_bottom_up():
    """Three nested rules that each differ from their parent need one unit per node, and the plan
    hands them back deepest first so each outer wrap excludes the inner ones.

        Rule(fqn="blocks.0",              gather=fp16)
        Rule(fqn="blocks.0.attn",         gather=fp32)
        Rule(fqn="blocks.0.attn.norm_q",  gather=default)   # carved back out

        blocks.0        [U] fp16   wrap order 3
        ├── linear          fp16   (inherits blocks.0)
        ├── norm            fp16   (inherits blocks.0)
        ├── attn        [U] fp32   wrap order 2
        │   └── norm_q  [U] bf16   wrap order 1, wraps first
        └── rope            buffer only, never gathered
        blocks.1        [U] bf16   block unit only, at the default dtype
    """
    model = Tiny()
    spec = PrecisionSpec(
        rules=(
            Rule(ModuleSel(fqn="blocks.0"), gather="fp16"),
            Rule(ModuleSel(fqn="blocks.0.attn"), gather="fp32"),
            Rule(ModuleSel(fqn="blocks.0.attn.norm_q"), gather="default"),
        )
    )
    compiled = compile_precision(model, spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == {
        "blocks.0.attn.norm_q": torch.bfloat16,
        "blocks.0.attn": torch.float32,
        "blocks.0": torch.float16,
    }
    assert _plan(model, compiled) == [
        ("blocks.0.attn.norm_q", torch.bfloat16),
        ("blocks.0.attn", torch.float32),
        ("blocks.0", torch.float16),
        ("blocks.1", torch.bfloat16),
    ]


def test_block_inside_an_override_wraps_at_the_override_dtype():
    """The rule sits above the block units, so the blocks wrap deeper than the override; at the
    default dtype they would be the innermost wrap and silently undo it.

    Rule(fqn="blocks", gather=fp32)

    blocks          [U] fp32   wrap order 3 (the override)
    ├── 0               fp32   wrap order 1, block unit forced to fp32
    └── 1               fp32   wrap order 2, block unit forced to fp32
    """
    model = Tiny()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="blocks"), gather="fp32"),))
    compiled = compile_precision(model, spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == {"blocks": torch.float32}
    assert _plan(model, compiled) == [
        ("blocks.0", torch.float32),
        ("blocks.1", torch.float32),
        ("blocks", torch.float32),
    ]


def test_paramless_module_gets_no_unit():
    """Buffers are never gathered, so pinning a buffer-only module lowers to nothing.

    Rule(cls="Rope", gather=fp32)

    blocks.0
    └── rope.freqs      buffer -> no unit
    """
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="Rope"), gather="fp32"),))
    compiled = compile_precision(Tiny(), spec, default_dtype=torch.bfloat16)
    assert compiled.wrap_units == []


def test_unmatched_rule_rejected():
    """A rule matching nothing is a typo'd pattern or class name, not a silent no-op."""
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="NoSuchModule"), gather="fp32"),))
    with pytest.raises(ValueError, match="matched no module"):
        compile_precision(Tiny(), spec, default_dtype=torch.bfloat16)
