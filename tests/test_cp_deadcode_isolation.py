"""阶段0 守卫：LLM-RL CP 死代码已标记 deprecated 且无活引用（AC-1）。

这些模块（training_utils/{loss,data,cp_utils,log_utils}.py）继承自 LLM RL，
带 causal/prompt/response 假设，diffusion 训练不调用。SP 落地稳定前先隔离，
此测试守住"无活引用"，防止后续误引入。
"""
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEAD_MODULES = ("loss", "data", "cp_utils", "log_utils")
DEAD_QUALNAMES = {f"miles.backends.training_utils.{m}" for m in DEAD_MODULES}
DEAD_FILES = {REPO_ROOT / "miles" / "backends" / "training_utils" / f"{m}.py" for m in DEAD_MODULES}


def _has_deprecated_marker(py_file: Path) -> bool:
    """静态检查 __deprecated__ = True；不 import——这些死代码在 diffusion fork 里已 import-broken
    （引用了不存在的 miles.utils.flops_utils），无法 import 反而是更强的死代码证据。"""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__deprecated__":
                    return isinstance(node.value, ast.Constant) and node.value.value is True
    return False


@pytest.mark.parametrize("mod", DEAD_MODULES)
def test_deadcode_marked_deprecated(mod):
    f = REPO_ROOT / "miles" / "backends" / "training_utils" / f"{mod}.py"
    assert _has_deprecated_marker(f), f"{mod} 未标记 __deprecated__ = True"


def _dead_import(py_file: Path):
    """返回该文件对死代码的引用串，无则 None。"""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    in_tu = "training_utils" in py_file.parts
    dead = set(DEAD_MODULES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in DEAD_QUALNAMES:
                    return a.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            hit = {a.name for a in node.names} & dead
            if mod.endswith("training_utils") and hit:  # from ...training_utils import loss
                return f"{mod}.{sorted(hit)}"
            if mod.split(".")[-1] in dead and ("training_utils" in mod or (node.level and in_tu)):
                return mod or f".{node.names[0].name}"  # from ...training_utils.loss import / from .loss import
            if node.level and in_tu and hit:  # from . import loss（包内）
                return f"relative:{sorted(hit)}"
    return None


def test_no_live_references_in_codebase():
    offenders = []
    for pkg in ("miles", "flow_grpo"):
        for py in (REPO_ROOT / pkg).rglob("*.py"):
            if py in DEAD_FILES:
                continue
            if _dead_import(py):
                offenders.append(f"{py.relative_to(REPO_ROOT)} -> {_dead_import(py)}")
    assert not offenders, "发现对已废弃 CP 死代码的活引用：\n" + "\n".join(offenders)
