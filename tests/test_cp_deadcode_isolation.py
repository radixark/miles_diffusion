"""阶段0 守卫：LLM-RL CP 死代码已**物理删除**且无活引用（AC-1 / AC-1.1）。

这些模块（training_utils/{loss,data,cp_utils,log_utils}.py）继承自 LLM RL，
带 causal/prompt/response 假设，diffusion 训练不调用（自带 PPO-clip loss 在
fsdp_utils/actor.py）。SP 稳定（AC-2~9 全绿）后按 AC-1.1 物理删除。

本测试守住两件事：
1. 这 4 个文件确已删除（import 失败应因文件不存在，而非 deprecated 标记）；
2. 全仓（miles/ + flow_grpo/）无对它们的活引用，防止后续误重新引入；
3. parallel.py（ParallelState + diffusion 实际使用的 dp_cp_* 兼容字段）保留。
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEAD_MODULES = ("loss", "data", "cp_utils", "log_utils")
DEAD_QUALNAMES = {f"miles.backends.training_utils.{m}" for m in DEAD_MODULES}
TU_DIR = REPO_ROOT / "miles" / "backends" / "training_utils"
DEAD_FILES = {TU_DIR / f"{m}.py" for m in DEAD_MODULES}


def test_deadcode_files_removed():
    """4 个 LLM CP 死代码模块文件已物理删除。"""
    still_present = sorted(f.relative_to(REPO_ROOT).as_posix() for f in DEAD_FILES if f.exists())
    assert not still_present, f"以下 LLM CP 死代码文件应已删除却仍存在：{still_present}"


def test_kept_modules_present():
    """保留项仍在：parallel.py（ParallelState/dp_cp_* 为 diffusion 活代码）。"""
    assert (TU_DIR / "parallel.py").exists(), "training_utils/parallel.py 被误删——ParallelState 是活代码"


def _dead_import(py_file: Path):
    """返回该文件对已删除死代码的引用串，无则 None（防重新引入）。"""
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
    """全仓无对已删除 CP 死代码的活引用（重新引入即失败）。"""
    offenders = []
    for pkg in ("miles", "flow_grpo"):
        for py in (REPO_ROOT / pkg).rglob("*.py"):
            ref = _dead_import(py)
            if ref:
                offenders.append(f"{py.relative_to(REPO_ROOT)} -> {ref}")
    assert not offenders, "发现对已删除 CP 死代码的引用：\n" + "\n".join(offenders)
