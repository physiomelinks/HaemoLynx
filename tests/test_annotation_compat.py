"""Guard against annotations that raise TypeError on older Pythons.

PEP 604 unions (``X | Y``) only became valid at runtime in Python 3.10.  A
module that writes them in an annotation position without
``from __future__ import annotations`` evaluates them at ``def`` time and
therefore fails to import on 3.9.  ``requires-python`` is ``>=3.10``, but the
future import keeps the sources importable on 3.9 and costs nothing, so this
test keeps them that way.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_DIRS = (REPO_ROOT / "src", REPO_ROOT / "examples")


def _has_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


class _UnionAnnotationFinder(ast.NodeVisitor):
    """Collect annotation positions that contain a PEP 604 union."""

    def __init__(self) -> None:
        self.hits: list[str] = []

    def _check(self, annotation, context: str) -> None:
        if annotation is None:
            return
        for node in ast.walk(annotation):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                self.hits.append(f"line {annotation.lineno}: {context}")
                return

    def visit_FunctionDef(self, node) -> None:
        args = node.args
        for arg in [
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            args.vararg,
            args.kwarg,
        ]:
            if arg is not None:
                self._check(arg.annotation, f"{node.name}(...{arg.arg})")
        self._check(node.returns, f"{node.name}() -> ...")
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AnnAssign(self, node) -> None:
        self._check(node.annotation, ast.unparse(node.target))
        self.generic_visit(node)


def _python_files() -> list[Path]:
    files: list[Path] = []
    for directory in SCANNED_DIRS:
        files.extend(
            path
            for path in sorted(directory.rglob("*.py"))
            if "__pycache__" not in path.parts
        )
    return files


@pytest.mark.parametrize(
    "path", _python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_pep604_unions_require_future_annotations(path: Path) -> None:
    tree = ast.parse(path.read_text())
    if _has_future_annotations(tree):
        return

    finder = _UnionAnnotationFinder()
    finder.visit(tree)

    assert not finder.hits, (
        f"{path.relative_to(REPO_ROOT)} uses PEP 604 `X | Y` annotations without "
        "`from __future__ import annotations`, so importing it raises TypeError "
        "on Python 3.9:\n  " + "\n  ".join(finder.hits)
    )


def test_scan_covers_the_package() -> None:
    scanned = {path.name for path in _python_files()}
    assert {"load.py", "poiseuille.py", "skeleton.py", "stats.py", "plot.py"} <= scanned
