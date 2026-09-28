"""The ``__main__`` block of an example driver must only call functions defined above it.

``cb_h2_error_propagation.py`` called ``boundary_sensitivity()`` from a main block placed above
the function's ``def``, so every run printed S10-S13 and then stopped with a NameError. The
AST check covers every driver in ``examples/``; the other two tests pin the S20 line that now
prints the within-specimen ratio ``main`` measured instead of a hard-coded 6.3%.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _is_main_guard(node):
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__")


def _calls_defined_after_main(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    defined_at = {node.name: node.lineno for node in tree.body
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    late = []
    for block in (node for node in tree.body if _is_main_guard(node)):
        for call in ast.walk(block):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and defined_at.get(call.func.id, 0) > block.lineno):
                late.append(f"{call.func.id} (called line {call.lineno}, "
                            f"defined line {defined_at[call.func.id]})")
    return late


@pytest.mark.parametrize("path", sorted(EXAMPLES.rglob("*.py")), ids=lambda p: p.name)
def test_main_block_calls_only_functions_defined_above_it(path):
    assert _calls_defined_after_main(path) == []


def test_the_check_catches_a_call_above_its_definition(tmp_path):
    script = tmp_path / "late.py"
    script.write_text('if __name__ == "__main__":\n    later()\n\n\ndef later():\n    pass\n')
    assert _calls_defined_after_main(script) == ["later (called line 2, defined line 5)"]


def _cross_network():
    """Centre node joined to one terminal on each of the six faces of a 10-unit box."""
    points = np.array([(5, 5, 5), (0, 5, 5), (10, 5, 5), (5, 0, 5),
                       (5, 10, 5), (5, 5, 0), (5, 5, 10)], dtype=float)
    u = np.zeros(6, int)
    v = np.arange(1, 7)
    length = np.full(6, 5.0)
    diameter = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 12.0])
    nodes = SimpleNamespace(points=points, point_data={
        "node_id": np.arange(7), "degree": np.array([6, 1, 1, 1, 1, 1, 1])})
    bounds = np.array([[0.0, 10.0]] * 3)
    return u, v, length, diameter, nodes, bounds


@pytest.fixture
def propagation(monkeypatch):
    import cb_h2_error_propagation as module
    monkeypatch.setattr(module, "load", lambda specimen_id: _cross_network())
    return module


def test_s20_prints_the_ratio_it_is_given_not_a_fixed_figure(propagation, capsys):
    propagation.boundary_sensitivity(4.7)
    out = capsys.readouterr().out
    assert "Calibre error moves the same ratio 4.7%" in out
    assert "6.3%" not in out


def test_main_returns_the_within_specimen_ratio_it_prints(propagation, capsys):
    ratio = propagation.main(0.69)
    out = capsys.readouterr().out
    assert np.isfinite(ratio)
    assert f"-> {ratio:.1f}%)" in out
