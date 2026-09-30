"""The ``__main__`` block of an example driver must only call functions defined above it.

``cb_h2_error_propagation.py`` called ``boundary_sensitivity()`` from a main block placed above
the function's ``def``, so every run printed S10-S13 and then stopped with a NameError. The
AST check covers every driver in ``examples/``; the other two tests pin the S20 line that now
prints the within-specimen ratio ``main`` measured instead of a hard-coded 6.3%.
"""
import ast
from pathlib import Path

import networkx as nx
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


def _cross_network(tmp_path):
    """Centre node joined to one terminal on each of the six ROI faces, in micrometres.

    One extra branch reaches the low axis-1 face through a parallel pair, so the MultiGraph path
    is exercised. Calibre and length come from a CSV, as they do from the batch output.
    """
    G = nx.MultiGraph()
    mid, top = 148.0, 296.0
    G.add_node(0, pos=np.array([mid, mid, mid]))
    faces = [(0.0, mid, mid), (top, mid, mid), (mid, 0.0, mid),
             (mid, top, mid), (mid, mid, 0.0), (mid, mid, top)]
    for i, pos in enumerate(faces, start=1):
        G.add_node(i, pos=np.array(pos))
        G.add_edge(0, i)
    G.add_node(7, pos=np.array([mid, 60.0, 100.0]))
    G.add_node(8, pos=np.array([mid, 0.5, 100.0]))
    G.add_edge(0, 7)
    G.add_edge(0, 7)
    G.add_edge(7, 8)
    path = tmp_path / "per_edge_morphometry.csv"
    lines = ["u,v,key,length_um,assigned_diameter_um"]
    for n, (a, b, k) in enumerate(G.edges(keys=True)):
        lines.append(f"{a},{b},{k},{50.0 + 5 * n},{4.0 + n}")
    path.write_text("\n".join(lines) + "\n")
    return G, path


@pytest.fixture
def propagation(monkeypatch, tmp_path):
    import cb_h2_error_propagation as module
    G, csv_path = _cross_network(tmp_path)
    monkeypatch.setattr(module, "load_network",
                        lambda specimen_id: (G, *module.network_arrays(G, csv_path)))
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


_CALIBRE_INPUTS = Path(__file__).resolve().parents[1] / "examples" / "outputs"


@pytest.mark.skipif(not (_CALIBRE_INPUTS / "cb_h1_sensitivity").is_dir()
                    or not (_CALIBRE_INPUTS / "cb_h1_batch").is_dir(),
                    reason="needs the batch and sensitivity outputs (gitignored)")
def test_the_propagation_default_is_the_measured_threshold_shift(capsys):
    """THRESHOLD_SHIFT_UM is copied from cb_h2_threshold_calibre.py; fail when it drifts."""
    import cb_h2_error_propagation
    import cb_h2_threshold_calibre

    shift = cb_h2_threshold_calibre.main()
    assert round(shift, 3) == cb_h2_error_propagation.THRESHOLD_SHIFT_UM
