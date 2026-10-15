"""The ``__main__`` block of an example driver must only call functions defined above it.

``cb_h2_error_propagation.py`` called ``boundary_sensitivity()`` from a main block placed above
the function's ``def``, so every run printed S10-S13 and then stopped with a NameError. The
AST check covers every driver in ``examples/``; the other two tests pin the S20 line that now
prints the within-specimen ratio ``main`` measured instead of a hard-coded 6.3%, and check that
every solve ``main`` makes uses the face rule's inlets and outlets (the 25% band rule it replaced,
package C, under any name, fails it).
"""
import ast
from pathlib import Path

import numpy as np
import pytest
from cb_h1_batch import SENSITIVITY_DIR
from conftest import cross_network

from ImageLynx import cb_settings
from ImageLynx.graph.boundaries import select_boundary_terminal_nodes_by_face
from ImageLynx.specimens import BATCH_RUN_ROOT, PROCESSING_VOXEL_UM

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
    """The shared six-face cross, with calibre and length from an edge table as the batch-run
    reader gives them. The widest edge is the axis-1 inlet's, so the shunt ratio is defined,
    and the interior dead end (node 8) lies in the old 25% band but on no face, so a band
    rule picks different inlets from the face rule."""
    return cross_network(diameter=lambda n, edge: (10.0 if edge[:2] == (0, 3) else 4.0) + 0.1 * n,
                         length=lambda n, edge: 50.0 + 5 * n)


@pytest.fixture
def propagation(monkeypatch, make_batch_run):
    import cb_h2_error_propagation as module
    G, lengths = _cross_network()
    diameters = {(a, b, k): d["assigned_diameter_um"]
                 for a, b, k, d in G.edges(keys=True, data=True)}
    run = make_batch_run(G=G, columns={"length_um": lengths,
                                       "assigned_diameter_um": diameters}).run
    monkeypatch.setattr(module, "open_batch_run", lambda specimen: run)
    return module


def test_s20_prints_the_ratio_it_is_given_not_a_fixed_figure(propagation, capsys):
    propagation.boundary_sensitivity(4.7)
    out = capsys.readouterr().out
    assert "Calibre error moves the same ratio 4.7%" in out
    assert "6.3%" not in out


def test_main_returns_the_within_specimen_ratio_it_prints(propagation, monkeypatch, capsys):
    solve = propagation.solve_edge_flows
    boundaries = []

    def spy(u, v, length, diameter, inlets, outlets, n_nodes):
        boundaries.append((sorted(inlets.tolist()), sorted(outlets.tolist())))
        return solve(u, v, length, diameter, inlets, outlets, n_nodes)

    monkeypatch.setattr(propagation, "solve_edge_flows", spy)
    ratio = propagation.main(0.69)
    out = capsys.readouterr().out
    assert np.isfinite(ratio)
    assert f"-> {ratio:.1f}%)" in out

    # Every solve main makes places pressure by the pipeline's face rule on the frozen axis.
    G, *_, index = propagation.load_network("WKY-A")
    inlet_nodes, outlet_nodes = select_boundary_terminal_nodes_by_face(
        G, cb_settings.ROI_VOXELS, axis=cb_settings.BOUNDARY_AXIS,
        face_tolerance_voxels=cb_settings.BOUNDARY_FACE_TOLERANCE_VOXELS,
        voxel_size=PROCESSING_VOXEL_UM)
    face_rule = (sorted(index[n] for n in inlet_nodes), sorted(index[n] for n in outlet_nodes))
    assert boundaries and boundaries == [face_rule] * len(boundaries)


@pytest.mark.realdata
@pytest.mark.slow
@pytest.mark.skipif(not SENSITIVITY_DIR.is_dir() or not BATCH_RUN_ROOT.is_dir(),
                    reason="needs the batch and sensitivity outputs (gitignored)")
def test_the_propagation_default_is_the_measured_threshold_shift(capsys):
    """THRESHOLD_SHIFT_UM is copied from cb_h2_threshold_calibre.py; fail when it drifts."""
    import cb_h2_error_propagation
    import cb_h2_threshold_calibre

    shift = cb_h2_threshold_calibre.main()
    assert round(shift, 3) == cb_h2_error_propagation.THRESHOLD_SHIFT_UM
