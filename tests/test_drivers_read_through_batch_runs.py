"""Drivers that read a batch run do it through ``batch_outputs.open_batch_run`` only.

A driver that unpickles the graph, opens the edge table or the TH ``.h5``, or places the ROI
itself has its own copy of a reading rule, and a rule fixed in the reader stays broken there.
The drivers are checked by their source, because what matters is that the old loaders are gone,
not what any one run returned (the old-vs-new comparison on real data covers that).
"""
import ast
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# Drivers switched to the reader so far (batch-run reader tickets 01 to 04).
ON_THE_READER = [
    "cb_h2_glomus_perfusion.py",
    "cb_h2_absolute_perfusion.py",
    "cb_h2_hypoxic_fraction.py",
    "cb_h2_vtk.py",
    "cb_h2_error_propagation.py",
    "cb_h2_boundary_selection.py",
    "cb_h1_th_metrics.py",
    "cb_h1_vtk.py",
]
# Modules a driver only needs if it reads batch-run files itself.
FILE_READERS = {"csv", "h5py", "pickle"}
# Calls the reader makes once at open; a driver making them again places the ROI twice.
ROI_CALLS = {"place_roi", "check_output_roi"}


def _tree(name):
    return ast.parse((EXAMPLES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ON_THE_READER)
def test_the_driver_opens_its_batch_runs_through_the_reader(name):
    calls = {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
             for n in ast.walk(_tree(name)) if isinstance(n, ast.Call)}
    assert "open_batch_run" in calls


@pytest.mark.parametrize("name", ON_THE_READER)
def test_the_driver_reads_no_batch_run_file_itself(name):
    tree = _tree(name)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            imported |= {alias.name for alias in node.names}
    assert imported & (FILE_READERS | ROI_CALLS) == set()

    globs = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) in ("glob", "rglob")]
    assert globs == []


@pytest.mark.parametrize("name", ON_THE_READER)
def test_the_driver_defines_no_batch_root_of_its_own(name):
    names = {target.id for node in _tree(name).body if isinstance(node, ast.Assign)
             for target in node.targets if isinstance(target, ast.Name)}
    assert "BATCH" not in names
