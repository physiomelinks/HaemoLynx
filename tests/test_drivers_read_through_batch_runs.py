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
PACKAGE = Path(__file__).resolve().parents[1] / "src"

# Drivers that read batch runs, all through the reader. A driver keeps only its own analysis
# filtering (``cb_h2_error_propagation.py`` drops self-loops and non-positive lengths or
# diameters, and raises on stranded nodes). ``cb_h1_renders.py`` is not listed: it reads
# ``cb_h1_vtk.py``'s ``.vtp`` files, not a batch run.
ON_THE_READER = [
    "cb_h2_glomus_perfusion.py",
    "cb_h2_absolute_perfusion.py",
    "cb_h2_hypoxic_fraction.py",
    "cb_h2_vtk.py",
    "cb_h2_error_propagation.py",
    "cb_h2_boundary_selection.py",
    "cb_h1_th_metrics.py",
    "cb_h1_vtk.py",
    "cb_h1_figures.py",
    "cb_h2_threshold_calibre.py",
    "cb_compare_batch_runs.py",
]
# Drivers that open their runs through another reader driver's public loader instead of calling
# ``open_batch_run`` themselves: driver -> (module, loader).
SHARED_LOADERS = {
    "cb_h2_boundary_selection.py": ("cb_h2_error_propagation", "load_network"),
}
# Drivers that open their runs through a package module's function, which itself calls
# ``open_batch_run``: driver -> (module, function).
PACKAGE_LOADERS = {
    "cb_compare_batch_runs.py": ("ImageLynx.batch_compare", "compare_roots"),
}
# Modules a driver only needs if it reads batch-run files itself.
FILE_READERS = {"csv", "h5py", "pickle"}
# Calls the reader makes once at open; a driver making them again places the ROI twice.
ROI_CALLS = {"place_roi", "check_output_roi"}
# Ways to read a batch-run file without the reader. JSON a driver wrote itself, such as the
# threshold selection, is read with ``Path.read_text`` and is not a batch-run file.
FILE_CALLS = {"read_csv", "read_pickle", "genfromtxt", "loadtxt", "fromfile", "read_bytes", "open"}
ARRAY_MODULES = {"np", "numpy"}
# Names and fragments that only the reader should know about.
BATCH_FILE_NAMES = {"per_edge_morphometry.csv", "network_graph.pkl", "skeleton.npy",
                    "vessel_mask.npy"}


def _tree(name):
    return ast.parse((EXAMPLES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ON_THE_READER)
def test_the_driver_opens_its_batch_runs_through_the_reader(name):
    tree = _tree(name)
    calls = {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
             for n in ast.walk(tree) if isinstance(n, ast.Call)}
    if name in PACKAGE_LOADERS:
        module, loader = PACKAGE_LOADERS[name]
        source = ast.parse((PACKAGE / Path(*module.split("."))).with_suffix(".py").read_text(
            encoding="utf-8"))
        assert "open_batch_run" in {getattr(n.func, "id", None) for n in ast.walk(source)
                                    if isinstance(n, ast.Call)}
        assert loader in calls
        return
    if name not in SHARED_LOADERS:
        assert "open_batch_run" in calls
        return
    module, loader = SHARED_LOADERS[name]
    assert f"{module}.py" in ON_THE_READER
    assert loader in calls
    assert any(isinstance(n, ast.ImportFrom) and n.module == module
               and loader in {alias.name for alias in n.names} for n in ast.walk(tree))


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
    assert not names & {"BATCH", "RESULTS", "OUTPUTS"}


def _code_strings(tree):
    """String constants in the code, leaving out docstrings, which only describe."""
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
                  and node.body and isinstance(node.body[0], ast.Expr)
                  and isinstance(node.body[0].value, ast.Constant)}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and isinstance(n.value, str) and id(n) not in docstrings]


def other_ways_to_read_a_batch_run(tree):
    """Calls and names that read, or locate, a batch-run file outside the reader."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            on_numpy = (isinstance(func, ast.Attribute) and name == "load"
                        and getattr(func.value, "id", None) in ARRAY_MODULES)
            if name in FILE_CALLS or on_numpy:
                found.append(f"line {node.lineno}: {name}(...)")
    for text in _code_strings(tree):
        if "_cache" in text or any(name in text for name in BATCH_FILE_NAMES):
            found.append(f"string {text!r}")
    return found


def private_names_taken_from_other_drivers(tree):
    """Underscore names reached in another ``cb_*`` driver, by import or by attribute."""
    found = []
    drivers = {(alias.asname or alias.name) for node in ast.walk(tree)
               if isinstance(node, ast.Import) for alias in node.names
               if alias.name.startswith("cb_")}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("cb_"):
            found += [f"line {node.lineno}: from {node.module} import {alias.name}"
                      for alias in node.names if alias.name.startswith("_")]
        elif (isinstance(node, ast.Attribute) and node.attr.startswith("_")
              and getattr(node.value, "id", None) in drivers):
            found.append(f"line {node.lineno}: {node.value.id}.{node.attr}")
    return found


# Calls that turn a string cell into a number.
NUMBER_CALLS = {"float", "int", "float64"}


def _is_nan(node):
    """``np.nan``, ``numpy.nan``, ``math.nan`` or ``float("nan")``."""
    if isinstance(node, ast.Attribute):
        return node.attr == "nan"
    return (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "float"
            and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
            and str(node.args[0].value).lower() == "nan")


def _catches_parse_errors(handler):
    caught = handler.type
    names = caught.elts if isinstance(caught, ast.Tuple) else [caught]
    return any(getattr(name, "id", None) in ("ValueError", "TypeError") for name in names)


def cells_turned_into_numbers_outside_the_reader(tree):
    """Ways a driver turns an edge-table cell into a number itself, where a blank or text cell
    would become NaN or raise without naming the run. ``BatchRun.numeric_column`` does it once."""
    found = []
    for node in ast.walk(tree):
        func = node.func if isinstance(node, ast.Call) else None
        if (func is not None
                and (getattr(func, "id", None) or getattr(func, "attr", None)) in NUMBER_CALLS
                and node.args and isinstance(node.args[0], ast.Subscript)
                and isinstance(node.args[0].slice, ast.Constant)
                and isinstance(node.args[0].slice.value, str)):
            found.append(f"line {node.lineno}: a number made from the cell "
                         f"{node.args[0].slice.value!r}")
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and any(
                isinstance(value, ast.Constant) and str(value.value).lower() == "nan"
                for value in node.values):
            found.append(f'line {node.lineno}: `or "nan"`')
        elif isinstance(node, ast.ExceptHandler) and node.type and _catches_parse_errors(node):
            if any(isinstance(n, ast.Return) and n.value is not None and _is_nan(n.value)
                   for n in ast.walk(node)):
                found.append(f"line {node.lineno}: a parse error handled by returning NaN")
    return found


@pytest.mark.parametrize("name", ON_THE_READER)
def test_the_driver_reads_no_batch_run_file_by_other_means(name):
    assert other_ways_to_read_a_batch_run(_tree(name)) == []


@pytest.mark.parametrize("name", ON_THE_READER)
def test_the_driver_takes_no_private_name_from_another_driver(name):
    assert private_names_taken_from_other_drivers(_tree(name)) == []


# --- Cells become numbers only in the reader --------------------------------------------------

@pytest.mark.parametrize("source", [
    'x = float(row["length_um"])',
    "x = float(row['edt_diameter_um'])",
    'x = float(row["length_um"] or "nan")',
    "x = value or 'NaN'",
    'n = int(row["n_centreline_points"])',
    'd = np.float64(row["edt_diameter_um"])',
    'd = numpy.float64(row["edt_diameter_um"])',
    "def _float(v):\n    try:\n        return float(v)\n    except (TypeError, ValueError):\n"
    "        return np.nan",
    "def parse(v):\n    try:\n        return float(v)\n    except ValueError:\n"
    "        return float('nan')",
], ids=["float of a cell", "single quotes", "or nan", "or NaN", "int of a cell",
        "np.float64 of a cell", "numpy.float64 of a cell", "_float helper",
        "float nan helper"])
def test_the_cell_check_flags_a_cell_turned_into_a_number_outside_the_reader(source):
    assert cells_turned_into_numbers_outside_the_reader(ast.parse(source)) != []


@pytest.mark.parametrize("source", [
    'x = float(data.get("flow_abs", np.nan))',
    "x = float(np.mean(values))",
    "x = flow / total if total else float('nan')",
    "x = float(values[k])",
    "def parse(v):\n    try:\n        return float(v)\n    except ValueError:\n        raise",
], ids=["graph attribute", "a computed result", "a nan result", "a non-string key",
        "a handler that raises"])
def test_the_cell_check_passes_numbers_that_are_not_edge_table_cells(source):
    assert cells_turned_into_numbers_outside_the_reader(ast.parse(source)) == []


@pytest.mark.parametrize("name", ON_THE_READER)
def test_the_driver_turns_no_cell_into_a_number_itself(name):
    assert cells_turned_into_numbers_outside_the_reader(_tree(name)) == []
