"""Writing a reference run and comparing a run folder with it, on tiny hand-made run folders.

Each test writes a run folder as the pipeline lays it out (``per_edge_morphometry.csv``, the
flow network ``.vtp`` and the perfusion ``.vti`` beside one ``*_cache`` folder with the mask,
skeleton and pickled graph: a three-edge graph on a 4^3 grid, and a 2^3-cell perfusion grid),
stores it as a reference run, then changes one thing on one side. The update script's refusal
of a dirty tree is tested on a throwaway git repo.
"""
import csv
import json
import math
import pickle
from functools import partial

import networkx as nx
import numpy as np
import pytest
import pyvista as pv

import update_benchmark_reference_run as update_script
from conftest import toy_git
from ImageLynx.batch_outputs import (CACHE_SUFFIX, EDGE_TABLE_NAME, GRAPH_NAME,
                                     PERFUSION_VTI_NAME, SKELETON_NAME, VESSEL_MASK_NAME,
                                     VESSELS_FLOW_VTP_NAME)
from ImageLynx.reference_run import (
    EDGE_TABLE, FLOW_NETWORK, GEOMETRY_RTOL, GRAPH, GRAPH_JSON_NAME, MANIFEST_NAME, MASKS_NAME,
    NETWORK_RTOL, PARTS, PERFUSION_FIELD, PERFUSION_RTOL, PERFUSION_TAGS, RHEOLOGY_STATUS,
    SKELETON, SUMMARY, SUMMARY_NAME, VESSEL_MASK, compare_with_reference, format_report,
    write_reference_run,
)

EDGES = [(0, 1, 0), (1, 2, 0), (1, 2, 1)]
SHAPE = (4, 4, 4)
COMMAND = ["--placed-input", f"{update_script.REPO_TOKEN}/sub.h5"]
CACHE_DIR = "sub" + CACHE_SUFFIX


def _columns():
    """The edge table after ``u, v, key``, as ``{column: {edge: cell}}``. Largest length 20."""
    return {"length_um": {edge: repr(10.0 * (edge[0] + 1)) for edge in EDGES},
            "assigned_diameter_um": {edge: repr(6.0 + edge[1] + edge[2]) for edge in EDGES},
            "n_centreline_points": dict.fromkeys(EDGES, "7"),
            "diameter_provenance": dict.fromkeys(EDGES, "measured_edt"),
            "fwhm_diameter_um": dict.fromkeys(EDGES, "")}


def _graph():
    G = nx.MultiGraph(voxel_size=(1.5, 1.5, 1.5))
    for u in (0, 1, 2):
        G.add_node(u, pos=np.array([1.0, 2.0, 0.5 * u]))
    for u, v, key in EDGES:
        G.add_edge(u, v, key=key, length=10.0 * (u + 1), voxels=[(1, 2, u), (1, 3, v)],
                   segment_id=u + key, centreline_smoothing="bspline", tortuosity=float("nan"))
    G.edges[1, 2, 1]["merged"] = True  # carried by one edge only, as the pipeline's are
    return G


def _mask():
    mask = np.zeros(SHAPE, bool)
    mask[1:3, 1:3, :] = True
    return mask


def _flow_network(**cell_arrays):
    """The flow network: one line per edge, with edge keys, a geometry array, three solve
    arrays and the rheology status in its field data. ``cell_arrays`` replace or add arrays."""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 0], [2, 1, 0]])
    mesh = pv.PolyData(points, lines=np.array([2, 0, 1, 2, 2, 3, 2, 4, 5]))
    edges = np.array(EDGES)
    arrays = {"edge_u": edges[:, 0], "edge_v": edges[:, 1], "edge_key": edges[:, 2],
              "assigned_diameter_um": np.array([7.0, 8.0, 9.0]),
              "flow_abs": np.array([100.0, 60.0, 40.0]),
              "flow_signed": np.array([100.0, -60.0, 40.0]),
              "hematocrit": np.array([0.45, 0.4, np.nan]),
              **cell_arrays}
    for name, values in arrays.items():
        mesh.cell_data[name] = values
    mesh.field_data["rheology_stop_reason"] = np.array(["converged"])
    mesh.field_data["rheology_iterations"] = np.array([330])
    mesh.field_data["rheology_relative_flow_change"] = np.array([9.9e-7])
    return mesh


def _perfusion_field(po2=None, **tags):
    """The Tier 3 perfusion field on 2^3 cells, PO2 50 to 85 and pH 7.4, with its tags in its
    field data. ``tags`` replace tags."""
    grid = pv.ImageData(dimensions=(3, 3, 3), spacing=(30.0, 30.0, 30.0))
    grid.cell_data["PO2_mmHg"] = np.linspace(50.0, 85.0, 8) if po2 is None else po2
    grid.cell_data["pH"] = np.full(8, 7.4)
    defaults = {"perfusion_tier": 3, "perfusion_solver": "solve_multi_species_perfusion",
                "perfusion_M_max": 0.05, "perfusion_converged": 1,
                "perfusion_picard_iterations": 49, "perfusion_final_residual": 5.8e-6}
    for name, value in {**defaults, **tags}.items():
        grid.field_data[name] = np.array([value])
    return grid


def _write_run(run_dir, graph=None, columns=None, mask=None, skeleton=None, flow=None,
               perfusion=None):
    """A run folder laid out as the pipeline writes one; returns it."""
    cache = run_dir / CACHE_DIR
    cache.mkdir(parents=True)
    (_flow_network() if flow is None else flow).save(run_dir / VESSELS_FLOW_VTP_NAME)
    (_perfusion_field() if perfusion is None else perfusion).save(run_dir / PERFUSION_VTI_NAME)
    np.save(cache / VESSEL_MASK_NAME, _mask() if mask is None else mask)
    skeleton_default = np.zeros(SHAPE, bool)
    skeleton_default[2, 2, :] = True
    np.save(cache / SKELETON_NAME, skeleton_default if skeleton is None else skeleton)
    with (cache / GRAPH_NAME).open("wb") as handle:
        pickle.dump(_graph() if graph is None else graph, handle)
    columns = _columns() if columns is None else columns
    with (run_dir / EDGE_TABLE_NAME).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["u", "v", "key", *columns])
        for edge in next(iter(columns.values())):
            writer.writerow([*edge, *(columns[column][edge] for column in columns)])
    return run_dir


@pytest.fixture
def reference(tmp_path):
    """A reference run written from the default run folder."""
    path = tmp_path / "reference"
    write_reference_run(_write_run(tmp_path / "made_from"), path, COMMAND, "abc123")
    return path


def _compare(tmp_path, reference, **changed):
    return compare_with_reference(_write_run(tmp_path / "run", **changed), reference)


def _differing(comparison) -> dict:
    return {verdict.part: verdict.detail for verdict in comparison.parts if not verdict.same}


# --- Writing ---------------------------------------------------------------------------------

def test_a_run_matches_the_reference_run_written_from_it(tmp_path, reference):
    comparison = _compare(tmp_path, reference)
    assert [verdict.part for verdict in comparison.parts] == list(PARTS)
    assert comparison.same, format_report(comparison)
    assert comparison.environment == ()


def test_the_masks_are_stored_bit_packed_and_the_manifest_records_the_run(reference):
    with np.load(reference / MASKS_NAME) as stored:
        assert stored["vessel_mask"].dtype == np.uint8
        assert stored["vessel_mask"].size == math.prod(SHAPE) // 8
        assert tuple(stored["skeleton_shape"]) == SHAPE
    manifest = json.loads((reference / MANIFEST_NAME).read_text())
    assert manifest["commit"] == "abc123" and manifest["command"] == COMMAND
    assert {"python", "packages"} <= set(manifest)
    assert (reference / EDGE_TABLE_NAME).read_bytes() == \
        (reference.parent / "made_from" / EDGE_TABLE_NAME).read_bytes()


def test_the_graph_is_stored_as_readable_json_on_forward_edges(reference):
    graph = json.loads((reference / GRAPH_JSON_NAME).read_text())
    assert graph["type"] == "MultiGraph"
    assert graph["graph"] == {"voxel_size": [1.5, 1.5, 1.5]}
    assert [(e["u"], e["v"], e["key"]) for e in graph["edges"]] == EDGES
    assert graph["nodes"][2] == {"id": 2, "attributes": {"pos": [1.0, 2.0, 1.0]}}


def test_rewriting_replaces_the_whole_folder(tmp_path, reference):
    (reference / "left_from_an_old_part.npz").write_bytes(b"x")
    write_reference_run(_write_run(tmp_path / "again"), reference, COMMAND, "def456")
    assert sorted(path.name for path in reference.iterdir()) == sorted(
        [MASKS_NAME, GRAPH_JSON_NAME, EDGE_TABLE_NAME, VESSELS_FLOW_VTP_NAME, PERFUSION_VTI_NAME,
         SUMMARY_NAME, MANIFEST_NAME])
    assert not reference.with_name(reference.name + ".new").exists()


def test_a_key_cell_that_is_not_an_integer_raises_naming_the_run(tmp_path, reference):
    run = _write_run(tmp_path / "run")
    table = run / EDGE_TABLE_NAME
    table.write_text(table.read_text().replace("\n1,2,0,", "\n1,two,0,"))
    with pytest.raises(ValueError, match=f"NEW: {EDGE_TABLE_NAME} line 3 has no integer u, v, key"):
        compare_with_reference(run, reference)


def test_writing_from_a_folder_with_no_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="_cache"):
        write_reference_run(tmp_path, tmp_path / "reference", COMMAND, "abc")


# --- Exact parts -----------------------------------------------------------------------------

@pytest.mark.parametrize("part, keyword", [(VESSEL_MASK, "mask"), (SKELETON, "skeleton")])
def test_one_changed_voxel_differs(tmp_path, reference, part, keyword):
    array = _mask()
    if part == SKELETON:
        array = np.zeros(SHAPE, bool)
        array[2, 2, :] = True
    array[0, 0, 0] = ~array[0, 0, 0]
    comparison = _compare(tmp_path, reference, **{keyword: array})
    assert _differing(comparison) == {part: f"1 of {math.prod(SHAPE)} voxels differ"}


def test_a_mask_of_another_shape_differs(tmp_path, reference):
    detail = _differing(_compare(tmp_path, reference, mask=np.zeros((4, 4, 5), bool)))
    assert detail == {VESSEL_MASK: "shape (4, 4, 4) in REFERENCE, (4, 4, 5) in NEW"}


def test_a_removed_edge_differs_in_the_graph_and_the_edge_table(tmp_path, reference):
    G = _graph()
    G.remove_edge(1, 2, key=1)
    columns = {column: {edge: cell for edge, cell in cells.items() if edge != (1, 2, 1)}
               for column, cells in _columns().items()}
    detail = _differing(_compare(tmp_path, reference, graph=G, columns=columns))
    assert detail[GRAPH] == "1 edges only in REFERENCE: [(1, 2, 1)]"
    assert detail[EDGE_TABLE] == "1 rows only in REFERENCE: [(1, 2, 1)]"


def test_a_changed_text_cell_differs(tmp_path, reference):
    columns = _columns()
    columns["diameter_provenance"][(0, 1, 0)] = "measured_fwhm"
    detail = _differing(_compare(tmp_path, reference, columns=columns))
    assert detail == {EDGE_TABLE: "diameter_provenance: 1 of 3 rows differ: [(0, 1, 0)]"}


def test_an_integer_differs_however_small_the_change(tmp_path, reference):
    columns = _columns()
    columns["n_centreline_points"][(0, 1, 0)] = "8"
    G = _graph()
    G.edges[0, 1, 0]["segment_id"] += 1
    detail = _differing(_compare(tmp_path, reference, graph=G, columns=columns))
    assert detail[EDGE_TABLE].startswith("n_centreline_points: 1 of 3 rows differ")
    assert detail[GRAPH] == "edge segment_id: 1 of 3 edges differ: [(0, 1, 0)]"


def test_a_float_that_became_an_integer_differs(tmp_path, reference):
    G = _graph()
    G.edges[0, 1, 0]["length"] = 10
    assert _differing(_compare(tmp_path, reference, graph=G))[GRAPH].startswith(
        "edge length: 1 of 3 edges differ")


def test_an_attribute_on_one_side_only_differs(tmp_path, reference):
    G = _graph()
    G.edges[0, 1, 0]["merged"] = True
    del G.edges[1, 2, 1]["merged"]
    assert _differing(_compare(tmp_path, reference, graph=G)) == {
        GRAPH: "edge merged: 2 of 3 edges differ: [(0, 1, 0), (1, 2, 1)]"}


def test_a_changed_node_set_differs(tmp_path, reference):
    G = _graph()
    G.add_node(7, pos=np.zeros(3))
    assert _differing(_compare(tmp_path, reference, graph=G)) == {
        GRAPH: "1 nodes only in NEW: [7]"}


# --- Geometry floats -------------------------------------------------------------------------

def _length_column(edge, value):
    columns = _columns()
    columns["length_um"][edge] = repr(value)
    return columns


def test_geometry_passes_inside_rtol_and_fails_outside(tmp_path, reference):
    # Edge (0, 1, 0) has length 10 and the column's largest is 20: the band is
    # rtol * 10 + rtol * 20 = 30 * rtol.
    inside = _compare(tmp_path / "a", reference,
                      columns=_length_column((0, 1, 0), 10.0 + 29 * GEOMETRY_RTOL))
    assert inside.same, format_report(inside)
    outside = _compare(tmp_path / "b", reference,
                       columns=_length_column((0, 1, 0), 10.0 + 31 * GEOMETRY_RTOL))
    detail = _differing(outside)[EDGE_TABLE]
    assert detail.startswith("length_um: 1 of 3 rows differ, max abs 3.1e-08, max rel 3.1e-09")


def test_the_absolute_floor_scales_with_the_largest_value(tmp_path, reference):
    """A node coordinate of 0 moved by less than rtol times the largest coordinate passes."""
    G = _graph()
    G.nodes[0]["pos"] = np.array([1.0, 2.0, 1.5 * GEOMETRY_RTOL])  # largest |pos| is 2
    assert _compare(tmp_path / "a", reference, graph=G).same
    G.nodes[0]["pos"] = np.array([1.0, 2.0, 2.5 * GEOMETRY_RTOL])
    assert _differing(_compare(tmp_path / "b", reference, graph=G))[GRAPH].startswith(
        "node pos: 1 of 3 nodes differ, max abs 2.5e-09, max rel inf: [0]")


def test_nan_equals_nan_and_not_a_number(tmp_path, reference):
    assert _compare(tmp_path / "a", reference).same  # every tortuosity is NaN on both sides
    G = _graph()
    G.edges[1, 2, 0]["tortuosity"] = 1.0
    assert _differing(_compare(tmp_path / "b", reference, graph=G)) == {
        GRAPH: "edge tortuosity: 1 of 3 edges differ: [(1, 2, 0)]"}


def test_a_blank_numeric_cell_raises(tmp_path, reference):
    columns = _columns()
    columns["assigned_diameter_um"][(1, 2, 0)] = ""
    with pytest.raises(ValueError, match="empty or non-finite assigned_diameter_um"):
        _compare(tmp_path, reference, columns=columns)


def test_a_text_numeric_cell_raises(tmp_path, reference):
    with pytest.raises(ValueError, match="non-finite length_um"):
        _compare(tmp_path, reference, columns=_length_column((0, 1, 0), "long"))


# --- Missing files and the report ------------------------------------------------------------

def test_a_part_file_missing_from_the_run_is_a_difference(tmp_path, reference):
    run = _write_run(tmp_path / "run")
    (run / CACHE_DIR / SKELETON_NAME).unlink()
    (run / EDGE_TABLE_NAME).unlink()
    assert _differing(compare_with_reference(run, reference)) == {
        SKELETON: f"missing in NEW: {SKELETON_NAME}",
        EDGE_TABLE: f"missing in NEW: {EDGE_TABLE_NAME}"}


def test_a_part_file_missing_from_the_reference_is_a_difference(tmp_path, reference):
    (reference / GRAPH_JSON_NAME).unlink()
    assert _differing(_compare(tmp_path, reference)) == {
        GRAPH: f"missing in REFERENCE: {GRAPH_JSON_NAME}"}


def test_a_run_folder_without_its_cache_differs_in_every_cached_part(tmp_path, reference):
    run = _write_run(tmp_path / "run")
    for path in (run / CACHE_DIR).iterdir():
        path.unlink()
    (run / CACHE_DIR).rmdir()
    assert set(_differing(compare_with_reference(run, reference))) == {
        VESSEL_MASK, SKELETON, GRAPH}


def test_no_reference_run_at_all_differs_in_every_part(tmp_path):
    comparison = compare_with_reference(_write_run(tmp_path / "run"), tmp_path / "none")
    assert set(_differing(comparison)) == set(PARTS)


def test_the_report_names_the_part_and_the_size_of_the_gap(tmp_path, reference):
    comparison = _compare(tmp_path, reference,
                          columns=_length_column((1, 2, 0), 20.0 * (1 + 1e-6)))
    report = format_report(comparison, label="benchmark run")
    assert report.splitlines()[0] == "benchmark run"
    assert ("  edge table      DIFFERENT: length_um: 1 of 3 rows differ, max abs 2e-05, "
            "max rel 1e-06: [(1, 2, 0)]") in report
    assert "  graph           same" in report
    assert report.splitlines()[-1] == f"{len(PARTS)} parts compared: 1 differ."


def test_a_different_environment_is_a_note_not_a_difference(tmp_path, reference):
    manifest = json.loads((reference / MANIFEST_NAME).read_text())
    manifest["python"] = "3.10.0"
    manifest["packages"]["numpy"] = "0.0"
    (reference / MANIFEST_NAME).write_text(json.dumps(manifest))
    comparison = _compare(tmp_path, reference)
    assert comparison.same
    report = format_report(comparison)
    assert "  environment     python 3.10.0 in REFERENCE" in report
    assert "  environment     numpy 0.0 in REFERENCE" in report


def test_a_different_command_is_a_note(tmp_path, reference):
    run = _write_run(tmp_path / "run")
    assert compare_with_reference(run, reference, COMMAND).environment == ()
    notes = compare_with_reference(run, reference, [*COMMAND, "--config", "x.yaml"]).environment
    assert len(notes) == 1 and notes[0].startswith(f"command {COMMAND} in REFERENCE")


def test_a_manifest_without_its_keys_raises(tmp_path, reference):
    (reference / MANIFEST_NAME).write_text(json.dumps({"commit": "abc123"}))
    with pytest.raises(ValueError, match="has no \\['python', 'packages', 'command'\\]"):
        _compare(tmp_path, reference)


# --- The update script -----------------------------------------------------------------------

def _toy_repo(path):
    path.mkdir()
    toy_git(path, "init", "-q")
    (path / "f.txt").write_text("a")
    toy_git(path, "add", "f.txt")
    toy_git(path, "commit", "-q", "-m", "init")
    return path


def test_the_update_script_refuses_a_tree_with_tracked_changes(tmp_path):
    repo = _toy_repo(tmp_path / "repo")
    (repo / "f.txt").write_text("changed")
    ran = []
    with pytest.raises(RuntimeError, match="tracked changes"):
        update_script.update(tmp_path / "reference", git=partial(toy_git, repo), run=ran.append)
    assert ran == [] and not (tmp_path / "reference").exists()


@pytest.mark.parametrize("folder", update_script.CODE_FOLDERS)
def test_the_update_script_refuses_untracked_code(tmp_path, folder):
    repo = _toy_repo(tmp_path / "repo")
    (repo / "notes.md").write_text("an untracked note elsewhere is fine")
    git, ran = partial(toy_git, repo), []
    assert update_script.refuse_a_dirty_tree(git) == git("rev-parse", "HEAD").strip()
    (repo / folder).mkdir()
    (repo / folder / "new_module.py").write_text("")
    with pytest.raises(RuntimeError, match="untracked code"):
        update_script.update(tmp_path / "reference", git=git, run=ran.append)
    assert ran == []


def test_the_update_script_writes_nothing_when_the_tree_changes_during_the_run(tmp_path):
    repo = _toy_repo(tmp_path / "repo")

    def run_and_edit(run_dir):
        _write_run(run_dir)
        (repo / "f.txt").write_text("edited mid-run")

    with pytest.raises(RuntimeError, match="tracked changes"):
        update_script.update(tmp_path / "reference", git=partial(toy_git, repo),
                             run=run_and_edit, out=lambda text: None)
    assert not (tmp_path / "reference").exists()


def test_the_update_script_writes_the_first_reference_run_then_reports_against_it(tmp_path):
    repo = _toy_repo(tmp_path / "repo")
    git, printed = partial(toy_git, repo), []
    reference = tmp_path / "reference"
    manifest = update_script.update(reference, git=git, run=_write_run, out=printed.append)
    assert printed == [f"No reference run at {reference}; writing the first one."]
    assert manifest["commit"] == git("rev-parse", "HEAD").strip()
    assert manifest["command"][0] == update_script.recorded_command(
        update_script.pipeline_command(tmp_path), tmp_path)[0]
    assert manifest["command"][-1] == update_script.RUN_TOKEN

    update_script.update(reference, git=git, run=_write_run, out=printed.append)
    assert printed[1].endswith(f"{len(PARTS)} parts compared: all match.")


def test_the_pipeline_command_runs_the_sub_volume_with_frozen_defaults(tmp_path):
    command = update_script.pipeline_command(tmp_path / "out")
    assert "--config" not in command
    assert command[command.index("--placed-input") + 1] == str(
        update_script.subvolume.SUBVOLUME_PATH)
    token = update_script.REPO_TOKEN

    def repo_path(path):
        return f"{token}/{path.relative_to(update_script.REPO).as_posix()}"

    assert update_script.recorded_command(command, tmp_path / "out") == [
        repo_path(update_script.PIPELINE), "--placed-input",
        repo_path(update_script.subvolume.SUBVOLUME_PATH),
        "--specimen", update_script.subvolume.SPECIMEN_ID,
        "--output-dir", update_script.RUN_TOKEN]


# --- The flow network and the perfusion field ------------------------------------------------

def _flow(**arrays):
    return {"flow": _flow_network(**arrays)}


def test_the_vtk_files_are_stored_as_written(reference):
    for name in (VESSELS_FLOW_VTP_NAME, PERFUSION_VTI_NAME):
        assert (reference / name).read_bytes() == (reference.parent / "made_from" / name).read_bytes()


def test_a_solve_array_passes_inside_the_network_band_and_fails_outside(tmp_path, reference):
    # Flow 60 against a largest |flow_abs| of 100: band 60 * rtol + 100 * rtol = 160 * rtol.
    inside = _compare(tmp_path / "a", reference,
                      **_flow(flow_abs=np.array([100.0, 60.0 + 159 * NETWORK_RTOL, 40.0])))
    assert FLOW_NETWORK not in _differing(inside)
    outside = _compare(tmp_path / "b", reference,
                       **_flow(flow_abs=np.array([100.0, 60.0 + 161 * NETWORK_RTOL, 40.0])))
    assert _differing(outside)[FLOW_NETWORK] == (
        "cell array 'flow_abs': 1 of 3 values differ, max abs 0.000161, max rel 2.68e-06")


def test_a_geometry_array_in_the_flow_network_keeps_the_geometry_band(tmp_path, reference):
    # Diameter 7 against a largest of 9: the geometry band is 16e-9, far inside the network one.
    inside = np.array([7.0 + 15 * GEOMETRY_RTOL, 8.0, 9.0])
    assert FLOW_NETWORK not in _differing(
        _compare(tmp_path / "a", reference, **_flow(assigned_diameter_um=inside)))
    outside = np.array([7.0 + 17 * GEOMETRY_RTOL, 8.0, 9.0])
    detail = _differing(_compare(tmp_path / "b", reference, **_flow(assigned_diameter_um=outside)))
    assert detail[FLOW_NETWORK].startswith("cell array 'assigned_diameter_um': 1 of 3 values")


def test_the_perfusion_field_passes_inside_its_band_and_fails_outside(tmp_path, reference):
    # PO2 50 against a largest of 85: band 135 * rtol.
    po2 = np.linspace(50.0, 85.0, 8)
    po2[0] = 50.0 + 134 * PERFUSION_RTOL
    inside = _compare(tmp_path / "a", reference, perfusion=_perfusion_field(po2=po2))
    assert PERFUSION_FIELD not in _differing(inside)
    po2[0] = 50.0 + 136 * PERFUSION_RTOL
    detail = _differing(_compare(tmp_path / "b", reference, perfusion=_perfusion_field(po2=po2)))
    assert detail[PERFUSION_FIELD].startswith("cell array 'PO2_mmHg': 1 of 8 values differ")


def test_an_array_on_one_side_only_is_a_difference(tmp_path, reference):
    detail = _differing(_compare(tmp_path, reference, **_flow(pressure_drop=np.zeros(3))))
    assert detail[FLOW_NETWORK] == "cell array 'pressure_drop' only in NEW"
    grid = _perfusion_field()
    grid.cell_data.remove("pH")
    detail = _differing(_compare(tmp_path / "b", reference, perfusion=grid))
    assert detail[PERFUSION_FIELD] == "cell array 'pH' only in REFERENCE"


def test_a_missing_flow_network_is_a_difference_in_every_part_read_from_it(tmp_path, reference):
    run = _write_run(tmp_path / "run")
    (run / VESSELS_FLOW_VTP_NAME).unlink()
    detail = _differing(compare_with_reference(run, reference))
    assert set(detail) == {FLOW_NETWORK, RHEOLOGY_STATUS, SUMMARY}
    assert detail[FLOW_NETWORK] == f"missing in NEW: {VESSELS_FLOW_VTP_NAME}"


# --- The rheology status and the perfusion tags ----------------------------------------------

def _with_field(mesh, **values):
    for name, value in values.items():
        mesh.field_data[name] = np.array([value])
    return mesh


def test_a_changed_tag_fails_in_its_status_part_only(tmp_path, reference):
    flow = _with_field(_flow_network(), rheology_stop_reason="max_iterations")
    assert _differing(_compare(tmp_path, reference, flow=flow)) == {
        RHEOLOGY_STATUS: "rheology_stop_reason: converged in REFERENCE, max_iterations in NEW"}


def test_the_rheology_iterations_are_exact(tmp_path, reference):
    flow = _with_field(_flow_network(), rheology_iterations=331)
    assert _differing(_compare(tmp_path, reference, flow=flow)) == {
        RHEOLOGY_STATUS: "rheology_iterations: 330 in REFERENCE, 331 in NEW"}


def test_a_rheology_residual_follows_the_network_rule(tmp_path, reference):
    # A lone value is its own largest: band 2 * rtol * 9.9e-7.
    inside = _with_field(_flow_network(), rheology_relative_flow_change=9.9e-7 * (1 + 1.9e-6))
    assert _compare(tmp_path / "a", reference, flow=inside).same
    outside = _with_field(_flow_network(), rheology_relative_flow_change=9.9e-7 * (1 + 2.1e-6))
    assert set(_differing(_compare(tmp_path / "b", reference, flow=outside))) == {
        RHEOLOGY_STATUS}


@pytest.mark.parametrize("iterations, same", [(48, True), (50, True), (47, False), (51, False)])
def test_picard_iterations_pass_at_one_either_way_and_fail_at_two(tmp_path, reference,
                                                                    iterations, same):
    comparison = _compare(tmp_path, reference,
                          perfusion=_perfusion_field(perfusion_picard_iterations=iterations))
    assert comparison.same is same
    if not same:
        assert _differing(comparison) == {
            PERFUSION_TAGS: f"perfusion_picard_iterations: 49 in REFERENCE, {iterations} in NEW"}


@pytest.mark.parametrize("tag, value", [("perfusion_tier", 2), ("perfusion_converged", 0),
                                        ("perfusion_solver", "solve_coupled_1d3d_perfusion"),
                                        ("perfusion_M_max", 0.05 * (1 + 1e-12))])
def test_the_tier_solver_converged_flag_and_settings_tags_are_exact(tmp_path, reference, tag,
                                                                    value):
    detail = _differing(_compare(tmp_path, reference, perfusion=_perfusion_field(**{tag: value})))
    assert set(detail) == {PERFUSION_TAGS} and detail[PERFUSION_TAGS].startswith(tag + ":")


def test_the_final_residual_follows_the_perfusion_rule(tmp_path, reference):
    inside = _perfusion_field(perfusion_final_residual=5.8e-6 * (1 + 1.9e-4))
    assert _compare(tmp_path / "a", reference, perfusion=inside).same
    outside = _perfusion_field(perfusion_final_residual=5.8e-6 * (1 + 2.1e-4))
    assert set(_differing(_compare(tmp_path / "b", reference, perfusion=outside))) == {
        PERFUSION_TAGS}


def test_a_tag_on_one_side_only_is_a_difference(tmp_path, reference):
    grid = _perfusion_field()
    grid.field_data.remove("perfusion_converged")
    assert _differing(_compare(tmp_path, reference, perfusion=grid)) == {
        PERFUSION_TAGS: "perfusion_converged only in REFERENCE"}


# --- The summary scalars ---------------------------------------------------------------------

def test_the_summary_holds_the_dashboards_numbers_at_full_precision(reference):
    summary = json.loads((reference / SUMMARY_NAME).read_text())
    assert summary[FLOW_NETWORK]["flow_signed"] == {
        "mean": 80.0 / 3, "median": 40.0, "min": -60.0, "max": 100.0}
    assert summary[FLOW_NETWORK]["hematocrit"]["mean"] == pytest.approx(0.425)  # NaN dropped
    assert set(summary[FLOW_NETWORK]) == {"assigned_diameter_um", "flow_abs", "flow_signed",
                                          "hematocrit"}
    assert summary[PERFUSION_FIELD]["PO2_mmHg"] == {
        "mean": 67.5, "median": 67.5, "min": 50.0, "max": 85.0}


def test_a_summary_number_follows_its_sources_rule_with_the_floor_from_its_largest_value(
        tmp_path, reference):
    # The median of flow_signed is 40 and the array's largest |value| is 100, so it may move by
    # 140 * NETWORK_RTOL. Moving the 40 moves the median and the mean (by a third as much).
    def signed(median):
        return _flow(flow_signed=np.array([100.0, -60.0, median]))

    inside = _compare(tmp_path / "a", reference, **signed(40.0 + 139 * NETWORK_RTOL))
    assert SUMMARY not in _differing(inside)
    outside = _differing(_compare(tmp_path / "b", reference, **signed(40.0 + 141 * NETWORK_RTOL)))
    assert outside[SUMMARY].startswith("flow network flow_signed: median, max abs 0.000141")


def test_a_summary_number_of_the_perfusion_field_uses_the_perfusion_rule(tmp_path, reference):
    # The max, 85, is its array's largest value: band 2 * PERFUSION_RTOL * 85.
    po2 = np.linspace(50.0, 85.0, 8)
    po2[-1] = 85.0 * (1 + 1.9e-4)
    assert SUMMARY not in _differing(_compare(tmp_path / "a", reference,
                                              perfusion=_perfusion_field(po2=po2)))
    po2[-1] = 85.0 * (1 + 2.1e-4)
    detail = _differing(_compare(tmp_path / "b", reference, perfusion=_perfusion_field(po2=po2)))
    assert detail[SUMMARY].startswith("perfusion field PO2_mmHg: max, max abs 0.0179, max rel 0.00021")


def test_a_summary_section_on_one_side_only_is_a_difference(tmp_path, reference):
    summary_path = reference / SUMMARY_NAME
    summary = json.loads(summary_path.read_text())
    del summary[PERFUSION_FIELD]
    summary_path.write_text(json.dumps(summary))
    detail = _differing(_compare(tmp_path, reference))
    assert detail == {SUMMARY: f"{PERFUSION_FIELD} only in NEW"}


def test_a_summary_statistic_on_one_side_only_is_a_difference(tmp_path, reference):
    summary_path = reference / SUMMARY_NAME
    summary = json.loads(summary_path.read_text())
    summary[FLOW_NETWORK]["flow_signed"]["p95"] = 95.0
    summary_path.write_text(json.dumps(summary))
    detail = _differing(_compare(tmp_path, reference))
    assert detail == {SUMMARY: f"{FLOW_NETWORK} flow_signed p95 only in REFERENCE"}


def test_every_tag_a_rule_names_is_in_the_committed_reference_run():
    """A tag the pipeline renamed would leave its rule naming nothing, and the renamed tag
    compared exactly; the committed reference run is what the pipeline writes today."""
    import ImageLynx.reference_run as reference_run
    stored = update_script.REFERENCE_RUN_DIR
    flow = reference_run._field_data(stored / VESSELS_FLOW_VTP_NAME)
    tags = reference_run._field_data(stored / PERFUSION_VTI_NAME)
    assert set(reference_run.RHEOLOGY_RTOL) <= set(flow)
    assert set(reference_run.PERFUSION_TAG_RTOL) | set(reference_run.PICARD_ITERATION_STEPS) <= \
        set(tags)
    arrays = pv.read(stored / VESSELS_FLOW_VTP_NAME).cell_data.keys()
    assert set(reference_run.NETWORK_SOLVE_ARRAYS) <= set(arrays)
