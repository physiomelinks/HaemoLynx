"""Writing a reference run and comparing a run folder with it, on tiny hand-made run folders.

Each test writes a run folder as the pipeline lays it out (``per_edge_morphometry.csv`` beside
one ``*_cache`` folder with the mask, skeleton and pickled graph: a three-edge graph on a 4^3
grid), stores it as a reference run, then changes one thing on one side. The update script's
refusal of a dirty tree is tested on a throwaway git repo.
"""
import csv
import json
import math
import pickle
from functools import partial

import networkx as nx
import numpy as np
import pytest

import update_benchmark_reference_run as update_script
from conftest import toy_git
from ImageLynx.batch_outputs import (EDGE_TABLE_NAME, GRAPH_NAME, SKELETON_NAME,
                                     VESSEL_MASK_NAME)
from ImageLynx.reference_run import (
    EDGE_TABLE, GEOMETRY_RTOL, GRAPH, GRAPH_JSON_NAME, MANIFEST_NAME, MASKS_NAME, PARTS,
    SKELETON, VESSEL_MASK, compare_with_reference, format_report, write_reference_run,
)

EDGES = [(0, 1, 0), (1, 2, 0), (1, 2, 1)]
SHAPE = (4, 4, 4)
COMMAND = ["--placed-input", "<REPO>/sub.h5"]


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


def _write_run(run_dir, graph=None, columns=None, mask=None, skeleton=None):
    """A run folder laid out as the pipeline writes one; returns it."""
    cache = run_dir / "sub_cache"
    cache.mkdir(parents=True)
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
        [MASKS_NAME, GRAPH_JSON_NAME, EDGE_TABLE_NAME, MANIFEST_NAME])
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
    (run / "sub_cache" / SKELETON_NAME).unlink()
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
    for path in (run / "sub_cache").iterdir():
        path.unlink()
    (run / "sub_cache").rmdir()
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
    assert ("  edge table   DIFFERENT: length_um: 1 of 3 rows differ, max abs 2e-05, "
            "max rel 1e-06: [(1, 2, 0)]") in report
    assert "  graph        same" in report
    assert report.splitlines()[-1] == "4 parts compared: 1 differ."


def test_a_different_environment_is_a_note_not_a_difference(tmp_path, reference):
    manifest = json.loads((reference / MANIFEST_NAME).read_text())
    manifest["python"] = "3.10.0"
    manifest["packages"]["numpy"] = "0.0"
    (reference / MANIFEST_NAME).write_text(json.dumps(manifest))
    comparison = _compare(tmp_path, reference)
    assert comparison.same
    report = format_report(comparison)
    assert "  environment  python 3.10.0 in REFERENCE" in report
    assert "  environment  numpy 0.0 in REFERENCE" in report


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
    assert printed[1].endswith("4 parts compared: all match.")


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
