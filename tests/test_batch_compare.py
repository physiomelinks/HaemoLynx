"""Comparing two batch runs by content, through the reader.

Each test builds tiny batch runs under two roots in ``tmp_path`` - a three-edge graph, its edge
table, skeleton, vessel mask and ``roi_placement.json`` - and changes one thing on one side.
``place_roi`` is replaced by a fixed small box, since the real one reads the specimen's data.
"""
import csv
import pickle
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

import ImageLynx.batch_outputs as batch_outputs
from ImageLynx import cb_settings
from ImageLynx.batch_compare import PARTS, RootsDoNotMatch, compare_roots, compare_runs
from ImageLynx.batch_outputs import open_batch_run
from ImageLynx.roi_placement import (
    RoiPlacement, centre_to_offsets, roi_record, write_roi_record,
)

h5py = pytest.importorskip("h5py")

SHAPE = (12, 10, 9)          # the specimen's full volume, z y x
SIZE = (4, 6, 5)             # the placed ROI
CENTRE = (6, 4, 5)
COLUMNS = ["u", "v", "key", "length_um", "assigned_diameter_um", "n_centreline_points",
           "diameter_provenance", "fwhm_diameter_um"]
EDGES = [(0, 1, 0), (1, 2, 0), (1, 2, 1)]


def _rows():
    return {edge: {"u": edge[0], "v": edge[1], "key": edge[2],
                   "length_um": f"{10.0 * (edge[0] + 1):g}",
                   "assigned_diameter_um": f"{6.0 + edge[1] + edge[2]:g}",
                   "n_centreline_points": "7", "diameter_provenance": "measured_edt",
                   "fwhm_diameter_um": ""}
            for edge in EDGES}


def _graph():
    G = nx.MultiGraph(voxel_size=(1.5, 1.5, 1.5))
    for u in (0, 1, 2):
        G.add_node(u, pos=np.array([1.0, 2.0, 0.5 * u]))
    for u, v, key in EDGES:
        G.add_edge(u, v, key=key, length=10.0 * (u + 1), voxels=[(1, 2, u), (1, 3, v)],
                   segment_id=u + key, centreline_smoothing="bspline")
    return G


@pytest.fixture
def specimens(tmp_path, monkeypatch):
    """Two stand-in specimens, their TH exports and a stubbed ``place_roi``."""
    found = {}
    for specimen_id in ("TEST-A", "TEST-B"):
        th_path = tmp_path / f"{specimen_id}_TH.h5"
        glomus = (np.arange(np.prod(SHAPE)) % 256).astype(np.uint8).reshape(SHAPE)
        with h5py.File(th_path, "w") as handle:
            handle.create_dataset("exported_data", data=np.stack([glomus, 255 - glomus], -1))
        found[specimen_id] = SimpleNamespace(specimen_id=specimen_id, shape_zyx=SHAPE,
                                             th_probabilities_path=th_path, batch_run_dir=None)
    placements = {specimen_id: RoiPlacement(
        specimen_id=specimen_id, centre_zyx=CENTRE, size_zyx=SIZE,
        offsets_zyx=centre_to_offsets(CENTRE, SHAPE), peak_slice=CENTRE[0], source="test")
        for specimen_id in found}
    monkeypatch.setattr(batch_outputs, "place_roi",
                        lambda s, size: placements[s.specimen_id]
                        if tuple(size) == tuple(cb_settings.ROI_VOXELS)
                        else pytest.fail(f"place_roi asked for {size}, not the frozen ROI"))
    return SimpleNamespace(by_id=found, placements=placements)


def write_run(run_dir, specimens, specimen_id, graph=None, rows=None, skeleton=None,
              vessel_mask=None, columns=COLUMNS):
    """One consistent batch run, with any part replaced by the caller's version."""
    cache = run_dir / f"{specimen_id}_vessels_ilastik_Probabilities_cache"
    cache.mkdir(parents=True)
    with (cache / batch_outputs.GRAPH_NAME).open("wb") as handle:
        pickle.dump(_graph() if graph is None else graph, handle)
    with (run_dir / batch_outputs.EDGE_TABLE_NAME).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows((_rows() if rows is None else rows).values())
    if skeleton is None:
        skeleton = np.zeros(SIZE, dtype=np.uint8)
        skeleton[2, 3, :] = 1
    np.save(cache / batch_outputs.SKELETON_NAME, skeleton)
    np.save(cache / batch_outputs.VESSEL_MASK_NAME,
            np.ones(SIZE, dtype=bool) if vessel_mask is None else vessel_mask)
    write_roi_record(run_dir, roi_record(specimens.placements[specimen_id], SHAPE, centred=False))
    return run_dir


def _open(specimens, run_dir, specimen_id="TEST-A"):
    return open_batch_run(specimens.by_id[specimen_id], run_dir)


def _pair(tmp_path, specimens, **new_parts):
    old = write_run(tmp_path / "old" / "TEST-A", specimens, "TEST-A")
    new = write_run(tmp_path / "new" / "TEST-A", specimens, "TEST-A", **new_parts)
    return _open(specimens, old), _open(specimens, new)


def _verdicts(comparison):
    return {verdict.part: verdict for verdict in comparison.parts}


# --- compare_runs ----------------------------------------------------------------------------

def test_two_identical_runs_match_in_all_six_parts(tmp_path, specimens):
    comparison = compare_runs(*_pair(tmp_path, specimens))
    assert [verdict.part for verdict in comparison.parts] == [
        "graph", "diameters", "edge table", "skeleton", "vessel mask", "TH mask"]
    assert list(PARTS) == [verdict.part for verdict in comparison.parts]
    assert all(verdict.same for verdict in comparison.parts)
    assert comparison.same


@pytest.mark.parametrize("part, change, expected", [
    ("skeleton", "skeleton", "8 of 120 voxels differ"),       # 5 voxels lost, 3 gained
    ("vessel mask", "vessel_mask", "3 of 120 voxels differ"),
])
def test_a_changed_array_is_reported_with_how_many_voxels_differ(tmp_path, specimens, part,
                                                                 change, expected):
    array = np.zeros(SIZE, dtype=bool) if change == "skeleton" else np.ones(SIZE, dtype=bool)
    array[0, 0, :3] = change == "skeleton"
    verdicts = _verdicts(compare_runs(*_pair(tmp_path, specimens, **{change: array})))
    assert (verdicts[part].same, verdicts[part].detail) == (False, expected)
    assert [p for p, v in verdicts.items() if not v.same] == [part]


def _changed_graph(change):
    G = _graph()
    if change == "node attribute":
        G.nodes[2]["pos"] = np.array([1.0, 2.0, 1.25])
    elif change == "edge attribute":
        G.edges[1, 2, 1]["voxels"] = [(1, 2, 1), (1, 4, 2)]
    elif change == "text attribute":
        G.edges[0, 1, 0]["centreline_smoothing"] = "raw_fallback"
    elif change == "graph attribute":
        G.graph["voxel_size"] = (1.5, 1.5, 1.6)
    elif change == "extra node":
        G.add_node(9, pos=np.zeros(3))
    elif change == "missing attribute":
        del G.edges[0, 1, 0]["segment_id"]
    return G


@pytest.mark.parametrize("change, expected", [
    ("node attribute", "node 2 pos"),
    ("edge attribute", "edge (1, 2, 1) voxels"),
    ("text attribute", "edge (0, 1, 0) centreline_smoothing"),
    ("graph attribute", "graph voxel_size"),
    ("extra node", "1 nodes only in NEW: [9]"),
    ("missing attribute", "edge (0, 1, 0) segment_id"),
])
def test_a_changed_graph_is_reported_by_what_changed(tmp_path, specimens, change, expected):
    verdicts = _verdicts(compare_runs(*_pair(tmp_path, specimens, graph=_changed_graph(change))))
    assert not verdicts["graph"].same
    assert expected in verdicts["graph"].detail
    assert [p for p, v in verdicts.items() if not v.same] == ["graph"]


def _rows_with(edge, column, cell):
    rows = _rows()
    rows[edge][column] = cell
    return rows


def test_a_changed_diameter_is_reported_by_edge_and_size(tmp_path, specimens):
    verdicts = _verdicts(compare_runs(*_pair(
        tmp_path, specimens, rows=_rows_with((1, 2, 1), "assigned_diameter_um", "10.5"))))
    assert (verdicts["diameters"].same, verdicts["diameters"].detail) == (
        False, "1 of 3 edges differ, max abs 1.5, max rel 0.167: [(1, 2, 1)]")
    assert "assigned_diameter_um: 1 of 3 rows differ" in verdicts["edge table"].detail


@pytest.mark.parametrize("column, cell, expected", [
    ("length_um", "20.5", "length_um: 1 of 3 rows differ, max abs 0.5, max rel 0.025: "
                          "[(1, 2, 0)]"),
    ("n_centreline_points", "8", "n_centreline_points: 1 of 3 rows differ"),
    ("diameter_provenance", "fallback", "diameter_provenance: 1 of 3 rows differ: [(1, 2, 0)]"),
    ("fwhm_diameter_um", "3.5", "fwhm_diameter_um: 1 of 3 rows differ: [(1, 2, 0)]"),
])
def test_a_changed_edge_table_cell_is_reported_by_column(tmp_path, specimens, column, cell,
                                                         expected):
    verdicts = _verdicts(compare_runs(*_pair(tmp_path, specimens,
                                             rows=_rows_with((1, 2, 0), column, cell))))
    assert not verdicts["edge table"].same
    assert verdicts["edge table"].detail.startswith(expected)
    assert [p for p, v in verdicts.items() if not v.same] == ["edge table"]


def test_a_column_on_one_side_only_is_reported(tmp_path, specimens):
    verdicts = _verdicts(compare_runs(*_pair(tmp_path, specimens, columns=COLUMNS[:-1])))
    assert verdicts["edge table"].detail == "columns only in OLD: ['fwhm_diameter_um']"


def test_an_edge_on_one_side_only_is_reported_in_the_graph_and_the_edge_table(tmp_path,
                                                                             specimens):
    G = _graph()
    G.remove_edge(1, 2, key=1)
    rows = _rows()
    del rows[(1, 2, 1)]
    verdicts = _verdicts(compare_runs(*_pair(tmp_path, specimens, graph=G, rows=rows)))
    assert "1 edges only in OLD: [(1, 2, 1)]" in verdicts["graph"].detail
    assert "1 rows only in OLD: [(1, 2, 1)]" in verdicts["edge table"].detail
    assert "1 edges only in OLD: [(1, 2, 1)]" in verdicts["diameters"].detail


def test_a_cell_that_is_not_a_number_is_refused_not_compared(tmp_path, specimens):
    with pytest.raises(ValueError, match="non-finite length_um"):
        compare_runs(*_pair(tmp_path, specimens, rows=_rows_with((0, 1, 0), "length_um", "")))


def _nudged():
    """Every float a hair (one part in 10^12) bigger: lengths in the table and the graph."""
    rows = _rows()
    for row in rows.values():
        row["length_um"] = repr(float(row["length_um"]) * (1 + 1e-12))
    G = _graph()
    for *_, data in G.edges(keys=True, data=True):
        data["length"] *= 1 + 1e-12
    return {"rows": rows, "graph": G}


def test_floats_must_match_exactly_by_default(tmp_path, specimens):
    verdicts = _verdicts(compare_runs(*_pair(tmp_path, specimens, **_nudged())))
    assert not verdicts["graph"].same and not verdicts["edge table"].same


def test_a_relative_tolerance_lets_floats_drift_within_it(tmp_path, specimens):
    assert compare_runs(*_pair(tmp_path, specimens, **_nudged()), rtol=1e-9).same


def test_a_relative_tolerance_never_loosens_text(tmp_path, specimens):
    rows = _rows_with((1, 2, 0), "diameter_provenance", "measured_edt ")
    assert not compare_runs(*_pair(tmp_path, specimens, rows=rows), rtol=0.5).same


def test_graphs_of_different_types_differ(tmp_path, specimens):
    G = nx.MultiDiGraph(voxel_size=(1.5, 1.5, 1.5))
    G.add_nodes_from(_graph().nodes(data=True))
    G.add_edges_from(_graph().edges(keys=True, data=True))
    verdicts = _verdicts(compare_runs(*_pair(tmp_path, specimens, graph=G)))
    assert verdicts["graph"].detail == "graph type MultiGraph in OLD, MultiDiGraph in NEW"


def test_a_relative_tolerance_never_loosens_an_integer_column(tmp_path, specimens):
    rows = _rows_with((1, 2, 0), "n_centreline_points", "8")
    assert not compare_runs(*_pair(tmp_path, specimens, rows=rows), rtol=0.5).same


@pytest.mark.parametrize("value", [np.float32(2.0), {"seed": np.array([1.0, 2.0])}],
                         ids=["numpy float", "dict holding an array"])
def test_numpy_floats_and_nested_dicts_are_compared_by_value(tmp_path, specimens, value):
    graphs = []
    for scale in (1.0, 1 + 1e-6):
        G = _graph()
        G.graph["extra"] = (value * np.float32(scale) if isinstance(value, np.floating)
                            else {"seed": value["seed"] * scale})
        graphs.append(G)
    old = write_run(tmp_path / "old" / "TEST-A", specimens, "TEST-A", graph=graphs[0])
    new = write_run(tmp_path / "new" / "TEST-A", specimens, "TEST-A", graph=graphs[1])
    runs = _open(specimens, old), _open(specimens, new)
    assert _verdicts(compare_runs(*runs))["graph"].detail == "graph extra"
    assert compare_runs(*runs, rtol=1e-5).same


def test_every_differing_edge_table_column_is_listed(tmp_path, specimens):
    extra = [f"c{n:02d}" for n in range(12)]
    rows = _rows()
    for row in rows.values():
        row.update({column: "1" for column in extra})
    old = write_run(tmp_path / "old" / "TEST-A", specimens, "TEST-A", rows=rows,
                    columns=COLUMNS + extra)
    for row in rows.values():
        row.update({column: "2" for column in extra})
    new = write_run(tmp_path / "new" / "TEST-A", specimens, "TEST-A", rows=rows,
                    columns=COLUMNS + extra)
    detail = _verdicts(compare_runs(_open(specimens, old), _open(specimens, new)))[
        "edge table"].detail
    assert [column for column in extra if f"{column}: 3 of 3 rows differ" in detail] == extra


# --- compare_roots ---------------------------------------------------------------------------

def _roots(tmp_path, specimens, old_runs, new_runs):
    """Write each ``(folder, specimen_id)`` run under an old and a new root."""
    for root, runs in (("old", old_runs), ("new", new_runs)):
        for folder, specimen_id in runs:
            write_run(tmp_path / root / folder / specimen_id, specimens, specimen_id)
    return tmp_path / "old", tmp_path / "new"


def _summary(comparisons):
    return [(c.label, c.same, c.only_in) for c in comparisons]


def test_batch_roots_are_compared_specimen_by_specimen(tmp_path, specimens):
    runs = [("", "TEST-A"), ("", "TEST-B")]
    old, new = _roots(tmp_path, specimens, runs, runs)
    comparisons = compare_roots(old, new, list(specimens.by_id.values()))
    assert _summary(comparisons) == [("TEST-A", True, None), ("TEST-B", True, None)]


def test_a_specimen_run_on_one_side_only_differs(tmp_path, specimens):
    old, new = _roots(tmp_path, specimens, [("", "TEST-A")], [("", "TEST-A"), ("", "TEST-B")])
    comparisons = compare_roots(old, new, list(specimens.by_id.values()))
    assert _summary(comparisons) == [("TEST-A", True, None), ("TEST-B", False, "NEW")]
    assert comparisons[1].parts == ()


def test_a_specimen_in_neither_root_is_refused_by_the_reader(tmp_path, specimens):
    old, new = _roots(tmp_path, specimens, [("", "TEST-A")], [("", "TEST-A")])
    with pytest.raises(FileNotFoundError, match="TEST-B"):
        compare_roots(old, new, list(specimens.by_id.values()))


def test_sensitivity_roots_are_compared_threshold_by_threshold(tmp_path, specimens):
    old, new = _roots(tmp_path, specimens,
                      [("t0.93", "TEST-A"), ("t0.97", "TEST-A")], [("t0.93", "TEST-A")])
    comparisons = compare_roots(old, new, [specimens.by_id["TEST-A"]])
    assert _summary(comparisons) == [("t0.93/TEST-A", True, None), ("t0.97", False, "OLD")]


def test_a_folder_that_is_not_a_threshold_does_not_make_a_sensitivity_root(tmp_path,
                                                                           specimens):
    runs = [("", "TEST-A")]
    old, new = _roots(tmp_path, specimens, runs, runs)
    (old / "tmp").mkdir()
    assert _summary(compare_roots(old, new, [specimens.by_id["TEST-A"]])) == [
        ("TEST-A", True, None)]


def test_a_sensitivity_root_against_a_batch_root_is_refused(tmp_path, specimens):
    old, new = _roots(tmp_path, specimens, [("t0.93", "TEST-A")], [("", "TEST-A")])
    with pytest.raises(RootsDoNotMatch, match="t<threshold>"):
        compare_roots(old, new, [specimens.by_id["TEST-A"]])


# --- The command line ------------------------------------------------------------------------

@pytest.fixture
def cli(specimens, monkeypatch):
    import cb_compare_batch_runs
    monkeypatch.setattr(cb_compare_batch_runs, "SPECIMENS", tuple(specimens.by_id.values()))
    return cb_compare_batch_runs


def test_the_command_exits_0_and_says_same_when_every_run_matches(tmp_path, specimens, cli,
                                                                  capsys):
    runs = [("", "TEST-A"), ("", "TEST-B")]
    old, new = _roots(tmp_path, specimens, runs, runs)
    assert cli.main([str(old), str(new)]) == 0
    out = capsys.readouterr().out
    assert "TEST-A\n  graph        same\n  diameters    same\n" in out
    assert "  TH mask      same (same source, same box)\n" in out
    assert out.endswith("2 runs compared: all match.\n")


def test_the_command_exits_1_and_says_what_differs(tmp_path, specimens, cli, capsys):
    old, new = _roots(tmp_path, specimens, [("", "TEST-A")], [("", "TEST-B")])
    rows = _rows_with((1, 2, 0), "length_um", "20.5")
    write_run(new / "TEST-A", specimens, "TEST-A", rows=rows)
    assert cli.main([str(old), str(new)]) == 1
    out = capsys.readouterr().out
    assert "  edge table   DIFFERENT: length_um: 1 of 3 rows differ" in out
    assert "TEST-B: DIFFERENT, only in NEW\n" in out
    assert out.endswith("2 runs compared: 2 differ.\n")


def test_the_command_compares_only_the_specimens_asked_for(tmp_path, specimens, cli, capsys):
    old, new = _roots(tmp_path, specimens, [("", "TEST-A")], [("", "TEST-A")])
    assert cli.main([str(old), str(new), "--specimen", "TEST-A"]) == 0
    assert "TEST-B" not in capsys.readouterr().out


def test_the_command_passes_its_tolerance_on(tmp_path, specimens, cli):
    old, new = _roots(tmp_path, specimens, [("", "TEST-A")], [])
    write_run(new / "TEST-A", specimens, "TEST-A", **_nudged())
    args = [str(old), str(new), "--specimen", "TEST-A"]
    assert cli.main(args) == 1
    assert cli.main(args + ["--rtol", "1e-9"]) == 0


@pytest.mark.parametrize("args", [["--specimen", "NOT-A-SPECIMEN"], ["--rtol", "-1"],
                                  ["--rtol", "nan"]])
def test_a_bad_option_exits_2(tmp_path, specimens, cli, args):
    old, new = _roots(tmp_path, specimens, [("", "TEST-A")], [("", "TEST-A")])
    with pytest.raises(SystemExit) as exit_:
        cli.main([str(old), str(new), *args])
    assert exit_.value.code == 2


def test_a_sensitivity_root_against_a_batch_root_exits_2(tmp_path, specimens, cli, capsys):
    old, new = _roots(tmp_path, specimens, [("", "TEST-A")], [("t0.93", "TEST-A")])
    with pytest.raises(SystemExit) as exit_:
        cli.main([str(old), str(new)])
    assert exit_.value.code == 2
    assert "not the same kind of root" in capsys.readouterr().err


def test_a_run_the_reader_refuses_exits_3_not_1(tmp_path, specimens, cli, capsys):
    old, new = _roots(tmp_path, specimens, [("", "TEST-A")], [("", "TEST-A")])
    assert cli.main([str(old), str(new)]) == 3     # TEST-B is in neither root
    err = capsys.readouterr().err
    assert "FileNotFoundError" in err and "TEST-B" in err
