"""Exports read back with third-party tools (`export.8b`), matched to the
data dictionary, and two runs' outputs compared (`repro.6c`, `repro.8b`)."""
from __future__ import annotations

import json
import pickle

import networkx as nx
import numpy as np
import pytest

from release_review import exports


class _Foreign:
    """Stands in for a HaemoLynx class a pickle should not need."""


def _graph(length=10.0):
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    graph.add_node(1, pos=np.array([0.0, 0.0, 10.0]))
    graph.add_edge(0, 1, length=length, diameter_um=float("nan"), voxels=[[0, 0, 0], [0, 0, 10]])
    graph.graph["voxel_size"] = (1.0, 1.0, 1.0)
    return graph


def _pickle(path, value):
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def test_a_graph_loads_with_networkx_and_numpy_only(tmp_path):
    _pickle(tmp_path / "g.pkl", _graph())
    assert exports.load_graph(tmp_path / "g.pkl").number_of_edges() == 1
    graph = _graph()
    graph.graph["model"] = _Foreign()
    _pickle(tmp_path / "foreign.pkl", graph)
    with pytest.raises(exports.ForeignClass, match="_Foreign"):
        exports.load_graph(tmp_path / "foreign.pkl")


def test_the_most_specific_dictionary_entry_names_a_file():
    entries = ["{stem}_statistics.csv", "{stem}_branch_statistics.csv", "*.png, *.html",
               "{name}_statistics.csv, {name}_branch_statistics.csv", "{arteriole|capillary}_sweep.csv"]
    assert exports.match_entry("E14_branch_statistics.csv", entries) == "{stem}_branch_statistics.csv"
    assert exports.match_entry("E14_statistics.csv", entries) == "{stem}_statistics.csv"
    assert exports.match_entry("raw_skeleton.png", entries) == "*.png, *.html"
    assert exports.match_entry("capillary_sweep.csv", entries) == "{arteriole|capillary}_sweep.csv"
    assert exports.match_entry("pericyte_sweep.csv", entries) is None
    assert exports.match_entry("E14_citations.txt", entries) is None


def _run_folder(folder, *, length=10.0, csv_text="a_um,b_pa\n1,2\n"):
    import pyvista

    folder.mkdir(parents=True)
    (folder / "x_statistics.csv").write_text(csv_text)
    _pickle(folder / "x_graph.pkl", _graph(length))
    np.save(folder / "x_skeleton.npy", np.zeros((3, 3, 3), dtype=np.uint8))
    mesh = pyvista.Line((0, 0, 0), (1, 0, 0))
    mesh.cell_data["flow_abs"] = np.array([1.5])
    mesh.save(str(folder / "run_vessels.vtp"))
    manifest = {"status": "finished", "started": "10:00", "stages": [{"stage": "segment", "wall_seconds": 1.0}],
                "settings": {"a": 1}}
    (folder / "x_run_manifest.json").write_text(json.dumps(manifest))
    (folder / "result.json").write_text("{}")  # the scorer's own
    return folder


def test_every_export_is_opened_by_its_reader_and_matched(tmp_path):
    folder = _run_folder(tmp_path / "run")
    (folder / "broken.csv").write_bytes(b"")
    entries = ["{stem}_statistics.csv", "{stem}_graph.pkl", "{stem}_skeleton.npy", "{stem}_vessels.vtp",
               "{stem}_run_manifest.json"]

    def undocumented(entry, names):
        return [name for name in names if name == "b_pa"] if entry == "{stem}_statistics.csv" else []

    found = {each.file: each for each in exports.read_back(folder, entries, undocumented)}
    assert "result.json" not in found
    assert found["x_graph.pkl"].opened and found["x_graph.pkl"].reader.startswith("NetworkX")
    assert found["run_vessels.vtp"].opened and found["run_vessels.vtp"].reader.startswith("VTK")
    assert found["x_skeleton.npy"].details["shape"] == [3, 3, 3]
    assert found["x_statistics.csv"].undocumented == ["b_pa"]
    assert not found["broken.csv"].opened and found["broken.csv"].entry is None
    assert found["x_run_manifest.json"].opened


def test_two_identical_runs_compare_equal_beyond_their_times(tmp_path):
    first = _run_folder(tmp_path / "a")
    second = _run_folder(tmp_path / "b")
    manifest = json.loads((second / "x_run_manifest.json").read_text())
    manifest["started"] = "11:00"
    manifest["stages"][0]["wall_seconds"] = 9.0
    (second / "x_run_manifest.json").write_text(json.dumps(manifest))
    assert exports.compare_folders(first, second) == []


def test_differences_are_found_in_each_kind_of_output(tmp_path):
    first = _run_folder(tmp_path / "a")
    second = _run_folder(tmp_path / "b", length=11.0, csv_text="a_um,b_pa\n1,3\n")
    manifest = json.loads((second / "x_run_manifest.json").read_text())
    manifest["settings"]["a"] = 2
    (second / "x_run_manifest.json").write_text(json.dumps(manifest))
    (second / "extra.png").write_bytes(b"x")
    found = {each.file: each for each in exports.compare_folders(first, second)}
    assert "length" in found["x_graph.pkl"].detail
    assert found["x_statistics.csv"].detail == "bytes differ"
    assert found["x_run_manifest.json"].detail == "differs beyond its times and memory"
    assert found["extra.png"].detail == "only in the second run"


def test_values_compare_with_nan_equal_and_arrays_by_element():
    assert exports.same_value(float("nan"), float("nan"))
    assert exports.same_value(np.array([1.0, np.nan]), np.array([1.0, np.nan]))
    assert not exports.same_value(np.array([1, 2]), np.array([1.0, 2.0]))  # an int array turned float
    assert not exports.same_value([1, 2], (1, 2))
    assert exports.same_value({"a": [np.float64(1.0)]}, {"a": [np.float64(1.0)]})


def test_a_csv_differing_in_its_export_timestamp_alone_is_not_counted(tmp_path):
    stamp = "Section,Metric,Value\nMetadata,Exported At (UTC),{}\nNetwork,Length,{}\n"
    first = _run_folder(tmp_path / "a", csv_text=stamp.format("2026-10-25T15:37:48", 10))
    second = _run_folder(tmp_path / "b", csv_text=stamp.format("2026-10-25T15:37:53", 10))
    (difference,) = exports.compare_folders(first, second)
    assert (difference.detail, difference.counted) == ("the export timestamp only", False)
    third = _run_folder(tmp_path / "c", csv_text=stamp.format("2026-10-25T15:37:53", 11))
    (difference,) = exports.compare_folders(first, third)
    assert (difference.detail, difference.counted) == ("bytes differ", True)


def test_a_plot_that_differs_is_reported_but_not_counted(tmp_path):
    first = _run_folder(tmp_path / "a")
    second = _run_folder(tmp_path / "b")
    (first / "plot.html").write_text("<div id='1'>")
    (second / "plot.html").write_text("<div id='2'>")
    (difference,) = exports.compare_folders(first, second)
    assert difference.file == "plot.html" and not difference.counted
