"""The fixture tier's collectors on fabricated run results, and the edge
cases' volumes and how a failure is classified (`robust.6a`)."""
from __future__ import annotations

import numpy as np
import pytest

from release_review import edge_cases, fixtures
from release_review.ladder import FAIL, PASS


def _result(status="finished", graph=None, error=None, **extra):
    return {"status": status, "seconds": 3.0, "graph": graph or {}, "error": error, **extra}


NERVE = _result(graph={
    "edges": 46, "length_um": 2130.8, "edges_with_diameter": 46, "diameter_sources": {"edt_mask": 40, "table": 6},
    "edges_with_flow": 46, "nodes_with_pressure": 40, "solved_um": 2130.8, "solved_fraction": 1.0,
})


def test_the_level_two_checks_read_the_nerve_crop():
    assert fixtures.graph_2a(NERVE).status == PASS
    assert fixtures.diameter_2a(NERVE).status == PASS
    assert fixtures.flow_2a(NERVE).status == PASS
    partly = _result(graph={**NERVE["graph"], "edges_with_diameter": 45})
    assert fixtures.diameter_2a(partly).status == FAIL
    unsolved = _result(graph={**NERVE["graph"], "solved_fraction": None, "edges_with_flow": 0})
    assert fixtures.flow_2a(unsolved).status == FAIL
    failed = _result("failed", error={"type": "ValueError", "message": "no inlets"})
    assert fixtures.graph_2a(failed).status == FAIL and "no inlets" in fixtures.graph_2a(failed).evidence
    assert fixtures.graph_2a(None).status == FAIL


def test_inputs_need_both_the_tiff_and_the_h5_to_run():
    assert fixtures.input_2a({"seven_vessel": _result(), "bundled_h5": _result()}).status == PASS
    assert fixtures.input_2a({"seven_vessel": _result(), "bundled_h5": _result("failed", error={})}).status == FAIL


def test_the_statistics_csv_and_the_exports_written():
    files = ["x_statistics.csv", "x_branch_statistics.csv", "run_vessels.vtp", "x_graph.pkl"]
    assert fixtures.stats_2a(_result(), files).status == PASS
    assert fixtures.stats_2a(_result(), ["x_branch_statistics.csv"]).status == FAIL
    assert fixtures.export_2a(_result(), files).status == PASS
    assert fixtures.export_2a(_result(), ["x_graph.pkl"]).status == FAIL


def test_settings_reload_and_provenance():
    results = {"a": _result(settings_reload_equal=True), "b": _result(settings_reload_equal=None)}
    assert fixtures.repro_2a(results).status == FAIL
    results["b"]["settings_reload_equal"] = True
    assert fixtures.repro_2a(results).status == PASS
    manifest = {"settings": {"a": 1}, "haemolynx_version": "0.1.0", "inputs": [{"checksum": "sha256:ab"}], "git": None}
    assert fixtures.repro_6a({"a": manifest}, "7f7e699c").status == PASS
    missing = fixtures.repro_6a({"a": {**manifest, "inputs": [{"checksum": None}]}, "b": None}, "7f7e699c")
    assert missing.status == FAIL and "a: no input checksums" in missing.evidence and "b: no manifest" in missing.evidence


def test_identity_counts_every_difference_but_a_plots():
    plot = {"file": "plots/x.html", "detail": "bytes differ (a plot)", "counted": False}
    graph = {"file": "x_graph.pkl", "detail": "3 edges' attributes differ: length", "counted": True}
    assert fixtures.repro_6c([plot], _result()).status == PASS
    assert fixtures.repro_6c([plot, graph], _result()).status == FAIL
    assert fixtures.repro_6c(None, None) is None


def test_the_worker_count_check_needs_a_pool_to_have_started():
    with_workers = _result(worker_pools_started=2)
    assert fixtures.repro_8b([], with_workers, _result()).status == PASS
    idle = fixtures.repro_8b([], _result(worker_pools_started=0), _result())
    assert idle.status == FAIL and "did not test" in idle.note


def test_every_export_must_open():
    readbacks = {"seven": [{"file": "a.csv", "opened": True, "reader": "pandas 3", "entry": "x", "error": ""}]}
    assert fixtures.export_8b(readbacks).status == PASS
    readbacks["nerve"] = [{"file": "b.vtp", "opened": False, "reader": "", "entry": None, "error": "bad"}]
    result = fixtures.export_8b(readbacks)
    assert result.status == FAIL and "nerve/b.vtp" in result.evidence
    assert fixtures.undocumented_outputs(readbacks) == ["b.vtp (no entry)"]


@pytest.mark.parametrize(
    "error_type, raised_in, expected",
    [
        (None, None, "finished"),
        ("ValueError", "haemolynx.pipeline.checks", "named"),
        ("ConfigError", "haemolynx.parsers.schema", "named"),
        ("RuntimeError", "haemolynx.graph.build", "named"),
        ("ValueError", "numpy", "unhandled"),
        ("IndexError", "haemolynx.graph.build", "unhandled"),
        ("KeyError", "haemolynx.pipeline.stages", "unhandled"),
        ("ZeroDivisionError", "haemolynx.statistics.network", "unhandled"),
    ],
)
def test_a_failure_is_named_only_when_haemolynx_raised_it_on_purpose(error_type, raised_in, expected):
    assert edge_cases.classify_failure(error_type, raised_in) == expected


def test_the_suite_passes_only_with_every_case_finished_or_named():
    cases = {
        "empty_mask": _result("failed", error={"type": "ValueError", "raised_in": "haemolynx.graph.build"}),
        "one_vessel": _result(),
    }
    assert fixtures.robust_6a(cases).status == PASS
    cases["two_dimensional"] = _result("failed", error={"type": "IndexError", "raised_in": "haemolynx.graph.build"})
    result = fixtures.robust_6a(cases)
    assert result.status == FAIL and result.value["two_dimensional"] == "unhandled"


def test_the_edge_case_volumes_are_what_they_say():
    volumes = {case.name: case.make() for case in edge_cases.CASES}
    assert set(volumes) == {"empty_mask", "one_vessel", "one_voxel_thick", "two_dimensional", "anisotropic",
                            "touching_border", "no_inlets_or_outlets"}
    assert not volumes["empty_mask"].any()
    assert volumes["two_dimensional"].ndim == 2 and volumes["two_dimensional"].any()
    thin = volumes["one_voxel_thick"]
    assert thin.sum() == thin.shape[2] - 8 and np.count_nonzero(thin.any(axis=(0, 2))) == 1
    border = volumes["touching_border"]
    assert border[:, :, 0].any() and border[:, :, -1].any() and border[:, 0, :].any()
    ring = volumes["no_inlets_or_outlets"]
    faces = [ring[0], ring[-1], ring[:, 0], ring[:, -1], ring[:, :, 0], ring[:, :, -1]]
    assert ring.any() and not any(face.any() for face in faces)
    one = volumes["one_vessel"]
    assert one[:, :, :8].sum() == 0 and one[:, :, -8:].sum() == 0
    assert edge_cases.CASES_BY_NAME["anisotropic"].settings["voxel_size_policy"] == "override"


def test_the_nerve_crops_boundary_boxes_are_the_integration_tests():
    settings = fixtures.nerve_crop_boundaries((63, 696, 941))
    assert settings["inlet_node_volumes"] == [[[0.0, 0.0, 0.0], [62.0, 277.0, 940.0]]]
    assert settings["outlet_node_volumes"] == [[[0.0, 278.0, 0.0], [62.0, 695.0, 940.0]]]
    identity = fixtures.identity_settings()
    assert "skeleton_min_branch_length" not in identity and identity["inlet_node_selection_method"] == "volume"
