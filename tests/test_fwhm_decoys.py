"""FWHM on a speckled image: the fewest accepted samples an edge needs, and
the decoy check that asks how often FWHM gives vessel-free tissue a width."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.haemodynamics import automated
from haemolynx.haemodynamics.fwhm_decoys import decoy_centrelines, fwhm_decoy_check
from test_raw_section_diameter import VOXEL, _graph, _render  # noqa: E402


def _vessel_at_one_sample_only(seed=41):
    """A 5 um vessel present only within 3 um of an edge's middle sample, the
    edge sampled every 7 um: exactly one sample can see it. Empty elsewhere --
    left with a faint shot-noise floor (0.2 counts), FWHM fitted a 2.3 um
    width to that too."""
    raw, mask, lines = _render([(5.0, (0, 0, 1), (0, 0, 0))], seed=seed)
    x = np.arange(raw.shape[2]) * VOXEL[2]
    away = np.abs(x - raw.shape[2] * VOXEL[2] / 2) > 3.0
    raw[:, :, away] = 0.0
    mask[:, :, away] = False
    return raw, mask, lines


def _fwhm(graph, raw, mask, **kwargs):
    return automated.measure_edge_diameters_fwhm_from_raw_tiff(
        graph, raw_tiff_path="unused", raw_volume=raw, vessel_mask=mask, voxel_size_zyx=VOXEL,
        sample_spacing_along_edge_um=7.0, transverse_profile_step_um=0.25,
        transverse_half_extent_um=6.0, **kwargs,
    )


# --- the fewest accepted samples ------------------------------------------------------


def test_one_accepted_sample_is_a_width_when_one_is_enough():
    raw, mask, lines = _vessel_at_one_sample_only()
    graph = _graph(lines, [5.0])

    _fwhm(graph, raw, mask, min_accepted_samples=1)

    assert len(graph[0][1][0]["fwhm_diameter_samples_um"]) == 1
    assert graph[0][1][0]["fwhm_status"] == "measured"


def test_one_accepted_sample_is_not_a_width_when_two_are_required():
    """A speck is one place, a lumen is all along the vessel: on a real
    speckled stack FWHM gave 28% of vessel-free decoys a width from one
    sample, 8% once two were required."""
    raw, mask, lines = _vessel_at_one_sample_only()
    graph = _graph(lines, [5.0])

    summary = _fwhm(graph, raw, mask, min_accepted_samples=2)

    data = graph[0][1][0]
    assert "fwhm_diameter_um" not in data
    assert data["fwhm_status"] == "failed:too_few_samples"
    assert summary["edges_skipped"][0][3] == "too_few_samples"


def test_the_pipeline_requires_two_samples_by_default():
    from haemolynx.pipeline import default_schema

    assert default_schema()["fwhm_min_accepted_samples"].default == 2


# --- where decoys go ----------------------------------------------------------------


def test_a_decoy_is_moved_clear_of_every_vessel():
    raw, mask, lines = _render([(5.0, (0, 0, 1), (0, -6.0, 0))], shape=(36, 60, 44), seed=43)
    graph = _graph(lines, [5.0])

    placed = decoy_centrelines(graph, [(0, 1, 0)], mask, VOXEL, rng=np.random.default_rng(0))

    ((edge, line, guide),) = placed
    assert edge == (0, 1, 0) and guide == 5.0
    vessel_voxels = np.argwhere(mask) * np.asarray(VOXEL)
    nearest = min(float(np.min(np.linalg.norm(vessel_voxels - p, axis=1))) for p in line)
    assert nearest >= 3.75  # three quarters of its 5 um guide
    shift = line - np.asarray(lines[0])
    assert np.allclose(shift, shift[0])  # moved, not bent
    assert abs(shift[0] @ np.array([0.0, 0.0, 1.0])) < 1e-9  # sideways, not along it


def test_an_edge_with_no_room_beside_it_gets_no_decoy():
    raw, mask, lines = _render([(5.0, (0, 0, 1), (0, 0, 0))], seed=47)
    graph = _graph(lines, [5.0])
    everywhere = np.ones_like(mask)

    assert decoy_centrelines(graph, [(0, 1, 0)], everywhere, VOXEL, rng=np.random.default_rng(0)) == []


# --- the check --------------------------------------------------------------------


def _measured_network(widths, seed=53):
    """Edges with FWHM widths already on them, well apart in an empty mask."""
    lines = [np.array([[20.0, 20.0 + 25.0 * i, x] for x in np.linspace(10.0, 40.0, 16)]) for i in range(len(widths))]
    graph = _graph(lines, [5.0] * len(widths))
    for i, width in enumerate(widths):
        graph[2 * i][2 * i + 1][0]["fwhm_diameter_um"] = width
    mask = np.zeros((40, 40 + 26 * len(widths), 50), dtype=bool)
    graph.graph["image_voxel_size_zyx"] = VOXEL
    return graph, mask


def _fake_measure(widths_for_decoys):
    """A stand-in for the run's FWHM: gives the i-th decoy the i-th width (None: fails)."""
    seen = []

    def measure(probe):
        seen.append(probe)
        for i, (_u, _v, data) in enumerate(probe.edges(data=True)):
            width = widths_for_decoys[i % len(widths_for_decoys)]
            if width is not None:
                data["fwhm_diameter_um"] = width
    return measure, seen


def test_the_check_reports_how_often_and_how_wide_fwhm_reads_texture():
    graph, mask = _measured_network([2.5, 3.0, 8.0, 12.0, 3.5, 20.0, 2.8, 9.0, 4.0, 6.0])
    measure, seen = _fake_measure([2.0, None, 3.0, 2.5, None, 4.0, 3.5, None, 2.2, 3.1])

    report = fwhm_decoy_check(graph, measure, vessel_mask=mask, voxel_size_zyx=VOXEL)

    assert report["decoys"] == 10
    assert report["decoys_measured"] == 7
    assert report["false_positive_rate"] == pytest.approx(0.7)
    low, high = report["speck_width_range_um"]
    assert low == pytest.approx(np.percentile([2.0, 3.0, 2.5, 4.0, 3.5, 2.2, 3.1], 5))
    assert high == pytest.approx(np.percentile([2.0, 3.0, 2.5, 4.0, 3.5, 2.2, 3.1], 95))
    flagged = {graph[u][v][k]["fwhm_diameter_um"] for u, v, k in graph.edges(keys=True)
               if graph[u][v][k]["fwhm_in_speck_width_range"]}
    assert flagged == {2.5, 3.0, 3.5, 2.8}
    assert report["measured_in_speck_width_range"] == pytest.approx(0.4)
    assert graph.graph["fwhm_decoy_check"] == report
    (probe,) = seen
    assert probe.graph["image_voxel_size_zyx"] == VOXEL  # measured at the run's own voxel size


def test_the_check_changes_no_diameter():
    graph, mask = _measured_network([2.5, 3.0, 8.0, 12.0, 3.5, 20.0])
    before = {e: graph.edges[e]["fwhm_diameter_um"] for e in graph.edges(keys=True)}
    measure, _seen = _fake_measure([3.0])

    fwhm_decoy_check(graph, measure, vessel_mask=mask, voxel_size_zyx=VOXEL)

    assert {e: graph.edges[e]["fwhm_diameter_um"] for e in graph.edges(keys=True)} == before


def test_too_few_decoy_widths_set_no_speck_range():
    graph, mask = _measured_network([2.5, 3.0, 8.0, 12.0])
    measure, _seen = _fake_measure([3.0, None, None, None])

    report = fwhm_decoy_check(graph, measure, vessel_mask=mask, voxel_size_zyx=VOXEL)

    assert report["decoys_measured"] == 1
    assert "speck_width_range_um" not in report
    assert not any("fwhm_in_speck_width_range" in d for _u, _v, d in graph.edges(data=True))


def test_the_check_needs_a_mask_and_something_fwhm_measured():
    graph, mask = _measured_network([3.0])
    measure, seen = _fake_measure([3.0])

    assert fwhm_decoy_check(graph, measure, vessel_mask=None, voxel_size_zyx=VOXEL)["skipped"]
    for u, v, k in graph.edges(keys=True):
        del graph[u][v][k]["fwhm_diameter_um"]
    assert fwhm_decoy_check(graph, measure, vessel_mask=mask, voxel_size_zyx=VOXEL)["skipped"]
    assert seen == []


def test_the_sample_size_caps_the_decoys():
    graph, mask = _measured_network([3.0] * 8)
    measure, _seen = _fake_measure([None])

    report = fwhm_decoy_check(graph, measure, vessel_mask=mask, voxel_size_zyx=VOXEL, sample_size=3)

    assert report["decoys"] == 3


# --- in a run ---------------------------------------------------------------------


def test_a_run_checks_its_fwhm_against_decoys_with_its_own_settings(monkeypatch):
    from haemolynx.haemodynamics import HaemodynamicsApplyConfig, apply, assign_edge_diameters

    graph, mask = _measured_network([3.0, 5.0, 7.0])
    for _u, _v, data in graph.edges(data=True):
        data["branch_order"] = "B01"
    calls = []

    def fake_fwhm(G, config, raw_volume=None, vessel_mask=None, **_kwargs):
        calls.append((G, config, vessel_mask))
        if G is graph:
            return {"edges_measured": 3}
        for _u, _v, data in G.edges(data=True):
            data["fwhm_diameter_um"] = 3.0
        return {}

    monkeypatch.setattr(apply, "_measure_fwhm_diameters", fake_fwhm)
    monkeypatch.setattr(apply, "load_fwhm_raw_volume", lambda _config: np.zeros((2, 2, 2), np.float32))
    config = HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": {"B01": 6.0}},
        fwhm={"use_fwhm_edge_diameters": True, "do_fwhm_measurement": True,
              "fwhm_decoy_check": True, "fwhm_decoy_check_sample_size": 2},
    )

    _graph_out, summary, _raw = assign_edge_diameters(graph, config, mask_volume=mask)

    assert [c[0] is graph for c in calls] == [True, False]
    probe, probe_config, probe_mask = calls[1]
    assert probe_config is config and probe_mask is not None
    assert probe.number_of_edges() == 2
    assert summary["fwhm_decoy_check"]["false_positive_rate"] == 1.0


def test_a_run_without_the_check_does_not_make_decoys(monkeypatch):
    from haemolynx.haemodynamics import HaemodynamicsApplyConfig, apply, assign_edge_diameters

    graph, mask = _measured_network([3.0])
    graph[0][1][0]["branch_order"] = "B01"
    calls = []
    monkeypatch.setattr(apply, "_measure_fwhm_diameters", lambda G, *a, **k: calls.append(G) or {})
    monkeypatch.setattr(apply, "load_fwhm_raw_volume", lambda _config: np.zeros((2, 2, 2), np.float32))
    config = HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": {"B01": 6.0}},
        fwhm={"use_fwhm_edge_diameters": True, "do_fwhm_measurement": True},
    )

    _graph_out, summary, _raw = assign_edge_diameters(graph, config, mask_volume=mask)

    assert len(calls) == 1
    assert "fwhm_decoy_check" not in summary


# --- a vessel running along z ---------------------------------------------------------


def test_fwhm_measures_a_vessel_running_along_z():
    """Regression: FWHM capped each line where another part of the same
    centreline came close, judged by the smaller of the 3D and the y-x
    distance. Along a vessel running in z every point further along lies
    straight above or below -- a y-x distance of about nothing -- so every
    line was capped at almost nothing and no z-running vessel was measured."""
    raw, mask, lines = _render([(6.0, (1, 0.05, 0.05), (0, 0, 0))], seed=61)
    graph = _graph(lines, [6.0])

    automated.measure_edge_diameters_fwhm_from_raw_tiff(
        graph, raw_tiff_path="unused", raw_volume=raw, vessel_mask=mask, voxel_size_zyx=VOXEL,
        sample_spacing_along_edge_um=2.0, transverse_profile_step_um=0.25,
        transverse_half_extent_um=6.0, min_accepted_samples=2,
    )

    data = graph[0][1][0]
    assert data["fwhm_status"] == "measured"
    assert data["fwhm_diameter_um"] == pytest.approx(6.0, rel=0.12)
