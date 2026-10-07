"""Log and CLI text from the 2026-09-29 re-run (package I: items 5, 11, 15, 19, 20, 26).

None of these change a number. Each one made a run's output say something that was not true,
or hid a fallback in a stats dict, so the text is pinned here.
"""
import logging

import networkx as nx
import numpy as np
import pytest


def test_hysteresis_log_prints_the_seed_unrounded(caplog):
    """The 0.999 seed printed as 'high=1.00', which reads as a cut at p >= 1 (item 5)."""
    from ImageLynx.preprocessing.image import hysteresis_threshold

    with caplog.at_level(logging.INFO, logger="ImageLynx.preprocessing.image"):
        hysteresis_threshold(np.zeros((4, 4, 4)), low=0.95, high=0.999)
    assert "low=0.95, high=0.999" in caplog.text
    assert "high=1.00" not in caplog.text


@pytest.mark.parametrize("name", ["compute_communities_summary",
                                  "compute_weighted_communities_summary"])
def test_community_fallback_is_logged(caplog, name):
    """Above max_nodes_exact the community count is a component count; say so (item 11)."""
    from ImageLynx.statistics import stats

    G = nx.path_graph(6)
    nx.set_edge_attributes(G, 1.0, "length")
    fn = getattr(stats, name)
    kwargs = {"source_attr": "length"} if "weighted" in name else {}

    with caplog.at_level(logging.WARNING, logger="ImageLynx.statistics.stats"):
        small = fn(G, max_nodes_exact=10, **kwargs)
    assert "connected components" not in caplog.text
    assert small["Community Method"].startswith("greedy_modularity")

    with caplog.at_level(logging.WARNING, logger="ImageLynx.statistics.stats"):
        large = fn(G, max_nodes_exact=3, **kwargs)
    assert large["Community Method"] == "connected_components_fallback"
    assert "6 nodes > max_nodes_exact=3" in caplog.text


_COMMUNITY_KEYS = ("Community Count", "Largest Community Size", "Mean Community Size")
_COMPONENT_KEYS = ("Connected Component Count", "Largest Component Size", "Mean Component Size")


@pytest.mark.parametrize("name", ["compute_communities_summary",
                                  "compute_weighted_communities_summary"])
def test_community_fallback_reports_components_not_communities(name):
    """The fallback counts components, so it must not label them communities (package S)."""
    from ImageLynx.statistics import stats

    G = nx.disjoint_union(nx.path_graph(4), nx.path_graph(2))
    nx.set_edge_attributes(G, 1.0, "length")
    fn = getattr(stats, name)
    kwargs = {"source_attr": "length"} if "weighted" in name else {}

    small = fn(G, max_nodes_exact=10, **kwargs)
    assert all(k in small for k in _COMMUNITY_KEYS)
    assert not any(k in small for k in _COMPONENT_KEYS)

    large = fn(G, max_nodes_exact=3, **kwargs)
    assert not any(k in large for k in _COMMUNITY_KEYS)
    assert large["Connected Component Count"] == 2
    assert large["Largest Component Size"] == 4
    assert large["Mean Component Size"] == 3.0


def _edge_graph(voxels):
    G = nx.MultiGraph()
    G.add_node("a", pos=np.array(voxels[0], float))
    G.add_node("b", pos=np.array(voxels[-1], float))
    G.add_edge("a", "b", voxels=voxels, length=float(len(voxels)))
    return G


def test_centreline_fallback_is_logged(caplog):
    """A raw-centreline fallback was visible only in the stats (item 11)."""
    from ImageLynx.graph._helpers import smooth_graph_edge_centerlines_continuous

    wobble = [[0.0, float(i), 2.0 * ((i % 2) - 0.5)] for i in range(20)]
    skeleton = np.zeros((3, 45, 9), dtype=bool)
    skeleton[0, 0, 0] = True          # no support along the edge, so every candidate fails

    G = _edge_graph(wobble)
    with caplog.at_level(logging.WARNING, logger="ImageLynx.graph._helpers"):
        stats = smooth_graph_edge_centerlines_continuous(G, skeleton, voxel_size=(1.0, 1.0, 1.0))
    assert stats["fallback_edges"] == 1
    assert stats["unchanged_edges"] == 0
    assert G["a"]["b"][0]["centreline_smoothing"] == "raw_fallback"
    assert "1 of 1 edges kept the raw centreline" in caplog.text


def test_no_fallback_no_warning(caplog):
    from ImageLynx.graph._helpers import smooth_graph_edge_centerlines_continuous

    G = _edge_graph([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])       # too short to spline
    skeleton = np.ones((1, 2, 1), dtype=bool)
    with caplog.at_level(logging.WARNING, logger="ImageLynx.graph._helpers"):
        stats = smooth_graph_edge_centerlines_continuous(G, skeleton, voxel_size=(1.0, 1.0, 1.0))
    assert stats["fallback_edges"] == 0
    assert "raw centreline" not in caplog.text


def test_tier3_first_iteration_diagnostics_are_debug_level():
    """They were logger.info with a 'DEBUG' prefix, so they filled every pipeline.log (item 11)."""
    import inspect
    from ImageLynx.haemodynamics import perfusion

    source = inspect.getsource(perfusion)
    assert "DEBUG Iteration" not in source
    assert 'logger.debug(f"Iteration 0:' in source


def test_batch_docstring_does_not_claim_a_group_comparison():
    """--stage run only runs the pipeline; the comparison is in the H1 scripts (item 15)."""
    import cb_h1_batch

    assert "compare the groups" not in cb_h1_batch.__doc__


# --- cb_h1_vtk.py (items 19, 20) ---------------------------------------------------------

pv = pytest.importorskip("pyvista")


def _box_grid(n=10):
    grid = pv.ImageData(dimensions=(n, n, n), spacing=(1.0, 1.0, 1.0))
    grid["vessel_mask"] = np.ones(grid.n_points, dtype=np.uint8)
    return grid


@pytest.mark.parametrize("lo, hi, reaches", [(-0.2, 9.2, True), (2.0, 7.0, False)])
def test_inset_note_only_when_the_network_stops_short(lo, hi, reaches):
    """A negative inset means the network overhangs every face; it did say 'does not reach'."""
    import cb_h1_vtk

    vessels = pv.Line((lo, lo, lo), (hi, hi, hi))
    notes = cb_h1_vtk.verify(None, vessels, None, _box_grid())
    inset = next(n for n in notes if n.startswith("inset"))
    assert ("network reaches every face" in inset) is reaches
    assert ("does not reach that face" in inset) is (not reaches)


def _stub_export(monkeypatch, tmp_path):
    import cb_h1_vtk

    class _Spec:
        specimen_id, group = "WKY-A", "WKY"

    out = tmp_path / "cb_h1_paraview"
    monkeypatch.setattr(cb_h1_vtk, "OUTPUT", out)
    monkeypatch.setattr(cb_h1_vtk, "SPECIMENS", [_Spec()])
    monkeypatch.setattr(cb_h1_vtk, "read_edges", lambda s: {})
    monkeypatch.setattr(cb_h1_vtk, "check_output_roi", lambda *a, **k: None)
    monkeypatch.setattr(cb_h1_vtk, "enrich_vessels", lambda s, e, r: pv.Line((0, 0, 0), (9, 9, 9)))
    monkeypatch.setattr(cb_h1_vtk, "build_nodes", lambda s, e, r: None)
    monkeypatch.setattr(cb_h1_vtk, "build_skeleton", lambda s, r: None)
    monkeypatch.setattr(cb_h1_vtk, "build_surface", lambda s, r: (None, _box_grid()))
    monkeypatch.setattr(cb_h1_vtk, "stamp", lambda mesh, s: mesh)
    return cb_h1_vtk, out


def test_h1_verify_writes_nothing(monkeypatch, tmp_path, capsys):
    """Same contract as cb_h2_vtk.py --verify: check, print, write no file (item 19)."""
    cb_h1_vtk, out = _stub_export(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["cb_h1_vtk.py", "--verify"])
    cb_h1_vtk.main()
    assert not out.exists()
    printed = capsys.readouterr().out
    assert "overhang beyond mask" in printed
    assert "nothing written" in printed


def test_h1_export_still_writes_without_verify(monkeypatch, tmp_path):
    cb_h1_vtk, out = _stub_export(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["cb_h1_vtk.py"])
    cb_h1_vtk.main()
    assert (out / "WKY-A_vessels.vtp").exists()
    assert (out / "WKY-A_mask.vti").exists()
    assert (out / "export_summary.json").exists()


# --- cb_h2_vtk.py (item 26) ---------------------------------------------------------------

@pytest.mark.parametrize("outside_pct, noted", [(0.0, False), (2.5, True)])
def test_h2_outside_grid_note_only_when_some_glomus_is_outside(monkeypatch, capsys,
                                                               outside_pct, noted):
    """The note printed at 0.0% whenever the grid box was short of the mask box."""
    import cb_h2_vtk

    class _Spec:
        specimen_id, group = "WKY-A", "WKY"

    check = {"ok": True, "penetrating_midpoint_inside_pct": 88.0,
             "non_penetrating_midpoint_inside_pct": 1.0, "transposed_control_pct": 20.0,
             "perfusion_grid_contains_glomus_volume": False,
             "glomus_outside_grid_pct": outside_pct}
    monkeypatch.setattr(cb_h2_vtk, "SPECIMENS", [_Spec()])
    monkeypatch.setattr(cb_h2_vtk, "solve", lambda *a, **k: {})
    monkeypatch.setattr(cb_h2_vtk, "verify", lambda s, state: check)
    monkeypatch.setattr("sys.argv", ["cb_h2_vtk.py", "--verify"])
    cb_h2_vtk.main()
    assert ("outside the perfusion grid" in capsys.readouterr().out) is noted
