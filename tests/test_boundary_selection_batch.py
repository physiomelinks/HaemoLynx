"""``cb_h2_boundary_selection.py`` must measure on the batch graph with the pipeline's rules (package O).

It used to read the H1 ParaView export through ``cb_h2_error_propagation.load``: the VTK frame, the
ParaView edge list, no ROI check, and its own hand-written band and face cuts. These pin the batch
MultiGraph (through ``load_network``, which checks the ROI) and the package's boundary functions.
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import cb_h2_boundary_selection as module
import cb_h2_error_propagation as propagation
from ImageLynx.graph.boundaries import (
    select_boundary_terminal_nodes,
    select_boundary_terminal_nodes_by_face,
)
from ImageLynx.specimens import PROCESSING_VOXEL_UM

SOURCE = Path(module.__file__).read_text()


def _cross_network(tmp_path, drop=None):
    """Centre node with a terminal on both faces of every axis, plus an interior dead end.

    ``drop`` removes one node before the morphometry CSV is written. The edge to node 3, the
    axis-1 inlet, is the widest, so on axis 1 the widest decile carries all the inlet flow.
    """
    G = nx.MultiGraph()
    mid, top = 148.0, 296.0
    G.add_node(0, pos=np.array([mid, mid, mid]))
    faces = [(0.0, mid, mid), (top, mid, mid), (mid, 0.0, mid),
             (mid, top, mid), (mid, mid, 0.0), (mid, mid, top)]
    for i, pos in enumerate(faces, start=1):
        G.add_node(i, pos=np.array(pos))
        G.add_edge(0, i)
    G.add_node(7, pos=np.array([mid, 60.0, 100.0]))
    G.add_node(8, pos=np.array([mid, 30.0, 100.0]))
    G.add_edge(0, 7)
    G.add_edge(0, 7)
    G.add_edge(7, 8)
    if drop is not None:
        G.remove_node(drop)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "per_edge_morphometry.csv"
    lines = ["u,v,key,length_um,assigned_diameter_um"]
    for n, (a, b, k) in enumerate(G.edges(keys=True)):
        lines.append(f"{a},{b},{k},{50.0 + 5 * n},{10.0 if (a, b) == (0, 3) else 4.0}")
    path.write_text("\n".join(lines) + "\n")
    return G, path


@pytest.fixture
def network(tmp_path):
    G, csv_path = _cross_network(tmp_path)
    return (G, *propagation.network_arrays(G, csv_path))


def test_the_script_no_longer_reads_the_paraview_export():
    assert "cb_h1_paraview" not in SOURCE
    assert "pyvista" not in SOURCE
    assert "load_network" in SOURCE
    assert not hasattr(propagation, "load")
    assert "pyvista" not in Path(propagation.__file__).read_text()


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_face_boundaries_are_the_package_face_rule(network, axis):
    G, *_, index = network
    inlets, outlets = module.boundaries(G, index, axis, "face", tol=1.0)
    expected_in, expected_out = select_boundary_terminal_nodes_by_face(
        G, module.ROI, axis=axis, face_tolerance_voxels=1.0, voxel_size=PROCESSING_VOXEL_UM)
    assert list(inlets) == [index[n] for n in expected_in]
    assert list(outlets) == [index[n] for n in expected_out]
    # The interior dead end is never a pressure boundary under the face rule.
    assert index[8] not in set(inlets) | set(outlets)


def test_band_boundaries_are_the_package_band_rule(network):
    G, *_, index = network
    inlets, outlets = module.boundaries(G, index, 1, "band", percent=25.0)
    expected_in, expected_out = select_boundary_terminal_nodes(
        G, module.ROI, edge_percent=25.0, end_percent=25.0, axis=1,
        voxel_size=PROCESSING_VOXEL_UM)
    assert list(inlets) == [index[n] for n in expected_in]
    assert list(outlets) == [index[n] for n in expected_out]


def test_face_sink_makes_every_other_terminal_an_outlet(network):
    G, *_, index = network
    inlets, outlets = module.boundaries(G, index, 1, "face_sink")
    terminals = {index[n] for n, d in G.degree() if d == 1}
    assert set(inlets) | set(outlets) == terminals
    assert not set(inlets) & set(outlets)


def test_an_empty_face_is_a_failed_solve_not_a_fallback(tmp_path):
    G, _ = _cross_network(tmp_path)
    G.remove_node(4)  # the only terminal on the high face of axis 1
    csv_path = _cross_network(tmp_path / "rebuilt", drop=4)[1]
    trimmed = (G, *propagation.network_arrays(G, csv_path))
    assert module.ratio(trimmed, 1, "face") is None
    assert module.ratio(trimmed, 0, "face") is not None


def test_ratio_is_widest_decile_flow_over_inlet_throughput(network):
    # Face rule: one inlet, whose edge is the whole widest decile, so the ratio is exactly one.
    assert module.ratio(network, 1, "face") == pytest.approx(1.0)
    # Band rule: the interior dead end (node 8) sits in the 25% band and becomes a second inlet,
    # so part of the inlet throughput bypasses the wide edge.
    G, *_, index = network
    assert index[8] in module.boundaries(G, index, 1, "band")[0]
    assert 0.0 < module.ratio(network, 1, "band") < 1.0
    # On axis 0 the wide edge is a side branch to an outlet, so it carries part of the flow only.
    assert 0.0 < module.ratio(network, 0, "face_sink") < 1.0


def test_main_reads_each_specimen_once_and_prints_every_section(monkeypatch, network, capsys):
    calls = []

    def fake_load(specimen_id):
        calls.append(specimen_id)
        return network

    monkeypatch.setattr(module, "load_network", fake_load)
    module.main()
    out = capsys.readouterr().out
    assert sorted(calls) == sorted(module.SPECIMENS)
    for header in ("mean axis spread", "=== total sensitivity", "=== which axes are usable",
                   "=== residual sensitivity"):
        assert header in out
    assert "graph axes (z, y, x)" in out


def test_load_network_checks_the_roi_before_reading(monkeypatch, tmp_path):
    monkeypatch.setattr(propagation, "BATCH", tmp_path)
    (tmp_path / "WKY-A").mkdir()

    def refuse(*args, **kwargs):
        raise ValueError("ROI mismatch")

    monkeypatch.setattr(propagation, "check_output_roi", refuse)
    with pytest.raises(ValueError, match="ROI mismatch"):
        propagation.load_network("WKY-A")
