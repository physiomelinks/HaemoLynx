"""Tests for VTK export and PyVista visualization helpers.

The exporter is the end of the pipeline: whatever it writes is what a user
measures in ParaView. Two things therefore have to be exact — the geometry
(node positions and edge polylines are physical ``(z, y, x)`` microns and must
be written through unchanged, so an anisotropic stack keeps its aspect ratio)
and the per-edge arrays (``resistance``/``conductance`` must land on the line
cell belonging to that edge, not a neighbouring one).
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from haemolynx.visualization import (
    derive_pericyte_points_from_graph,
    graph_to_vtk,
    visualize_vtk_network,
)


def _line_cell_point_ids(mesh) -> list[list[int]]:
    """Split a PolyData `lines` array into one id list per line cell."""
    lines = np.asarray(mesh.lines, dtype=int)
    cells: list[list[int]] = []
    cursor = 0
    while cursor < len(lines):
        count = int(lines[cursor])
        cells.append([int(i) for i in lines[cursor + 1 : cursor + 1 + count]])
        cursor += count + 1
    return cells


def _straight_edge_graph(length_um: float, step_um: float = 1.0) -> nx.MultiGraph:
    """One edge running along array axis 0, sampled every `step_um` microns."""
    voxels = [
        (float(z), 0.0, 0.0)
        for z in np.arange(0.0, length_um + step_um / 2.0, step_um)
    ]
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([length_um, 0.0, 0.0]))
    G.add_edge(0, 1, voxels=voxels, length=length_um, branch_order="BO1")
    return G


def test_derive_pericyte_points_from_graph(simple_graph):
    out = derive_pericyte_points_from_graph(
        simple_graph,
        constriction_spacing=1.0,
        constriction_length=0.4,
    )
    assert "points" in out
    assert out["points"].ndim == 2
    assert out["points"].shape[1] == 3
    assert len(out["points"]) > 0
    assert len(out["edge_u"]) == len(out["points"])


def test_graph_to_vtk_exports_files(simple_graph, tmp_path):
    prefix = tmp_path / "network"
    out = graph_to_vtk(
        simple_graph,
        prefix,
        constriction_spacing=1.0,
        constriction_length=0.4,
    )

    vessels_path = Path(out["vessels_path"])
    pericytes_path = Path(out["pericytes_path"])
    nodes_path = Path(out["nodes_path"])

    assert vessels_path.exists()
    assert pericytes_path.exists()
    assert nodes_path.exists()
    assert out["vessel_line_count"] > 0
    assert out["node_count"] == 3
    assert out["pericyte_count"] >= 0

    vessels = pv.read(vessels_path)
    pericytes = pv.read(pericytes_path)
    nodes = pv.read(nodes_path)
    assert vessels.n_cells > 0
    assert "branch_order" in vessels.cell_data
    assert pericytes.n_points == out["pericyte_count"]
    assert "node_id" in nodes.point_data


def test_graph_to_vtk_uses_shared_junction_points(simple_graph, tmp_path):
    prefix = tmp_path / "connected_network"
    out = graph_to_vtk(
        simple_graph,
        prefix,
        constriction_spacing=1.0,
        constriction_length=0.4,
    )
    vessels = pv.read(out["vessels_path"])
    conn = vessels.connectivity()
    region_ids = conn.cell_data["RegionId"]
    # For simple_graph (0-1-2 chain), two line cells should share junction at node 1.
    assert len(np.unique(region_ids)) == 1


def test_visualize_vtk_network_smoke(simple_graph, tmp_path):
    prefix = tmp_path / "network_for_view"
    out = graph_to_vtk(
        simple_graph,
        prefix,
        constriction_spacing=1.0,
        constriction_length=0.4,
    )
    plotter = visualize_vtk_network(
        out["vessels_path"],
        out["pericytes_path"],
        out["nodes_path"],
        show_nodes=True,
        show=False,
    )
    assert plotter is not None


def test_graph_to_vtk_orients_reversed_voxel_paths(tmp_path):
    import networkx as nx

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([10.0, 0.0, 0.0]))
    reversed_voxels = [(10.0, 0.0, 0.0), (9.0, 0.0, 0.0), (8.0, 0.0, 0.0), (7.0, 0.0, 0.0),
                       (6.0, 0.0, 0.0), (5.0, 0.0, 0.0), (4.0, 0.0, 0.0), (3.0, 0.0, 0.0),
                       (2.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    G.add_edge(0, 1, voxels=reversed_voxels, branch_order="BO1", weight=1.0)

    out = graph_to_vtk(G, tmp_path / "reversed_path")
    vessels = pv.read(out["vessels_path"])

    line = vessels.lines
    npts = int(line[0])
    ids = line[1 : 1 + npts]
    pts = vessels.points[ids]
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)

    # If path orientation is wrong before endpoint snapping, exporter creates
    # long endpoint jumps (e.g. 0->9 and 1->10). Guard against that.
    assert float(np.max(seg_lengths)) <= 2.0


# --- geometry is written through in physical (z, y, x) microns --------------


def test_exported_points_keep_the_node_positions_component_for_component(tmp_path):
    """Node coordinates go out as-is; reordering them to (x, y, z) would mirror the model."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([14.0, 1.5, 1.2]))
    G.add_node(1, pos=np.array([28.0, 1.5, 1.2]))
    G.add_edge(0, 1, voxels=[(14.0, 1.5, 1.2), (21.0, 1.5, 1.2), (28.0, 1.5, 1.2)])

    out = graph_to_vtk(G, tmp_path / "geom")
    nodes = pv.read(out["nodes_path"])

    exported = {tuple(np.round(p, 6)) for p in np.asarray(nodes.points)}
    assert (14.0, 1.5, 1.2) in exported
    assert (28.0, 1.5, 1.2) in exported


def test_anisotropic_stack_exports_with_its_true_physical_extent(tmp_path):
    """A vessel 8 voxels long in z spans 7 * 2.0 um, not 7 * 0.4 um.

    Positions reach the exporter already in microns, so a z/x voxel-size swap
    upstream shows up here as a bounding box with the wrong aspect ratio. All
    three spacings differ, so no swap can produce the expected numbers.
    """
    pytest.importorskip("skan")
    from haemolynx.graph import build_graph_from_skeleton

    voxel_size_zyx = (2.0, 0.5, 0.4)
    skeleton = np.zeros((12, 7, 7), dtype=bool)
    skeleton[2:10, 3, 3] = True

    G = build_graph_from_skeleton(
        skeleton,
        voxel_size=voxel_size_zyx,
        min_stub_length=0.0,
        cluster_collapse_distance=0.0,
    )
    out = graph_to_vtk(G, tmp_path / "anisotropic")
    vessels = pv.read(out["vessels_path"])

    z_min, z_max, y_min, y_max, x_min, x_max = vessels.bounds
    assert (z_max - z_min) == pytest.approx(7 * 2.0, rel=1e-9)
    assert y_min == pytest.approx(3 * 0.5) and y_max == pytest.approx(3 * 0.5)
    assert x_min == pytest.approx(3 * 0.4) and x_max == pytest.approx(3 * 0.4)


def test_polyline_endpoints_are_snapped_onto_the_node_positions(tmp_path):
    """Skeleton voxels stop short of the node; unsnapped ends leave gaps at junctions."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([10.0, 0.0, 0.0]))
    G.add_edge(0, 1, voxels=[(0.3, 0.0, 0.0), (5.0, 0.0, 0.0), (9.7, 0.0, 0.0)])

    out = graph_to_vtk(G, tmp_path / "snapped")
    vessels = pv.read(out["vessels_path"])
    (ids,) = _line_cell_point_ids(vessels)
    points = np.asarray(vessels.points)[ids]

    assert points[0] == pytest.approx([0.0, 0.0, 0.0])
    assert points[-1] == pytest.approx([10.0, 0.0, 0.0])


def test_edges_meeting_at_a_node_share_one_exported_point(tmp_path):
    """Duplicating the junction point would export a topologically disconnected tree."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([5.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([5.0, 5.0, 0.0]))
    G.add_edge(0, 1, voxels=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
    G.add_edge(1, 2, voxels=[(5.0, 0.0, 0.0), (5.0, 5.0, 0.0)])

    out = graph_to_vtk(G, tmp_path / "junction")
    vessels = pv.read(out["vessels_path"])

    assert out["vessel_line_count"] == 2
    assert vessels.n_points == 3


def test_point_merge_decimals_controls_how_close_counts_as_the_same_point(tmp_path):
    """Sub-nanometre coordinate noise must not split a junction into two points."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([5.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([5.0 + 1e-9, 0.0, 0.0]))
    G.add_node(3, pos=np.array([10.0, 0.0, 0.0]))
    G.add_edge(0, 1, voxels=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
    G.add_edge(2, 3, voxels=[(5.0 + 1e-9, 0.0, 0.0), (10.0, 0.0, 0.0)])

    merged = pv.read(graph_to_vtk(G, tmp_path / "merged")["vessels_path"])
    split = pv.read(
        graph_to_vtk(G, tmp_path / "split", point_merge_decimals=12)["vessels_path"]
    )

    assert merged.n_points == 3
    assert split.n_points == 4


# --- per-edge arrays land on the right line cell ---------------------------


def test_resistance_and_conductance_follow_their_own_edge(tmp_path):
    """Cell arrays are positional; a shifted append would attribute the wrong physics."""
    G = nx.MultiGraph()
    for node, pos in enumerate([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)]):
        G.add_node(node, pos=np.array(pos))
    G.add_edge(0, 1, voxels=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
               resistance=1.0e16, conductance=1.0e-16, branch_order="BO1")
    G.add_edge(1, 2, voxels=[(5.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
               resistance=4.0e16, conductance=2.5e-17, branch_order="BO2")

    vessels = pv.read(graph_to_vtk(G, tmp_path / "physics")["vessels_path"])

    by_edge = {
        (int(u), int(v)): index
        for index, (u, v) in enumerate(
            zip(vessels.cell_data["edge_u"], vessels.cell_data["edge_v"])
        )
    }
    assert vessels.cell_data["resistance"][by_edge[(0, 1)]] == pytest.approx(1.0e16)
    assert vessels.cell_data["resistance"][by_edge[(1, 2)]] == pytest.approx(4.0e16)
    assert vessels.cell_data["conductance"][by_edge[(1, 2)]] == pytest.approx(2.5e-17)
    assert vessels.cell_data["branch_order"][by_edge[(1, 2)]] == "BO2"


def test_edges_without_haemodynamics_export_nan_rather_than_zero(tmp_path):
    """Zero resistance reads as a short circuit; NaN reads as 'not solved'."""
    G = _straight_edge_graph(10.0)
    vessels = pv.read(graph_to_vtk(G, tmp_path / "unsolved")["vessels_path"])

    assert np.isnan(vessels.cell_data["resistance"]).all()
    assert np.isnan(vessels.cell_data["conductance"]).all()


def test_flow_arrays_are_omitted_until_a_flow_solve_has_run(tmp_path):
    """An all-NaN pressure field in ParaView looks like a solve that produced garbage."""
    G = _straight_edge_graph(10.0)
    vessels = pv.read(graph_to_vtk(G, tmp_path / "no_flow")["vessels_path"])

    for name in ("pressure_u", "pressure_v", "pressure_drop", "flow_signed", "flow_abs"):
        assert name not in vessels.cell_data


def test_flow_arrays_appear_once_any_edge_carries_a_flow(tmp_path):
    G = _straight_edge_graph(10.0)
    G.add_node(2, pos=np.array([20.0, 0.0, 0.0]))
    G.add_edge(1, 2, voxels=[(10.0, 0.0, 0.0), (20.0, 0.0, 0.0)])
    for _u, _v, data in G.edges(data=True):
        if _u == 0:
            data["flow_abs"] = 3.5e-15
            data["pressure_drop"] = 120.0

    vessels = pv.read(graph_to_vtk(G, tmp_path / "with_flow")["vessels_path"])

    flow = vessels.cell_data["flow_abs"]
    assert np.nanmax(flow) == pytest.approx(3.5e-15)
    # The unsolved edge stays NaN instead of being filled with a fake zero.
    assert int(np.isnan(flow).sum()) == 1


# --- degenerate inputs are dropped, not exported as broken cells ------------


def test_an_edge_with_no_geometry_at_all_is_skipped(tmp_path):
    """Neither voxels nor node positions means there is nothing to draw."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([5.0, 0.0, 0.0]))
    G.add_node(2)  # no 'pos'
    G.add_node(3)  # no 'pos'
    G.add_edge(0, 1, voxels=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
    G.add_edge(2, 3)

    out = graph_to_vtk(G, tmp_path / "partial")

    assert out["vessel_line_count"] == 1
    assert out["node_count"] == 2


def test_a_zero_length_edge_is_not_exported_as_a_degenerate_line(tmp_path):
    """Both endpoints on one point collapses to a single id; a 1-point line is invalid."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([2.0, 2.0, 2.0]))
    G.add_node(1, pos=np.array([2.0, 2.0, 2.0]))
    G.add_edge(0, 1, voxels=[(2.0, 2.0, 2.0), (2.0, 2.0, 2.0)])

    out = graph_to_vtk(G, tmp_path / "degenerate")
    vessels = pv.read(out["vessels_path"])

    assert out["vessel_line_count"] == 0
    assert vessels.n_lines == 0


def test_export_creates_a_missing_output_directory(tmp_path, simple_graph):
    out = graph_to_vtk(simple_graph, tmp_path / "nested" / "deeper" / "net")
    assert Path(out["vessels_path"]).is_file()
    assert Path(out["vessels_path"]).parent == tmp_path / "nested" / "deeper"


def test_a_plain_graph_exports_the_same_geometry_as_a_multigraph(tmp_path):
    """Callers pass nx.Graph in the tutorial; the non-multigraph branch must match."""
    multi = _straight_edge_graph(10.0)
    plain = nx.Graph()
    plain.add_nodes_from(multi.nodes(data=True))
    for u, v, data in multi.edges(data=True):
        plain.add_edge(u, v, **data)

    multi_out = graph_to_vtk(multi, tmp_path / "multi")
    plain_out = graph_to_vtk(plain, tmp_path / "plain")

    assert multi_out["vessel_line_count"] == plain_out["vessel_line_count"] == 1
    assert np.allclose(
        pv.read(multi_out["vessels_path"]).points,
        pv.read(plain_out["vessels_path"]).points,
    )


# --- pericyte placement -----------------------------------------------------


def test_pericyte_centres_sit_at_half_a_constriction_then_every_spacing(tmp_path):
    """Positions are the model's constriction sites, so the arithmetic must be exact."""
    G = _straight_edge_graph(100.0)

    out = derive_pericyte_points_from_graph(
        G, constriction_spacing=30.0, constriction_length=40.0
    )

    # First centre at 40/2 = 20 um, then 50 and 80; 110 overruns the 100 um edge.
    assert np.allclose(out["points"][:, 0], [20.0, 50.0, 80.0])
    assert np.allclose(out["points"][:, 1:], 0.0)
    assert list(out["branch_order"]) == ["BO1", "BO1", "BO1"]
    assert list(out["edge_u"]) == [0, 0, 0]
    assert list(out["edge_v"]) == [1, 1, 1]


def test_an_edge_shorter_than_half_a_constriction_gets_no_pericyte():
    """Placing one anyway would put a constriction outside the vessel it belongs to."""
    G = _straight_edge_graph(10.0)

    out = derive_pericyte_points_from_graph(
        G, constriction_spacing=100.0, constriction_length=40.0
    )

    assert out["points"].shape == (0, 3)


def test_pericyte_spacing_is_measured_along_the_path_not_end_to_end():
    """A tortuous vessel carries more pericytes than its straight-line distance implies."""
    zig = [(0.0, 0.0, 0.0)]
    for step in range(1, 21):
        zig.append((float(step), 5.0 if step % 2 else 0.0, 0.0))
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array(zig[0]))
    G.add_node(1, pos=np.array(zig[-1]))
    G.add_edge(0, 1, voxels=zig)

    path_length = float(
        np.sum(np.linalg.norm(np.diff(np.asarray(zig), axis=0), axis=1))
    )
    out = derive_pericyte_points_from_graph(
        G, constriction_spacing=10.0, constriction_length=4.0
    )

    assert path_length > 100.0  # ~102 um of path across only 20 um of z
    expected = int((path_length - 2.0) // 10.0) + 1
    assert len(out["points"]) == expected


@pytest.mark.parametrize(
    "spacing,length", [(0.0, 40.0), (-1.0, 40.0), (100.0, 0.0), (100.0, -5.0)]
)
def test_non_positive_constriction_geometry_is_rejected(simple_graph, spacing, length):
    with pytest.raises(ValueError, match="must be > 0"):
        derive_pericyte_points_from_graph(
            simple_graph, constriction_spacing=spacing, constriction_length=length
        )


def test_pericyte_count_in_the_summary_matches_the_written_file(tmp_path):
    G = _straight_edge_graph(100.0)
    out = graph_to_vtk(
        G, tmp_path / "counted", constriction_spacing=30.0, constriction_length=40.0
    )
    pericytes = pv.read(out["pericytes_path"])

    assert out["pericyte_count"] == 3
    assert pericytes.n_points == 3
    assert np.allclose(np.sort(pericytes.points[:, 0]), [20.0, 50.0, 80.0])
