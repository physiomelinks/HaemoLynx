"""Tests for VTK export and PyVista visualization helpers."""
from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from ImageLynx.visualization import (
    derive_pericyte_points_from_graph,
    graph_to_vtk,
    visualize_vtk_network,
)


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
    G.add_edge(0, 1, voxels=reversed_voxels, branch_order="BO1", resistance=1.0)

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
