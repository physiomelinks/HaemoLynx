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


# --- Morphometry on the geometry (#98) ----------------------------------------------------

def _morphometry_graph():
    """Two edges carrying the quantities H1 reports, plus their provenance tags."""
    import networkx as nx
    import numpy as np

    G = nx.MultiGraph()
    for n, pos in {0: (0, 0, 0), 1: (0, 0, 10), 2: (0, 10, 10)}.items():
        G.add_node(n, pos=np.array(pos, dtype=float))
    G.add_edge(0, 1, length=14.0, weight=14.0, branch_order="B01",
               voxels=[[0, 0, 0], [0, 3, 5], [0, 0, 10]],
               assigned_diameter_um=6.37, edt_diameter_um=6.37, fwhm_diameter_um=8.2,
               diameter_provenance="measured_edt", edt_junction_trim="trimmed",
               centreline_smoothing="bspline", resistance=1.0)
    G.add_edge(1, 2, length=10.0, weight=10.0, branch_order="B02",
               voxels=[[0, 0, 10], [0, 5, 10], [0, 10, 10]],
               assigned_diameter_um=4.0, diameter_provenance="synthetic_branch_order",
               edt_junction_trim="untrimmed_too_short",
               centreline_smoothing="raw_fallback", resistance=2.0)
    return G


def test_vessel_export_carries_the_quantities_h1_reports(tmp_path):
    """The geometry was exported without the morphometry measured on it.

    vessels.vtp carried an assigned diameter and a branch order, so the network could not be
    coloured by tortuosity, by the EDT diameter section 1.2 actually reports, or by which
    edges retained a known-biased radius. All of it is already on the graph at export time.
    """
    import pyvista as pv

    from ImageLynx.visualization.vtk_io import graph_to_vtk

    result = graph_to_vtk(_morphometry_graph(), tmp_path / "net")
    mesh = pv.read(result["vessels_path"])

    for column in ("length_um", "tortuosity", "edt_diameter_um", "assigned_diameter_um"):
        assert column in mesh.cell_data, column
    # Tortuosity is path length over endpoint separation; edge 0-1 detours 14 um over 10.
    order = list(zip(mesh.cell_data["edge_u"], mesh.cell_data["edge_v"]))
    detour = order.index((0, 1))
    assert mesh.cell_data["tortuosity"][detour] == pytest.approx(1.4)
    assert mesh.cell_data["edt_diameter_um"][detour] == pytest.approx(6.37)


def test_provenance_tags_are_exported_with_numeric_codes(tmp_path):
    """ParaView cannot colour by a string array, so each tag needs a code beside it."""
    import pyvista as pv

    from ImageLynx.visualization.vtk_io import graph_to_vtk

    result = graph_to_vtk(_morphometry_graph(), tmp_path / "net")
    mesh = pv.read(result["vessels_path"])

    for tag in ("diameter_provenance", "edt_junction_trim", "centreline_smoothing"):
        assert tag in mesh.cell_data, tag
        assert f"{tag}_code" in mesh.cell_data, tag
        codes = mesh.cell_data[f"{tag}_code"]
        assert codes.min() >= 0, f"{tag} produced an unmapped level"
    # A synthetic diameter must not be indistinguishable from a measured one.
    order = list(zip(mesh.cell_data["edge_u"], mesh.cell_data["edge_v"]))
    assert (mesh.cell_data["diameter_provenance_code"][order.index((0, 1))]
            != mesh.cell_data["diameter_provenance_code"][order.index((1, 2))])


def test_nodes_carry_degree_which_is_the_section_1_1_readout(tmp_path):
    """Section 1.1 counts branch points; nodes.vtp carried only an id."""
    import pyvista as pv

    from ImageLynx.visualization.vtk_io import graph_to_vtk

    result = graph_to_vtk(_morphometry_graph(), tmp_path / "net")
    nodes = pv.read(result["nodes_path"])

    assert "degree" in nodes.point_data
    by_id = dict(zip(nodes.point_data["node_id"], nodes.point_data["degree"]))
    assert by_id[0] == 1 and by_id[1] == 2 and by_id[2] == 1
    assert "is_branch_node" in nodes.point_data
