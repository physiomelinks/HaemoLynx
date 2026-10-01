"""Tests for small arteriole/venule mask overlap boundary labelling on graphs."""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest
import numpy as np

# Allow running this test without an editable package install.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx.graph import (
    infer_boundary_nodes_from_small_vessel_masks,
    write_small_vessel_mask_boundary_labelling_3d_html,
)
from browser_diagnostics import open_diagnostic_html

# Checked-in interactive demo (regenerated when this test runs).
DEMO_HTML_PATH = REPO_ROOT / "examples" / "plots" / "small_vessel_mask_boundary_labelling_demo_3d.html"


def _voxel_polyline_samples(
    p0: np.ndarray, p1: np.ndarray, *, count: int = 24
) -> list[tuple[float, float, float]]:
    """Dense (z, y, x) samples along a segment for mask overlap scoring."""
    t = np.linspace(0.0, 1.0, int(count), dtype=float)
    pts = (1.0 - t).reshape(-1, 1) * p0.reshape(1, 3) + t.reshape(-1, 1) * p1.reshape(1, 3)
    uniq = np.unique(np.rint(pts).astype(int), axis=0)
    out: list[tuple[float, float, float]] = [tuple(float(x) for x in row) for row in uniq]
    return out


def build_synthetic_small_vessel_boundary_model() -> tuple[
    nx.MultiGraph,
    np.ndarray,
    np.ndarray,
    dict[str, list[int]],
]:
    """3D synthetic capillary bed: arteriole slab, gap, venule slab, with a side branch.

    Node positions are physical (z, y, x) with voxel_size 1; mask voxels align with indices.
    """
    nz, ny, n_vox_x = 16, 16, 22
    shape = (nz, ny, n_vox_x)
    small_arteriole_mask = np.zeros(shape, dtype=bool)
    small_venule_mask = np.zeros(shape, dtype=bool)
    # Non-overlapping thick slabs (vessel-adjacent tissue) in z,y; separated in x.
    small_arteriole_mask[6:12, 6:12, 0:7] = True
    small_venule_mask[6:12, 6:12, 10:n_vox_x] = True

    G = nx.MultiGraph()
    # Main trunk at y=8, z=8 along +x through art -> capillary gap -> venule.
    trunk_xs = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    for i, x in enumerate(trunk_xs):
        G.add_node(i, pos=np.array([8.0, 8.0, x], dtype=float))

    for a, b in zip(range(len(trunk_xs) - 1), range(1, len(trunk_xs))):
        pa = np.asarray(G.nodes[a]["pos"], dtype=float)
        pb = np.asarray(G.nodes[b]["pos"], dtype=float)
        vox = _voxel_polyline_samples(pa, pb, count=32)
        G.add_edge(a, b, voxels=vox, length=float(np.linalg.norm(pb - pa)), weight=1.0)

    # Side branch from arteriole interior: steps in +z while staying inside art slab.
    br = len(trunk_xs)
    G.add_node(br, pos=np.array([12.0, 8.0, 4.0], dtype=float))
    pa = np.asarray(G.nodes[1]["pos"], dtype=float)
    pb = np.asarray(G.nodes[br]["pos"], dtype=float)
    G.add_edge(
        1,
        br,
        voxels=_voxel_polyline_samples(pa, pb, count=24),
        length=float(np.linalg.norm(pb - pa)),
        weight=1.0,
    )

    expected = {
        "arteriole_boundary_nodes": [2],
        "venule_boundary_nodes": [4],
    }
    return G, small_arteriole_mask, small_venule_mask, expected


@pytest.mark.plotting
def test_infer_boundary_nodes_from_small_vessel_masks(tmp_path):
    """Synthetic 3D graph + masks; boundaries at art/cap and cap/ven transitions."""
    pytest.importorskip("plotly.graph_objects")

    G, art_mask, ven_mask, expected = build_synthetic_small_vessel_boundary_model()

    result = infer_boundary_nodes_from_small_vessel_masks(
        G,
        small_arteriole_mask=art_mask,
        small_venule_mask=ven_mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        minimum_overlap_fraction=0.5,
    )

    assert result["arteriole_boundary_nodes"] == expected["arteriole_boundary_nodes"]
    assert result["venule_boundary_nodes"] == expected["venule_boundary_nodes"]
    assert result["arteriole_edge_count"] >= 2
    assert result["venule_edge_count"] >= 1
    assert G.nodes[2]["mask_vessel_type"] == "arteriole"
    assert G.nodes[4]["mask_vessel_type"] == "venule"

    html_tmp = tmp_path / "small_vessel_mask_boundary_labelling_3d.html"
    ok = write_small_vessel_mask_boundary_labelling_3d_html(
        G,
        small_arteriole_mask=art_mask,
        small_venule_mask=ven_mask,
        arteriole_boundary_nodes=result["arteriole_boundary_nodes"],
        venule_boundary_nodes=result["venule_boundary_nodes"],
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_html_path=html_tmp,
        title="Synthetic small-vessel boundary model (3D)",
    )
    assert ok is True
    assert html_tmp.is_file()
    assert html_tmp.stat().st_size > 1000

    DEMO_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    ok_demo = write_small_vessel_mask_boundary_labelling_3d_html(
        G,
        small_arteriole_mask=art_mask,
        small_venule_mask=ven_mask,
        arteriole_boundary_nodes=result["arteriole_boundary_nodes"],
        venule_boundary_nodes=result["venule_boundary_nodes"],
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_html_path=DEMO_HTML_PATH,
        title="Synthetic small-vessel boundary model (3D)",
    )
    assert ok_demo is True
    assert DEMO_HTML_PATH.is_file()

    open_diagnostic_html(html_tmp)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --- one under-covered edge is not the arteriole ending twice -------------------


def _arteriole_chain(middle_covered_from_x: int):
    """Five nodes along x joined in a chain, then a capillary; the arteriole
    mask covers the first four edges except the start of the middle one."""
    shape = (5, 5, 60)
    G = nx.MultiGraph()
    xs = [0.0, 10.0, 20.0, 30.0, 40.0, 55.0]
    for i, x in enumerate(xs):
        G.add_node(i, pos=np.array([2.0, 2.0, x]))
    for i in range(len(xs) - 1):
        G.add_edge(i, i + 1, voxels=[[2.0, 2.0, float(x)] for x in range(int(xs[i]), int(xs[i + 1]) + 1)])
    arteriole = np.zeros(shape, dtype=bool)
    arteriole[:, :, 0:10] = True
    arteriole[:, :, middle_covered_from_x:41] = True  # edge 1-2 spans x 10..20
    venule = np.zeros(shape, dtype=bool)
    return G, arteriole, venule


def test_an_under_covered_edge_between_two_arteriole_edges_is_arteriole():
    """Regression: 40% of edge 1-2 in the mask fell short of 50%, so the
    arteriole read as meeting the capillary bed at nodes 1 and 2 -- in the
    middle of the vessel -- as well as at node 4, where it really does."""
    G, arteriole, venule = _arteriole_chain(middle_covered_from_x=16)  # x 16..20: 5 of 11 voxels

    result = infer_boundary_nodes_from_small_vessel_masks(
        G, arteriole, venule, voxel_size_zyx=(1.0, 1.0, 1.0), minimum_overlap_fraction=0.5
    )

    assert G.edges[1, 2, 0]["mask_vessel_type"] == "arteriole"
    assert result["gap_filled_edge_count"] == 1
    assert result["arteriole_boundary_nodes"] == [4]


def test_an_uncovered_edge_between_two_arteriole_edges_is_left_a_gap():
    """Hysteresis, not a blanket fill: an edge with too little of itself in
    the mask -- a capillary linking two arteriole branches -- keeps no type."""
    G, arteriole, venule = _arteriole_chain(middle_covered_from_x=21)  # nothing of x 10..20

    result = infer_boundary_nodes_from_small_vessel_masks(
        G, arteriole, venule, voxel_size_zyx=(1.0, 1.0, 1.0), minimum_overlap_fraction=0.5
    )

    assert "mask_vessel_type" not in G.edges[1, 2, 0]
    assert result["gap_filled_edge_count"] == 0
    assert set(result["arteriole_boundary_nodes"]) == {1, 2, 4}


def test_a_simple_graph_is_labelled_the_same_way_as_a_multigraph():
    """The two used to be two copies of the same forty lines."""
    multi, arteriole, venule = _arteriole_chain(middle_covered_from_x=16)
    simple = nx.Graph(multi)

    a = infer_boundary_nodes_from_small_vessel_masks(
        multi, arteriole, venule, voxel_size_zyx=(1.0, 1.0, 1.0)
    )
    b = infer_boundary_nodes_from_small_vessel_masks(
        simple, arteriole, venule, voxel_size_zyx=(1.0, 1.0, 1.0)
    )

    assert a == b


def test_the_overlap_fraction_is_a_fraction_of_the_vessels_length():
    """Regression (audit): the fraction counted centreline vertices, one per
    voxel -- four times as dense per micron along a 0.5 um x as along a 2 um
    z -- so the part of a vessel running in z carried a quarter of its weight.
    Here an L-shaped edge runs 20 um in-plane (inside the arteriole mask) then
    20 um along z (outside it): half its length is in the mask, but 41 of its
    51 vertices were."""
    from haemolynx.graph.automated_vessel_assignment import (
        _edge_sample_points_from_data,
        _sample_overlap_fraction,
    )

    spacing = (2.0, 0.5, 0.5)
    in_plane = [[0.0, 2.0, x * 0.5] for x in range(41)]  # 40 voxels, 20 um
    along_z = [[z * 2.0, 2.0, 20.0] for z in range(1, 11)]  # 10 slices, 20 um
    edge = {"voxels": in_plane + along_z}
    mask = np.zeros((12, 6, 44), dtype=bool)
    mask[0, 4, :41] = True  # the in-plane leg, at y = 2 um

    vertices = _edge_sample_points_from_data(edge, (np.zeros(3), np.zeros(3)))
    evenly = _edge_sample_points_from_data(edge, (np.zeros(3), np.zeros(3)), step_um=0.5)

    assert _sample_overlap_fraction(vertices, mask, voxel_size_zyx=spacing) == pytest.approx(41 / 51)
    # Half, to within the first micron of the z leg, which rounds into slice 0.
    assert _sample_overlap_fraction(evenly, mask, voxel_size_zyx=spacing) == pytest.approx(0.5, abs=0.04)
