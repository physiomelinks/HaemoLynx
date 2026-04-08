"""Tests for small arteriole/venule mask overlap boundary labelling on graphs."""
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

import networkx as nx
import pytest
import numpy as np

# Allow running this test without an editable package install.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ImageLynx.graph import (
    infer_boundary_nodes_from_small_vessel_masks,
    infer_boundary_nodes_from_small_vessel_masks_progressive_dilation,
    write_small_vessel_mask_boundary_labelling_3d_html,
)

# Checked-in interactive demo (regenerated when this test runs).
DEMO_HTML_PATH = REPO_ROOT / "examples" / "plots" / "small_vessel_mask_boundary_labelling_demo_3d.html"
TEST_PLOT_DIR = REPO_ROOT / "tests" / "plots" / "small_vessel_mask_boundary_labelling"


def _voxel_polyline_samples(
    p0: np.ndarray, p1: np.ndarray, *, count: int = 24
) -> list[tuple[float, float, float]]:
    """Dense physical (x, y, z) samples along a segment for mask overlap scoring."""
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

    Node positions are physical (x, y, z) with voxel_size 1; mask voxels align with indices.
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
        G.add_node(i, pos=np.array([x, 8.0, 8.0], dtype=float))

    for a, b in zip(range(len(trunk_xs) - 1), range(1, len(trunk_xs))):
        pa = np.asarray(G.nodes[a]["pos"], dtype=float)
        pb = np.asarray(G.nodes[b]["pos"], dtype=float)
        vox = _voxel_polyline_samples(pa, pb, count=32)
        G.add_edge(a, b, voxels=vox, length=float(np.linalg.norm(pb - pa)), weight=1.0)

    # Side branch from arteriole interior: steps in +z while staying inside art slab.
    br = len(trunk_xs)
    G.add_node(br, pos=np.array([4.0, 8.0, 12.0], dtype=float))
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
        voxel_size_xyz=(1.0, 1.0, 1.0),
        minimum_overlap_fraction=0.5,
    )

    assert result["arteriole_boundary_nodes"] == expected["arteriole_boundary_nodes"]
    assert result["venule_boundary_nodes"] == expected["venule_boundary_nodes"]
    assert result["arteriole_edge_count"] >= 2
    assert result["venule_edge_count"] >= 1
    assert G.nodes[2]["mask_vessel_type"] == "arteriole"
    assert G.nodes[4]["mask_vessel_type"] == "venule"

    TEST_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    html_tmp = TEST_PLOT_DIR / "small_vessel_mask_boundary_labelling_3d.html"
    ok = write_small_vessel_mask_boundary_labelling_3d_html(
        G,
        small_arteriole_mask=art_mask,
        small_venule_mask=ven_mask,
        arteriole_boundary_nodes=result["arteriole_boundary_nodes"],
        venule_boundary_nodes=result["venule_boundary_nodes"],
        voxel_size_xyz=(1.0, 1.0, 1.0),
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
        voxel_size_xyz=(1.0, 1.0, 1.0),
        output_html_path=DEMO_HTML_PATH,
        title="Synthetic small-vessel boundary model (3D)",
    )
    assert ok_demo is True
    assert DEMO_HTML_PATH.is_file()

    try:
        webbrowser.open_new_tab(html_tmp.resolve().as_uri())
    except OSError:
        pass


def test_progressive_small_vessel_boundary_assignment_locks_earlier_nodes():
    """Boundary nodes assigned earlier remain fixed across later dilation steps."""
    G = nx.MultiGraph()
    # 0-1-2-3 trunk with degree-1 terminals at 0 and 3.
    G.add_node(0, pos=np.array([1.0, 1.0, 1.0], dtype=float))
    G.add_node(1, pos=np.array([3.0, 1.0, 1.0], dtype=float))
    G.add_node(2, pos=np.array([5.0, 1.0, 1.0], dtype=float))
    G.add_node(3, pos=np.array([7.0, 1.0, 1.0], dtype=float))
    for u, v in [(0, 1), (1, 2), (2, 3)]:
        pu = np.asarray(G.nodes[u]["pos"], dtype=float)
        pv = np.asarray(G.nodes[v]["pos"], dtype=float)
        G.add_edge(
            u,
            v,
            voxels=_voxel_polyline_samples(pu, pv, count=24),
            length=float(np.linalg.norm(pv - pu)),
            weight=1.0,
        )

    art_mask = np.zeros((12, 12, 12), dtype=bool)
    ven_mask = np.zeros((12, 12, 12), dtype=bool)
    art_mask[1, 1, 1] = True
    ven_mask[1, 1, 7] = True

    result = infer_boundary_nodes_from_small_vessel_masks_progressive_dilation(
        G,
        small_arteriole_mask=art_mask,
        small_venule_mask=ven_mask,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        max_dilation_microns=10.0,
        dilation_step_microns=5.0,
        minimum_overlap_fraction=0.5,
        allow_overlap=False,
    )

    # Initial step labels node 1 as arteriole boundary and node 2 as venule boundary.
    # Later dilation increases overlap but must not reassign these boundary nodes.
    assert result["arteriole_boundary_nodes"] == [1]
    assert result["venule_boundary_nodes"] == [2]
    assert G.nodes[1].get("mask_vessel_type") == "arteriole"
    assert G.nodes[2].get("mask_vessel_type") == "venule"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
