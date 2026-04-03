"""Tests for automated terminal-node vessel assignment."""
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

import networkx as nx
import numpy as np

# Allow running this test without an editable package install.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ImageLynx.graph import (
    compute_overlapping_terminal_assignment_metrics,
    dilate_large_vessel_masks_by_microns,
    select_terminal_nodes_from_large_vessel_masks,
)

TEST_PLOT_DIR = REPO_ROOT / "tests" / "plots" / "automated_vessel_detection"


def _write_rotatable_assignment_graph(
    G: nx.Graph,
    input_nodes: list[int],
    output_nodes: list[int],
    arteriole_mask: np.ndarray,
    venule_mask: np.ndarray,
    html_path: Path,
    voxel_size_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    title: str = "Automated Vessel Detection (3D)",
    annotation_lines: list[str] | None = None,
) -> None:
    """Write interactive 3D HTML with masks and color-coded IO nodes."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return

    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        return
    if arteriole_mask.shape != venule_mask.shape:
        return

    # Draw edges as 3D line segments. Stored pos is (z, y, x); plotly uses (x, y, z).
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    if isinstance(G, nx.MultiGraph):
        edge_iter = G.edges(keys=True, data=True)
        for u, v, _k, _data in edge_iter:
            pu = np.asarray(pos[u], dtype=float)
            pv = np.asarray(pos[v], dtype=float)
            edge_x += [float(pu[2]), float(pv[2]), None]
            edge_y += [float(pu[1]), float(pv[1]), None]
            edge_z += [float(pu[0]), float(pv[0]), None]
    else:
        edge_iter = G.edges(data=True)
        for u, v, _data in edge_iter:
            pu = np.asarray(pos[u], dtype=float)
            pv = np.asarray(pos[v], dtype=float)
            edge_x += [float(pu[2]), float(pv[2]), None]
            edge_y += [float(pu[1]), float(pv[1]), None]
            edge_z += [float(pu[0]), float(pv[0]), None]

    input_set = set(input_nodes)
    output_set = set(output_nodes)
    other_nodes = [n for n in G.nodes if n not in input_set and n not in output_set]

    def _coords(nodes: list[int]) -> tuple[list[float], list[float], list[float]]:
        xs = [float(np.asarray(pos[n], dtype=float)[2]) for n in nodes]
        ys = [float(np.asarray(pos[n], dtype=float)[1]) for n in nodes]
        zs = [float(np.asarray(pos[n], dtype=float)[0]) for n in nodes]
        return xs, ys, zs

    def _add_volume_trace(mask: np.ndarray, *, name: str, color: str) -> None:
        # Plotly volume expects a regular grid. Use binary values and render the
        # occupied voxels as a semi-transparent isovalue volume.
        if not np.any(mask):
            return
        z_scale, y_scale, x_scale = voxel_size_xyz
        zz, yy, xx = np.indices(mask.shape, dtype=float)
        fig.add_trace(
            go.Volume(
                x=(xx * float(x_scale)).ravel(),
                y=(yy * float(y_scale)).ravel(),
                z=(zz * float(z_scale)).ravel(),
                value=mask.astype(float).ravel(),
                isomin=0.5,
                isomax=1.0,
                opacity=0.12,
                surface_count=1,
                caps=dict(x_show=False, y_show=False, z_show=False),
                colorscale=[[0.0, color], [1.0, color]],
                showscale=False,
                name=name,
            )
        )

    fig = go.Figure()
    _add_volume_trace(arteriole_mask, name="Arteriole Mask Volume", color="#00FF7F")
    _add_volume_trace(venule_mask, name="Venule Mask Volume", color="#FF3EA5")
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="rgba(0, 200, 255, 0.7)", width=5),
            name="Edges",
        )
    )

    if other_nodes:
        ox, oy, oz = _coords(other_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="markers",
                marker=dict(size=5, color="#9E9E9E"),
                name="Other Nodes",
            )
        )
    if input_nodes:
        ix, iy, iz = _coords(input_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=ix,
                y=iy,
                z=iz,
                mode="markers",
                marker=dict(size=8, color="#00FF7F"),
                name="Input Nodes",
            )
        )
    if output_nodes:
        ox, oy, oz = _coords(output_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="markers",
                marker=dict(size=8, color="#FF3EA5"),
                name="Output Nodes",
            )
        )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        showlegend=True,
    )
    if annotation_lines:
        fig.add_annotation(
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            text="<br>".join(annotation_lines),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#666",
            borderwidth=1,
            font=dict(size=12),
        )
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    # Best-effort auto-open for local interactive debugging.
    try:
        webbrowser.open_new_tab(html_path.resolve().as_uri())
    except Exception:
        # Keep test robust in headless/CI environments.
        pass


def _parallel_cylinder_mask_along_x(
    shape: tuple[int, int, int],
    *,
    center_z: float,
    center_y: float,
    radius: float,
    x_start: int,
    x_end: int,
) -> np.ndarray:
    """Create a binary cylinder running along +x within [x_start, x_end]."""
    zz, yy, xx = np.indices(shape, dtype=float)
    radial_dist = np.sqrt((zz - float(center_z)) ** 2 + (yy - float(center_y)) ** 2)
    mask = radial_dist <= float(radius)
    return mask & (xx >= int(x_start)) & (xx <= int(x_end))


def test_automated_vessel_detection(_tmp_path):
    """Assign degree-1 nodes via mask overlap, with and without dilation."""
    TEST_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    # Build a richer graph with multiple degree>1 pass-through/junction nodes.
    # Terminals should still be the only selectable input/output nodes.
    G = nx.MultiGraph()
    # Terminals intended for automated assignment.
    G.add_node(0, pos=np.array([1.0, 1.0, 1.0]))  # terminal in arteriole volume
    G.add_node(1, pos=np.array([3.0, 1.0, 0.0]))  # terminal near arteriole volume (kept distinct from trunk nodes)
    G.add_node(2, pos=np.array([5.0, 1.0, 1.0]))  # terminal in venule volume
    # Non-terminal backbone and branch nodes (degree > 1).
    G.add_node(10, pos=np.array([2.0, 1.0, 1.0]))
    G.add_node(11, pos=np.array([3.0, 1.0, 1.0]))
    G.add_node(12, pos=np.array([4.0, 1.0, 1.0]))
    G.add_node(20, pos=np.array([3.0, 2.0, 1.0]))  # branch junction
    G.add_node(21, pos=np.array([3.0, 3.0, 1.0]))  # branch terminal
    G.add_node(22, pos=np.array([4.0, 4.0, 1.0]))  # branch terminal (kept away from venule dilation)

    # Main trunk traversing both vessel volumes.
    G.add_edge(0, 10, length=1.0, weight=1.0)
    G.add_edge(10, 11, length=1.0, weight=1.0)
    G.add_edge(11, 12, length=1.0, weight=1.0)
    G.add_edge(12, 2, length=1.0, weight=1.0)
    # Side branches to create additional degree>1 structure.
    G.add_edge(11, 20, length=1.0, weight=1.0)
    G.add_edge(20, 21, length=1.0, weight=1.0)
    G.add_edge(20, 22, length=1.0, weight=1.0)
    # Keep node 1 as an isolated terminal branch near arteriole.
    G.add_edge(1, 10, length=1.0, weight=1.0)

    # Masks in voxel space (voxel_size=1 means pos maps directly to indices).
    # Keep arteriole and venule as separated 3D volumes (non-overlapping).
    arteriole_mask = np.zeros((8, 8, 8), dtype=bool)
    venule_mask = np.zeros((8, 8, 8), dtype=bool)
    # Proximal arteriole volume around terminal 0 and nearby trunk voxels.
    arteriole_mask[0:3, 0:3, 0:3] = True
    # Distal venule volume around terminal 2 and nearby trunk voxels.
    venule_mask[5:8, 0:3, 0:3] = True

    # Non-dilated assignment should only pick directly overlapping terminals.
    start_nodes, out_nodes = select_terminal_nodes_from_large_vessel_masks(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        allow_overlap=False,
    )
    assert start_nodes == [0]
    assert out_nodes == [2]
    _write_rotatable_assignment_graph(
        G=G,
        input_nodes=start_nodes,
        output_nodes=out_nodes,
        arteriole_mask=arteriole_mask,
        venule_mask=venule_mask,
        html_path=TEST_PLOT_DIR / "automated_vessel_detection_before_dilation_3d.html",
        voxel_size_xyz=(1.0, 1.0, 1.0),
        title="Before Dilation",
    )

    # After arteriole dilation by >=1 micron, terminal node 1 (z=3) is included.
    dilated_arteriole_mask, dilated_venule_mask = dilate_large_vessel_masks_by_microns(
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        dilation_microns=1.0,
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    start_nodes_dilated, out_nodes_dilated = select_terminal_nodes_from_large_vessel_masks(
        G,
        large_arteriole_mask=dilated_arteriole_mask,
        large_venule_mask=dilated_venule_mask,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        allow_overlap=False,
    )
    assert start_nodes_dilated == [0, 1]
    assert out_nodes_dilated == [2]

    _write_rotatable_assignment_graph(
        G=G,
        input_nodes=start_nodes_dilated,
        output_nodes=out_nodes_dilated,
        arteriole_mask=dilated_arteriole_mask,
        venule_mask=dilated_venule_mask,
        html_path=TEST_PLOT_DIR / "automated_vessel_detection_after_dilation_3d.html",
        voxel_size_xyz=(1.0, 1.0, 1.0),
        title="After Dilation",
    )


def test_overlap_resolution_prefers_cross_section_midline_distance(_tmp_path):
    """Overlapping cylinders: cross-section midline distance is evaluated first."""
    TEST_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([6.0, 6.0, 2.0]))   # overlapping terminal
    G.add_node(1, pos=np.array([6.0, 6.0, 12.0]))  # junction
    G.add_node(2, pos=np.array([6.0, 8.0, 12.0]))  # second terminal
    G.add_edge(
        0,
        1,
        length=10.0,
        weight=10.0,
        voxels=[(6.0, 6.0, float(x)) for x in range(2, 13)],
    )
    G.add_edge(1, 2, length=2.0, weight=2.0, voxels=[(6.0, 6.0, 12.0), (6.0, 8.0, 12.0)])

    shape = (16, 16, 24)
    arteriole_mask = _parallel_cylinder_mask_along_x(
        shape, center_z=6.0, center_y=5.0, radius=2.0, x_start=2, x_end=5
    )
    venule_mask = _parallel_cylinder_mask_along_x(
        shape, center_z=6.0, center_y=8.0, radius=2.0, x_start=2, x_end=10
    )

    start_nodes, out_nodes = select_terminal_nodes_from_large_vessel_masks(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        allow_overlap=False,
    )
    assert start_nodes == [0]
    assert out_nodes == []
    metrics = compute_overlapping_terminal_assignment_metrics(
        G,
        0,
        node_pos=np.asarray(G.nodes[0]["pos"], dtype=float),
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    assert (
        metrics["arteriole_cross_section_midpoint_distance"]
        < metrics["venule_cross_section_midpoint_distance"]
    )

    _write_rotatable_assignment_graph(
        G=G,
        input_nodes=start_nodes,
        output_nodes=out_nodes,
        arteriole_mask=arteriole_mask,
        venule_mask=venule_mask,
        html_path=TEST_PLOT_DIR / "overlap_resolution_cross_section_priority_3d.html",
        voxel_size_xyz=(1.0, 1.0, 1.0),
        title="Overlap Resolution: Cross-section Priority",
        annotation_lines=[
            f"Node 0 cross-section distance -> arteriole={metrics['arteriole_cross_section_midpoint_distance']:.3f}, "
            f"venule={metrics['venule_cross_section_midpoint_distance']:.3f}",
            f"Node 0 overlap -> arteriole={metrics['arteriole_overlap_fraction']:.3f}, "
            f"venule={metrics['venule_overlap_fraction']:.3f}",
            f"Node 0 midpoint distance -> arteriole={metrics['arteriole_midpoint_distance']:.3f}, "
            f"venule={metrics['venule_midpoint_distance']:.3f}",
        ],
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
