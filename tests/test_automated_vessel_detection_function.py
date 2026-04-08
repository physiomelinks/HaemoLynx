"""Tests for automated terminal-node vessel assignment."""
from __future__ import annotations

import os
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
    assess_large_vessel_assignment_quality,
    compute_overlapping_terminal_assignment_metrics,
    dilate_large_vessel_masks_by_microns,
    select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence,
    select_terminal_nodes_from_large_vessel_masks_progressive_dilation,
    select_terminal_nodes_from_large_vessel_masks,
)


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

    # Draw edges as 3D line segments. Stored pos is physical (x, y, z).
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    if isinstance(G, nx.MultiGraph):
        edge_iter = G.edges(keys=True, data=True)
        for u, v, _k, _data in edge_iter:
            pu = np.asarray(pos[u], dtype=float)
            pv = np.asarray(pos[v], dtype=float)
            edge_x += [float(pu[0]), float(pv[0]), None]
            edge_y += [float(pu[1]), float(pv[1]), None]
            edge_z += [float(pu[2]), float(pv[2]), None]
    else:
        edge_iter = G.edges(data=True)
        for u, v, _data in edge_iter:
            pu = np.asarray(pos[u], dtype=float)
            pv = np.asarray(pos[v], dtype=float)
            edge_x += [float(pu[0]), float(pv[0]), None]
            edge_y += [float(pu[1]), float(pv[1]), None]
            edge_z += [float(pu[2]), float(pv[2]), None]

    # Enforce terminal-only visual IO labels to avoid ambiguous debug plots.
    input_nodes = [n for n in input_nodes if int(G.degree(n)) == 1]
    output_nodes = [n for n in output_nodes if int(G.degree(n)) == 1]

    input_set = set(input_nodes)
    output_set = set(output_nodes)
    other_nodes = [n for n in G.nodes if n not in input_set and n not in output_set]

    def _coords(nodes: list[int]) -> tuple[list[float], list[float], list[float]]:
        xs = [float(np.asarray(pos[n], dtype=float)[0]) for n in nodes]
        ys = [float(np.asarray(pos[n], dtype=float)[1]) for n in nodes]
        zs = [float(np.asarray(pos[n], dtype=float)[2]) for n in nodes]
        return xs, ys, zs

    def _add_volume_trace(mask: np.ndarray, *, name: str, color: str) -> None:
        # Plotly volume expects a regular grid. Use binary values and render the
        # occupied voxels as a semi-transparent isovalue volume.
        if not np.any(mask):
            return
        x_scale, y_scale, z_scale = voxel_size_xyz
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
        other_text = [
            f"node={int(n)}<br>degree={int(G.degree(n))}<br>role=other"
            for n in other_nodes
        ]
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="markers",
                marker=dict(size=5, color="#9E9E9E"),
                name="Other Nodes",
                text=other_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )
    if input_nodes:
        ix, iy, iz = _coords(input_nodes)
        input_text = [
            f"node={int(n)}<br>degree={int(G.degree(n))}<br>role=input"
            for n in input_nodes
        ]
        fig.add_trace(
            go.Scatter3d(
                x=ix,
                y=iy,
                z=iz,
                mode="markers",
                marker=dict(size=8, color="#00FF7F"),
                name="Input Nodes",
                text=input_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )
    if output_nodes:
        ox, oy, oz = _coords(output_nodes)
        output_text = [
            f"node={int(n)}<br>degree={int(G.degree(n))}<br>role=output"
            for n in output_nodes
        ]
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="markers",
                marker=dict(size=8, color="#FF3EA5"),
                name="Output Nodes",
                text=output_text,
                hovertemplate="%{text}<extra></extra>",
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


def test_automated_vessel_detection(tmp_path):
    """Assign degree-1 nodes via mask overlap, with and without dilation."""
    # Build a richer graph with multiple degree>1 pass-through/junction nodes.
    # Terminals should still be the only selectable input/output nodes.
    G = nx.MultiGraph()
    # Terminals intended for automated assignment.
    G.add_node(0, pos=np.array([1.0, 1.0, 1.0]))  # terminal in arteriole volume
    G.add_node(1, pos=np.array([0.0, 1.0, 3.0]))  # terminal near arteriole volume (kept distinct from trunk nodes)
    G.add_node(2, pos=np.array([1.0, 1.0, 5.0]))  # terminal in venule volume
    # Non-terminal backbone and branch nodes (degree > 1).
    G.add_node(10, pos=np.array([1.0, 1.0, 2.0]))
    G.add_node(11, pos=np.array([1.0, 1.0, 3.0]))
    G.add_node(12, pos=np.array([1.0, 1.0, 4.0]))
    G.add_node(20, pos=np.array([1.0, 2.0, 3.0]))  # branch junction
    G.add_node(21, pos=np.array([1.0, 3.0, 3.0]))  # branch terminal
    G.add_node(22, pos=np.array([1.0, 4.0, 4.0]))  # branch terminal (kept away from venule dilation)

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
        html_path=tmp_path / "automated_vessel_detection_before_dilation_3d.html",
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
        html_path=tmp_path / "automated_vessel_detection_after_dilation_3d.html",
        voxel_size_xyz=(1.0, 1.0, 1.0),
        title="After Dilation",
    )


def test_overlap_resolution_prefers_cross_section_midline_distance(tmp_path):
    """Overlapping cylinders: cross-section midline distance is evaluated first."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([2.0, 6.0, 6.0]))   # overlapping terminal
    G.add_node(1, pos=np.array([12.0, 6.0, 6.0]))  # junction
    G.add_node(2, pos=np.array([12.0, 8.0, 6.0]))  # second terminal
    G.add_edge(
        0,
        1,
        length=10.0,
        weight=10.0,
        voxels=[(float(x), 6.0, 6.0) for x in range(2, 13)],
    )
    G.add_edge(1, 2, length=2.0, weight=2.0, voxels=[(12.0, 6.0, 6.0), (12.0, 8.0, 6.0)])

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
        html_path=tmp_path / "overlap_resolution_cross_section_priority_3d.html",
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


def test_large_vessel_overlap_exclusion_assignment_graph(tmp_path):
    """Write overlap-assignment debug HTML from large-vessel test geometry."""
    G = nx.MultiGraph()
    # Arrange a clear terminal-connector-terminal chain in Z so visuals are unambiguous.
    G.add_node(0, pos=np.array([5.0, 5.0, 4.0]))  # degree-1 terminal (lower)
    G.add_node(1, pos=np.array([5.0, 5.0, 6.0]))  # degree-1 terminal (upper)
    G.add_node(2, pos=np.array([5.0, 5.0, 5.0]))  # degree-2 connector (middle)
    G.add_edge(0, 2, length=1.0, weight=1.0)
    G.add_edge(1, 2, length=1.0, weight=1.0)

    arteriole_mask = np.zeros((12, 12, 12), dtype=bool)
    venule_mask = np.zeros((12, 12, 12), dtype=bool)
    arteriole_mask[2:10, 2:10, 2:10] = True
    venule_mask[5:7, 5:7, 5:8] = True

    start_nodes, out_nodes = select_terminal_nodes_from_large_vessel_masks(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        allow_overlap=False,
        exclude_smaller_overlapping_volumes=True,
    )
    assert start_nodes == [0, 1]
    assert out_nodes == []

    html_path = tmp_path / "large_vessel_overlap_exclusion_assignment.html"
    if os.environ.get("IMAGELYNX_WRITE_TEST_PLOTLY", "").strip().lower() in {"1", "true", "yes"}:
        html_path = (
            Path(__file__).resolve().parents[1]
            / "tests"
            / "outputs"
            / "overlap_assignment"
            / "large_vessel_overlap_exclusion_assignment.html"
        )
    _write_rotatable_assignment_graph(
        G=G,
        input_nodes=start_nodes,
        output_nodes=out_nodes,
        arteriole_mask=arteriole_mask,
        venule_mask=venule_mask,
        html_path=html_path,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        title="Large-vessel Overlap Exclusion Assignment",
        annotation_lines=[
            "Green nodes are assigned inputs (degree-1 only).",
            "Middle node is the degree-2 connector; end nodes are degree-1 terminals.",
            f"Node degrees: 0->{int(G.degree(0))}, 1->{int(G.degree(1))}, 2->{int(G.degree(2))}.",
        ],
    )


def test_progressive_dilation_assignment_locks_earlier_nodes():
    """Nodes assigned early remain fixed across later dilation steps."""
    G = nx.MultiGraph()
    # Terminal near arteriole and venule volumes with staged overlap behavior.
    G.add_node(0, pos=np.array([1.0, 1.0, 1.0]))   # near arteriole at 0 microns
    G.add_node(1, pos=np.array([7.0, 1.0, 1.0]))   # outside both at 0, venule at +5, arteriole at +10
    G.add_node(2, pos=np.array([9.0, 1.0, 1.0]))   # venule at 0 microns
    # Internal connector nodes to keep terminals degree-1.
    G.add_node(10, pos=np.array([2.0, 1.0, 1.0]))
    G.add_node(11, pos=np.array([7.0, 1.0, 1.0]))
    G.add_edge(0, 10, length=1.0, weight=1.0)
    G.add_edge(10, 11, length=5.0, weight=5.0)
    G.add_edge(11, 2, length=2.0, weight=2.0)
    G.add_edge(1, 10, length=5.0, weight=5.0)

    arteriole_mask = np.zeros((12, 12, 12), dtype=bool)
    venule_mask = np.zeros((12, 12, 12), dtype=bool)
    arteriole_mask[1, 1, 1] = True
    venule_mask[1, 1, 9] = True

    # At max dilation=10 with 5-micron steps:
    # - node 0 is input at 0 microns
    # - node 2 is output at 0 microns
    # - node 1 becomes newly output at 5 microns
    #   (and must remain output, even though arteriole reaches it at 10 microns)
    start_nodes, out_nodes = select_terminal_nodes_from_large_vessel_masks_progressive_dilation(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        max_dilation_microns=10.0,
        dilation_step_microns=5.0,
        allow_overlap=False,
    )
    assert start_nodes == [0]
    assert out_nodes == [1, 2]


def test_confidence_mode_replaces_exact_tie_with_unresolved():
    """Equal arteriole/venule evidence should be flagged unresolved."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([2.0, 2.0, 2.0]))  # terminal in overlap (exact tie)
    G.add_node(1, pos=np.array([3.0, 2.0, 2.0]))  # connector (degree 2)
    G.add_node(2, pos=np.array([4.0, 2.0, 2.0]))  # interior (degree 2)
    G.add_node(3, pos=np.array([5.0, 2.0, 2.0]))  # distal terminal: venule-only, not in overlap
    G.add_edge(0, 1, length=1.0, weight=1.0, voxels=[(2.0, 2.0, 2.0), (3.0, 2.0, 2.0)])
    G.add_edge(1, 2, length=1.0, weight=1.0, voxels=[(3.0, 2.0, 2.0), (4.0, 2.0, 2.0)])
    G.add_edge(2, 3, length=1.0, weight=1.0, voxels=[(4.0, 2.0, 2.0), (5.0, 2.0, 2.0)])

    arteriole_mask = np.zeros((8, 8, 8), dtype=bool)
    venule_mask = np.zeros((8, 8, 8), dtype=bool)
    arteriole_mask[1:4, 1:4, 1:4] = True
    venule_mask[1:4, 1:4, 1:4] = True
    # x = 4 slice: venule only (clear venule output for node 3, no arteriole tie).
    venule_mask[1:4, 1:4, 4:6] = True

    result = select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        max_dilation_microns=0.0,
        confidence_margin=0.05,
        minimum_confidence=0.05,
        topology_penalty=0.0,
    )
    assert result["input_nodes"] == []
    assert result["output_nodes"] == [3]
    assert result["unresolved_nodes"] == [0]
    assert result["node_confidence"][0]["reason"] in {"exact_tie", "low_score_gap"}


def test_confidence_mode_topology_penalty_biases_label():
    """Topology support should penalize physiologically implausible labels."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([2.0, 2.0, 2.0]))   # terminal candidate
    G.add_node(1, pos=np.array([3.0, 2.0, 2.0]))   # neighbor junction
    G.add_node(2, pos=np.array([4.0, 2.0, 2.0]))   # distal terminal
    G.add_edge(
        0,
        1,
        length=1.0,
        weight=1.0,
        branch_order="Ven1",
        vessel_type="venule",
        voxels=[(2.0, 2.0, 2.0), (3.0, 2.0, 2.0)],
    )
    G.add_edge(1, 2, length=1.0, weight=1.0, branch_order="Ven2", vessel_type="venule")

    arteriole_mask = np.zeros((8, 8, 8), dtype=bool)
    venule_mask = np.zeros((8, 8, 8), dtype=bool)
    arteriole_mask[1:4, 1:4, 1:4] = True
    venule_mask[1:4, 1:4, 1:4] = True

    result = select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        max_dilation_microns=0.0,
        confidence_margin=0.0,
        minimum_confidence=0.01,
        topology_penalty=0.2,
    )
    assert 0 in result["output_nodes"]
    assert 0 not in result["input_nodes"]
    assert result["node_confidence"][0]["decision"] == "output"


def test_quality_gate_can_trigger_conservative_mode():
    """High overlap/fragmentation should trigger conservative robust mode."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([2.0, 2.0, 2.0]))
    G.add_node(1, pos=np.array([3.0, 2.0, 2.0]))
    G.add_edge(0, 1, length=1.0, weight=1.0)

    arteriole_mask = np.zeros((16, 16, 16), dtype=bool)
    venule_mask = np.zeros((16, 16, 16), dtype=bool)
    # Deliberately make heavy overlap and many disconnected fragments.
    arteriole_mask[2:6, 2:6, 2:6] = True
    venule_mask[2:6, 2:6, 2:6] = True
    for idx in range(8, 15):
        arteriole_mask[idx, idx % 16, 1] = True
        venule_mask[idx, idx % 16, 2] = True

    quality = assess_large_vessel_assignment_quality(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        quality_max_overlap_fraction=0.05,
        quality_min_terminal_coverage=0.9,
        quality_max_component_count=2,
    )
    assert quality["poor_quality"] is True

    result = select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        max_dilation_microns=50.0,
        confidence_margin=0.08,
        minimum_confidence=0.12,
        quality_max_overlap_fraction=0.05,
        quality_min_terminal_coverage=0.9,
        quality_max_component_count=2,
        conservative_max_dilation_microns=15.0,
    )
    assert result["conservative_mode"] is True
    assert result["effective_max_dilation_microns"] <= 15.0


if __name__ == "__main__":
    import pytest

    os.environ["IMAGELYNX_WRITE_TEST_PLOTLY"] = "1"
    raise SystemExit(
        pytest.main(
            [
                __file__,
                "-q",
                "-k",
                "test_large_vessel_overlap_exclusion_assignment_graph",
            ]
        )
    )
