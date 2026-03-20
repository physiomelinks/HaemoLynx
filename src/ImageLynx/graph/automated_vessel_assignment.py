"""Automatic terminal-node assignment from arteriole/venule masks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


def _sort_nodes(nodes: set[Any]) -> list[Any]:
    return sorted(nodes, key=lambda n: (str(type(n)), str(n)))


def _terminal_nodes_with_positions(G: nx.Graph) -> list[tuple[Any, np.ndarray]]:
    node_pos = nx.get_node_attributes(G, "pos")
    terminals: list[tuple[Any, np.ndarray]] = []
    for node_id, degree in G.degree():
        if degree != 1 or node_id not in node_pos:
            continue
        terminals.append((node_id, np.asarray(node_pos[node_id], dtype=float)))
    return terminals


def _position_to_mask_index(
    position_xyz: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
    mask_shape: tuple[int, ...],
) -> tuple[int, int, int] | None:
    voxel_size = np.asarray(voxel_size_xyz, dtype=float)
    if voxel_size.shape != (3,) or np.any(voxel_size <= 0):
        raise ValueError(
            f"voxel_size_xyz must be three positive values, got {voxel_size_xyz}."
        )
    if len(mask_shape) != 3:
        raise ValueError(f"Expected a 3D mask shape, got {mask_shape}.")

    voxel_index = np.rint(position_xyz / voxel_size).astype(int)
    if np.any(voxel_index < 0):
        return None
    if np.any(voxel_index >= np.asarray(mask_shape, dtype=int)):
        return None
    return (int(voxel_index[0]), int(voxel_index[1]), int(voxel_index[2]))


def _terminal_edge_sample_points(
    G: nx.Graph,
    node_id: Any,
    node_pos: np.ndarray,
    *,
    max_sample_points: int = 25,
) -> np.ndarray:
    """Collect sample points near a terminal node along its incident edge."""
    if max_sample_points <= 0:
        return np.asarray([node_pos], dtype=float)

    edge_voxels: np.ndarray | None = None
    if isinstance(G, nx.MultiGraph):
        incident_edges = list(G.edges(node_id, keys=True, data=True))
        if incident_edges:
            edge_data = incident_edges[0][3]
            voxels = edge_data.get("voxels")
            if voxels is not None:
                arr = np.asarray(voxels, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 3 and arr.size > 0:
                    edge_voxels = arr
    else:
        incident_edges = list(G.edges(node_id, data=True))
        if incident_edges:
            edge_data = incident_edges[0][2]
            voxels = edge_data.get("voxels")
            if voxels is not None:
                arr = np.asarray(voxels, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 3 and arr.size > 0:
                    edge_voxels = arr

    if edge_voxels is None:
        return np.asarray([node_pos], dtype=float)

    distances = np.linalg.norm(edge_voxels - node_pos.reshape(1, 3), axis=1)
    nearest_idx = np.argsort(distances)[: max_sample_points]
    samples = edge_voxels[nearest_idx]
    samples = np.vstack([samples, node_pos.reshape(1, 3)])
    return np.unique(samples, axis=0)


def _mask_midpoint_physical(
    mask: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
) -> np.ndarray:
    points_zyx = np.argwhere(mask.astype(bool, copy=False))
    if points_zyx.size == 0:
        return np.asarray([np.inf, np.inf, np.inf], dtype=float)
    voxel_size = np.asarray(voxel_size_xyz, dtype=float)
    return np.mean(points_zyx.astype(float), axis=0) * voxel_size


def _overlap_fraction_and_intersection(
    sample_points: np.ndarray,
    mask: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
    node_pos: np.ndarray,
) -> tuple[float, np.ndarray | None]:
    valid_points: list[np.ndarray] = []
    in_mask_points: list[np.ndarray] = []
    for point in sample_points:
        mask_index = _position_to_mask_index(
            point,
            voxel_size_xyz=voxel_size_xyz,
            mask_shape=mask.shape,
        )
        if mask_index is None:
            continue
        valid_points.append(point)
        if mask[mask_index]:
            in_mask_points.append(point)
    if not valid_points:
        return 0.0, None
    overlap_fraction = float(len(in_mask_points)) / float(len(valid_points))
    if not in_mask_points:
        return overlap_fraction, None
    in_mask_arr = np.asarray(in_mask_points, dtype=float)
    dists = np.linalg.norm(in_mask_arr - node_pos.reshape(1, 3), axis=1)
    return overlap_fraction, in_mask_arr[int(np.argmin(dists))]


def resolve_overlapping_terminal_node_assignment(
    G: nx.Graph,
    node_id: Any,
    *,
    node_pos: np.ndarray,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
    max_sample_points: int = 25,
) -> str:
    """Resolve input/output assignment for a terminal node in both masks.

    Decision rule:
    1) Prefer the vessel with the higher local overlap percentage near the node.
    2) If tied, choose the vessel with the shorter distance from its overlap
       intersection point to the vessel-volume midpoint.
    """
    samples = _terminal_edge_sample_points(
        G,
        node_id,
        node_pos,
        max_sample_points=max_sample_points,
    )
    arteriole_overlap, arteriole_intersection = _overlap_fraction_and_intersection(
        samples, large_arteriole_mask, voxel_size_xyz, node_pos
    )
    venule_overlap, venule_intersection = _overlap_fraction_and_intersection(
        samples, large_venule_mask, voxel_size_xyz, node_pos
    )
    if arteriole_overlap > venule_overlap:
        return "input"
    if venule_overlap > arteriole_overlap:
        return "output"

    arteriole_mid = _mask_midpoint_physical(large_arteriole_mask, voxel_size_xyz)
    venule_mid = _mask_midpoint_physical(large_venule_mask, voxel_size_xyz)
    arteriole_dist = np.inf
    venule_dist = np.inf
    if arteriole_intersection is not None and np.all(np.isfinite(arteriole_mid)):
        arteriole_dist = float(np.linalg.norm(arteriole_intersection - arteriole_mid))
    if venule_intersection is not None and np.all(np.isfinite(venule_mid)):
        venule_dist = float(np.linalg.norm(venule_intersection - venule_mid))
    if arteriole_dist < venule_dist:
        return "input"
    if venule_dist < arteriole_dist:
        return "output"
    # Final deterministic tie-break.
    return "input"


def select_terminal_nodes_from_large_vessel_masks(
    G: nx.Graph,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    *,
    voxel_size_xyz: tuple[float, float, float],
    allow_overlap: bool = False,
) -> tuple[list[Any], list[Any]]:
    """Assign degree-1 nodes to input/output groups by vessel-mask overlap."""
    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )

    arteriole_mask = large_arteriole_mask.astype(bool, copy=False)
    venule_mask = large_venule_mask.astype(bool, copy=False)
    terminal_nodes = _terminal_nodes_with_positions(G)
    if not terminal_nodes:
        return [], []

    starting_nodes: set[Any] = set()
    output_nodes: set[Any] = set()
    for node_id, node_pos in terminal_nodes:
        index_zyx = _position_to_mask_index(
            node_pos,
            voxel_size_xyz=voxel_size_xyz,
            mask_shape=arteriole_mask.shape,
        )
        if index_zyx is None:
            continue
        in_arteriole = bool(arteriole_mask[index_zyx])
        in_venule = bool(venule_mask[index_zyx])
        if in_arteriole and in_venule and not allow_overlap:
            assignment = resolve_overlapping_terminal_node_assignment(
                G,
                node_id,
                node_pos=node_pos,
                large_arteriole_mask=arteriole_mask,
                large_venule_mask=venule_mask,
                voxel_size_xyz=voxel_size_xyz,
            )
            if assignment == "input":
                starting_nodes.add(node_id)
            else:
                output_nodes.add(node_id)
            continue
        if in_arteriole:
            starting_nodes.add(node_id)
        if in_venule:
            output_nodes.add(node_id)

    if not allow_overlap:
        output_nodes -= starting_nodes

    return _sort_nodes(starting_nodes), _sort_nodes(output_nodes)


def write_automated_vessel_assignment_3d_html(
    G: nx.Graph,
    *,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    input_nodes: list[Any],
    output_nodes: list[Any],
    voxel_size_xyz: tuple[float, float, float],
    output_html_path: str | Path,
) -> bool:
    """Write interactive 3D HTML showing masks, graph, and selected nodes."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return False

    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        raise ValueError("Graph has no node positions ('pos').")
    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )

    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    # Edges from node positions; graph positions are stored as (z, y, x).
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

    def _coords(nodes: list[Any]) -> tuple[list[float], list[float], list[float]]:
        xs = [float(np.asarray(pos[n], dtype=float)[2]) for n in nodes if n in pos]
        ys = [float(np.asarray(pos[n], dtype=float)[1]) for n in nodes if n in pos]
        zs = [float(np.asarray(pos[n], dtype=float)[0]) for n in nodes if n in pos]
        return xs, ys, zs

    def _add_volume_trace(mask: np.ndarray, *, name: str, color: str, fig: Any) -> None:
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
    _add_volume_trace(
        large_arteriole_mask.astype(bool, copy=False),
        name="Arteriole Mask Volume",
        color="#00FF7F",
        fig=fig,
    )
    _add_volume_trace(
        large_venule_mask.astype(bool, copy=False),
        name="Venule Mask Volume",
        color="#FF3EA5",
        fig=fig,
    )
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
                marker=dict(size=4, color="#9E9E9E"),
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
        title="Automated Vessel Assignment (3D)",
        showlegend=True,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
    )
    fig.write_html(str(output_html_path), include_plotlyjs="cdn")
    return True
