"""3D visualization for automated large-vessel input/output assignment."""
from __future__ import annotations

from typing import Any, Optional

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from .plot import _is_pytest_runtime

#: Plotly ``Volume`` styling for mask overlays. Pipeline HTML and the GUI
#: final-graph writer share these so a selected arteriole/venule volume looks
#: the same in both files.
VESSEL_VOLUME_TRACE_STYLES: dict[str, dict[str, Any]] = {
    "large_arteriole": {
        "name": "Large arteriole mask",
        "color": "#B71C1C",
        "opacity": 0.22,
    },
    "large_venule": {
        "name": "Large venule mask",
        "color": "#1B5E20",
        "opacity": 0.22,
    },
    "small_arteriole": {
        "name": "Small arteriole mask",
        "color": "#FF3B30",
        "opacity": 0.12,
    },
    "small_venule": {
        "name": "Small venule mask",
        "color": "#2ECC71",
        "opacity": 0.12,
    },
}


def _zyx_points_to_xyz(points: np.ndarray) -> np.ndarray:
    """Convert physical (z, y, x) points to plotly (x, y, z)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return pts
    return np.column_stack([pts[:, 2], pts[:, 1], pts[:, 0]])


def _nonzero_bbox_slices_zyx(mask: np.ndarray) -> tuple[slice, slice, slice] | None:
    """Return tight z/y/x bounding slices for nonzero mask voxels."""
    coords = np.argwhere(mask.astype(bool, copy=False))
    if coords.size == 0:
        return None
    mins = coords.min(axis=0).astype(int)
    maxs = coords.max(axis=0).astype(int) + 1
    return (
        slice(int(mins[0]), int(maxs[0])),
        slice(int(mins[1]), int(maxs[1])),
        slice(int(mins[2]), int(maxs[2])),
    )


def _downsample_binary_mask_max(
    mask: np.ndarray,
    stride: int,
) -> np.ndarray:
    """Downsample a 3D binary mask via block max-pooling."""
    if stride <= 1:
        return mask.astype(bool, copy=False)

    z, y, x = mask.shape
    pad_z = (-z) % stride
    pad_y = (-y) % stride
    pad_x = (-x) % stride
    if pad_z or pad_y or pad_x:
        padded = np.pad(
            mask.astype(bool, copy=False),
            ((0, pad_z), (0, pad_y), (0, pad_x)),
            mode="constant",
            constant_values=False,
        )
    else:
        padded = mask.astype(bool, copy=False)

    z2, y2, x2 = padded.shape
    pooled = padded.reshape(
        z2 // stride,
        stride,
        y2 // stride,
        stride,
        x2 // stride,
        stride,
    )
    return np.max(pooled, axis=(1, 3, 5))


def add_binary_mask_volume_trace(
    fig: go.Figure,
    mask: np.ndarray,
    *,
    name: str,
    color: str,
    opacity: float,
    voxel_size_zyx: tuple[float, float, float],
    volume_downsample_stride: int = 1,
) -> bool:
    """Add one pipeline-style Plotly ``Volume`` trace for a binary mask.

    Crops to the nonzero bounding box then max-pools by ``stride`` — the same
    path ``visualize_3d_plotly_large_vessel_assignment`` uses. Returns True
    when a trace was added.
    """
    mask_bool = mask.astype(bool, copy=False)
    bbox = _nonzero_bbox_slices_zyx(mask_bool)
    if bbox is None:
        return False
    z_scale, y_scale, x_scale = (
        float(voxel_size_zyx[0]),
        float(voxel_size_zyx[1]),
        float(voxel_size_zyx[2]),
    )
    z_slice, y_slice, x_slice = bbox
    cropped = mask_bool[z_slice, y_slice, x_slice]
    stride = max(1, int(volume_downsample_stride))
    downsampled = _downsample_binary_mask_max(cropped, stride)
    if not np.any(downsampled):
        # Safety fallback for very sparse masks.
        downsampled = cropped
        effective_stride = 1
    else:
        effective_stride = stride
    zz, yy, xx = np.indices(downsampled.shape, dtype=float)
    xx = (xx * float(effective_stride)) + float(x_slice.start)
    yy = (yy * float(effective_stride)) + float(y_slice.start)
    zz = (zz * float(effective_stride)) + float(z_slice.start)
    fig.add_trace(
        go.Volume(
            x=(xx * x_scale).ravel(),
            y=(yy * y_scale).ravel(),
            z=(zz * z_scale).ravel(),
            value=downsampled.astype(float).ravel(),
            isomin=0.5,
            isomax=1.0,
            opacity=float(opacity),
            surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale=[[0.0, color], [1.0, color]],
            showscale=False,
            name=name,
            hoverinfo="skip",
        )
    )
    return True


def visualize_3d_plotly_large_vessel_assignment(
    G: nx.Graph,
    *,
    large_arteriole_mask: np.ndarray | None,
    large_venule_mask: np.ndarray | None,
    small_arteriole_mask: np.ndarray | None = None,
    small_venule_mask: np.ndarray | None = None,
    input_nodes: list[Any],
    output_nodes: list[Any],
    arteriole_boundary_nodes: list[Any] | None = None,
    venule_boundary_nodes: list[Any] | None = None,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    volume_downsample_stride: int = 1,
    title: str = "Final Graph with Automated Large-Vessel Assignment (3D)",
    save_html_path: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Render graph + large/small vessel volumes + I/O nodes + vessel/branch labels."""
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        raise ValueError("Graph has no node positions ('pos').")
    if (large_arteriole_mask is None) != (large_venule_mask is None):
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must be both set or both None."
        )
    if (
        large_arteriole_mask is not None
        and large_venule_mask is not None
        and large_arteriole_mask.shape != large_venule_mask.shape
    ):
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )
    if (small_arteriole_mask is None) != (small_venule_mask is None):
        raise ValueError(
            "small_arteriole_mask and small_venule_mask must be both set or both None."
        )
    if (
        small_arteriole_mask is not None
        and small_venule_mask is not None
        and small_arteriole_mask.shape != small_venule_mask.shape
    ):
        raise ValueError(
            "small_arteriole_mask and small_venule_mask must share a shape. "
            f"Got {small_arteriole_mask.shape} and {small_venule_mask.shape}."
        )

    stride = max(1, int(volume_downsample_stride))

    def _add_volume_trace(
        mask: np.ndarray,
        *,
        name: str,
        color: str,
        opacity: float,
        fig: go.Figure,
    ) -> bool:
        return add_binary_mask_volume_trace(
            fig,
            mask,
            name=name,
            color=color,
            opacity=opacity,
            voxel_size_zyx=voxel_size_zyx,
            volume_downsample_stride=stride,
        )

    def _empty_line_lists() -> tuple[list[float | None], list[float | None], list[float | None]]:
        return [], [], []

    edge_segments: dict[str, tuple[list[float | None], list[float | None], list[float | None]]] = {
        "arteriole": _empty_line_lists(),
        "capillary": _empty_line_lists(),
        "venule": _empty_line_lists(),
    }
    branch_label_x: list[float] = []
    branch_label_y: list[float] = []
    branch_label_z: list[float] = []
    branch_label_text: list[str] = []

    def _normalize_edge_vessel_type(edge_data: dict[str, Any]) -> str:
        branch_order = edge_data.get("branch_order")
        if branch_order is not None:
            bo = str(branch_order).strip()
            if bo.startswith("Art"):
                return "arteriole"
            if bo.startswith("Ven"):
                return "venule"
            if bo.startswith("B"):
                return "capillary"
        vessel_type = edge_data.get("vessel_type")
        if vessel_type is None:
            vessel_type = edge_data.get("mask_vessel_type")
        vt = str(vessel_type).strip().lower() if vessel_type is not None else ""
        if vt in {"arteriole", "venule", "capillary"}:
            return vt
        if vt in {"art", "arterial"}:
            return "arteriole"
        if vt in {"ven", "venous"}:
            return "venule"
        return "capillary"

    def _push_segment(kind: str, xs: list[float], ys: list[float], zs: list[float]) -> None:
        lx, ly, lz = edge_segments[kind]
        for x, y, z in zip(xs, ys, zs):
            lx.append(float(x))
            ly.append(float(y))
            lz.append(float(z))
        lx.append(None)
        ly.append(None)
        lz.append(None)

    def _add_branch_label(edge_data: dict[str, Any], pts_xyz: np.ndarray) -> None:
        branch_order = edge_data.get("branch_order")
        if branch_order is None or pts_xyz.size == 0:
            return
        mid_idx = int(pts_xyz.shape[0] // 2)
        mid = pts_xyz[mid_idx]
        branch_label_x.append(float(mid[0]))
        branch_label_y.append(float(mid[1]))
        branch_label_z.append(float(mid[2]))
        branch_label_text.append(str(branch_order))

    if isinstance(G, nx.MultiGraph):
        edge_iter = G.edges(keys=True, data=True)
        for u, v, _k, edge_data in edge_iter:
            kind = _normalize_edge_vessel_type(edge_data)
            voxels = edge_data.get("voxels", [])
            if len(voxels) > 1:
                pts_xyz = _zyx_points_to_xyz(np.asarray(voxels, dtype=float))
                _push_segment(
                    kind,
                    pts_xyz[:, 0].tolist(),
                    pts_xyz[:, 1].tolist(),
                    pts_xyz[:, 2].tolist(),
                )
                _add_branch_label(edge_data, pts_xyz)
            else:
                pu = _zyx_points_to_xyz(np.asarray(pos[u], dtype=float).reshape(1, 3))[0]
                pv = _zyx_points_to_xyz(np.asarray(pos[v], dtype=float).reshape(1, 3))[0]
                _push_segment(
                    kind,
                    [float(pu[0]), float(pv[0])],
                    [float(pu[1]), float(pv[1])],
                    [float(pu[2]), float(pv[2])],
                )
                _add_branch_label(edge_data, np.vstack([pu, pv]))
    else:
        for u, v, edge_data in G.edges(data=True):
            kind = _normalize_edge_vessel_type(edge_data)
            voxels = edge_data.get("voxels", [])
            if len(voxels) > 1:
                pts_xyz = _zyx_points_to_xyz(np.asarray(voxels, dtype=float))
                _push_segment(
                    kind,
                    pts_xyz[:, 0].tolist(),
                    pts_xyz[:, 1].tolist(),
                    pts_xyz[:, 2].tolist(),
                )
                _add_branch_label(edge_data, pts_xyz)
            else:
                pu = _zyx_points_to_xyz(np.asarray(pos[u], dtype=float).reshape(1, 3))[0]
                pv = _zyx_points_to_xyz(np.asarray(pos[v], dtype=float).reshape(1, 3))[0]
                _push_segment(
                    kind,
                    [float(pu[0]), float(pv[0])],
                    [float(pu[1]), float(pv[1])],
                    [float(pu[2]), float(pv[2])],
                )
                _add_branch_label(edge_data, np.vstack([pu, pv]))

    input_set = set(input_nodes)
    output_set = set(output_nodes)
    art_boundary_set = set(arteriole_boundary_nodes or [])
    ven_boundary_set = set(venule_boundary_nodes or [])
    boundary_set = art_boundary_set | ven_boundary_set
    other_nodes = [
        n for n in G.nodes if n not in input_set and n not in output_set and n not in boundary_set
    ]

    def _coords_with_ids(
        nodes: list[Any],
    ) -> tuple[list[float], list[float], list[float], list[str]]:
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        node_ids: list[str] = []
        for node_id in nodes:
            if node_id not in pos:
                continue
            node_pos = _zyx_points_to_xyz(
                np.asarray(pos[node_id], dtype=float).reshape(1, 3)
            )[0]
            xs.append(float(node_pos[0]))
            ys.append(float(node_pos[1]))
            zs.append(float(node_pos[2]))
            node_ids.append(str(node_id))
        return xs, ys, zs, node_ids

    fig = go.Figure()
    volume_trace_indices: list[int] = []
    if large_arteriole_mask is not None and large_venule_mask is not None:
        large_art_style = VESSEL_VOLUME_TRACE_STYLES["large_arteriole"]
        large_ven_style = VESSEL_VOLUME_TRACE_STYLES["large_venule"]
        if _add_volume_trace(
            large_arteriole_mask.astype(bool, copy=False),
            name=str(large_art_style["name"]),
            color=str(large_art_style["color"]),
            opacity=float(large_art_style["opacity"]),
            fig=fig,
        ):
            volume_trace_indices.append(len(fig.data) - 1)
        if _add_volume_trace(
            large_venule_mask.astype(bool, copy=False),
            name=str(large_ven_style["name"]),
            color=str(large_ven_style["color"]),
            opacity=float(large_ven_style["opacity"]),
            fig=fig,
        ):
            volume_trace_indices.append(len(fig.data) - 1)
    if small_arteriole_mask is not None and small_venule_mask is not None:
        small_art_style = VESSEL_VOLUME_TRACE_STYLES["small_arteriole"]
        small_ven_style = VESSEL_VOLUME_TRACE_STYLES["small_venule"]
        if _add_volume_trace(
            small_arteriole_mask.astype(bool, copy=False),
            name=str(small_art_style["name"]),
            color=str(small_art_style["color"]),
            opacity=float(small_art_style["opacity"]),
            fig=fig,
        ):
            volume_trace_indices.append(len(fig.data) - 1)
        if _add_volume_trace(
            small_venule_mask.astype(bool, copy=False),
            name=str(small_ven_style["name"]),
            color=str(small_ven_style["color"]),
            opacity=float(small_ven_style["opacity"]),
            fig=fig,
        ):
            volume_trace_indices.append(len(fig.data) - 1)
    vessel_styles = {
        "arteriole": dict(color="rgba(255, 59, 48, 0.9)", name="Edges (arteriole)"),
        "capillary": dict(color="rgba(0, 200, 255, 0.75)", name="Edges (capillary)"),
        "venule": dict(color="rgba(46, 204, 113, 0.9)", name="Edges (venule)"),
    }
    for kind in ("arteriole", "capillary", "venule"):
        ex, ey, ez = edge_segments[kind]
        if not ex:
            continue
        style = vessel_styles[kind]
        fig.add_trace(
            go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                line=dict(color=style["color"], width=5),
                name=style["name"],
            )
        )
    if branch_label_text:
        fig.add_trace(
            go.Scatter3d(
                x=branch_label_x,
                y=branch_label_y,
                z=branch_label_z,
                mode="markers+text",
                marker=dict(size=2, color="rgba(255,255,255,0.35)"),
                text=branch_label_text,
                textposition="top center",
                textfont=dict(size=9, color="#FFFFFF"),
                name="Branch order labels",
                hovertemplate="Branch %{text}<extra></extra>",
            )
        )

    if other_nodes:
        ox, oy, oz, oid = _coords_with_ids(other_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="markers",
                marker=dict(size=4, color="#9E9E9E"),
                name="Other nodes",
                customdata=oid,
                hovertemplate="Node %{customdata}<extra></extra>",
            )
        )
    if input_nodes:
        ix, iy, iz, iid = _coords_with_ids(input_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=ix,
                y=iy,
                z=iz,
                mode="markers",
                marker=dict(size=9, color="#FF3B30"),
                name="Input nodes",
                customdata=iid,
                hovertemplate="Input node %{customdata}<extra></extra>",
            )
        )
    if output_nodes:
        ox, oy, oz, oid = _coords_with_ids(output_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="markers",
                marker=dict(size=9, color="#2ECC71"),
                name="Output nodes",
                customdata=oid,
                hovertemplate="Output node %{customdata}<extra></extra>",
            )
        )
    if boundary_set:
        bx, by, bz, bid = _coords_with_ids(sorted(boundary_set))
        fig.add_trace(
            go.Scatter3d(
                x=bx,
                y=by,
                z=bz,
                mode="markers",
                marker=dict(size=8, color="#000000"),
                name="Boundary nodes",
                customdata=bid,
                hovertemplate="Boundary node %{customdata}<extra></extra>",
            )
        )
    # Add an invisible hover hitbox over all nodes so node ids are easy to inspect.
    all_nodes = list(G.nodes)
    if all_nodes:
        hx, hy, hz, hid = _coords_with_ids(all_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=hx,
                y=hy,
                z=hz,
                mode="markers",
                marker=dict(size=11, color="rgba(0,0,0,0.0)"),
                name="Node IDs (hover)",
                showlegend=False,
                customdata=hid,
                hovertemplate="<b>Node %{customdata}</b><extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        showlegend=True,
        hovermode="closest",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.01,
                y=1.08,
                showactive=True,
                buttons=[
                    dict(
                        label="Show all",
                        method="restyle",
                        args=[{"visible": True}, volume_trace_indices],
                    ),
                    dict(
                        label="Hide volumes",
                        method="restyle",
                        args=[{"visible": "legendonly"}, volume_trace_indices],
                    ),
                ],
            )
        ],
        hoverlabel=dict(bgcolor="rgba(20,20,20,0.9)", font=dict(color="#FFFFFF")),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
    )
    if save_html_path:
        fig.write_html(str(save_html_path), include_plotlyjs="cdn")
    if show and not _is_pytest_runtime():
        fig.show()
    return fig


def _edge_direction_sign_from_attributes(
    edge_data: dict[str, Any],
    *,
    signed_flow_attr: str,
    direction_attr: str,
    positive_flow_means_u_to_v: bool,
) -> Optional[int]:
    """Infer edge direction sign (+1 for u->v, -1 for v->u, None unknown)."""
    flow_val = edge_data.get(signed_flow_attr)
    if flow_val is not None:
        try:
            flow_float = float(flow_val)
            if np.isfinite(flow_float) and flow_float != 0.0:
                if flow_float > 0.0:
                    return 1 if positive_flow_means_u_to_v else -1
                return -1 if positive_flow_means_u_to_v else 1
        except (TypeError, ValueError):
            pass

    dir_val = edge_data.get(direction_attr)
    if dir_val is None:
        return None
    text = str(dir_val).strip().lower()
    if text in {"u_to_v", "uv", "forward", "fwd", "+"}:
        return 1
    if text in {"v_to_u", "vu", "reverse", "rev", "-"}:
        return -1
    return None


def _edge_points_xyz(
    u: Any, v: Any, edge_data: dict[str, Any], pos: dict[Any, Any]
) -> Optional[np.ndarray]:
    """Return edge polyline points in plotly (x, y, z) from physical (z, y, x)."""
    voxels = edge_data.get("voxels", [])
    if len(voxels) > 1:
        pts = np.asarray(voxels, dtype=float)
        if pts.ndim == 2 and pts.shape[1] >= 3:
            return _zyx_points_to_xyz(pts[:, :3])
    if u not in pos or v not in pos:
        return None
    pu = np.asarray(pos[u], dtype=float).reshape(1, 3)
    pv = np.asarray(pos[v], dtype=float).reshape(1, 3)
    return _zyx_points_to_xyz(np.vstack([pu, pv]))


def _polyline_length(points: np.ndarray) -> float:
    """Return Euclidean polyline length."""
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _edge_arrow_anchor_and_vector(
    points: np.ndarray,
    *,
    direction_sign: int,
    arrow_length: float,
    lateral_offset: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Compute a nearby arrow anchor and direction vector for an edge."""
    if points.shape[0] < 2:
        return None
    if direction_sign < 0:
        points = points[::-1, :]

    mid_idx = int(points.shape[0] // 2)
    lo = max(0, mid_idx - 1)
    hi = min(points.shape[0] - 1, mid_idx + 1)
    tangent = points[hi] - points[lo]
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        tangent = points[-1] - points[0]
        tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        return None
    tangent_unit = tangent / tangent_norm

    # Offset arrow laterally so it is visible next to the edge.
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    lateral = np.cross(tangent_unit, ref)
    lat_norm = float(np.linalg.norm(lateral))
    if lat_norm <= 1e-12:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
        lateral = np.cross(tangent_unit, ref)
        lat_norm = float(np.linalg.norm(lateral))
    if lat_norm <= 1e-12:
        return None
    lateral_unit = lateral / lat_norm

    anchor = points[mid_idx] + (lateral_offset * lateral_unit)
    vector = tangent_unit * float(arrow_length)
    return anchor, vector


def visualize_3d_plotly_large_vessel_assignment_flow_direction(
    G: nx.Graph,
    *,
    large_arteriole_mask: np.ndarray | None,
    large_venule_mask: np.ndarray | None,
    small_arteriole_mask: np.ndarray | None = None,
    small_venule_mask: np.ndarray | None = None,
    input_nodes: list[Any],
    output_nodes: list[Any],
    arteriole_boundary_nodes: list[Any] | None = None,
    venule_boundary_nodes: list[Any] | None = None,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    volume_downsample_stride: int = 1,
    signed_flow_attr: str = "flow_signed",
    direction_attr: str = "edge_direction",
    positive_flow_means_u_to_v: bool = True,
    arrow_color: str = "#FFD54F",
    arrow_length_scale: float = 0.18,
    arrow_offset_scale: float = 0.08,
    title: str = "Final Graph with Flow Direction (3D)",
    save_html_path: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Render final large/small-vessel assignment view with edge direction arrows.

    Direction source priority:
    1) ``signed_flow_attr`` numeric sign (default ``flow_signed``)
    2) ``direction_attr`` string labels (e.g., ``u_to_v`` / ``v_to_u``)
    """
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        raise ValueError("Graph has no node positions ('pos').")

    fig = visualize_3d_plotly_large_vessel_assignment(
        G,
        large_arteriole_mask=large_arteriole_mask,
        large_venule_mask=large_venule_mask,
        small_arteriole_mask=small_arteriole_mask,
        small_venule_mask=small_venule_mask,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        arteriole_boundary_nodes=arteriole_boundary_nodes,
        venule_boundary_nodes=venule_boundary_nodes,
        voxel_size_zyx=voxel_size_zyx,
        volume_downsample_stride=volume_downsample_stride,
        title=title,
        save_html_path=None,
        show=False,
    )

    edge_iter = (
        G.edges(keys=True, data=True)
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
        else G.edges(data=True)
    )

    directed_points: list[np.ndarray] = []
    direction_signs: list[int] = []
    for edge_item in edge_iter:
        u = edge_item[0]
        v = edge_item[1]
        edge_data = edge_item[-1]
        points = _edge_points_xyz(u, v, edge_data, pos)
        if points is None:
            continue
        direction_sign = _edge_direction_sign_from_attributes(
            edge_data,
            signed_flow_attr=signed_flow_attr,
            direction_attr=direction_attr,
            positive_flow_means_u_to_v=positive_flow_means_u_to_v,
        )
        if direction_sign is None:
            continue
        directed_points.append(points)
        direction_signs.append(int(direction_sign))

    edge_lengths = [_polyline_length(pts) for pts in directed_points]
    valid_lengths = [float(v) for v in edge_lengths if v > 0]
    base_len = float(np.median(valid_lengths)) if valid_lengths else 1.0
    arrow_length = max(0.25, base_len * float(arrow_length_scale))
    lateral_offset = max(0.10, base_len * float(arrow_offset_scale))

    arrow_x: list[float] = []
    arrow_y: list[float] = []
    arrow_z: list[float] = []
    arrow_u: list[float] = []
    arrow_v: list[float] = []
    arrow_w: list[float] = []

    for points, direction_sign in zip(directed_points, direction_signs):
        result = _edge_arrow_anchor_and_vector(
            points,
            direction_sign=direction_sign,
            arrow_length=arrow_length,
            lateral_offset=lateral_offset,
        )
        if result is None:
            continue
        anchor, vector = result
        arrow_x.append(float(anchor[0]))
        arrow_y.append(float(anchor[1]))
        arrow_z.append(float(anchor[2]))
        arrow_u.append(float(vector[0]))
        arrow_v.append(float(vector[1]))
        arrow_w.append(float(vector[2]))

    if arrow_x:
        fig.add_trace(
            go.Cone(
                x=arrow_x,
                y=arrow_y,
                z=arrow_z,
                u=arrow_u,
                v=arrow_v,
                w=arrow_w,
                anchor="tail",
                sizemode="absolute",
                sizeref=max(arrow_length * 0.8, 0.1),
                colorscale=[[0.0, arrow_color], [1.0, arrow_color]],
                cmin=0.0,
                cmax=1.0,
                showscale=False,
                name=f"Flow direction arrows ({len(arrow_x)})",
                hoverinfo="skip",
            )
        )

    fig.update_layout(title=title)
    if save_html_path:
        fig.write_html(str(save_html_path), include_plotlyjs="cdn")
    if show and not _is_pytest_runtime():
        fig.show()
    return fig
