"""3D visualization for automated large-vessel input/output assignment."""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from .plot import _is_pytest_runtime


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
    voxel_size_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    volume_downsample_stride: int = 1,
    title: str = "Final Graph with Automated Large-Vessel Assignment (3D)",
    save_html_path: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Render graph + large/small vessel volumes + I/O nodes + vessel/branch labels."""
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        raise ValueError("Graph has no node positions ('pos').")
    if large_arteriole_mask is None or large_venule_mask is None:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask are required for this view."
        )
    if large_arteriole_mask.shape != large_venule_mask.shape:
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
    ) -> None:
        mask_bool = mask.astype(bool, copy=False)
        bbox = _nonzero_bbox_slices_zyx(mask_bool)
        if bbox is None:
            return
        x_scale, y_scale, z_scale = (
            float(voxel_size_xyz[0]),
            float(voxel_size_xyz[1]),
            float(voxel_size_xyz[2]),
        )
        z_slice, y_slice, x_slice = bbox
        cropped = mask_bool[z_slice, y_slice, x_slice]
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
            )
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
                pts = np.asarray(voxels, dtype=float)
                _push_segment(kind, pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist())
                _add_branch_label(edge_data, pts)
            else:
                pu = np.asarray(pos[u], dtype=float)
                pv = np.asarray(pos[v], dtype=float)
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
                pts = np.asarray(voxels, dtype=float)
                _push_segment(kind, pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist())
                _add_branch_label(edge_data, pts)
            else:
                pu = np.asarray(pos[u], dtype=float)
                pv = np.asarray(pos[v], dtype=float)
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
            node_pos = np.asarray(pos[node_id], dtype=float)
            xs.append(float(node_pos[0]))
            ys.append(float(node_pos[1]))
            zs.append(float(node_pos[2]))
            node_ids.append(str(node_id))
        return xs, ys, zs, node_ids

    fig = go.Figure()
    _add_volume_trace(
        large_arteriole_mask.astype(bool, copy=False),
        name="Large arteriole mask",
        color="#B71C1C",
        opacity=0.22,
        fig=fig,
    )
    _add_volume_trace(
        large_venule_mask.astype(bool, copy=False),
        name="Large venule mask",
        color="#1B5E20",
        opacity=0.22,
        fig=fig,
    )
    if small_arteriole_mask is not None and small_venule_mask is not None:
        _add_volume_trace(
            small_arteriole_mask.astype(bool, copy=False),
            name="Small arteriole mask",
            color="#FF3B30",
            opacity=0.12,
            fig=fig,
        )
        _add_volume_trace(
            small_venule_mask.astype(bool, copy=False),
            name="Small venule mask",
            color="#2ECC71",
            opacity=0.12,
            fig=fig,
        )
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

    fig.update_layout(
        title=title,
        showlegend=True,
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
