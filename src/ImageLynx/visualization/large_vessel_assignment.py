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


def visualize_3d_plotly_large_vessel_assignment(
    G: nx.Graph,
    *,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    input_nodes: list[Any],
    output_nodes: list[Any],
    voxel_size_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    title: str = "Final Graph with Automated Large-Vessel Assignment (3D)",
    save_html_path: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Render final graph + large-vessel volumes + assigned input/output nodes."""
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        raise ValueError("Graph has no node positions ('pos').")
    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )

    def _add_volume_trace(mask: np.ndarray, *, name: str, color: str, fig: go.Figure) -> None:
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
        zz, yy, xx = np.indices(cropped.shape, dtype=float)
        xx = xx + float(x_slice.start)
        yy = yy + float(y_slice.start)
        zz = zz + float(z_slice.start)
        fig.add_trace(
            go.Volume(
                x=(xx * x_scale).ravel(),
                y=(yy * y_scale).ravel(),
                z=(zz * z_scale).ravel(),
                value=cropped.astype(float).ravel(),
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

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    if isinstance(G, nx.MultiGraph):
        edge_iter = G.edges(keys=True, data=True)
        for u, v, _k, edge_data in edge_iter:
            voxels = edge_data.get("voxels", [])
            if len(voxels) > 1:
                for pt in voxels:
                    edge_x.append(float(pt[0]))
                    edge_y.append(float(pt[1]))
                    edge_z.append(float(pt[2]))
                edge_x.append(None)
                edge_y.append(None)
                edge_z.append(None)
            else:
                pu = np.asarray(pos[u], dtype=float)
                pv = np.asarray(pos[v], dtype=float)
                edge_x += [float(pu[0]), float(pv[0]), None]
                edge_y += [float(pu[1]), float(pv[1]), None]
                edge_z += [float(pu[2]), float(pv[2]), None]
    else:
        for u, v, edge_data in G.edges(data=True):
            voxels = edge_data.get("voxels", [])
            if len(voxels) > 1:
                for pt in voxels:
                    edge_x.append(float(pt[0]))
                    edge_y.append(float(pt[1]))
                    edge_z.append(float(pt[2]))
                edge_x.append(None)
                edge_y.append(None)
                edge_z.append(None)
            else:
                pu = np.asarray(pos[u], dtype=float)
                pv = np.asarray(pos[v], dtype=float)
                edge_x += [float(pu[0]), float(pv[0]), None]
                edge_y += [float(pu[1]), float(pv[1]), None]
                edge_z += [float(pu[2]), float(pv[2]), None]

    input_set = set(input_nodes)
    output_set = set(output_nodes)
    other_nodes = [n for n in G.nodes if n not in input_set and n not in output_set]

    def _coords(nodes: list[Any]) -> tuple[list[float], list[float], list[float]]:
        xs = [float(np.asarray(pos[n], dtype=float)[0]) for n in nodes if n in pos]
        ys = [float(np.asarray(pos[n], dtype=float)[1]) for n in nodes if n in pos]
        zs = [float(np.asarray(pos[n], dtype=float)[2]) for n in nodes if n in pos]
        return xs, ys, zs

    fig = go.Figure()
    _add_volume_trace(
        large_arteriole_mask.astype(bool, copy=False),
        name="Large arteriole mask",
        color="#00FF7F",
        fig=fig,
    )
    _add_volume_trace(
        large_venule_mask.astype(bool, copy=False),
        name="Large venule mask",
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
                name="Other nodes",
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
                marker=dict(size=9, color="#00FF7F"),
                name="Input nodes",
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
                marker=dict(size=9, color="#FF3EA5"),
                name="Output nodes",
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
