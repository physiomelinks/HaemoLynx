"""Plotting functions for vascular networks."""
from typing import Optional, Tuple, Any
import os

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
import plotly.graph_objects as go

from ._helpers import (
    sort_branch_orders_numerically,
    create_color_mapping,
    group_branch_orders_for_legend,
)


def _resolve_voxel_size(
    G: Optional[nx.Graph] = None,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float, float]:
    """Resolve voxel size from explicit input or graph metadata."""
    if voxel_size is not None:
        return tuple(float(v) for v in voxel_size)
    if G is not None:
        meta = G.graph.get("voxel_size")
        if meta is not None and len(meta) == 3:
            return tuple(float(v) for v in meta)
    return (1.0, 1.0, 1.0)


def _projection_extent(
    projection_shape: Tuple[int, int],
    voxel_size: Tuple[float, float, float],
) -> Tuple[float, float, float, float]:
    """Return imshow extent for (Y, X) projection in physical units."""
    y_size, x_size = projection_shape
    vy = float(voxel_size[1])
    vx = float(voxel_size[2])
    # Keep top-left origin semantics used by existing overlays.
    return (0.0, x_size * vx, y_size * vy, 0.0)


def _projection_extent_xz(
    projection_shape: Tuple[int, int],
    voxel_size: Tuple[float, float, float],
) -> Tuple[float, float, float, float]:
    """Return imshow extent for (Z, X) projection in physical units."""
    z_size, x_size = projection_shape
    vz = float(voxel_size[0])
    vx = float(voxel_size[2])
    return (0.0, x_size * vx, z_size * vz, 0.0)


def _compute_two_angle_projections(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return YX and XZ max projections for 2D/3D images."""
    if image.ndim == 3:
        projection_yx = np.max(image, axis=0)
        projection_xz = np.max(image, axis=1)
        return projection_yx, projection_xz
    if image.ndim == 2:
        projection = image
        return projection, projection
    raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")


def _two_angle_figure(
    projection_yx: np.ndarray,
    projection_xz: np.ndarray,
    *,
    voxel_size: Tuple[float, float, float],
    figsize: Tuple[float, float],
    dpi: Optional[int] = None,
    cmap: str = "gray",
) -> Tuple[Any, np.ndarray]:
    """Create a two-angle subplot figure and draw both projection backgrounds."""
    subplot_kwargs = {"figsize": figsize}
    if dpi is not None:
        subplot_kwargs["dpi"] = dpi
    fig, axes = plt.subplots(1, 2, **subplot_kwargs)
    extent_yx = _projection_extent(projection_yx.shape, voxel_size)
    extent_xz = _projection_extent_xz(projection_xz.shape, voxel_size)
    axes[0].imshow(projection_yx, cmap=cmap, extent=extent_yx)
    axes[1].imshow(projection_xz, cmap=cmap, extent=extent_xz)
    return fig, axes


def _show_matplotlib_non_blocking(pause_s: float = 0.001) -> None:
    """Show matplotlib figures without blocking script execution."""
    plt.show(block=False)
    plt.pause(pause_s)


def _is_pytest_runtime() -> bool:
    """Return True when running under pytest."""
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def plot_node_degree_distribution(
    G: nx.Graph,
    title: str = "Node Degree Distribution",
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
    show_after_save: bool = False,
    block: bool = False,
) -> dict:
    """Plot histogram of node degrees."""
    degrees = [d for _, d in G.degree()]
    plt.figure(figsize=(10, 6))
    plt.hist(
        degrees,
        bins=range(1, max(degrees) + 2) if degrees else [1],
        align="left",
        rwidth=0.8,
        alpha=0.7,
    )
    plt.xlabel("Node Degree")
    plt.ylabel("Count")
    plt.title(f"{title} (N={G.number_of_nodes()}, E={G.number_of_edges()})")
    plt.grid(True, alpha=0.3)
    degree_counts = {}
    for d in degrees:
        degree_counts[d] = degree_counts.get(d, 0) + 1
    stats_text = "Degree distribution:\n" + "".join(
        f"  {d}: {degree_counts[d]} nodes\n" for d in sorted(degree_counts)
    )
    plt.text(
        0.98,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=9,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show and show_after_save:
            if block:
                plt.show()
            else:
                _show_matplotlib_non_blocking()
        else:
            plt.close()
    elif show:
        if block:
            plt.show()
        else:
            _show_matplotlib_non_blocking()
    else:
        plt.close()
    return degree_counts


def visualize_edges_and_nodes(image: np.ndarray, G: nx.Graph, label_nodes: bool = False, 
                              save_path: Optional[str] = None,
                              show_coordinates_degree_1: bool = False,
                              voxel_size: Optional[Tuple[float, float, float]] = None,
                              show: bool = True,
                              show_after_save: bool = False,
                              block: bool = False) -> None:
    """Overlay edges and nodes on YX/XZ projections of image.

    Set label_nodes=True to draw node IDs.
    """
    projection_yx, projection_xz = _compute_two_angle_projections(image)
    pos = nx.get_node_attributes(G, "pos")
    resolved_voxel_size = _resolve_voxel_size(G, voxel_size)
    fig, axes = _two_angle_figure(
        projection_yx,
        projection_xz,
        voxel_size=resolved_voxel_size,
        figsize=(14, 6),
        cmap="gray",
    )
    for u, v, d in G.edges(data=True):
        path = d.get("voxels", [])
        if len(path) > 1:
            path_arr = np.array(path, dtype=float)
            axes[0].plot(path_arr[:, 2], path_arr[:, 1], color="cyan", linewidth=0.5)
            axes[1].plot(path_arr[:, 2], path_arr[:, 0], color="cyan", linewidth=0.5)
    if pos:
        coords = np.array(list(pos.values()))
        axes[0].scatter(coords[:, 2], coords[:, 1], c="red", s=3)
        axes[1].scatter(coords[:, 2], coords[:, 0], c="red", s=3)
        if label_nodes:
            for node_id, node_pos in pos.items():
                axes[0].text(
                    float(node_pos[2]) + 1.0,
                    float(node_pos[1]) + 1.0,
                    str(node_id),
                    color="yellow",
                    fontsize=3,
                )
                axes[1].text(
                    float(node_pos[2]) + 1.0,
                    float(node_pos[0]) + 1.0,
                    str(node_id),
                    color="yellow",
                    fontsize=3,
                )
        if show_coordinates_degree_1:
            for node_id, node_pos in pos.items():
                if G.degree(node_id) == 1:
                    x = float(node_pos[2])
                    y = float(node_pos[1])
                    z = float(node_pos[0])
                    axes[0].text(
                        float(node_pos[2]) + 1.0,
                        float(node_pos[1]) + 1.0,
                        f"({x:.1f}, {y:.1f}, {z:.1f})",
                        color="blue",
                        fontsize=3,
                    )
                    axes[1].text(
                        float(node_pos[2]) + 1.0,
                        float(node_pos[0]) + 1.0,
                        f"({x:.1f}, {y:.1f}, {z:.1f})",
                        color="blue",
                        fontsize=3,
                    )
    axes[0].set_title("YX projection (max over Z)")
    axes[1].set_title("XZ projection (max over Y)")
    axes[0].axis("off")
    axes[1].axis("off")
    fig.suptitle("Overlay: Edges and Nodes")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show and show_after_save:
            if block:
                plt.show()
            else:
                _show_matplotlib_non_blocking()
        else:
            plt.close(fig)
    elif show:
        if block:
            plt.show()
        else:
            _show_matplotlib_non_blocking()
    else:
        plt.close(fig)


def visualize_geometry_with_branch_orders(
    image: np.ndarray,
    G: nx.MultiGraph,
    figsize=(12, 10),
    color_palette="viridis",
    node_color="red",
    node_size=3,
    edge_linewidth=0.8,
    show_legend=True,
    background_cmap="gray",
    save_path=None,
    dpi=300,
    alpha=0.8,
    reverse_gradient=True,
    group_above=None,
    voxel_size: Optional[Tuple[float, float, float]] = None,
    show=True,
    show_after_save: bool = False,
    block: bool = False,
):
    """Plot network colored by branch order."""
    projection_yx, projection_xz = _compute_two_angle_projections(image)
    resolved_voxel_size = _resolve_voxel_size(G, voxel_size)
    all_branch_orders = set()
    edge_branch_orders = {}
    edge_paths = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        bo = data.get("branch_order", "No_BO")
        path = data.get("voxels", [])
        if bo != "No_BO":
            all_branch_orders.add(bo)
        if bo != "No_BO" and len(path) > 1:
            edge_branch_orders[(u, v, key)] = bo
            edge_paths[(u, v, key)] = path
    visualizable = {
        (u, v, k) for (u, v, k), bo in edge_branch_orders.items()
        if len(edge_paths[(u, v, k)]) > 1
    }
    all_bo = sort_branch_orders_numerically(list(all_branch_orders))
    branch_orders = all_bo
    color_mapping = create_color_mapping(
        branch_orders, color_palette, reverse_gradient, group_above
    )
    actual_edge_counts = {}
    for bo in branch_orders:
        actual_edge_counts[bo] = sum(
            1
            for (u, v, k), b in edge_branch_orders.items()
            if b == bo and len(edge_paths.get((u, v, k), [])) > 1
        )
    legend_orders, legend_counts = group_branch_orders_for_legend(
        branch_orders, group_above, actual_edge_counts
    )
    subplot_figsize = (float(figsize[0]) * 1.8, float(figsize[1]))
    fig, axes = _two_angle_figure(
        projection_yx,
        projection_xz,
        voxel_size=resolved_voxel_size,
        figsize=subplot_figsize,
        dpi=dpi,
        cmap=background_cmap,
    )
    for bo in branch_orders:
        paths = [
            np.array(edge_paths[(u, v, k)])
            for (u, v, k), b in edge_branch_orders.items()
            if b == bo and len(edge_paths[(u, v, k)]) > 1
        ]
        color = color_mapping.get(bo, "gray")
        for path in paths:
            axes[0].plot(
                path[:, 2], path[:, 1],
                color=color,
                linewidth=edge_linewidth,
                alpha=alpha,
            )
            axes[1].plot(
                path[:, 2], path[:, 0],
                color=color,
                linewidth=edge_linewidth,
                alpha=alpha,
            )
    pos = nx.get_node_attributes(G, "pos")
    if pos:
        coords = np.array(list(pos.values()))
        axes[0].scatter(coords[:, 2], coords[:, 1], c=node_color, s=node_size)
        axes[1].scatter(coords[:, 2], coords[:, 0], c=node_color, s=node_size)
    if show_legend and legend_orders:
        handles = [
            plt.Line2D(
                [0], [0], color=color_mapping.get(bo, "gray"), lw=3,
                label=f"{bo} ({legend_counts.get(bo, 0)} edges)",
                alpha=alpha,
            )
            for bo in legend_orders
            if bo in color_mapping
        ]
        axes[0].legend(handles=handles, title="Branch Orders", loc="upper right")
    axes[0].set_title("YX projection (max over Z)")
    axes[1].set_title("XZ projection (max over Y)")
    axes[0].axis("off")
    axes[1].axis("off")
    fig.suptitle("Network Geometry with Branch Order Colors")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show and show_after_save:
            if block:
                plt.show()
            else:
                _show_matplotlib_non_blocking()
        else:
            plt.close(fig)
    elif show:
        if block:
            plt.show()
        else:
            _show_matplotlib_non_blocking()
    else:
        plt.close(fig)
    return fig, axes[0], color_mapping


def visualize_geometry_with_edge_weights(
    image: np.ndarray,
    G: nx.MultiGraph,
    figsize=(12, 10),
    color_palette="viridis",
    node_color="red",
    node_size=3,
    edge_linewidth=0.8,
    show_legend=True,
    background_cmap="gray",
    save_path=None,
    dpi=300,
    alpha=0.8,
    min_weight=None,
    max_weight=None,
    legend_bins=5,
    reverse_gradient=False,
    use_inverse=True,
    voxel_size: Optional[Tuple[float, float, float]] = None,
    show=True,
    show_after_save: bool = False,
    block: bool = False,
):
    """Plot network colored by edge resistance."""
    projection_yx, projection_xz = _compute_two_angle_projections(image)
    resolved_voxel_size = _resolve_voxel_size(G, voxel_size)
    edge_resistances = {}
    edge_paths = {}
    resistance_values = []
    for u, v, key, data in G.edges(keys=True, data=True):
        resistance = data.get("resistance", data.get("weight"))
        path = data.get("voxels", [])
        if resistance is not None:
            proc = 1.0 / resistance if use_inverse else resistance
            if use_inverse and resistance == 0:
                proc = None
            else:
                resistance_values.append(proc)
        else:
            proc = None
        edge_resistances[(u, v, key)] = proc
        edge_paths[(u, v, key)] = path
    if not resistance_values:
        return None, None, None, None
    data_min = min(resistance_values)
    data_max = max(resistance_values)
    vmin = min_weight if min_weight is not None else data_min
    vmax = max_weight if max_weight is not None else data_max
    cmap = plt.get_cmap(color_palette)
    if reverse_gradient:
        cmap = cmap.reversed()
    norm = Normalize(vmin=vmin, vmax=vmax)
    subplot_figsize = (float(figsize[0]) * 1.8, float(figsize[1]))
    fig, axes = _two_angle_figure(
        projection_yx,
        projection_xz,
        voxel_size=resolved_voxel_size,
        figsize=subplot_figsize,
        dpi=dpi,
        cmap=background_cmap,
    )
    for (u, v, key), resistance in edge_resistances.items():
        if resistance is not None:
            path = edge_paths[(u, v, key)]
            if len(path) > 1:
                path_arr = np.array(path)
                color = cmap(norm(resistance))
                axes[0].plot(
                    path_arr[:, 2], path_arr[:, 1],
                    color=color,
                    linewidth=edge_linewidth,
                    alpha=alpha,
                )
                axes[1].plot(
                    path_arr[:, 2], path_arr[:, 0],
                    color=color,
                    linewidth=edge_linewidth,
                    alpha=alpha,
                )
    pos = nx.get_node_attributes(G, "pos")
    if pos:
        coords = np.array(list(pos.values()))
        axes[0].scatter(coords[:, 2], coords[:, 1], c=node_color, s=node_size)
        axes[1].scatter(coords[:, 2], coords[:, 0], c=node_color, s=node_size)
    if show_legend:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, aspect=24)
        cbar.set_label("Conductance (1/Resistance)" if use_inverse else "Edge Resistance", rotation=270)
    axes[0].set_title("YX projection (max over Z)")
    axes[1].set_title("XZ projection (max over Y)")
    axes[0].axis("off")
    axes[1].axis("off")
    fig.suptitle("Network Geometry Colored by Edge Resistance")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show and show_after_save:
            if block:
                plt.show()
            else:
                _show_matplotlib_non_blocking()
        else:
            plt.close(fig)
    elif show:
        if block:
            plt.show()
        else:
            _show_matplotlib_non_blocking()
    else:
        plt.close(fig)
    return fig, axes[0], (vmin, vmax), cmap


def visualize_3d_plotly(
    G: nx.Graph,
    title: str = "3D Network",
    save_html_path: Optional[str] = None,
    show: bool = True,
) -> go.Figure:
    """Interactive 3D graph rendering using Plotly.

    Uses edge voxel polylines when present, otherwise falls back to node-to-node
    straight segments. Coordinates are interpreted as (z, y, x) in graph
    metadata and mapped to Plotly axes as (x, y, z).
    """
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        raise ValueError("Graph has no node positions ('pos').")
    edge_x, edge_y, edge_z = [], [], []
    if isinstance(G, nx.MultiGraph):
        edge_iter = G.edges(keys=True, data=True)
        for u, v, _k, edge_data in edge_iter:
            voxels = edge_data.get("voxels", [])
            if len(voxels) > 1:
                for pt in voxels:
                    # Stored as (z, y, x)
                    edge_x.append(float(pt[2]))
                    edge_y.append(float(pt[1]))
                    edge_z.append(float(pt[0]))
                edge_x.append(None)
                edge_y.append(None)
                edge_z.append(None)
            elif u in pos and v in pos:
                pu, pv = pos[u], pos[v]
                edge_x += [float(pu[2]), float(pv[2]), None]
                edge_y += [float(pu[1]), float(pv[1]), None]
                edge_z += [float(pu[0]), float(pv[0]), None]
    else:
        for u, v, edge_data in G.edges(data=True):
            voxels = edge_data.get("voxels", [])
            if len(voxels) > 1:
                for pt in voxels:
                    edge_x.append(float(pt[2]))
                    edge_y.append(float(pt[1]))
                    edge_z.append(float(pt[0]))
                edge_x.append(None)
                edge_y.append(None)
                edge_z.append(None)
            elif u in pos and v in pos:
                pu, pv = pos[u], pos[v]
                edge_x += [float(pu[2]), float(pv[2]), None]
                edge_y += [float(pu[1]), float(pv[1]), None]
                edge_z += [float(pu[0]), float(pv[0]), None]

    # Stored as (z, y, x) -> plot as (x, y, z)
    node_x = [float(p[2]) for p in pos.values()]
    node_y = [float(p[1]) for p in pos.values()]
    node_z = [float(p[0]) for p in pos.values()]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="cyan", width=2),
        name="Edges",
    ))
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode="markers",
        marker=dict(size=3, color="red"),
        name="Nodes",
    ))
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


def visualize_3d_plotly_vessel_types(
    G: nx.Graph,
    title: str = "3D Vessel Types",
    save_html_path: Optional[str] = None,
    show: bool = True,
) -> go.Figure:
    """
    Interactive 3D graph rendering colored by vessel class from branch_order.

    Mapping:
    - Art* -> arteriole (red)
    - B*   -> capillary (green)
    - Ven* -> venule (blue)
    """
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        raise ValueError("Graph has no node positions ('pos').")

    def _vessel_type(branch_order: Any) -> str:
        label = str(branch_order or "")
        if label.startswith("Art"):
            return "arteriole"
        if label.startswith("Ven"):
            return "venule"
        if label.startswith("B"):
            return "capillary"
        return "unknown"

    type_to_color = {
        "arteriole": "#d62728",  # red
        "capillary": "#2ca02c",  # green
        "venule": "#1f77b4",  # blue
        "unknown": "#7f7f7f",
    }
    type_to_label = {
        "arteriole": "Arterioles",
        "capillary": "Capillaries",
        "venule": "Venules",
        "unknown": "Unknown",
    }

    # Collect edge polylines per vessel type.
    per_type_coords: dict[str, dict[str, list[float | None]]] = {
        k: {"x": [], "y": [], "z": []} for k in type_to_color
    }
    per_type_counts = {k: 0 for k in type_to_color}

    if isinstance(G, nx.MultiGraph):
        edge_iter = G.edges(keys=True, data=True)
        for u, v, _k, edge_data in edge_iter:
            vessel_type = _vessel_type(edge_data.get("branch_order"))
            voxels = edge_data.get("voxels", [])
            if len(voxels) > 1:
                for pt in voxels:
                    per_type_coords[vessel_type]["x"].append(float(pt[2]))
                    per_type_coords[vessel_type]["y"].append(float(pt[1]))
                    per_type_coords[vessel_type]["z"].append(float(pt[0]))
                per_type_coords[vessel_type]["x"].append(None)
                per_type_coords[vessel_type]["y"].append(None)
                per_type_coords[vessel_type]["z"].append(None)
                per_type_counts[vessel_type] += 1
            elif u in pos and v in pos:
                pu, pv = pos[u], pos[v]
                per_type_coords[vessel_type]["x"] += [float(pu[2]), float(pv[2]), None]
                per_type_coords[vessel_type]["y"] += [float(pu[1]), float(pv[1]), None]
                per_type_coords[vessel_type]["z"] += [float(pu[0]), float(pv[0]), None]
                per_type_counts[vessel_type] += 1
    else:
        for u, v, edge_data in G.edges(data=True):
            vessel_type = _vessel_type(edge_data.get("branch_order"))
            voxels = edge_data.get("voxels", [])
            if len(voxels) > 1:
                for pt in voxels:
                    per_type_coords[vessel_type]["x"].append(float(pt[2]))
                    per_type_coords[vessel_type]["y"].append(float(pt[1]))
                    per_type_coords[vessel_type]["z"].append(float(pt[0]))
                per_type_coords[vessel_type]["x"].append(None)
                per_type_coords[vessel_type]["y"].append(None)
                per_type_coords[vessel_type]["z"].append(None)
                per_type_counts[vessel_type] += 1
            elif u in pos and v in pos:
                pu, pv = pos[u], pos[v]
                per_type_coords[vessel_type]["x"] += [float(pu[2]), float(pv[2]), None]
                per_type_coords[vessel_type]["y"] += [float(pu[1]), float(pv[1]), None]
                per_type_coords[vessel_type]["z"] += [float(pu[0]), float(pv[0]), None]
                per_type_counts[vessel_type] += 1

    node_x = [float(p[2]) for p in pos.values()]
    node_y = [float(p[1]) for p in pos.values()]
    node_z = [float(p[0]) for p in pos.values()]

    fig = go.Figure()
    for vessel_type in ("arteriole", "capillary", "venule", "unknown"):
        coords = per_type_coords[vessel_type]
        if not coords["x"]:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=coords["x"],
                y=coords["y"],
                z=coords["z"],
                mode="lines",
                line=dict(color=type_to_color[vessel_type], width=3),
                name=f"{type_to_label[vessel_type]} ({per_type_counts[vessel_type]})",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            marker=dict(size=3, color="black"),
            name="Nodes",
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


def visualize_skeleton(
    skeleton: np.ndarray,
    save_path: Optional[str] = None,
    dpi: int = 150,
    voxel_color: str = "cyan",
    background_color: str = "black",
    point_size: float = 3.0,
    voxel_size: Optional[Tuple[float, float, float]] = None,
    show: bool = True,
    block: bool = False,
) -> None:
    """Visualize a 3D skeleton in an interactive PyVista 3D view.

    Each foreground voxel is rendered as a point cloud. For 2D skeletons, or
    when save_path is supplied (headless/CI use), falls back to a flat
    Z-projection saved with matplotlib instead.

    Parameters
    ----------
    skeleton:
        2D or 3D boolean array produced by the pipeline.
    save_path:
        When given, saves a 2D Z-projection PNG instead of opening the 3D
        viewer (useful for automated runs without a display).
    dpi:
        Resolution used when saving the 2D fallback figure.
    voxel_color:
        Colour of skeleton voxel points in the 3D view.
    background_color:
        Plotter background colour.
    point_size:
        Rendered sphere radius for each voxel point.
    show:
        Pass False to suppress the interactive window (e.g. tests).
    block:
        When True, use blocking display calls; default False keeps the script
        running while windows stay open.
    """
    if skeleton.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2D or 3D skeleton, got shape {skeleton.shape}"
        )

    # 2D skeleton or headless save → matplotlib projection fallback.
    if skeleton.ndim == 2 or save_path is not None:
        projection_yx, projection_xz = _compute_two_angle_projections(
            skeleton.astype(float)
        )
        resolved_voxel_size = voxel_size or (1.0, 1.0, 1.0)
        fig, axes = _two_angle_figure(
            projection_yx,
            projection_xz,
            voxel_size=resolved_voxel_size,
            figsize=(14, 6),
            cmap="gray",
        )
        axes[0].set_title("YX projection (max over Z)")
        axes[1].set_title("XZ projection (max over Y)")
        axes[0].axis("off")
        axes[1].axis("off")
        fig.suptitle(
            f"Skeleton projections  —  shape: {skeleton.shape}  "
            f"voxels: {int(skeleton.sum())}"
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            if block:
                plt.show()
            else:
                _show_matplotlib_non_blocking()
        return

    # 3D skeleton → PyVista interactive viewer.
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "pyvista is required for 3D skeleton visualization. "
            "Install with `pip install pyvista`."
        ) from exc

    # Get (Z, Y, X) foreground voxel coordinates → convert to (X, Y, Z).
    coords = np.argwhere(skeleton).astype(float)
    xyz = coords[:, [2, 1, 0]]

    cloud = pv.PolyData(xyz)
    plotter = pv.Plotter(
        title=f"Skeleton — shape: {skeleton.shape}, voxels: {int(skeleton.sum())}"
    )
    plotter.set_background(background_color)
    plotter.add_mesh(
        cloud,
        color=voxel_color,
        point_size=point_size,
        render_points_as_spheres=True,
    )
    plotter.add_axes()
    if show:
        if block:
            plotter.show()
        else:
            plotter.show(auto_close=False, interactive_update=True)


def visualize_volume(
    volume: np.ndarray,
    title: str = "3D Volume Mask",
    background_color: str = "black",
    vessel_color: str = "salmon",
    opacity: float = 1.0,
    show: bool = True,
    save_path: str | None = None,
) -> Any:
    """Visualize a 3D binary volume as a smooth surface mesh using PyVista.

    Parameters
    ----------
    volume:
        3D boolean or integer array.
    title:
        Window title.
    vessel_color:
        Color of the rendered vessel surface.
    save_path:
        Optional path to save a screenshot of the visualization.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "pyvista is required for 3D volume visualization. "
            "Install with `pip install pyvista`."
        ) from exc

    if volume.ndim != 3:
        raise ValueError(f"visualize_volume expects a 3D array, got {volume.ndim}D")

    # Create a uniform grid from the voxel data
    # Note: PyVista expects (X, Y, Z) ordering for dimensions
    # Our data is (Z, Y, X), so we transpose to (X, Y, Z)
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.transpose(2, 1, 0).shape) + 1
    grid.cell_data["values"] = volume.transpose(2, 1, 0).flatten(order="F")

    # Extract the surface where value >= 0.5 (the vessel boundary)
    # Note: contour filter requires point data, so we convert cell data to point data (ctp)
    surface = grid.ctp().contour([0.5], scalars="values")

    plotter = pv.Plotter(title=title, off_screen=(not show))
    plotter.set_background(background_color)
    plotter.add_mesh(
        surface,
        color=vessel_color,
        opacity=opacity,
        smooth_shading=True,
        show_edges=False,
    )
    plotter.add_axes()
    
    if show:
        plotter.show()
    
    if save_path:
        plotter.screenshot(save_path)
    
    return plotter


def visualize_volume_vedo(
    volume: np.ndarray,
    title: str = "Vedo 3D Volume",
    mode: str = "iso",
    spacing: tuple = (1.0, 1.0, 1.0),
    vessel_color: str = "salmon",
    background_color: str = "white",
    alpha: float = 1.0,
    smooth_iter: int = 0,
    show: bool = True,
):
    """Visualize a 3D volume using Vedo (image_to_model style).

    Parameters
    ----------
    mode:
        'iso' for smooth surface mesh, 'lego' for raw voxel blocks.
    spacing:
        (z, y, x) voxel dimensions.
    smooth_iter:
        Iterations of Laplacian smoothing (only for 'iso' mode).
    show:
        Whether to immediately display the visualization window.
    """
    try:
        import vedo
    except ImportError as exc:
        raise ImportError("vedo is required. Install with `pip install vedo`.")

    # Vedo Volume expects (Z, Y, X) data but spacing usually maps to (X, Y, Z)
    # in terms of how it stretches. To match ImageLynx/image_to_model convention:
    # Transpose Z,Y,X -> X,Y,Z for internal consistency
    vol_data = volume.transpose(2, 1, 0)
    # Re-order spacing to match the transposed dims (x, y, z)
    vedo_spacing = (spacing[2], spacing[1], spacing[0])
    
    # Auto-threshold for surface extraction
    vmin = 0.5 if volume.dtype == bool else np.mean(volume)
    
    vol = vedo.Volume(vol_data, spacing=vedo_spacing)
    
    if mode.lower() == "lego":
        actor = vol.legosurface(vmin=vmin).color(vessel_color).alpha(alpha)
    else:
        actor = vol.isosurface(vmin).color(vessel_color).alpha(alpha)
        if smooth_iter > 0:
            actor.smooth(niter=smooth_iter)
    
    plt_vedo = vedo.Plotter(title=title, bg=background_color, offscreen=not show)
    plt_vedo.add(actor)
    if show:
        plt_vedo.show(interactive=True)
    return plt_vedo


def visualize_overlay_vedo(
    volume: np.ndarray,
    skeleton: np.ndarray,
    title: str = "3D Skeleton Overlay (Vedo)",
    mode: str = "iso",
    spacing: tuple = (1.0, 1.0, 1.0),
    vessel_color: str = "salmon",
    skeleton_color: str = "cyan",
    background_color: str = "white",
    alpha: float = 0.3,
    smooth_iter: int = 0,
    skeleton_point_size: float = 5.0,
    show: bool = True,
    separate_windows: bool = False,
    G: Optional[nx.Graph] = None,
):
    """Visualize a 3D skeleton overlaid on its parent volume mesh using Vedo."""
    try:
        import vedo
    except ImportError as exc:
        raise ImportError("vedo is required. Install with `pip install vedo`.")

    vol_data = volume.transpose(2, 1, 0)
    vedo_spacing = (spacing[2], spacing[1], spacing[0])
    
    vmin = 0.5 if volume.dtype == bool else np.mean(volume)
    
    vol = vedo.Volume(vol_data, spacing=vedo_spacing)
    
    if mode.lower() == "lego":
        actor = vol.legosurface(vmin=vmin).color(vessel_color).alpha(alpha)
    else:
        actor = vol.isosurface(vmin).color(vessel_color).alpha(alpha)
        if smooth_iter > 0:
            actor.smooth(niter=smooth_iter)
            
    coords = np.argwhere(skeleton).astype(float)
    if coords.size > 0:
        xyz = coords[:, [2, 1, 0]]
        xyz = xyz * np.array(vedo_spacing)
        pts = vedo.Points(xyz, r=skeleton_point_size).color(skeleton_color).alpha(0.75)
    else:
        pts = None

    graph_actors = []
    if G is not None:
        for u, v, d in G.edges(data=True):
            path = d.get("voxels", [])
            if len(path) > 1:
                path = np.array(path)
                path_xyz = path[:, [2, 1, 0]] * np.array(vedo_spacing)
                graph_actors.append(vedo.Line(path_xyz).color("yellow").lw(2))
        
        pos = nx.get_node_attributes(G, "pos")
        if pos:
            nodes_coords = np.array(list(pos.values()))
            nodes_xyz = nodes_coords[:, [2, 1, 0]] * np.array(vedo_spacing)
            graph_actors.append(vedo.Points(nodes_xyz, r=skeleton_point_size * 2.0).color("red"))

    if separate_windows:
        # Window 1: Mask only (30% opacity)
        actor_mask = actor.clone().alpha(0.3)
        plt1 = vedo.Plotter(title="1. Post-Processed Mask", bg=background_color, offscreen=not show, pos=(0, 0))
        plt1.add(actor_mask)
        
        # Window 2: Skeleton only
        plt2 = vedo.Plotter(title="2. Skeleton Only", bg=background_color, offscreen=not show, pos=(500, 0))
        if pts is not None:
            pts_skel = pts.clone().alpha(0.75)
            plt2.add(pts_skel)
            
        # Window 3: Overlay
        plt3 = vedo.Plotter(title="3. Overlay", bg=background_color, offscreen=not show, pos=(1000, 0))
        plt3.add(actor)
        if pts is not None:
            plt3.add(pts)
            
        plts = [plt1, plt2, plt3]
        
        if G is not None:
            # Window 4: Graph nodes and edges
            plt4 = vedo.Plotter(title="4. Optimized Graph", bg=background_color, offscreen=not show, pos=(1500, 0))
            plt4.add(actor.clone().alpha(0.1)) # faint mask for context
            for ga in graph_actors:
                plt4.add(ga)
            plts.append(plt4)
            
        if show:
            for p in plts[:-1]:
                p.show(interactive=False)
            plts[-1].show(interactive=True)
            
        return plts
        
    else:
        plt_vedo = vedo.Plotter(title=title, bg=background_color, offscreen=not show)
        plt_vedo.add(actor)
        if pts is not None:
            plt_vedo.add(pts)
        if G is not None:
            for ga in graph_actors:
                plt_vedo.add(ga)

        if show:
            plt_vedo.show(interactive=True)
        return plt_vedo


def visualize_overlay(
    volume: np.ndarray,
    skeleton: np.ndarray,
    title: str = "3D Skeleton Overlay",
    background_color: str = "black",
    vessel_color: str = "salmon",
    skeleton_color: str = "cyan",
    vessel_opacity: float = 0.3,
    skeleton_point_size: float = 5.0,
    show: bool = True,
) -> Any:
    """Visualize a 3D skeleton overlaid on its parent volume mesh."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista is required for 3D overlay. Install with `pip install pyvista`.")

    plotter = pv.Plotter(title=title)
    plotter.set_background(background_color)
    
    # Enable depth peeling for correct transparency rendering
    plotter.enable_depth_peeling()

    # 1. Add Vessel Surface
    vol_transposed = volume.transpose(2, 1, 0)
    grid = pv.ImageData()
    grid.dimensions = np.array(vol_transposed.shape) + 1
    # Half-voxel offset ensures voxel center (integer) aligns with surface center
    grid.origin = (-0.5, -0.5, -0.5)
    grid.spacing = (1, 1, 1)
    grid.cell_data["values"] = vol_transposed.flatten(order="F")
    surface = grid.ctp().contour([0.5], scalars="values")
    
    plotter.add_mesh(surface, color=vessel_color, opacity=vessel_opacity, smooth_shading=True, label="Vessel Mask")

    # 2. Add Skeleton Points
    coords = np.argwhere(skeleton).astype(float)
    if coords.size > 0:
        xyz = coords[:, [2, 1, 0]]
        plotter.add_points(
            xyz, 
            color=skeleton_color, 
            point_size=skeleton_point_size, 
            render_points_as_spheres=True, 
            label="Skeleton Centerline"
        )

    plotter.add_axes()
    plotter.add_legend()
    if show:
        plotter.show()
    return plotter


def visualize_volume_rendering(
    volume: np.ndarray,
    title: str = "3D Volume Rendering",
    cmap: str = "bone",
    opacity: str = "linear",
    show: bool = True,
) -> Any:
    """Visualize a 3D volume using direct volume rendering (ray casting)."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista is required for volume rendering. Install with `pip install pyvista`.")

    vol_transposed = volume.transpose(2, 1, 0)
    grid = pv.ImageData()
    grid.dimensions = np.array(vol_transposed.shape) + 1
    grid.cell_data["values"] = vol_transposed.flatten(order="F")

    plotter = pv.Plotter(title=title)
    plotter.add_volume(grid, scalars="values", cmap=cmap, opacity=opacity, blending="composite")
    plotter.add_axes()
    if show:
        plotter.show()
    return plotter


# British-spelling alias used in the example script.
visualise_skeleton = visualize_skeleton
