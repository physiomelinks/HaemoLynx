"""Plotting functions for vascular networks."""
from typing import Optional, Tuple, Any
import os

import numpy as np
import matplotlib
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


def _overlay_z_projection(image: np.ndarray) -> np.ndarray:
    """Return a Z-projection suitable for graph overlays.

    For low-cardinality integer label maps (e.g., 0/1, 0/255, 1/2), use a
    foreground occupancy projection instead of raw max-intensity projection.
    This avoids flat backgrounds for 1/2 segmentations where max projection is
    often constant (all 2s).
    """
    arr = np.asarray(image)
    if arr.ndim < 3:
        return np.asarray(arr)

    if arr.dtype == bool:
        return np.max(arr, axis=0).astype(float)

    if np.issubdtype(arr.dtype, np.integer):
        values, counts = np.unique(arr, return_counts=True)
        if values.size == 1:
            return np.max(arr, axis=0)
        if values.size == 2:
            if 0 in values:
                fg_value = values[values != 0][0]
            else:
                fg_value = values[int(np.argmin(counts))]
            return np.max(arr == fg_value, axis=0).astype(float)
        if values.size <= 4:
            nonzero_values = values[values != 0]
            if nonzero_values.size > 0:
                nonzero_counts = np.array(
                    [counts[np.where(values == v)[0][0]] for v in nonzero_values]
                )
                fg_value = nonzero_values[int(np.argmin(nonzero_counts))]
                return np.max(arr == fg_value, axis=0).astype(float)

    return np.max(arr, axis=0)


def backend_can_display() -> bool:
    """True when the active matplotlib backend can actually open a window.

    Under a non-interactive backend (``Agg`` in tests and CI, and whatever a GUI
    host installs) ``plt.show`` cannot display anything and only emits
    ``UserWarning: FigureCanvasAgg is non-interactive``. Callers use this to skip
    display entirely rather than warn once per figure.
    """
    backend = matplotlib.get_backend().lower()
    try:  # matplotlib >= 3.9
        from matplotlib.backends import BackendFilter, backend_registry

        interactive = backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    except ImportError:  # pragma: no cover - older matplotlib
        interactive = matplotlib.rcsetup.interactive_bk
    return backend in {name.lower() for name in interactive}


def _show_matplotlib_blocking() -> None:
    """Show figures and block, when the backend can display."""
    if not backend_can_display():
        plt.close("all")
        return
    plt.show()


def _show_matplotlib_non_blocking(pause_s: float = 0.001) -> None:
    """Show matplotlib figures without blocking script execution."""
    if not backend_can_display():
        plt.close("all")
        return
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
                _show_matplotlib_blocking()
            else:
                _show_matplotlib_non_blocking()
        else:
            plt.close()
    elif show:
        if block:
            _show_matplotlib_blocking()
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
    """Overlay edges and nodes on Z-projection of image.

    Set label_nodes=True to draw node IDs.
    """
    projection = _overlay_z_projection(image)
    pos = nx.get_node_attributes(G, "pos")
    resolved_voxel_size = _resolve_voxel_size(G, voxel_size)
    extent = _projection_extent(projection.shape, resolved_voxel_size)
    plt.figure(figsize=(10, 10))
    plt.imshow(projection, cmap="gray", extent=extent)
    for u, v, d in G.edges(data=True):
        path = d.get("voxels", [])
        if len(path) > 1:
            path = np.array(path)
            plt.plot(path[:, 2], path[:, 1], color="cyan", linewidth=0.5)
    if pos:
        coords = np.array(list(pos.values()))
        plt.scatter(coords[:, 2], coords[:, 1], c="red", s=3)
        if label_nodes:
            for node_id, node_pos in pos.items():
                plt.text(
                    float(node_pos[2]) + 1.0,
                    float(node_pos[1]) + 1.0,
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
                    plt.text(
                        float(node_pos[2]) + 1.0,
                        float(node_pos[1]) + 1.0,
                        f"({x:.1f}, {y:.1f}, {z:.1f})",
                        color="blue",
                        fontsize=3,
                    )
    plt.title("Overlay: Edges and Nodes on Z-Projection")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show and show_after_save:
            if block:
                _show_matplotlib_blocking()
            else:
                _show_matplotlib_non_blocking()
        else:
            plt.close()
    elif show:
        if block:
            _show_matplotlib_blocking()
        else:
            _show_matplotlib_non_blocking()
    else:
        plt.close()


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
    projection = _overlay_z_projection(image)
    resolved_voxel_size = _resolve_voxel_size(G, voxel_size)
    extent = _projection_extent(projection.shape, resolved_voxel_size)
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
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(projection, cmap=background_cmap, extent=extent)
    for bo in branch_orders:
        paths = [
            np.array(edge_paths[(u, v, k)])
            for (u, v, k), b in edge_branch_orders.items()
            if b == bo and len(edge_paths[(u, v, k)]) > 1
        ]
        color = color_mapping.get(bo, "gray")
        for path in paths:
            ax.plot(
                path[:, 2], path[:, 1],
                color=color,
                linewidth=edge_linewidth,
                alpha=alpha,
            )
    pos = nx.get_node_attributes(G, "pos")
    if pos:
        coords = np.array(list(pos.values()))
        ax.scatter(coords[:, 2], coords[:, 1], c=node_color, s=node_size)
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
        ax.legend(handles=handles, title="Branch Orders", loc="upper right")
    ax.set_title("Network Geometry with Branch Order Colors")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show and show_after_save:
            if block:
                _show_matplotlib_blocking()
            else:
                _show_matplotlib_non_blocking()
        else:
            plt.close(fig)
    elif show:
        if block:
            _show_matplotlib_blocking()
        else:
            _show_matplotlib_non_blocking()
    else:
        plt.close(fig)
    return fig, ax, color_mapping


def visualize_geometry_with_edge_resistance(
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
    """Plot network coloured by haemodynamic edge resistance.

    Reads the ``resistance`` edge attribute (Pa.s/m^3). ``use_inverse=True``
    colours by conductance instead.
    """
    projection = _overlay_z_projection(image)
    resolved_voxel_size = _resolve_voxel_size(G, voxel_size)
    extent = _projection_extent(projection.shape, resolved_voxel_size)
    edge_weights = {}
    edge_paths = {}
    weights_list = []
    for u, v, key, data in G.edges(keys=True, data=True):
        weight = data.get("resistance")
        path = data.get("voxels", [])
        if weight is not None:
            proc = 1.0 / weight if use_inverse else weight
            if use_inverse and weight == 0:
                proc = None
            else:
                weights_list.append(proc)
        else:
            proc = None
        edge_weights[(u, v, key)] = proc
        edge_paths[(u, v, key)] = path
    if not weights_list:
        return None, None, None, None
    data_min = min(weights_list)
    data_max = max(weights_list)
    vmin = min_weight if min_weight is not None else data_min
    vmax = max_weight if max_weight is not None else data_max
    cmap = plt.get_cmap(color_palette)
    if reverse_gradient:
        cmap = cmap.reversed()
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(projection, cmap=background_cmap, extent=extent)
    for (u, v, key), weight in edge_weights.items():
        if weight is not None:
            path = edge_paths[(u, v, key)]
            if len(path) > 1:
                path_arr = np.array(path)
                color = cmap(norm(weight))
                ax.plot(
                    path_arr[:, 2], path_arr[:, 1],
                    color=color,
                    linewidth=edge_linewidth,
                    alpha=alpha,
                )
    pos = nx.get_node_attributes(G, "pos")
    if pos:
        coords = np.array(list(pos.values()))
        ax.scatter(coords[:, 2], coords[:, 1], c=node_color, s=node_size)
    if show_legend:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label("1/Weight" if use_inverse else "Edge Weight", rotation=270)
    ax.set_title("Network Geometry Colored by Edge Weight")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show and show_after_save:
            if block:
                _show_matplotlib_blocking()
            else:
                _show_matplotlib_non_blocking()
        else:
            plt.close(fig)
    elif show:
        if block:
            _show_matplotlib_blocking()
        else:
            _show_matplotlib_non_blocking()
    else:
        plt.close(fig)
    return fig, ax, (vmin, vmax), cmap


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
        if backend_can_display():
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
        if backend_can_display():
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

    # 2D skeleton or headless save → matplotlib Z-projection fallback.
    if skeleton.ndim == 2 or save_path is not None:
        projection = (
            np.max(skeleton, axis=0).astype(float)
            if skeleton.ndim == 3
            else skeleton.astype(float)
        )
        resolved_voxel_size = voxel_size or (1.0, 1.0, 1.0)
        extent = _projection_extent(projection.shape, resolved_voxel_size)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(projection, cmap="gray", interpolation="nearest", extent=extent)
        ax.set_title(
            f"Skeleton Z-projection  —  shape: {skeleton.shape}  "
            f"voxels: {int(skeleton.sum())}"
        )
        ax.axis("off")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            if block:
                _show_matplotlib_blocking()
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


# British-spelling alias used in the example script.
visualise_skeleton = visualize_skeleton
