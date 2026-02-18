"""Plotting functions for vascular networks."""
from typing import Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize

from ._helpers import (
    sort_branch_orders_numerically,
    create_color_mapping,
    group_branch_orders_for_legend,
)


def plot_node_degree_distribution(
    G: nx.Graph, title: str = "Node Degree Distribution"
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
    plt.show()
    return degree_counts


def visualize_edges_and_nodes(image: np.ndarray, G: nx.Graph) -> None:
    """Overlay edges and nodes on Z-projection of image."""
    projection = np.max(image, axis=0)
    pos = nx.get_node_attributes(G, "pos")
    plt.figure(figsize=(10, 10))
    plt.imshow(projection, cmap="gray")
    for u, v, d in G.edges(data=True):
        path = d.get("voxels", [])
        if len(path) > 1:
            path = np.array(path)
            plt.plot(path[:, 2], path[:, 1], color="cyan", linewidth=0.5)
    if pos:
        coords = np.array(list(pos.values()))
        plt.scatter(coords[:, 2], coords[:, 1], c="red", s=3)
    plt.title("Overlay: Edges and Nodes on Z-Projection")
    plt.axis("off")
    plt.show()


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
):
    """Plot network colored by branch order."""
    projection = np.max(image, axis=0)
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
    ax.imshow(projection, cmap=background_cmap)
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
    plt.show()
    return fig, ax, color_mapping


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
):
    """Plot network colored by edge weight."""
    projection = np.max(image, axis=0)
    edge_weights = {}
    edge_paths = {}
    weights_list = []
    for u, v, key, data in G.edges(keys=True, data=True):
        weight = data.get("weight")
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
    ax.imshow(projection, cmap=background_cmap)
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
    plt.show()
    return fig, ax, (vmin, vmax), cmap
