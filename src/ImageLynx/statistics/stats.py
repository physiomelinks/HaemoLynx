"""Vessel network statistics."""
from typing import Dict, Any, Optional, Union

import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean
from networkx.algorithms.community import greedy_modularity_communities


def compute_basic_statistics(
    G: Union[nx.Graph, nx.MultiGraph], is_multigraph: bool
) -> Dict[str, Any]:
    """Compute basic graph statistics."""
    if is_multigraph:
        edge_weights = [
            d.get("weight", d.get("length"))
            for _, _, _, d in G.edges(keys=True, data=True)
            if d.get("weight") is not None or d.get("length") is not None
        ]
    else:
        edge_weights = [
            d.get("weight", d.get("length"))
            for _, _, d in G.edges(data=True)
            if d.get("weight") is not None or d.get("length") is not None
        ]
    node_degrees = [G.degree(n) for n in G.nodes()]
    return {
        "Total Nodes": G.number_of_nodes(),
        "Total Edges": G.number_of_edges(),
        "Total Edge Length (microns)": sum(edge_weights) if edge_weights else 0,
        "Average Edge Length (microns)": (
            sum(edge_weights) / len(edge_weights) if edge_weights else 0
        ),
        "Average Degree": sum(node_degrees) / len(node_degrees) if node_degrees else 0,
    }

#Update with newer tortuosity values HD gave to Anna
def compute_tortuosity_measures(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict],
    is_multigraph: bool,
) -> Dict[str, Any]:
    """Compute tortuosity index and curvature."""
    if node_positions is None:
        return {
            "Average Tortuosity Index": "N/A (no position data)",
            "Average Curvature": "N/A (no position data)",
        }
    tortuosity_indices = []
    curvatures = []
    it = G.edges(keys=True, data=True) if is_multigraph else G.edges(data=True)
    for item in it:
        u, v = item[0], item[1]
        edge_data = item[-1]
        if u in node_positions and v in node_positions:
            pos_u = np.array(node_positions[u])
            pos_v = np.array(node_positions[v])
            straight = euclidean(pos_u, pos_v)
            path_length = edge_data.get("weight", edge_data.get("length", straight))
            if straight > 0:
                tortuosity_indices.append(path_length / straight)
                if path_length > 0:
                    curvatures.append((path_length - straight) / path_length)
    return {
        "Average Tortuosity Index": (
            np.mean(tortuosity_indices) if tortuosity_indices else 0
        ),
        "Average Curvature": np.mean(curvatures) if curvatures else 0,
    }


def compute_branching_statistics(
    G: nx.Graph, node_positions: Optional[dict]
) -> Dict[str, Any]:
    """Compute average branching angle."""
    if node_positions is None:
        return {"Average Branching Angle (degrees)": "N/A (no position data)"}
    branching_angles = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) >= 3:
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if (
                        node in node_positions
                        and neighbors[i] in node_positions
                        and neighbors[j] in node_positions
                    ):
                        c = np.array(node_positions[node])
                        p1 = np.array(node_positions[neighbors[i]])
                        p2 = np.array(node_positions[neighbors[j]])
                        v1 = p1 - c
                        v2 = p2 - c
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if n1 > 0 and n2 > 0:
                            cos_a = np.clip(
                                np.dot(v1, v2) / (n1 * n2), -1, 1
                            )
                            branching_angles.append(np.degrees(np.arccos(cos_a)))
    return {
        "Average Branching Angle (degrees)": (
            np.mean(branching_angles) if branching_angles else 0
        ),
        "Number of Branching Points": len(
            [n for n in G.nodes() if G.degree(n) > 2]
        ),
    }


def compute_tree_asymmetry(G: nx.Graph) -> Dict[str, Any]:
    """Compute tree asymmetry index."""

    def _asym(node, parent=None):
        children = [n for n in G.neighbors(node) if n != parent]
        if not children:
            return 0, 1
        child_sizes = []
        total = 0
        for child in children:
            a, s = _asym(child, node)
            child_sizes.append(s)
            total += a
        node_a = max(child_sizes) - min(child_sizes) if len(child_sizes) >= 2 else 0
        return total + node_a, sum(child_sizes) + 1

    if not nx.is_tree(G):
        if nx.is_connected(G):
            G = nx.minimum_spanning_tree(G.copy())
        else:
            return {"Tree Asymmetry Index": "N/A (disconnected graph)"}
    root = max(G.nodes(), key=G.degree)
    asymmetry, size = _asym(root)
    return {
        "Tree Asymmetry Index": asymmetry / size if size > 0 else 0
    }


def compute_fractal_dimension(
    G: nx.Graph, node_positions: Optional[dict]
) -> Dict[str, Any]:
    """Compute fractal dimension via box-counting."""
    if node_positions is None or len(node_positions) < 2:
        return {"Fractal Dimension": "N/A (insufficient position data)"}
    positions = np.array(
        [node_positions[n] for n in G.nodes() if n in node_positions]
    )
    if len(positions) < 2:
        return {"Fractal Dimension": "N/A (insufficient position data)"}
    max_range = np.max(positions.max(axis=0) - positions.min(axis=0))
    min_bs = max_range / 100
    max_bs = max_range / 2
    box_sizes, box_counts = [], []
    for bs in np.logspace(np.log10(min_bs), np.log10(max_bs), 10):
        min_c = positions.min(axis=0)
        indices = ((positions - min_c) / bs).astype(int)
        box_sizes.append(bs)
        box_counts.append(len(set(tuple(i) for i in indices)))
    if len(box_sizes) > 1 and all(c > 0 for c in box_counts):
        fd = -np.polyfit(np.log(box_sizes), np.log(box_counts), 1)[0]
    else:
        fd = 0
    return {"Fractal Dimension": fd}


def compute_path_efficiency(
    G: Union[nx.Graph, nx.MultiGraph], is_multigraph: bool
) -> Dict[str, Any]:
    """Compute path efficiency from weighted shortest-path lengths.

    Path efficiency is defined here as the inverse of the mean weighted
    shortest-path length across all unique node pairs.
    """
    if is_multigraph:
        # Reduce a MultiGraph to a simple graph by keeping the lightest edge
        # between each unordered node pair.
        G_s = nx.Graph()
        G_s.add_nodes_from(G.nodes())
        ew = {}
        for u, v, k, d in G.edges(keys=True, data=True):
            w = d.get("weight", d.get("length", 1))
            uv = (u, v)
            if uv not in ew or w < ew[uv]:
                ew[uv] = w
        for (u, v), w in ew.items():
            G_s.add_edge(u, v, weight=w)
    else:
        G_s = G

    # Efficiency is undefined for disconnected graphs because some node pairs
    # have infinite path length.
    if not nx.is_connected(G_s):
        return {"Path Efficiency": "N/A (disconnected graph)"}

    # For a connected graph, every pair should have a valid path.
    # Unexpected failures should surface as errors rather than being silently
    # swallowed, otherwise statistics can look valid while being wrong.
    path_lengths = []
    nodes = list(G_s.nodes())
    for i, src in enumerate(nodes):
        for tgt in nodes[i + 1 :]:
            try:
                pl = nx.shortest_path_length(G_s, src, tgt, weight="weight")
            except nx.NetworkXNoPath as exc:
                raise RuntimeError(
                    f"No path between connected-graph nodes {src} and {tgt}"
                ) from exc
            path_lengths.append(pl)

    avg_path_length = np.mean(path_lengths) if path_lengths else 0
    efficiency = 1 / avg_path_length if avg_path_length > 0 else 0
    return {
        "Path Efficiency": efficiency,
        "Average Shortest Path Length (microns)": avg_path_length,
    }


def compute_vessel_density(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict],
    voxel_size,
    image_dimensions,
    is_multigraph: bool,
) -> Dict[str, Any]:
    """Compute vessel density."""
    if is_multigraph:
        lengths = [
            d.get("weight", d.get("length", 0))
            for _, _, _, d in G.edges(keys=True, data=True)
        ]
    else:
        lengths = [
            d.get("weight", d.get("length", 0))
            for _, _, d in G.edges(data=True)
        ]
    total_length = sum(lengths)
    out = {"Total Vessel Length (microns)": total_length}
    if node_positions and len(node_positions) > 0:
        positions = np.array(
            [node_positions[n] for n in G.nodes() if n in node_positions]
        )
        if len(positions) > 0:
            vol = np.prod(
                positions.max(axis=0) - positions.min(axis=0)
            )
            out["Vessel Density in Tissue (microns/micron³)"] = (
                total_length / vol if vol > 0 else 0
            )
            out["Vessel-Occupied Volume (micron³)"] = vol
        else:
            out["Vessel Density in Tissue (microns/micron³)"] = (
                "N/A (no position data)"
            )
    else:
        out["Vessel Density in Tissue (microns/micron³)"] = "N/A (no position data)"

    if image_dimensions is not None and voxel_size is not None:
        img_vol = np.prod(
            [d * v for d, v in zip(image_dimensions, voxel_size)]
        )
        out["Vessel Density in Whole Image (microns/micron³)"] = (
            total_length / img_vol if img_vol > 0 else 0
        )
        out["Total Image Volume (micron³)"] = img_vol
    else:
        out["Vessel Density in Whole Image (microns/micron³)"] = (
            "N/A (no image dimension data)"
        )
    return out
    
def compute_communities(G):
     #need to add resistance as weight
    return list(greedy_modularity_communities(G))
    
def compute_betweenness(G):
    #need to add resistance as weight
    return nx.betweenness_centrality(G)

def compute_comprehensive_vessel_statistics(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict] = None,
    voxel_size=(1.0, 1.0, 1.0),
    image_dimensions=None,
) -> Dict[str, Any]:
    """Combine all vessel statistics."""
    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    G_simple = (
        (nx.Graph(G) if not G.is_directed() else nx.DiGraph(G))
        if is_mg
        else G
    )
    communities = compute_communities(G)
    return {
        **compute_basic_statistics(G, is_mg),
        **compute_tortuosity_measures(G, node_positions, is_mg),
        **compute_branching_statistics(G_simple, node_positions),
        **compute_tree_asymmetry(G_simple),
        **compute_fractal_dimension(G_simple, node_positions),
        **compute_path_efficiency(G, is_mg),
        "communities": communities,
        **compute_betweenness(G),
        **compute_vessel_density(
            G, node_positions, voxel_size, image_dimensions, is_mg
        ),
    }
