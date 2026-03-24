"""Vessel network statistics."""
from typing import Dict, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import re

import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean
from networkx.algorithms.community import greedy_modularity_communities

#Need to add in bifurcation ratios

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
    G: Union[nx.Graph, nx.MultiGraph],
    is_multigraph: bool,
    max_pairs: Optional[int] = 5000,
    rng_seed: int = 42,
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
    total_pairs = (len(nodes) * (len(nodes) - 1)) // 2

    # For large graphs, estimate efficiency from a bounded random sample of
    # node pairs to avoid very long runtimes.
    if max_pairs is not None and total_pairs > max_pairs:
        rng = np.random.default_rng(rng_seed)
        sampled_pairs = set()
        while len(sampled_pairs) < max_pairs:
            i = int(rng.integers(0, len(nodes)))
            j = int(rng.integers(0, len(nodes)))
            if i == j:
                continue
            if i > j:
                i, j = j, i
            sampled_pairs.add((i, j))

        pairs = [(nodes[i], nodes[j]) for i, j in sampled_pairs]
    else:
        pairs = []
        for i, src in enumerate(nodes):
            for tgt in nodes[i + 1 :]:
                pairs.append((src, tgt))

    for src, tgt in pairs:
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
        "Path Efficiency Pair Sample Size": len(path_lengths),
        "Path Efficiency Pair Coverage": (
            len(path_lengths) / total_pairs if total_pairs > 0 else 0
        ),
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
    
def compute_communities_summary(
    G: nx.Graph, max_nodes_exact: int = 1500
) -> Dict[str, Any]:
    """Compute community statistics with runtime guards."""
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return {"Community Count": 0}

    if n_nodes <= max_nodes_exact:
        communities = list(greedy_modularity_communities(G))
        sizes = [len(c) for c in communities]
        return {
            "Community Count": len(communities),
            "Largest Community Size": max(sizes) if sizes else 0,
            "Mean Community Size": float(np.mean(sizes)) if sizes else 0,
            "Community Method": "greedy_modularity",
        }

    # Fallback for large graphs: connected components are fast and stable.
    components = list(nx.connected_components(G))
    sizes = [len(c) for c in components]
    return {
        "Community Count": len(components),
        "Largest Community Size": max(sizes) if sizes else 0,
        "Mean Community Size": float(np.mean(sizes)) if sizes else 0,
        "Community Method": "connected_components_fallback",
    }


def compute_communities(G: nx.Graph):
    # TODO: consider weighted community detection for resistance-aware grouping.
    return list(greedy_modularity_communities(G))


def compute_betweenness_summary(
    G: nx.Graph,
    max_nodes_exact: int = 1000,
    approx_k: int = 128,
    seed: int = 42,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Compute compact betweenness summary, avoiding huge outputs."""
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return {"Betweenness Mean": 0.0, "Betweenness Max": 0.0}

    if n_nodes <= max_nodes_exact:
        bet = nx.betweenness_centrality(G)
        method = "exact"
    else:
        k = min(approx_k, n_nodes)
        bet = nx.betweenness_centrality(G, k=k, seed=seed)
        method = f"approx_k={k}"

    values = list(bet.values())
    top = sorted(bet.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {
        "Betweenness Mean": float(np.mean(values)) if values else 0.0,
        "Betweenness Max": float(np.max(values)) if values else 0.0,
        "Betweenness Top Nodes": top,
        "Betweenness Method": method,
    }


def compute_betweenness(G: nx.Graph):
    # TODO: consider weighted betweenness using resistance.
    return nx.betweenness_centrality(G)


def _simple_graph_with_edge_attr(
    G: Union[nx.Graph, nx.MultiGraph],
    source_attr: str,
    transform: Optional[Callable[[float], float]] = None,
    target_attr: str = "analysis_weight",
) -> nx.Graph:
    """Build a simple graph carrying one transformed edge attribute.

    For MultiGraph inputs, parallel edges are collapsed by taking the smallest
    transformed value, which is appropriate for path-based distance weights.
    """
    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    G_s = nx.Graph()
    G_s.add_nodes_from(G.nodes())

    if transform is None:
        transform = lambda x: x  # noqa: E731

    if is_mg:
        best = {}
        for u, v, _, data in G.edges(keys=True, data=True):
            raw = data.get(source_attr)
            if raw is None or raw <= 0:
                continue
            transformed = transform(raw)
            if transformed is None or transformed <= 0:
                continue
            uv = tuple(sorted((u, v)))
            if uv not in best or transformed < best[uv]:
                best[uv] = float(transformed)
        for (u, v), val in best.items():
            G_s.add_edge(u, v, **{target_attr: val})
    else:
        for u, v, data in G.edges(data=True):
            raw = data.get(source_attr)
            if raw is None or raw <= 0:
                continue
            transformed = transform(raw)
            if transformed is None or transformed <= 0:
                continue
            existing = G_s.get_edge_data(u, v, default={}).get(target_attr)
            if existing is None or transformed < existing:
                G_s.add_edge(u, v, **{target_attr: float(transformed)})
    return G_s


def compute_weighted_betweenness_summary(
    G: Union[nx.Graph, nx.MultiGraph],
    source_attr: str,
    inverse_source_attr: bool = False,
    max_nodes_exact: int = 1000,
    approx_k: int = 128,
    seed: int = 42,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Compute weighted betweenness summary from a chosen edge attribute."""
    transform = (lambda x: 1.0 / x) if inverse_source_attr else None
    G_s = _simple_graph_with_edge_attr(
        G, source_attr=source_attr, transform=transform, target_attr="analysis_weight"
    )
    n_nodes = G_s.number_of_nodes()
    if n_nodes == 0:
        return {"Betweenness Mean": 0.0, "Betweenness Max": 0.0}

    if n_nodes <= max_nodes_exact:
        bet = nx.betweenness_centrality(G_s, weight="analysis_weight")
        method = "exact_weighted"
    else:
        k = min(approx_k, n_nodes)
        bet = nx.betweenness_centrality(
            G_s, k=k, seed=seed, weight="analysis_weight"
        )
        method = f"approx_weighted_k={k}"

    values = list(bet.values())
    top = sorted(bet.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {
        "Betweenness Mean": float(np.mean(values)) if values else 0.0,
        "Betweenness Max": float(np.max(values)) if values else 0.0,
        "Betweenness Top Nodes": [
            {"node": node, "value": float(value)} for node, value in top
        ],
        "Betweenness Method": method,
    }


def compute_weighted_communities_summary(
    G: Union[nx.Graph, nx.MultiGraph],
    source_attr: str,
    inverse_source_attr: bool = False,
    max_nodes_exact: int = 1500,
) -> Dict[str, Any]:
    """Compute weighted community summary using greedy modularity."""
    transform = (lambda x: 1.0 / x) if inverse_source_attr else None
    G_s = _simple_graph_with_edge_attr(
        G, source_attr=source_attr, transform=transform, target_attr="analysis_weight"
    )
    n_nodes = G_s.number_of_nodes()
    if n_nodes == 0:
        return {"Community Count": 0}

    if n_nodes <= max_nodes_exact:
        communities = list(greedy_modularity_communities(G_s, weight="analysis_weight"))
        sizes = [len(c) for c in communities]
        return {
            "Community Count": len(communities),
            "Largest Community Size": max(sizes) if sizes else 0,
            "Mean Community Size": float(np.mean(sizes)) if sizes else 0,
            "Community Method": "greedy_modularity_weighted",
        }

    components = list(nx.connected_components(G_s))
    sizes = [len(c) for c in components]
    return {
        "Community Count": len(components),
        "Largest Community Size": max(sizes) if sizes else 0,
        "Mean Community Size": float(np.mean(sizes)) if sizes else 0,
        "Community Method": "connected_components_fallback",
    }


def compute_betweenness_and_community_measurements(
    G: Union[nx.Graph, nx.MultiGraph],
) -> Dict[str, Dict[str, Any]]:
    """Compute weighted betweenness/community using two edge distance models."""
    inverse_weight_results = {
        "Betweenness": compute_weighted_betweenness_summary(
            G, source_attr="weight", inverse_source_attr=True
        ),
        "Communities": compute_weighted_communities_summary(
            G, source_attr="weight", inverse_source_attr=True
        ),
    }
    edge_length_results = {
        "Betweenness": compute_weighted_betweenness_summary(
            G, source_attr="length", inverse_source_attr=False
        ),
        "Communities": compute_weighted_communities_summary(
            G, source_attr="length", inverse_source_attr=False
        ),
    }
    return {
        "inverse_edge_weight": inverse_weight_results,
        "edge_length": edge_length_results,
    }


def compute_comprehensive_vessel_statistics(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict] = None,
    voxel_size=(1.0, 1.0, 1.0),
    image_dimensions=None,
    statistics_mode: str = "fast",
) -> Dict[str, Any]:
    """Combine all vessel statistics."""
    valid_modes = {"fast", "full"}
    if statistics_mode not in valid_modes:
        raise ValueError(
            f"Invalid statistics_mode='{statistics_mode}'. "
            f"Choose one of {sorted(valid_modes)}."
        )

    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    G_simple = (
        (nx.Graph(G) if not G.is_directed() else nx.DiGraph(G))
        if is_mg
        else G
    )

    base = {
        **compute_basic_statistics(G, is_mg),
        **compute_tortuosity_measures(G, node_positions, is_mg),
        **compute_branching_statistics(G_simple, node_positions),
        **compute_tree_asymmetry(G_simple),
        **compute_fractal_dimension(G_simple, node_positions),
        **compute_vessel_density(
            G, node_positions, voxel_size, image_dimensions, is_mg
        ),
    }

    if statistics_mode == "full":
        return {
            **base,
            **compute_path_efficiency(G, is_mg, max_pairs=None),
            "communities": compute_communities(G_simple),
            **compute_betweenness(G_simple),
            "Statistics Mode": "full",
        }

    return {
        **base,
        **compute_path_efficiency(G, is_mg),
        **compute_communities_summary(G_simple),
        **compute_betweenness_summary(G_simple),
        "Statistics Mode": "fast",
    }


def _flatten_statistics_dict(
    stats: Dict[str, Any], prefix: str = ""
) -> list[tuple[str, Any]]:
    """Flatten nested statistics dict for tabular export."""
    flattened: list[tuple[str, Any]] = []
    for key, value in stats.items():
        full_key = f"{prefix} > {key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.extend(_flatten_statistics_dict(value, full_key))
        else:
            flattened.append((full_key, value))
    return flattened


def _split_metric_path(metric_path: str) -> tuple[str, str]:
    """Split flattened key path into section and metric."""
    if " > " not in metric_path:
        return "Core Statistics", metric_path
    section, metric = metric_path.rsplit(" > ", 1)
    return section, metric


def _extract_unit(metric_name: str) -> tuple[str, str]:
    """Extract trailing '(unit)' from metric labels when present."""
    m = re.search(r"\(([^()]+)\)\s*$", metric_name)
    if not m:
        return metric_name, ""
    unit = m.group(1).strip()
    clean_metric = re.sub(r"\s*\([^()]+\)\s*$", "", metric_name).strip()
    return clean_metric, unit


def _format_stat_value(metric_name: str, value: Any) -> tuple[str, str]:
    """Format metric value plus a short human-readable note."""
    if isinstance(value, str):
        if value.startswith("N/A"):
            return value, "Not computable with current inputs."
        return value, ""

    if isinstance(value, (bool, np.bool_)):
        return ("True" if bool(value) else "False"), ""

    if isinstance(value, (int, np.integer)):
        return str(int(value)), ""

    if isinstance(value, (float, np.floating)):
        f_val = float(value)
        if "Coverage" in metric_name:
            return f"{f_val:.2%}", "Fraction of all node-pairs included."
        return f"{f_val:.6g}", ""

    if isinstance(value, (list, tuple, set)):
        serializable = list(value) if not isinstance(value, list) else value
        try:
            return json.dumps(serializable), "Serialized collection value."
        except TypeError:
            return str(serializable), "Collection converted to text."

    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True), "Serialized nested object."
        except TypeError:
            return str(value), "Nested object converted to text."

    return str(value), ""


def _annotation_for_metric(metric_name: str) -> str:
    """Return short metric-specific annotation used in CSV notes."""
    if metric_name == "Statistics Mode":
        return "fast=sampled/compact metrics, full=heavier/exact metrics."
    if "Tortuosity" in metric_name:
        return "Higher values indicate less straight vessels."
    if "Curvature" in metric_name:
        return "Relative deviation from straight vessel segments."
    if "Branching Angle" in metric_name:
        return "Average local bifurcation/trifurcation angle."
    if "Asymmetry" in metric_name:
        return "Tree imbalance estimate after simplification to a tree."
    if "Fractal Dimension" in metric_name:
        return "Box-counting estimate of structural complexity."
    if "Path Efficiency" in metric_name:
        return "Inverse of mean shortest-path distance."
    if "Betweenness" in metric_name:
        return "Centrality based on shortest-path traffic."
    if "Community" in metric_name:
        return "Subnetwork partition summary."
    if "Density" in metric_name:
        return "Total vessel length normalized by available volume."
    return ""


def export_statistics_to_csv(
    stats: Dict[str, Any],
    output_csv_path: Union[str, Path],
) -> Path:
    """Export statistics dictionary to a readable annotated CSV file.

    The CSV is long-format with one row per metric and includes:
    section, metric name, value, parsed unit, and notes.
    """
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flattened = _flatten_statistics_dict(stats)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Metric", "Value", "Unit", "Notes"])
        writer.writerow(
            [
                "Metadata",
                "Exported At (UTC)",
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "",
                "Generated by ImageLynx statistics exporter.",
            ]
        )
        writer.writerow(
            [
                "Metadata",
                "Metric Count",
                str(len(flattened)),
                "",
                "Number of metric rows in this file.",
            ]
        )

        for metric_path, value in flattened:
            section, metric = _split_metric_path(metric_path)
            metric_clean, unit = _extract_unit(metric)
            formatted_value, value_note = _format_stat_value(metric_clean, value)
            metric_note = _annotation_for_metric(metric_clean)
            notes = " ".join(n for n in [value_note, metric_note] if n).strip()
            writer.writerow([section, metric_clean, formatted_value, unit, notes])

    return output_path


def _normalize_branch_order_tag(tag: Any) -> Optional[str]:
    """Normalize branch-order labels to ArtN / BON / VenN where possible."""
    if tag is None:
        return None
    label = str(tag).strip()
    if not label:
        return None
    m = re.match(r"^(art|ven|bo|b)\s*0*(\d+)$", label, flags=re.IGNORECASE)
    if not m:
        return label
    prefix = m.group(1).lower()
    n = int(m.group(2))
    if prefix == "art":
        return f"Art{n}"
    if prefix == "ven":
        return f"Ven{n}"
    return f"BO{n}"


def _branch_order_sort_key(tag: str) -> tuple[int, int, str]:
    """Sort as Art1..ArtN, BO1..BON, Ven1..VenN, then unknown labels."""
    m = re.match(r"^(art|ven|bo)\s*(\d+)$", str(tag), flags=re.IGNORECASE)
    if not m:
        return (3, 0, str(tag))
    prefix = m.group(1).lower()
    n = int(m.group(2))
    group = {"art": 0, "bo": 1, "ven": 2}.get(prefix, 3)
    return (group, n, str(tag))


def compute_branch_order_statistics(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute mean length/tortuosity per branch-order edge tag.

    Returns a dictionary keyed by branch-order label with a compact summary.
    """
    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    edge_iter = G.edges(keys=True, data=True) if is_mg else G.edges(data=True)

    by_tag: Dict[str, Dict[str, Any]] = {}
    for item in edge_iter:
        u, v = item[0], item[1]
        data = item[-1]
        normalized_tag = _normalize_branch_order_tag(data.get("branch_order"))
        if not normalized_tag:
            continue

        length = data.get("length", data.get("weight", 0.0))
        try:
            length_f = float(length)
        except (TypeError, ValueError):
            continue
        if length_f <= 0:
            continue

        if normalized_tag not in by_tag:
            by_tag[normalized_tag] = {
                "Branch Order": normalized_tag,
                "Edge Count": 0,
                "Mean Length (microns)": 0.0,
                "Mean Tortuosity Index": "N/A (no position data)",
                "Tortuosity Sample Count": 0,
            }
        rec = by_tag[normalized_tag]
        rec["Edge Count"] += 1
        rec["Mean Length (microns)"] += length_f

        if (
            node_positions is not None
            and u in node_positions
            and v in node_positions
        ):
            pos_u = np.array(node_positions[u], dtype=float)
            pos_v = np.array(node_positions[v], dtype=float)
            straight = euclidean(pos_u, pos_v)
            if straight > 0:
                tort = length_f / straight
                if rec["Mean Tortuosity Index"] == "N/A (no position data)":
                    rec["Mean Tortuosity Index"] = 0.0
                rec["Mean Tortuosity Index"] += tort
                rec["Tortuosity Sample Count"] += 1

    for rec in by_tag.values():
        edge_count = int(rec["Edge Count"])
        if edge_count > 0:
            rec["Mean Length (microns)"] = rec["Mean Length (microns)"] / edge_count
        t_samples = int(rec["Tortuosity Sample Count"])
        if t_samples > 0 and isinstance(rec["Mean Tortuosity Index"], (int, float)):
            rec["Mean Tortuosity Index"] = rec["Mean Tortuosity Index"] / t_samples
        elif t_samples == 0:
            rec["Mean Tortuosity Index"] = "N/A (insufficient position data)"

    ordered = {
        k: by_tag[k]
        for k in sorted(by_tag.keys(), key=_branch_order_sort_key)
    }
    return ordered


def export_branch_order_statistics_to_csv(
    branch_order_stats: Dict[str, Dict[str, Any]],
    output_csv_path: Union[str, Path],
) -> Path:
    """Export per-branch-order summary statistics to readable CSV."""
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Branch Order",
                "Edge Count",
                "Mean Length (microns)",
                "Mean Tortuosity Index",
                "Notes",
            ]
        )
        writer.writerow(
            [
                "# Ordered by vessel class",
                "",
                "",
                "",
                "Art* first, then BO*, then Ven*.",
            ]
        )
        for branch_tag in sorted(branch_order_stats.keys(), key=_branch_order_sort_key):
            rec = branch_order_stats[branch_tag]
            mean_len = rec.get("Mean Length (microns)", 0.0)
            if isinstance(mean_len, (int, float, np.integer, np.floating)):
                mean_len_s = f"{float(mean_len):.6g}"
            else:
                mean_len_s = str(mean_len)
            mean_tort = rec.get("Mean Tortuosity Index", "N/A")
            if isinstance(mean_tort, (int, float, np.integer, np.floating)):
                mean_tort_s = f"{float(mean_tort):.6g}"
                note = "Mean tortuosity is path length / straight distance."
            else:
                mean_tort_s = str(mean_tort)
                note = "Tortuosity unavailable (missing/insufficient node positions)."
            writer.writerow(
                [
                    branch_tag,
                    int(rec.get("Edge Count", 0)),
                    mean_len_s,
                    mean_tort_s,
                    note,
                ]
            )

    return output_path
