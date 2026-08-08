"""Diagnostics utilities for graph topology cleanup."""
from typing import Any, Dict, List, Union

import networkx as nx

from ._helpers import get_all_edge_data


def diagnose_degree2_nodes(
    G: Union[nx.Graph, nx.MultiGraph],
    max_degree: int = 4,
    sample_limit: int = 10,
) -> Dict[str, Any]:
    """Summarize remaining degree-2 nodes and why smart cleanup may skip them."""
    if max_degree < 1:
        raise ValueError("max_degree must be >= 1")
    if sample_limit < 1:
        raise ValueError("sample_limit must be >= 1")

    degree2_nodes: List[Any] = [n for n in G.nodes() if G.degree[n] == 2]
    reason_nodes: Dict[str, List[Any]] = {
        "neighbors_not_2": [],
        "high_degree_neighbor": [],
        "missing_pos": [],
        "missing_edge_data": [],
        "eligible_for_smart_removal": [],
    }

    for node in degree2_nodes:
        neighbors = list(G.neighbors(node))
        if len(neighbors) != 2:
            reason_nodes["neighbors_not_2"].append(node)
            continue

        n1, n2 = neighbors
        if G.degree[n1] >= max_degree or G.degree[n2] >= max_degree:
            reason_nodes["high_degree_neighbor"].append(node)
            continue

        node_pos = G.nodes[node].get("pos")
        n1_pos = G.nodes[n1].get("pos")
        n2_pos = G.nodes[n2].get("pos")
        if node_pos is None or n1_pos is None or n2_pos is None:
            reason_nodes["missing_pos"].append(node)
            continue

        edge1_data_list = get_all_edge_data(G, node, n1)
        edge2_data_list = get_all_edge_data(G, node, n2)
        if not edge1_data_list or not edge2_data_list:
            reason_nodes["missing_edge_data"].append(node)
            continue

        reason_nodes["eligible_for_smart_removal"].append(node)

    reason_counts = {k: len(v) for k, v in reason_nodes.items()}
    reason_examples = {k: v[:sample_limit] for k, v in reason_nodes.items() if v}

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "total_degree2": len(degree2_nodes),
        "max_degree_threshold": max_degree,
        "reason_counts": reason_counts,
        "reason_examples": reason_examples,
    }


def format_degree2_diagnostics_report(report: Dict[str, Any]) -> str:
    """Format degree-2 diagnostics into a compact multiline report."""
    lines = [
        "Degree-2 diagnostics:",
        (
            f"  total_nodes={report.get('total_nodes', 0)}, "
            f"total_edges={report.get('total_edges', 0)}, "
            f"total_degree2={report.get('total_degree2', 0)}, "
            f"max_degree_threshold={report.get('max_degree_threshold', 0)}"
        ),
    ]

    counts = report.get("reason_counts", {})
    examples = report.get("reason_examples", {})
    for key in (
        "neighbors_not_2",
        "high_degree_neighbor",
        "missing_pos",
        "missing_edge_data",
        "eligible_for_smart_removal",
    ):
        count = counts.get(key, 0)
        sample = examples.get(key, [])
        lines.append(f"  {key}: {count}" + (f" (sample={sample})" if sample else ""))

    return "\n".join(lines)
