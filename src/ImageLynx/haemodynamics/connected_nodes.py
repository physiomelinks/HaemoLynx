"""Helpers for selecting connected node pairs in vascular graphs."""
from __future__ import annotations

from typing import Iterable

import networkx as nx


def find_connected_start_output_pairs(
    G: nx.Graph,
    starting_nodes: Iterable[int],
    output_nodes: Iterable[int],
) -> list[tuple[int, int]]:
    """Return all start/output node pairs connected in the graph.

    Pairs are emitted in deterministic order following the provided
    ``starting_nodes`` and ``output_nodes`` iteration order.
    """
    component_index: dict[int, int] = {}
    for component_id, component_nodes in enumerate(nx.connected_components(G)):
        for node_id in component_nodes:
            component_index[int(node_id)] = int(component_id)

    pairs: list[tuple[int, int]] = []
    for start_node in starting_nodes:
        start_id = int(start_node)
        start_component = component_index.get(start_id)
        if start_component is None:
            continue
        for output_node in output_nodes:
            output_id = int(output_node)
            if component_index.get(output_id) == start_component:
                pairs.append((start_id, output_id))
    return pairs
