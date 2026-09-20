"""Pure graph-topology statistics: no positions, no geometry, no weights."""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Union

import networkx as nx


def compute_basic_statistics(
    G: Union[nx.Graph, nx.MultiGraph], is_multigraph: bool
) -> Dict[str, Any]:
    """Compute basic graph statistics."""
    if is_multigraph:
        edge_lengths = [
            d["length"]
            for _, _, _, d in G.edges(keys=True, data=True)
            if d.get("length") is not None
        ]
    else:
        edge_lengths = [
            d["length"]
            for _, _, d in G.edges(data=True)
            if d.get("length") is not None
        ]
    node_degrees = [G.degree(n) for n in G.nodes()]
    return {
        "Total Nodes": G.number_of_nodes(),
        "Total Edges": G.number_of_edges(),
        "Total Edge Length (microns)": sum(edge_lengths) if edge_lengths else 0,
        "Average Edge Length (microns)": (
            sum(edge_lengths) / len(edge_lengths) if edge_lengths else 0
        ),
        "Average Degree": sum(node_degrees) / len(node_degrees) if node_degrees else 0,
    }


def compute_branching_statistics(G: nx.Graph) -> Dict[str, Any]:
    """Count branching points (degree > 2 nodes).

    This used to also report "Average Branching Angle (degrees)": every
    pair of a junction's neighbours, angled node-to-node in a straight line
    and pooled into one network-wide mean. compute_branch_order_statistics's
    "Mean Emergence Angle (degrees)" (compute_emergence_angles_by_branch_order)
    replaced it with a more rigorous measurement -- one parent versus each
    daughter, along the local centreline tangent rather than a straight
    node-to-node line, broken down per branch order rather than pooled --
    so the cruder version was removed rather than left alongside it.
    """
    return {
        "Number of Branching Points": len(
            [n for n in G.nodes() if G.degree(n) > 2]
        ),
    }


def _tree_asymmetry_from_root(G: nx.Graph, root: Any) -> tuple[float, int]:
    """Iterative post-order asymmetry/subtree-size accumulation, rooted at *root*.

    A recursive version of this (one call per node, following the tree down
    from the root) hits Python's default recursion limit on a long,
    sparsely-branching stretch of vessel -- a real shape, not a pathological
    one, once a network has been simplified to a spanning tree. BFS order
    reversed is a safe post-order for a tree: every node's children appear
    strictly later than it in BFS, so processing nodes in reverse guarantees
    each node's children are already resolved.
    """
    parent: Dict[Any, Any] = {root: None}
    order: list[Any] = [root]
    queue: deque = deque([root])
    while queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor == parent[node] or neighbor in parent:
                continue
            parent[neighbor] = node
            order.append(neighbor)
            queue.append(neighbor)

    result: Dict[Any, tuple[float, int]] = {}
    for node in reversed(order):
        children = [
            result[neighbor]
            for neighbor in G.neighbors(node)
            if neighbor != parent[node]
        ]
        if not children:
            result[node] = (0.0, 1)
            continue
        child_sizes = [size for _asymmetry, size in children]
        total = sum(asymmetry for asymmetry, _size in children)
        node_a = max(child_sizes) - min(child_sizes) if len(child_sizes) >= 2 else 0
        result[node] = (total + node_a, sum(child_sizes) + 1)
    return result[root]


def compute_tree_asymmetry(G: nx.Graph) -> Dict[str, Any]:
    """Compute tree asymmetry index."""
    if G.number_of_nodes() == 0:
        return {"Tree Asymmetry Index": "N/A (empty graph)"}

    if not nx.is_tree(G):
        if nx.is_connected(G):
            # There is no "weight" edge attribute on these graphs (see
            # graph.assert_no_forbidden_edge_attributes); without an explicit
            # weight=, every edge is treated as weight 1 and the "minimum"
            # spanning tree is really an arbitrary one picked by Kruskal's
            # tie-breaking, not the vessel network's own shortest-path
            # skeleton. A real vascular graph has loops (capillary beds
            # anastomose), so this branch is the common case, not an edge
            # case.
            G = nx.minimum_spanning_tree(G.copy(), weight="length")
        else:
            return {"Tree Asymmetry Index": "N/A (disconnected graph)"}
    root = max(G.nodes(), key=G.degree)
    asymmetry, size = _tree_asymmetry_from_root(G, root)
    return {
        "Tree Asymmetry Index": asymmetry / size if size > 0 else 0
    }


def compute_network_robustness(
    G: Union[nx.Graph, nx.MultiGraph],
) -> Dict[str, Any]:
    """How much of the network sits on a single point of failure.

    A *bridge* is an edge whose removal (occluding that one vessel) splits
    the network into more pieces; a parallel vessel between the same two
    junctions means neither edge of that pair is a bridge, which
    ``nx.bridges`` already accounts for correctly on a ``MultiGraph``. An
    *articulation point* is a node whose removal (occluding every vessel
    passing through it) does the same, regardless of how many vessels meet
    there. Both are found in O(V+E) via chain decomposition, in preference
    to exhaustively counting independent inlet-to-outlet paths -- this is
    the topology half of an occlusion/stroke-risk framing: where a single
    occlusion would fragment perfusion rather than just reroute it.
    """
    total_edges = G.number_of_edges()
    total_nodes = G.number_of_nodes()
    bridge_count = sum(1 for _ in nx.bridges(G))
    articulation_point_count = sum(1 for _ in nx.articulation_points(G))
    return {
        "Bridge Edge Count": bridge_count,
        "Bridge Edge Fraction": (
            bridge_count / total_edges if total_edges else "N/A (no edges)"
        ),
        "Articulation Point Count": articulation_point_count,
        "Articulation Point Fraction": (
            articulation_point_count / total_nodes
            if total_nodes
            else "N/A (no nodes)"
        ),
    }
