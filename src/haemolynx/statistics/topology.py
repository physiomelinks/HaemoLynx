"""Pure graph-topology statistics: no positions, no geometry, no weights."""
from __future__ import annotations

import warnings
from collections import deque
from typing import Any, Dict, Iterable, Optional, Union

import networkx as nx
import numpy as np


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


def _spanning_forest_bfs_order(G: Union[nx.Graph, nx.MultiGraph]) -> tuple[dict, list]:
    """BFS spanning forest (one tree per connected component) as a
    ``{node: parent}`` map (root maps to ``None``) plus the BFS visiting
    order -- every bridge of *G* is guaranteed to be a tree edge of any
    spanning tree (removing a bridge disconnects *G*, so no spanning tree
    can omit it and stay connected), so this one forest is enough to
    classify every bridge, not just the ones on whichever tree a particular
    root happens to produce.
    """
    parent: dict = {}
    order: list = []
    for component in nx.connected_components(G):
        root = next(iter(component))
        parent[root] = None
        order.append(root)
        queue: deque = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor in G.neighbors(node):
                if neighbor in parent:
                    continue
                parent[neighbor] = node
                order.append(neighbor)
                queue.append(neighbor)
    return parent, order


def _subtree_membership_counts(
    G: Union[nx.Graph, nx.MultiGraph], parent: dict, order: list, marked: set
) -> dict:
    """For every node, how many of *marked* lie in its own spanning-tree
    subtree (itself included) -- one post-order pass over *order* reversed,
    the same technique :func:`_tree_asymmetry_from_root` uses, generalised
    to a forest instead of one tree.
    """
    counts: dict = {}
    for node in reversed(order):
        total = 1 if node in marked else 0
        for neighbor in G.neighbors(node):
            if parent.get(neighbor) == node:
                total += counts[neighbor]
        counts[node] = total
    return counts


def _perfusion_critical_failure_points(
    G: Union[nx.Graph, nx.MultiGraph],
    bridges: list,
    articulation_pts: list,
    inlets: set,
    outlets: set,
) -> Dict[str, Any]:
    """Which *bridges*/*articulation_pts* actually separate an inlet from an
    outlet if removed, rather than just fragmenting the topology somewhere
    that never carried perfusion in the first place (a bridge feeding a
    single dead-end capillary spur is a point of failure for that spur
    alone).

    Bridges are classified in one O(V+E) pass (see
    :func:`_spanning_forest_bfs_order`/:func:`_subtree_membership_counts`):
    a bridge is exactly a spanning-tree edge, so its own two sides are the
    tree-child's subtree and everything else. Articulation points are
    classified by actually removing each one and checking whether the
    remaining components split the inlets from the outlets -- more
    expensive (O(V+E) per point), but articulation points are typically far
    fewer than bridges in a vessel network (most junctions have a
    redundant path; most bridges are dead-end capillary tips).
    """
    total_inlets = len(inlets)
    total_outlets = len(outlets)

    parent, order = _spanning_forest_bfs_order(G)
    inlet_subtree = _subtree_membership_counts(G, parent, order, inlets)
    outlet_subtree = _subtree_membership_counts(G, parent, order, outlets)

    critical_bridges = 0
    for u, v in bridges:
        child = u if parent.get(u) == v else v
        inlets_in = inlet_subtree[child]
        outlets_in = outlet_subtree[child]
        inlets_out = total_inlets - inlets_in
        outlets_out = total_outlets - outlets_in
        if (inlets_in > 0 and outlets_out > 0) or (outlets_in > 0 and inlets_out > 0):
            critical_bridges += 1

    critical_articulation_points = 0
    for point in articulation_pts:
        H = G.copy()
        H.remove_node(point)
        separates = False
        for component in nx.connected_components(H):
            if (inlets & component) and (outlets - component):
                separates = True
                break
            if (outlets & component) and (inlets - component):
                separates = True
                break
        if separates:
            critical_articulation_points += 1

    return {
        "Critical Bridge Edge Count": critical_bridges,
        "Critical Bridge Edge Fraction": (
            critical_bridges / len(bridges) if bridges else "N/A (no bridges)"
        ),
        "Critical Articulation Point Count": critical_articulation_points,
        "Critical Articulation Point Fraction": (
            critical_articulation_points / len(articulation_pts)
            if articulation_pts
            else "N/A (no articulation points)"
        ),
    }


def compute_network_robustness(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    inlet_nodes: Optional[Iterable[Any]] = None,
    outlet_nodes: Optional[Iterable[Any]] = None,
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

    *inlet_nodes*/*outlet_nodes*, when both given (and non-empty), also
    classify each bridge/articulation point as "critical" when removing it
    actually separates some inlet from some outlet -- the perfusion-relevant
    subset of "a single point of failure," since a bridge or junction that
    only isolates a dead-end capillary spur is a much smaller concern than
    one that cuts off an entire route from arteriole to venule. Omitted (the
    default) skips this classification and returns exactly the counts above.
    """
    total_edges = G.number_of_edges()
    total_nodes = G.number_of_nodes()
    bridges = list(nx.bridges(G))
    articulation_pts = list(nx.articulation_points(G))
    bridge_count = len(bridges)
    articulation_point_count = len(articulation_pts)
    result = {
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

    inlets = set(inlet_nodes) if inlet_nodes is not None else set()
    outlets = set(outlet_nodes) if outlet_nodes is not None else set()
    if inlets and outlets:
        result.update(
            _perfusion_critical_failure_points(G, bridges, articulation_pts, inlets, outlets)
        )
    return result


def compute_cyclomatic_number(G: Union[nx.Graph, nx.MultiGraph]) -> Dict[str, Any]:
    """How many independent loops (anastomoses) this network has.

    The cyclomatic number / circuit rank, ``E - N + C`` (edges minus nodes
    plus connected components), counts exactly the number of independent
    cycles a graph has -- 0 for a pure tree/forest. Computed directly on
    *G* rather than via ``nx.cycle_basis`` (which needs a simple graph):
    collapsing a MultiGraph's parallel edges first would merge two parallel
    vessels between the same pair of junctions into one edge and silently
    erase the loop they form. For a vascular network, every independent
    loop is a collateral/anastomotic route perfusion can reroute through if
    one of the vessels on it is occluded -- 0 means every vessel is a
    single point of failure for everything downstream of it.
    """
    n = G.number_of_nodes()
    e = G.number_of_edges()
    if n == 0:
        return {"Cyclomatic Number": 0, "Cyclomatic Number Per Node": "N/A (no nodes)"}
    c = nx.number_connected_components(G)
    cyclomatic_number = e - n + c
    return {
        "Cyclomatic Number": cyclomatic_number,
        "Cyclomatic Number Per Node": cyclomatic_number / n,
    }


def compute_degree_assortativity(G: Union[nx.Graph, nx.MultiGraph]) -> Dict[str, Any]:
    """Whether high-degree junctions tend to connect to other high-degree
    junctions (positive), to low-degree ones (negative), or neither (~0).

    Vascular trees are normally disassortative -- a large vessel branches
    into smaller ones, not into another large one -- so a value drifting
    toward zero or positive is worth a second look: capillary-capillary
    shunting, or a segmentation artefact merging vessels of very different
    calibre into the same junction. ``nx.degree_assortativity_coefficient``
    returns NaN when there is no degree variance to correlate (too few
    edges, or a perfectly regular graph); reported as "N/A" rather than NaN.
    """
    if G.number_of_edges() < 2:
        return {"Degree Assortativity": "N/A (fewer than 2 edges)"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = nx.degree_assortativity_coefficient(G)
    if not np.isfinite(value):
        return {"Degree Assortativity": "N/A (no degree variance)"}
    return {"Degree Assortativity": float(value)}


def compute_rich_club_coefficient(G: Union[nx.Graph, nx.MultiGraph]) -> Dict[str, Any]:
    """Whether the highest-degree vessels/junctions preferentially connect
    to each other, rather than to lower-degree ones -- a "rich club" of
    large-branch-order vessels forming their own tightly-connected backbone.

    ``nx.rich_club_coefficient`` needs a simple graph (parallel edges add no
    extra information for it), and only its raw, unnormalised form here
    (``normalized=False``): the normalised form compares against random
    graphs built by degree-preserving edge rewiring, which is both
    expensive and non-deterministic unless seeded, for a report whose other
    measures are all direct, not null-model-relative. Reports the
    coefficient at the tightest tier available (the highest degree threshold
    the graph has any node above), the club of the most-connected vessels.
    """
    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    G_simple = nx.Graph(G) if is_mg else G
    if G_simple.number_of_edges() == 0:
        return {"Rich Club Coefficient": "N/A (no edges)"}
    curve = nx.rich_club_coefficient(G_simple, normalized=False)
    if not curve:
        return {"Rich Club Coefficient": "N/A (no edges)"}
    tightest_k = max(curve)
    return {
        "Rich Club Coefficient": float(curve[tightest_k]),
        "Rich Club Coefficient Degree Threshold": tightest_k,
    }


def compute_k_core_structure(G: Union[nx.Graph, nx.MultiGraph]) -> Dict[str, Any]:
    """How onion-layered (redundant core plus peripheral shell) vs. purely
    tree-like (every vessel a single point of failure) this network is.

    A node's *core number* is the largest k for which some k-core (the
    maximal subgraph where every node has degree >= k) still contains it.
    The innermost core -- the highest core number reached -- is the most
    robustly interconnected part of the network; a network that is mostly
    pendant capillary tips off a thin backbone has a low maximum core
    number regardless of its size, while a denser capillary mesh reaches
    higher. Needs a simple graph, like ``compute_rich_club_coefficient``.
    """
    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    G_simple = nx.Graph(G) if is_mg else G
    if G_simple.number_of_nodes() == 0:
        return {"Max Core Number": 0, "Mean Core Number": "N/A (no nodes)", "Innermost Core Size": 0}
    core_numbers = nx.core_number(G_simple)
    values = list(core_numbers.values())
    max_core = max(values)
    innermost_size = sum(1 for v in values if v == max_core)
    return {
        "Max Core Number": max_core,
        "Mean Core Number": float(np.mean(values)),
        "Innermost Core Size": innermost_size,
    }
