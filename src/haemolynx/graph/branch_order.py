"""Assign branch orders to graph edges via BFS from boundary nodes."""
from __future__ import annotations

import logging
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

import networkx as nx

from ._helpers import edge_id

logger = logging.getLogger(__name__)

PostAssignCallback = Callable[[nx.MultiGraph], None]


class MissingSmallVesselAssignmentWarning(UserWarning):
    """Strict branch-order mode ran without small arteriole/venule terminals.

    Capillary ``B*`` orders are still assigned from the inlet nodes; Art*/Ven*
    labelling is skipped because no small-vessel (or manual A/V) terminals
    were provided.
    """


def _compute_node_distances(
    G: nx.MultiGraph,
    inlet_nodes: list[int],
    stop_nodes: set[int] | None = None,
) -> dict[int, int]:
    """Compute unweighted BFS node distances with optional stop nodes."""
    stop_nodes = stop_nodes or set()
    node_distances: dict[int, int] = {}
    queue: deque[tuple[int, int]] = deque()

    for inlet_node in inlet_nodes:
        if inlet_node in G.nodes():
            if inlet_node not in node_distances:
                node_distances[inlet_node] = 0
                queue.append((inlet_node, 0))
        else:
            logger.warning("Inlet node %s not found in graph", inlet_node)

    while queue:
        current_node, distance = queue.popleft()
        if (
            current_node in node_distances
            and node_distances[current_node] < distance
        ):
            continue
        if current_node in stop_nodes:
            continue
        for neighbor in G.neighbors(current_node):
            new_distance = distance + 1
            if (
                neighbor not in node_distances
                or node_distances[neighbor] > new_distance
            ):
                node_distances[neighbor] = new_distance
                queue.append((neighbor, new_distance))
    return node_distances


def assign_branch_orders(
    G: nx.MultiGraph,
    inlet_nodes: list[int],
    prefix: str = "B",
    stop_nodes: set[int] | None = None,
    excluded_edges: set[tuple[int, int, int]] | None = None,
    included_edges: set[tuple[int, int, int]] | None = None,
) -> dict:
    """Assign branch-order labels to edges from BFS distance to inlet nodes."""
    node_distances = _compute_node_distances(
        G,
        inlet_nodes,
        stop_nodes=stop_nodes,
    )
    excluded_edges = excluded_edges or set()
    included_edges = included_edges or set()

    results = {
        "edges_assigned": 0,
        "edges_skipped": 0,
        "branch_order_counts": defaultdict(int),
        "unreachable_edges": [],
        "excluded_edges": 0,
    }

    for u, v, key, data in G.edges(keys=True, data=True):
        edge_identifier = edge_id(u, v, key)
        if included_edges and edge_identifier not in included_edges:
            results["edges_skipped"] += 1
            continue
        if edge_identifier in excluded_edges:
            results["excluded_edges"] += 1
            results["edges_skipped"] += 1
            continue
        u_dist = node_distances.get(u, float("inf"))
        v_dist = node_distances.get(v, float("inf"))
        if u_dist == float("inf") and v_dist == float("inf"):
            results["unreachable_edges"].append((u, v, key))
            results["edges_skipped"] += 1
            continue
        edge_distance = min(u_dist, v_dist) + 1
        branch_order = f"{prefix}{edge_distance:02d}" if prefix == "B" else f"{prefix}{edge_distance}"
        G[u][v][key]["branch_order"] = branch_order
        results["edges_assigned"] += 1
        results["branch_order_counts"][branch_order] += 1
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Edge (%s, %s, %s): %s (u_dist=%s, v_dist=%s)",
                u,
                v,
                key,
                branch_order,
                u_dist,
                v_dist,
            )
    return results


def assign_hierarchical_branch_orders(
    G: nx.MultiGraph,
    inlet_nodes: list[int],
    outlet_nodes: list[int],
    arteriole_boundary_nodes: list[int],
    venule_boundary_nodes: list[int],
) -> dict:
    """Assign Art*/Ven*/B* branch orders using arteriole and venule boundaries."""
    arteriole_boundary_set = set(arteriole_boundary_nodes)
    venule_boundary_set = set(venule_boundary_nodes)

    arteriole_node_distances = _compute_node_distances(
        G,
        inlet_nodes,
        stop_nodes=arteriole_boundary_set,
    )
    arteriole_nodes = set(arteriole_node_distances.keys())
    arteriole_edges = {
        edge_id(u, v, key)
        for u, v, key in G.edges(keys=True)
        if u in arteriole_nodes and v in arteriole_nodes
    }
    arteriole_results = assign_branch_orders(
        G,
        inlet_nodes,
        prefix="Art",
        stop_nodes=arteriole_boundary_set,
        included_edges=arteriole_edges,
    )

    venule_node_distances = _compute_node_distances(
        G,
        outlet_nodes,
        stop_nodes=venule_boundary_set,
    )
    venule_nodes = set(venule_node_distances.keys())
    venule_edges = {
        edge_id(u, v, key)
        for u, v, key in G.edges(keys=True)
        if u in venule_nodes and v in venule_nodes
    }
    venule_results = assign_branch_orders(
        G,
        outlet_nodes,
        prefix="Ven",
        stop_nodes=venule_boundary_set,
        excluded_edges=arteriole_edges,
        included_edges=venule_edges,
    )

    capillary_start_nodes = arteriole_boundary_nodes if arteriole_boundary_nodes else inlet_nodes
    capillary_excluded_edges = arteriole_edges | venule_edges
    capillary_results = assign_branch_orders(
        G,
        capillary_start_nodes,
        prefix="B",
        excluded_edges=capillary_excluded_edges,
    )

    return {
        "arteriole": arteriole_results,
        "venule": venule_results,
        "capillary": capillary_results,
        "arteriole_edge_count": len(arteriole_edges),
        "venule_edge_count": len(venule_edges),
        "excluded_capillary_edge_count": len(capillary_excluded_edges),
    }


def assign_vessel_branch_orders(
    G: nx.MultiGraph,
    inlet_nodes: list[int],
    *,
    outlet_nodes: list[int] | None = None,
    arteriole_boundary_nodes: list[int] | None = None,
    venule_boundary_nodes: list[int] | None = None,
    strict_hierarchical: bool = False,
    expects_hierarchical: bool = False,
    post_assign_callback: PostAssignCallback | None = None,
) -> dict[str, Any]:
    """
    Assign branch orders on ``G`` using capillary-only or hierarchical rules.

    When arteriole boundary nodes, venule boundary nodes, and outlet nodes are
    all non-empty, uses :func:`assign_hierarchical_branch_orders`; otherwise
    assigns capillary ``B*`` orders from ``inlet_nodes`` only.

    Parameters
    ----------
    G
        Vascular graph (modified in place).
    inlet_nodes
        Inlet / arteriole-side seed nodes for capillary or hierarchical assignment.
    outlet_nodes, arteriole_boundary_nodes, venule_boundary_nodes
        Optional node sets for hierarchical assignment.
    strict_hierarchical
        When hierarchical prerequisites are missing: raise if small-vessel
        (or manual A/V) assignment was expected, otherwise warn and fall back
        to capillary ``B*`` orders.
    expects_hierarchical
        Set when small-vessel masks or manual arteriole/venule methods were
        supposed to supply hierarchical terminals.
    post_assign_callback
        Optional ``callback(G)`` after orders are written (e.g. for plotting).

    Returns
    -------
    dict
        Summary with ``mode`` of ``"hierarchical"``, ``"capillary"``, or ``"skipped"``.
    """
    if not inlet_nodes:
        return {"mode": "skipped", "reason": "no_inlet_nodes"}

    outlet_nodes = list(outlet_nodes or [])
    arteriole_boundary_nodes = list(arteriole_boundary_nodes or [])
    venule_boundary_nodes = list(venule_boundary_nodes or [])

    use_hierarchical = bool(
        arteriole_boundary_nodes and venule_boundary_nodes and outlet_nodes
    )
    if strict_hierarchical and not use_hierarchical:
        if expects_hierarchical:
            raise ValueError(
                "Strict branch-order assignment is enabled, but hierarchical "
                "assignment prerequisites are missing. "
                f"Need non-empty outlet_nodes, arteriole_boundary_nodes, and "
                f"venule_boundary_nodes. Got counts: "
                f"outlet_nodes={len(outlet_nodes)}, "
                f"arteriole_boundary_nodes={len(arteriole_boundary_nodes)}, "
                f"venule_boundary_nodes={len(venule_boundary_nodes)}. "
                "Fix mask inputs/thresholds or disable strict_hierarchical."
            )
        warnings.warn(
            "Strict branch-order assignment is enabled, but small arterioles "
            "and venules were not assigned (no arteriole/venule boundary "
            "nodes). Assigning capillary branch orders (B01, B02, …) from "
            "inlet nodes only.",
            MissingSmallVesselAssignmentWarning,
            stacklevel=2,
        )

    if use_hierarchical:
        branch_results = assign_hierarchical_branch_orders(
            G,
            inlet_nodes=inlet_nodes,
            outlet_nodes=outlet_nodes,
            arteriole_boundary_nodes=arteriole_boundary_nodes,
            venule_boundary_nodes=venule_boundary_nodes,
        )
        summary: dict[str, Any] = {"mode": "hierarchical", **branch_results}
    else:
        capillary_results = assign_branch_orders(G, inlet_nodes)
        summary = {"mode": "capillary", "capillary": capillary_results}

    if post_assign_callback is not None:
        post_assign_callback(G)

    return summary
