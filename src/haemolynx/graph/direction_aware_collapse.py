"""A direction-aware alternative to :func:`collapse.collapse_node_clusters`.

That function collapses every node within ``distance_threshold`` of another
into one representative, by taking the *connected components* of a proximity
graph -- single-linkage clustering with no limit on how far a merge can
chain. In a densely braided or looped region (many short segments, many
nearby branch points -- exactly what thickness-gated skeletonisation's Lee
thinning path can produce), that chaining can sweep a whole tangled patch
into one node over its ten repair passes, each carrying every edge from
every node it absorbed. That is a cartwheel hub by construction -- see
``cartwheel_guard`` for the diagnostic that names the shape, and its own
docstring for why it cannot undo the damage after the fact: once collapsed,
the cluster's original, separate positions are gone.

This module stops the damage before it happens instead of reporting it
afterwards. It reuses ``cartwheel_guard``'s own geometry (never
reimplemented here) as a *gate* on the merge itself: growing a cluster is
allowed only as long as what it would spoke out to stays directionally
coherent. A candidate merge that would create a hub cartwheel_guard could
flag is skipped -- the nodes involved are left as separate, nearby
representatives instead of one. Merges that pass are additionally checked
for redundant parallel wiring to the same neighbour (several original nodes
in a cluster reaching the same external node by slightly different
skeleton-noise paths), keeping only the shorter of any resulting duplicates
-- collapsing a hairball should not inflate a real neighbour's edge count
either.

Opt-in via ``cluster_collapse_method="direction_aware"`` (default
``"distance_only"``, the unmodified legacy behaviour); deliberately kept in
its own module, reusing only read-only helpers from ``collapse``/
``cartwheel_guard``, so it can be deleted along with its one call site and
two settings without touching either of those modules if it does not help.
"""
from __future__ import annotations

import logging
from typing import Any, Union

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from .cartwheel_guard import (
    DEFAULT_TANGENT_LENGTH_UM,
    _incident_edge_items,
    _spoke_direction_and_length,
    hub_radial_dispersion,
)
from .collapse import _patch_voxel_endpoint

logger = logging.getLogger(__name__)

#: Below this many external edges, a merge is never blocked -- an ordinary
#: bifurcation or trifurcation cannot look wheel-shaped, and this matches
#: cartwheel_guard's own default floor for the same reason.
DEFAULT_MIN_DEGREE_FOR_DISPERSION_CHECK = 6

#: Merging must not drop a hub's radial dispersion to or below this --
#: matches cartwheel_guard's own default "spread out enough to be a
#: cartwheel" line, kept as a separate setting so the two features (one
#: diagnostic, one corrective) can be tuned or removed independently.
DEFAULT_MAX_RADIAL_DISPERSION = 0.5


def _merge_is_direction_safe(
    G: Union[nx.Graph, nx.MultiGraph],
    candidate_members: set,
    *,
    min_degree_for_dispersion_check: int,
    max_radial_dispersion: float,
    tangent_length_um: float,
) -> bool:
    """Whether merging *candidate_members* into one node stays coherent.

    Every edge with exactly one end in *candidate_members* becomes a spoke
    of the merged node; edges with both ends inside it are absorbed by the
    merge and are not spokes. Safe when there are too few spokes to judge
    fairly (fewer than *min_degree_for_dispersion_check*, or too few with a
    resolvable direction), or when the spokes that do resolve still agree
    well enough (dispersion above *max_radial_dispersion*).
    """
    directions: list[np.ndarray] = []
    degree = 0
    for node in candidate_members:
        for neighbor, _key, data in _incident_edge_items(G, node):
            if neighbor in candidate_members:
                continue
            degree += 1
            direction, _length = _spoke_direction_and_length(
                G, node, neighbor, data, tangent_length_um=tangent_length_um
            )
            if direction is not None:
                directions.append(direction)

    if degree < min_degree_for_dispersion_check:
        return True
    if len(directions) < min_degree_for_dispersion_check:
        # Too many spokes have no resolvable direction (missing pos, a
        # duplicate point) to judge fairly -- do not block on missing data.
        return True
    return hub_radial_dispersion(directions) > max_radial_dispersion


def _rewire_edges_deduplicating(
    G: Union[nx.Graph, nx.MultiGraph],
    old_node: Any,
    new_node: Any,
    is_multi: bool,
) -> None:
    """Move every edge incident to *old_node* onto *new_node*.

    Same as ``collapse._rewire_edges``, plus: when *new_node* already
    reaches a neighbour some other way (its own original edge, or one moved
    there earlier in this same cluster's merge), keep only the shorter of
    the two rather than both -- a cluster's internal nodes reaching the same
    external neighbour by slightly different noise paths should not inflate
    that neighbour's apparent number of distinct connections.
    """
    old_pos = np.asarray(G.nodes[old_node].get("pos", [0, 0, 0]), dtype=float)
    new_pos = np.asarray(G.nodes[new_node].get("pos", [0, 0, 0]), dtype=float)

    if is_multi:
        edges = list(G.edges(old_node, data=True, keys=True))
        for u, v, _key, data in edges:
            neighbor = v if u == old_node else u
            if neighbor == new_node:
                continue
            patched = _patch_voxel_endpoint(data, old_pos, new_pos)
            new_length = patched.get("length", float("inf"))
            existing = G.get_edge_data(new_node, neighbor) or {}
            if existing:
                shortest_key = min(
                    existing, key=lambda k: existing[k].get("length", float("inf"))
                )
                shortest_length = existing[shortest_key].get("length", float("inf"))
                if new_length < shortest_length:
                    G.remove_edge(new_node, neighbor, key=shortest_key)
                    G.add_edge(new_node, neighbor, **patched)
                # else: the new parallel edge is redundant and longer; drop it.
            else:
                G.add_edge(new_node, neighbor, **patched)
    else:
        edges = list(G.edges(old_node, data=True))
        for u, v, data in edges:
            neighbor = v if u == old_node else u
            if neighbor == new_node:
                continue
            patched = _patch_voxel_endpoint(data, old_pos, new_pos)
            if not G.has_edge(new_node, neighbor):
                G.add_edge(new_node, neighbor, **patched)
            else:
                existing_len = G[new_node][neighbor].get("length", float("inf"))
                new_len = patched.get("length", float("inf"))
                if new_len < existing_len:
                    G.remove_edge(new_node, neighbor)
                    G.add_edge(new_node, neighbor, **patched)


def collapse_node_clusters_direction_aware(
    G: Union[nx.Graph, nx.MultiGraph],
    distance_threshold: float = 5.0,
    *,
    max_radial_dispersion: float = DEFAULT_MAX_RADIAL_DISPERSION,
    min_degree_for_dispersion_check: int = DEFAULT_MIN_DEGREE_FOR_DISPERSION_CHECK,
    tangent_length_um: float = DEFAULT_TANGENT_LENGTH_UM,
    debug: bool = False,
    max_iterations: int = 10,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Collapse nearby node clusters, refusing a merge that would cartwheel.

    Same proximity discovery as ``collapse_node_clusters`` (a cKDTree over
    node positions, pairs within *distance_threshold*), but candidate pairs
    are agglomerated closest-first with union-find rather than taken as
    whole connected components: two (possibly already-grown) groups are
    merged only when :func:`_merge_is_direction_safe` says the combined
    node's spokes would still agree well enough. A chained proximity
    component that would otherwise become one cartwheel-shaped hub instead
    stops growing where it would start looking like one, leaving separate
    representative nodes for what proximity alone could not tell apart.

    Raises
    ------
    ValueError
        If *max_radial_dispersion* is outside ``[0.0, 1.0]`` or
        *min_degree_for_dispersion_check* is below 2 -- mirroring
        ``cartwheel_guard.detect_cartwheel_hubs``'s own validation, since
        both read the same kind of value.
    """
    if not 0.0 <= max_radial_dispersion <= 1.0:
        raise ValueError("max_radial_dispersion must be in [0.0, 1.0]")
    if min_degree_for_dispersion_check < 2:
        raise ValueError("min_degree_for_dispersion_check must be >= 2")

    G = G.copy()
    is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    total_merged = 0

    for iteration in range(max_iterations):
        nodes_with_pos = [
            (n, np.array(G.nodes[n]["pos"], dtype=float))
            for n in G.nodes()
            if "pos" in G.nodes[n]
        ]
        if len(nodes_with_pos) < 2:
            break

        node_ids = [n for n, _ in nodes_with_pos]
        coords = np.array([p for _, p in nodes_with_pos])

        tree = cKDTree(coords)
        pairs = tree.query_pairs(distance_threshold)
        if not pairs:
            break

        pair_list = sorted(
            pairs, key=lambda ij: float(np.linalg.norm(coords[ij[0]] - coords[ij[1]]))
        )

        parent = {n: n for n in node_ids}
        members: dict[Any, set] = {n: {n} for n in node_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        blocked_this_iter = 0
        for i, j in pair_list:
            root_a, root_b = find(node_ids[i]), find(node_ids[j])
            if root_a == root_b:
                continue
            candidate = members[root_a] | members[root_b]
            if not _merge_is_direction_safe(
                G,
                candidate,
                min_degree_for_dispersion_check=min_degree_for_dispersion_check,
                max_radial_dispersion=max_radial_dispersion,
                tangent_length_um=tangent_length_um,
            ):
                blocked_this_iter += 1
                continue
            if len(members[root_a]) < len(members[root_b]):
                root_a, root_b = root_b, root_a
            parent[root_b] = root_a
            members[root_a] |= members[root_b]
            del members[root_b]

        merged_this_iter = 0
        for group in members.values():
            if len(group) < 2:
                continue
            group = [n for n in group if G.has_node(n)]
            if len(group) < 2:
                continue

            rep = max(group, key=lambda n: (G.degree(n), -n))
            others = [n for n in group if n != rep]

            if debug:
                logger.info(
                    "Collapsing cluster of %d nodes %s -> representative %s "
                    "(direction-aware)",
                    len(group), group, rep,
                )

            cluster_positions = np.array(
                [G.nodes[n]["pos"] for n in group if "pos" in G.nodes[n]]
            )
            G.nodes[rep]["pos"] = cluster_positions.mean(axis=0)

            for other in others:
                if not G.has_node(other):
                    continue
                _rewire_edges_deduplicating(G, other, rep, is_multi)
                G.remove_node(other)
                merged_this_iter += 1

        total_merged += merged_this_iter
        if debug:
            logger.info(
                "Iteration %d: merged %d nodes, blocked %d cartwheel-shaped "
                "merges (%d total merged)",
                iteration + 1, merged_this_iter, blocked_this_iter, total_merged,
            )
        if merged_this_iter == 0:
            break

    self_loops = list(nx.selfloop_edges(G, keys=True)) if is_multi else [
        (u, v) for u, v in nx.selfloop_edges(G)
    ]
    if self_loops:
        if is_multi:
            for u, v, k in self_loops:
                G.remove_edge(u, v, key=k)
        else:
            G.remove_edges_from(self_loops)
        if debug:
            logger.info("Removed %d self-loops after collapsing", len(self_loops))

    if debug or total_merged > 0:
        logger.info(
            "collapse_node_clusters_direction_aware: merged %d nodes, "
            "graph now has %d nodes / %d edges",
            total_merged, G.number_of_nodes(), G.number_of_edges(),
        )
    return G
