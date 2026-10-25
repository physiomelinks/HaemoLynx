"""Degree-2 node removal and edge merging."""
import logging
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import networkx as nx

from .thick_vessel_junctions import IS_ZERO_RESISTANCE, THICK_TAPER, taper_meets_vessel

#: What smoothing did to a vessel (``smoothing.smooth_graph_centrelines``),
#: least smoothed first: a merged vessel says the least of its two halves.
_SMOOTHING_ORDER = ("kept_raw", "relaxed", "smoothed", "too_short")


def _merged_smoothing(d1: dict, d2: dict):
    """The ``centreline_smoothing`` a vessel merged from *d1* and *d2*
    carries, or None when neither says: dropped at a merge, a clean-up after
    smoothing left vessels that said nothing of what happened to them."""
    known = [d.get("centreline_smoothing") for d in (d1, d2) if d.get("centreline_smoothing") in _SMOOTHING_ORDER]
    return min(known, key=_SMOOTHING_ORDER.index) if known else None
from ._helpers import (
    get_all_edge_data,
    create_merged_edge_attributes,
    voxel_path_overlap_ratio,
    add_edge_safe,
    has_edge_safe,
    remove_edge_safe,
    are_paths_similar,
    is_path_curved,
    merge_curved_edges,
    should_add_merged_edge,
    calculate_path_length,
    merge_edge_voxels_at_node,
)

logger = logging.getLogger(__name__)

#: The share of a vessel's length that mask recovery drew (``recovered``
#: edges) and that bridges drew (edges with a ``bridge_kind``), and the kinds
#: of those bridges joined by "+": what a merged vessel keeps of where its
#: halves came from. A share, not microns, since a split copies its edge's
#: attributes onto both pieces.
RECOVERED_FRACTION = "recovered_fraction"
BRIDGED_FRACTION = "bridged_fraction"
BRIDGE_KINDS = "bridge_kinds"


def _drawn_fraction(data: dict, fraction: str, whole: bool) -> float:
    if fraction in data:
        return float(data[fraction] or 0.0)
    return 1.0 if whole else 0.0


def _bridge_kinds(data: dict) -> set:
    kinds = data.get(BRIDGE_KINDS) or data.get("bridge_kind") or ""
    return {kind for kind in str(kinds).split("+") if kind}


def carry_merged_provenance(merged: dict, d1: dict, d2: dict) -> dict:
    """Set, in place on *merged* (the attributes of the vessel merged from
    *d1* and *d2*), what it keeps of where its halves came from, and return
    it. Merges dropped it, so a report could not tell a recovered or bridged
    stretch once the clean-up had merged it into its vessel.

    ``recovered`` and ``bridge_kind`` (with ``bridge_background_um``) describe
    the whole edge, so they stay only where both halves had them, one kind of
    bridge: recovery's own trims (``mask_recovery.keep_two_ended_recovery``,
    ``_trim_recovered_spur``) take out a ``recovered`` edge whole, and must
    not take a vessel with it. :data:`RECOVERED_FRACTION`,
    :data:`BRIDGED_FRACTION` and :data:`BRIDGE_KINDS` say how much of it
    either drew, however it was merged, weighted by each half's ``length``."""
    for key in ("recovered", "bridge_kind", "bridge_background_um", RECOVERED_FRACTION, BRIDGED_FRACTION, BRIDGE_KINDS):
        merged.pop(key, None)
    halves = (d1, d2)
    if all(d.get("recovered") for d in halves):
        merged["recovered"] = True
    kinds = [d.get("bridge_kind") for d in halves]
    if kinds[0] and kinds[0] == kinds[1]:
        merged["bridge_kind"] = kinds[0]
        backgrounds = [d["bridge_background_um"] for d in halves if d.get("bridge_background_um") is not None]
        if backgrounds:
            merged["bridge_background_um"] = float(max(backgrounds))
    lengths = np.array([float(d.get("length") or 0.0) for d in halves])
    total = float(lengths.sum())
    weights = lengths / total if total > 0 else np.full(2, 0.5)
    for fraction, whole in (
        (RECOVERED_FRACTION, lambda d: bool(d.get("recovered"))),
        (BRIDGED_FRACTION, lambda d: bool(d.get("bridge_kind"))),
    ):
        share = float(sum(w * _drawn_fraction(d, fraction, whole(d)) for w, d in zip(weights, halves)))
        if share > 0.0:
            merged[fraction] = min(share, 1.0)
    every_kind = _bridge_kinds(d1) | _bridge_kinds(d2)
    if every_kind:
        merged[BRIDGE_KINDS] = "+".join(sorted(every_kind))
    return merged


def drawn_um(data: dict, fraction: str) -> float:
    """How many um of an edge mask recovery (:data:`RECOVERED_FRACTION`) or
    bridges (:data:`BRIDGED_FRACTION`) drew: all of it for an edge that is
    one, the carried share of a vessel merged from one."""
    whole = bool(data.get("recovered") if fraction == RECOVERED_FRACTION else data.get("bridge_kind"))
    try:
        length = float(data.get("length") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return length * _drawn_fraction(data, fraction, whole)


def _compute_skeleton_overlap(
    voxels: List, skeleton_data: np.ndarray
) -> float:
    """Fraction of voxel path coordinates that lie on the skeleton."""
    if not voxels or skeleton_data is None or skeleton_data.size == 0:
        return 0.0
    shape = skeleton_data.shape
    on_skeleton = 0
    total = 0
    for v in voxels:
        coords = tuple(int(round(c)) for c in v)
        if all(0 <= coords[i] < shape[i] for i in range(len(coords))):
            total += 1
            if skeleton_data[coords]:
                on_skeleton += 1
        else:
            total += 1
    return on_skeleton / total if total > 0 else 0.0


def _should_replace_existing_simple_edge(
    G: nx.Graph,
    n1: Any,
    n2: Any,
    merged_voxels: List,
    overlap_threshold: float = 0.9,
) -> bool:
    """Only replace existing edge when voxel overlap is significant."""
    if not G.has_edge(n1, n2):
        return True
    existing = G.get_edge_data(n1, n2) or {}
    existing_voxels = existing.get("voxels", [])
    overlap = voxel_path_overlap_ratio(existing_voxels, merged_voxels)
    return overlap >= overlap_threshold


def safer_simple_remove_all_degree2_nodes(
    G: Union[nx.Graph, nx.MultiGraph],
    max_degree: int = 4,
    debug: bool = False,
    max_iterations: int = 100,
    max_edge_length_ratio: float = 2.0,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Remove degree-2 nodes with safety check for shortcut creation."""
    total_removed = 0
    skipped_long_edges = 0
    is_multigraph = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))

    if debug:
        initial_degree2 = len([n for n in G.nodes() if G.degree[n] == 2])
        logger.info("Starting safer cleanup with %d degree-2 nodes", initial_degree2)

    for iteration in range(max_iterations):
        removed_this_iter = 0
        degree2_nodes = [n for n in G.nodes() if G.degree[n] == 2]
        if not degree2_nodes:
            break
        for node in degree2_nodes:
            if not G.has_node(node) or G.degree[node] != 2:
                continue
            neighbors = list(G.neighbors(node))
            if len(neighbors) != 2:
                continue
            n1, n2 = neighbors
            if G.degree[n1] >= max_degree or G.degree[n2] >= max_degree:
                continue
            edge1_data_list = get_all_edge_data(G, node, n1)
            edge2_data_list = get_all_edge_data(G, node, n2)
            node_pos = G.nodes[node].get("pos", None)
            if not edge1_data_list or not edge2_data_list:
                continue
            if "pos" in G.nodes[n1] and "pos" in G.nodes[n2] and node_pos is not None:
                pos1 = np.array(G.nodes[n1]["pos"])
                pos2 = np.array(G.nodes[n2]["pos"])
                node_pos_arr = np.array(node_pos)
                current_path_length = np.linalg.norm(pos1 - node_pos_arr) + np.linalg.norm(
                    node_pos_arr - pos2
                )
                direct_distance = np.linalg.norm(pos2 - pos1)
                if current_path_length > direct_distance * max_edge_length_ratio:
                    if debug:
                        logger.debug(
                            "Skipped removing node %s: would create shortcut", node
                        )
                    skipped_long_edges += 1
                    continue
            G.remove_node(node)
            if is_multigraph:
                for edge1_data in edge1_data_list:
                    for edge2_data in edge2_data_list:
                        merged_attrs = carry_merged_provenance(
                            create_merged_edge_attributes(edge1_data, edge2_data, node_pos),
                            edge1_data,
                            edge2_data,
                        )
                        add_edge_safe(G, n1, n2, **merged_attrs)
            else:
                edge1_data = edge1_data_list[0]
                edge2_data = edge2_data_list[0]
                merged_attrs = carry_merged_provenance(
                    create_merged_edge_attributes(edge1_data, edge2_data, node_pos),
                    edge1_data,
                    edge2_data,
                )
                if has_edge_safe(G, n1, n2):
                    if _should_replace_existing_simple_edge(
                        G, n1, n2, merged_attrs.get("voxels", [])
                    ):
                        remove_edge_safe(G, n1, n2)
                    else:
                        if debug:
                            logger.debug(
                                "Preserved existing edge %s-%s (low overlap with merged path)",
                                n1,
                                n2,
                            )
                        continue
                add_edge_safe(G, n1, n2, **merged_attrs)
            removed_this_iter += 1
            total_removed += 1
        if debug and removed_this_iter > 0:
            remaining = len([n for n in G.nodes() if G.degree[n] == 2])
            logger.debug(
                "Iteration %d: removed %d, %d remain",
                iteration + 1,
                removed_this_iter,
                remaining,
            )
        if removed_this_iter == 0:
            break

    logger.info(
        "Safer cleanup: removed %d, skipped %d long edges, final degree-2: %d",
        total_removed,
        skipped_long_edges,
        len([n for n in G.nodes() if G.degree[n] == 2]),
    )
    return G


def trivial_remove_all_degree2_nodes(
    G: Union[nx.Graph, nx.MultiGraph],
    max_degree: int = 4,
    debug: bool = False,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Remove all degree-2 nodes by merging adjacent edges."""
    total_removed = 0
    is_multigraph = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    for iteration in range(100):
        degree2_nodes = [n for n in G.nodes() if G.degree[n] == 2]
        if not degree2_nodes:
            break
        removed_this_iter = 0
        for node in degree2_nodes:
            if not G.has_node(node) or G.degree[node] != 2:
                continue
            neighbors = list(G.neighbors(node))
            if len(neighbors) != 2:
                continue
            n1, n2 = neighbors
            if G.degree[n1] >= max_degree or G.degree[n2] >= max_degree:
                continue
            node_pos = G.nodes[node].get("pos", None)
            edge1_data_list = get_all_edge_data(G, node, n1)
            edge2_data_list = get_all_edge_data(G, node, n2)
            if not edge1_data_list or not edge2_data_list:
                continue
            G.remove_node(node)
            if is_multigraph:
                for edge1_data in edge1_data_list:
                    for edge2_data in edge2_data_list:
                        merged_edge = create_trivial_merged_edge(
                            edge1_data, edge2_data, node_pos
                        )
                        G.add_edge(n1, n2, **merged_edge)
            else:
                edge1_data = edge1_data_list[0]
                edge2_data = edge2_data_list[0]
                merged_edge = create_trivial_merged_edge(
                    edge1_data, edge2_data, node_pos
                )
                if G.has_edge(n1, n2):
                    if _should_replace_existing_simple_edge(
                        G, n1, n2, merged_edge.get("voxels", [])
                    ):
                        G.remove_edge(n1, n2)
                    else:
                        if debug:
                            logger.debug(
                                "Preserved existing edge %s-%s (low overlap with trivial merged path)",
                                n1,
                                n2,
                            )
                        continue
                G.add_edge(n1, n2, **merged_edge)
            removed_this_iter += 1
            total_removed += 1
            if debug:
                logger.debug("Removed degree-2 node %s, connected %s-%s", node, n1, n2)
        if removed_this_iter == 0:
            break
    logger.info("Trivial removal: total removed %d", total_removed)
    return G


def create_trivial_merged_edge(
    edge1_data: dict, edge2_data: dict, removed_node_pos: Any
) -> dict:
    """Create merged edge with exact topology preservation."""
    voxels1 = edge1_data.get("voxels", [])
    voxels2 = edge2_data.get("voxels", [])
    merged_voxels = merge_edge_voxels_at_node(voxels1, voxels2, removed_node_pos)
    merged_attributes = {
        # Recomputed from the merged path, matching smart_multigraph_degree2_removal.
        # The additive sum goes stale whenever a path is re-routed upstream.
        "length": calculate_path_length(merged_voxels),
        "voxels": merged_voxels,
        "merged": True,
        "trivial_merge": True,
        "removed_node_pos": removed_node_pos,
    }
    for key, value in edge1_data.items():
        if key not in merged_attributes:
            merged_attributes[key] = value
    # A taper only with a taper (taper_meets_vessel): its protection in the
    # clean-up is not the whole vessel's.
    if edge1_data.get(THICK_TAPER) and edge2_data.get(THICK_TAPER):
        merged_attributes[THICK_TAPER] = True
    else:
        merged_attributes.pop(THICK_TAPER, None)
    smoothing = _merged_smoothing(edge1_data, edge2_data)
    if smoothing is not None:
        merged_attributes["centreline_smoothing"] = smoothing
    return carry_merged_provenance(merged_attributes, edge1_data, edge2_data)


def duplicate_parallel_edges(
    G: nx.MultiGraph,
    inside_lumen: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> List[Tuple[Any, Any, Any]]:
    """The ``(u, v, key)`` edges that repeat another edge between the same two
    nodes: the same vessel twice (``are_paths_similar``: within 3 um of each
    other or, given *inside_lumen*, through one lumen).

    Of each such group the edge kept -- left out of the list -- is a curved
    one over a straight one, then the shortest, as when degree-2 merging
    meets a duplicate: a straight edge is usually a bridge across a gap, not
    the vessel. An edge without a path is never a duplicate. Self-loops are
    not considered.
    """
    duplicates: List[Tuple[Any, Any, Any]] = []
    seen = set()
    for u, v in G.edges():
        if u == v or (u, v) in seen or G.number_of_edges(u, v) < 2:
            continue
        seen.update({(u, v), (v, u)})
        preferred_first = sorted(
            G[u][v].items(),
            key=lambda item: (
                not is_path_curved(item[1].get("voxels") or []),
                item[1].get("length", float("inf")),
            ),
        )
        kept: List[Any] = []
        for key, data in preferred_first:
            voxels = data.get("voxels") or []
            if any(
                are_paths_similar(voxels, G[u][v][other].get("voxels") or [], inside_lumen=inside_lumen)
                for other in kept
            ):
                duplicates.append((u, v, key))
            else:
                kept.append(key)
    return duplicates


def remove_duplicate_parallel_edges(
    G: nx.MultiGraph,
    inside_lumen: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> int:
    """Remove every edge :func:`duplicate_parallel_edges` names; how many.

    Never disconnects anything: each removed edge leaves the one it repeats.
    """
    duplicates = duplicate_parallel_edges(G, inside_lumen)
    for u, v, key in duplicates:
        G.remove_edge(u, v, key=key)
    return len(duplicates)


def duplicate_vessel_routes(
    G: nx.MultiGraph,
    inside_lumen: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> List[Any]:
    """The degree-2 nodes whose route -- their two edges joined -- repeats an
    edge already joining their two neighbours (``are_paths_similar``, as for
    :func:`duplicate_parallel_edges`): the same vessel twice, once as an edge
    and once through the node.

    Degree-2 merging leaves none; this is the check that it did.
    """
    routes: List[Any] = []
    for node in G.nodes():
        if G.degree[node] != 2 or G.nodes[node].get("pos") is None:
            continue
        edges = list(G.edges(node, data=True))
        if len(edges) != 2:
            continue
        (_, n1, d1), (_, n2, d2) = edges
        voxels1, voxels2 = d1.get("voxels") or [], d2.get("voxels") or []
        if n1 == n2 or not G.has_edge(n1, n2) or len(voxels1) < 2 or len(voxels2) < 2:
            continue
        route = merge_curved_edges(voxels1, voxels2, np.asarray(G.nodes[node]["pos"], dtype=float))
        if any(
            are_paths_similar(route, data.get("voxels") or [], inside_lumen=inside_lumen)
            for data in G[n1][n2].values()
        ):
            routes.append(node)
    return routes


def smart_multigraph_degree2_removal(
    G: nx.MultiGraph,
    skeleton_data: np.ndarray = None,
    max_degree: int = 4,
    debug: bool = False,
    max_iterations: int = 500,
    inside_lumen: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> nx.MultiGraph:
    """Degree-2 removal for MultiGraphs: each removed node's two edges become
    one, whose path is the two edges' own paths joined at the node.

    Where the node's neighbours are already joined, the merged path is
    compared with that edge (``should_add_merged_edge``): a different vessel
    is added beside it, a duplicate replaces it or, when the existing edge is
    the better of the two, is dropped with the node. *inside_lumen*, the
    segmentation the skeleton came from (``assemble.mask_lumen_test``), lets
    "duplicate" mean one lumen as well as "within 3 um": Lee thinning leaves
    strands 5-10 um apart through one wide vessel.

    Each round of merges starts by removing every duplicate parallel edge
    already there (:func:`remove_duplicate_parallel_edges`) -- the loops
    skeletonisation leaves round one lumen, which no merge ever compares, and
    whatever the steps between passes join twice -- and the last round merges
    nothing, so adds none. A degree-2 node beside a junction of *max_degree*
    or more, otherwise left alone, still goes when its route repeats an edge
    joining its neighbours: a graph this returns has no vessel twice between
    the same two nodes (:func:`duplicate_parallel_edges` and
    :func:`duplicate_vessel_routes` are empty).

    *skeleton_data* is accepted and not read. Straight legs used to be
    re-traced through the skeleton by A*, whose route was not tied to the
    edges being merged: across a gap bridge, which has no skeleton, it went
    round through other vessels -- often back down the leg it had just come
    up -- and elsewhere it cut through other junctions' voxels. Either way the
    merged edge ran over voxels that it, or another edge, already covered.
    """
    if not isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("This function is designed for MultiGraphs")

    total_removed = duplicates_removed = 0
    for iteration in range(max_iterations):
        duplicates_removed += remove_duplicate_parallel_edges(G, inside_lumen)
        removed_this_iter = 0

        for node in list(G.nodes()):
            if not G.has_node(node) or G.degree[node] != 2:
                continue

            edges = list(G.edges(node, keys=True, data=True))
            if len(edges) != 2:
                continue

            _, n1, k1, d1 = edges[0]
            _, n2, k2, d2 = edges[1]

            # A thick-vessel bridge is not vessel: merged with one, the
            # vessel would lose its resistance or the bridge gain one.
            if d1.get(IS_ZERO_RESISTANCE) or d2.get(IS_ZERO_RESISTANCE):
                continue
            # Nor is a taper merged with a vessel: the clean-up's protection of
            # the taper would cover the whole vessel, and spread with each merge.
            if taper_meets_vessel(d1, d2):
                continue

            # Degenerate case: both incident edges point to the same neighbor.
            # Removing the node here can collapse/erase valid parallel paths.
            if n1 == n2:
                continue

            # Beside a junction of max_degree or more the node stays -- unless
            # its route repeats an edge already joining n1 and n2, which is
            # resolved below whatever the degrees: one vessel, kept once.
            busy = G.degree[n1] >= max_degree or G.degree[n2] >= max_degree
            if busy and not G.has_edge(n1, n2):
                continue

            node_pos = G.nodes[node].get("pos", None)
            n1_pos = G.nodes[n1].get("pos", None)
            n2_pos = G.nodes[n2].get("pos", None)
            if node_pos is None or n1_pos is None or n2_pos is None:
                continue

            voxels1 = d1.get("voxels", [])
            voxels2 = d2.get("voxels", [])

            merged_voxels = merge_curved_edges(voxels1, voxels2, np.array(node_pos), debug)
            merged_attrs = {
                "length": calculate_path_length(merged_voxels),
                "voxels": merged_voxels,
                "merged": True,
                "original_edges": 2,
            }
            if d1.get(THICK_TAPER) and d2.get(THICK_TAPER):
                merged_attrs[THICK_TAPER] = True
            smoothing = _merged_smoothing(d1, d2)
            if smoothing is not None:
                merged_attrs["centreline_smoothing"] = smoothing
            carry_merged_provenance(merged_attrs, d1, d2)

            should_add, replace_key = should_add_merged_edge(
                G, n1, n2, merged_voxels, merged_attrs, debug, inside_lumen=inside_lumen
            )
            if busy and should_add and replace_key is None:
                continue  # a separate vessel beside the edge, at a busy junction
            # The node goes either way: its route becomes one edge or, when
            # refused as a duplicate of a better n1-n2 edge, is dropped.
            G.remove_node(node)
            if not should_add:
                removed_this_iter += 1
                total_removed += 1
                continue
            if replace_key is not None:
                G.remove_edge(n1, n2, key=replace_key)
            G.add_edge(n1, n2, **merged_attrs)

            removed_this_iter += 1
            total_removed += 1

        if removed_this_iter == 0:
            break

    logger.info(
        "Smart removal: %d removed (and %d duplicate parallel edges), "
        "graph now has %d nodes / %d edges",
        total_removed,
        duplicates_removed,
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G