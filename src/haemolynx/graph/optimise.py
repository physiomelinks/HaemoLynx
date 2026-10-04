"""Optimise graph topology: terminal reconnection with skeleton validation."""
import logging
import heapq
from typing import Optional

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from ._helpers import add_edge_safe, calculate_path_length
from .build import (
    MAX_GAP_BRIDGE_TURN_DEG,
    gap_bridge_continues_terminal,
    gap_bridge_pairs_following_ends,
)
from .reconnect import MaskBridges
from .cartwheel_guard import (
    DEFAULT_MAX_RADIAL_DISPERSION,
    DEFAULT_MIN_DEGREE,
    DEFAULT_TANGENT_LENGTH_UM,
    _incident_edge_items,
    _spoke_direction_and_length,
    hub_radial_dispersion,
)
from .validate import validate_skeleton_connection

logger = logging.getLogger(__name__)


def _physical_path_length(points) -> float:
    """Compute 3D polyline length in physical units."""
    if not points or len(points) < 2:
        return 0.0
    return float(calculate_path_length(points))


def optimise_graph_topology_fixed(
    G,
    voxel_loops,
    loop_edges,
    skeleton_data=None,
    debug=False,
    reconnect_threshold=3.0,
    use_spatial_index=True,
    remove_degree2_nodes=True,
    consolidation_threshold=2.0,
    improve_junctions=True,
    preserve_multigraph=True,
    validate_reconnections=True,
    aggressive_degree2_cleanup_level=1,
    max_bridge_turn_deg=MAX_GAP_BRIDGE_TURN_DEG,
    mask_support=None,
):
    """Reconnect nearby terminals with optional skeleton validation.

    A pair is not joined when the join turns more than *max_bridge_turn_deg*
    from either terminal's end direction, or either terminal ends too short a
    vessel to have one -- the rule ``build.build_graph_segment_skan_stitched_loops``
    bridges by; ``None`` turns it off. With *mask_support* (a
    ``preprocessing.MaskSupport``) a join passing those checks is also put to
    ``reconnect.MaskBridges``: the skeleton path, else a route through the
    mask, must cross no more background than it allows and not run beside
    another edge in the same lumen.
    """
    vs = tuple(G.graph.get("voxel_size", (1.0, 1.0, 1.0)))

    if reconnect_threshold and reconnect_threshold > 0:
        valid_nodes = [n for n in G.nodes() if "pos" in G.nodes[n]]
        terminals = [n for n in valid_nodes if G.degree[n] == 1]

        if len(terminals) > 1:
            if use_spatial_index and len(terminals) > 10:
                terminal_coords = np.array([G.nodes[n]["pos"] for n in terminals])
                tree = cKDTree(terminal_coords)
                pairs_indices = tree.query_pairs(reconnect_threshold)
                pairs = []
                for i, j in pairs_indices:
                    src, tgt = terminals[i], terminals[j]
                    edge_norm = tuple(sorted([src, tgt]))
                    if (
                        G.has_edge(src, tgt)
                        or edge_norm in loop_edges
                        or G.degree[src] > 1
                        or G.degree[tgt] > 1
                    ):
                        continue
                    dist = np.linalg.norm(terminal_coords[i] - terminal_coords[j])
                    pairs.append((dist, src, tgt))
            else:
                pairs = []
                for i, src in enumerate(terminals):
                    if "pos" not in G.nodes[src]:
                        continue
                    for j in range(i + 1, len(terminals)):
                        tgt = terminals[j]
                        if "pos" not in G.nodes[tgt]:
                            continue
                        edge_norm = tuple(sorted([src, tgt]))
                        if (
                            edge_norm in loop_edges
                            or G.degree[src] > 1
                            or G.degree[tgt] > 1
                        ):
                            continue
                        src_pos = np.array(G.nodes[src]["pos"])
                        tgt_pos = np.array(G.nodes[tgt]["pos"])
                        dist = np.linalg.norm(src_pos - tgt_pos)
                        if dist <= reconnect_threshold:
                            pairs.append((dist, src, tgt))

            if max_bridge_turn_deg is not None:
                pairs = gap_bridge_pairs_following_ends(
                    G, pairs, max_turn_deg=max_bridge_turn_deg
                )
            heapq.heapify(pairs)
            bridges = None if mask_support is None else MaskBridges(G, mask_support, "optimise")
            reconnected = 0
            while pairs:
                dist, src, tgt = heapq.heappop(pairs)
                if (
                    not G.has_node(src)
                    or not G.has_node(tgt)
                    or "pos" not in G.nodes[src]
                    or "pos" not in G.nodes[tgt]
                    or G.has_edge(src, tgt)
                    or G.degree[src] > 1
                    or G.degree[tgt] > 1
                ):
                    continue
                src_pos = np.array(G.nodes[src]["pos"])
                tgt_pos = np.array(G.nodes[tgt]["pos"])

                if validate_reconnections and skeleton_data is not None:
                    connection_valid, voxel_path = validate_skeleton_connection(
                        skeleton_data, src_pos, tgt_pos, max_gap=reconnect_threshold,
                        voxel_size=vs,
                    )
                    if not connection_valid:
                        if debug:
                            logger.debug("Skipped reconnection %s-%s: no skeleton path", src, tgt)
                        continue
                    vs_arr = np.asarray(vs, dtype=float)
                    if voxel_path:
                        phys_path = [(np.array(p, dtype=float) * vs_arr).tolist() for p in voxel_path]
                    else:
                        phys_path = [src_pos.tolist(), tgt_pos.tolist()]
                    tags = {}
                    if bridges is not None:
                        found = bridges.bridge(src_pos, tgt_pos, preferred=phys_path)
                        if found is None:
                            continue
                        phys_path, tags = found[0].tolist(), bridges.attributes(found[1])
                    path_length = _physical_path_length(phys_path)
                    add_edge_safe(
                        G,
                        src,
                        tgt,
                        length=path_length if path_length > 0 else dist,
                        voxels=phys_path,
                        reconnected=True,
                        validated=True,
                        **tags,
                    )
                else:
                    conservative_threshold = min(reconnect_threshold * 0.5, 1.5)
                    if dist > conservative_threshold:
                        if debug:
                            logger.debug(
                                "Skipped reconnection %s-%s: dist %.2f > threshold %.2f",
                                src,
                                tgt,
                                dist,
                                conservative_threshold,
                            )
                        continue
                    phys_path, tags = [src_pos.tolist(), tgt_pos.tolist()], {}
                    if bridges is not None:
                        found = bridges.bridge(src_pos, tgt_pos)
                        if found is None:
                            continue
                        phys_path, tags = found[0].tolist(), bridges.attributes(found[1])
                    add_edge_safe(
                        G,
                        src,
                        tgt,
                        length=dist if not tags else _physical_path_length(phys_path),
                        voxels=phys_path,
                        reconnected=True,
                        conservative=True,
                        **tags,
                    )
                reconnected += 1
                if debug:
                    logger.debug("Reconnected %s-%s, d=%.2f", src, tgt, dist)
            if bridges is not None:
                bridges.log_summary()
            if debug and reconnected > 0:
                logger.info("Reconnected %d terminal pairs", reconnected)

    return G, voxel_loops


def _reconnection_is_direction_safe(
    G: nx.MultiGraph,
    tgt,
    tgt_pos: np.ndarray,
    src_pos: np.ndarray,
    *,
    min_degree_for_dispersion_check: int,
    max_radial_dispersion: float,
    tangent_length_um: float,
) -> bool:
    """Whether adding a new edge from *tgt* toward *src_pos* would keep
    *tgt*'s own spoke pattern directionally coherent.

    The same question ``direction_aware_collapse._merge_is_direction_safe``
    asks of a cluster merge, asked here of a single new spoke instead: this
    is the gap that guard does not cover on its own -- it only protects the
    one collapse step it wraps, while orphan/dangling reconnection runs
    later in the same topology pipeline with no cartwheel-awareness at all,
    and can independently attach many separate dangling stubs to the same
    nearby node (``max_new_edges_per_node`` caps how many edges *one
    source* contributes, not how many different sources one *target*
    accepts), recreating exactly the wheel shape the collapse step was
    built to prevent.

    Queries *tgt*'s incident edges fresh from *G* rather than a
    precomputed snapshot: unlike ``direction_aware_collapse``'s single
    static pass, this function's caller adds edges one at a time within
    the same loop, so an earlier reconnection to the same *tgt* in this
    same pass must count toward this check, not be invisible to it.
    """
    directions: list[np.ndarray] = []
    degree = 0
    for neighbor, _key, data in _incident_edge_items(G, tgt):
        degree += 1
        direction, _length = _spoke_direction_and_length(
            G, tgt, neighbor, data, tangent_length_um=tangent_length_um
        )
        if direction is not None:
            directions.append(direction)

    degree += 1
    new_vector = src_pos - tgt_pos
    norm = float(np.linalg.norm(new_vector))
    if norm > 0.0:
        directions.append(new_vector / norm)

    if degree < min_degree_for_dispersion_check:
        return True
    if len(directions) < min_degree_for_dispersion_check:
        # Too many spokes (existing or the candidate) have no resolvable
        # direction to judge fairly -- do not block on missing data.
        return True
    return hub_radial_dispersion(directions) > max_radial_dispersion


def reconnect_orphan_and_dangling_nodes(
    G: nx.MultiGraph,
    skeleton_data=None,
    reconnect_threshold: float = 3.0,
    include_degree1: bool = True,
    max_new_edges_per_node: int = 1,
    validate_reconnections: bool = True,
    debug: bool = False,
    *,
    direction_aware: bool = False,
    max_radial_dispersion: float = DEFAULT_MAX_RADIAL_DISPERSION,
    min_degree_for_dispersion_check: int = DEFAULT_MIN_DEGREE,
    tangent_length_um: float = DEFAULT_TANGENT_LENGTH_UM,
    max_bridge_turn_deg: Optional[float] = MAX_GAP_BRIDGE_TURN_DEG,
    mask_support=None,
) -> nx.MultiGraph:
    """Reconnect degree-0/degree-1 nodes to nearby nodes via skeleton path.

    *direction_aware*, when set, additionally refuses a reconnection that
    would push its target's own incident-edge dispersion at or below
    *max_radial_dispersion* -- see :func:`_reconnection_is_direction_safe`
    for why this step needs its own copy of that check rather than relying
    on ``cluster_collapse_method="direction_aware"``'s own guard on the
    earlier collapse step alone. Off by default, matching every other
    caller of this function that predates the option.

    A reconnection that does not continue a dangling end it starts or lands
    on (either end of degree 1) -- turning more than *max_bridge_turn_deg*
    from that vessel's end direction, or from an end too short to have one --
    is refused, as gap bridges are in
    ``build.build_graph_segment_skan_stitched_loops``; ``None`` turns that off.

    With *mask_support* (a ``preprocessing.MaskSupport``) each reconnection
    is also put to ``reconnect.MaskBridges`` and drawn along the path it
    accepts, as ``optimise_graph_topology_fixed`` does.
    """
    if reconnect_threshold <= 0:
        return G
    if max_new_edges_per_node < 1:
        return G
    if not isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("This function is designed for MultiGraphs")

    vs = tuple(G.graph.get("voxel_size", (1.0, 1.0, 1.0)))
    vs_arr = np.asarray(vs, dtype=float)

    valid_nodes = [n for n in G.nodes if "pos" in G.nodes[n]]
    if len(valid_nodes) < 2:
        return G

    target_nodes = []
    for node in valid_nodes:
        degree = G.degree[node]
        if degree >= 1:
            target_nodes.append(node)
    if len(target_nodes) < 1:
        return G

    source_nodes = []
    for node in valid_nodes:
        degree = G.degree[node]
        if degree == 0 or (include_degree1 and degree == 1):
            source_nodes.append(node)
    if not source_nodes:
        return G

    target_coords = np.array([G.nodes[n]["pos"] for n in target_nodes], dtype=float)
    tree = cKDTree(target_coords)

    candidate_pairs = []
    for src in source_nodes:
        src_pos = np.array(G.nodes[src]["pos"], dtype=float)
        idxs = tree.query_ball_point(src_pos, reconnect_threshold)
        for idx in idxs:
            tgt = target_nodes[idx]
            if tgt == src:
                continue
            if G.has_edge(src, tgt):
                continue
            dist = float(np.linalg.norm(src_pos - np.array(G.nodes[tgt]["pos"], dtype=float)))
            if dist <= reconnect_threshold:
                candidate_pairs.append((dist, src, tgt))

    if not candidate_pairs:
        return G

    heapq.heapify(candidate_pairs)
    bridges = None if mask_support is None else MaskBridges(G, mask_support, "orphan")
    added_edges_per_node = {n: 0 for n in source_nodes}
    reconnect_count = 0

    while candidate_pairs:
        dist, src, tgt = heapq.heappop(candidate_pairs)
        if not G.has_node(src) or not G.has_node(tgt):
            continue
        if "pos" not in G.nodes[src] or "pos" not in G.nodes[tgt]:
            continue
        if G.has_edge(src, tgt):
            continue
        if added_edges_per_node.get(src, 0) >= max_new_edges_per_node:
            continue
        if G.degree[src] > 1 and include_degree1:
            continue

        src_pos = np.array(G.nodes[src]["pos"], dtype=float)
        tgt_pos = np.array(G.nodes[tgt]["pos"], dtype=float)
        voxel_path = None

        # Cheap direction checks first: no point paying for skeleton-path
        # validation below on a reconnection these would reject anyway. End
        # directions are read now, not up front: an earlier reconnection in
        # this loop can have given either node a second edge.
        if max_bridge_turn_deg is not None and not (
            gap_bridge_continues_terminal(G, src, tgt_pos, max_turn_deg=max_bridge_turn_deg)
            and gap_bridge_continues_terminal(G, tgt, src_pos, max_turn_deg=max_bridge_turn_deg)
        ):
            continue
        if direction_aware and not _reconnection_is_direction_safe(
            G, tgt, tgt_pos, src_pos,
            min_degree_for_dispersion_check=min_degree_for_dispersion_check,
            max_radial_dispersion=max_radial_dispersion,
            tangent_length_um=tangent_length_um,
        ):
            continue

        if validate_reconnections and skeleton_data is not None:
            connection_valid, voxel_path = validate_skeleton_connection(
                skeleton_data,
                src_pos,
                tgt_pos,
                max_gap=reconnect_threshold,
                voxel_size=vs,
            )
            if not connection_valid:
                continue
            if voxel_path:
                phys_path = [(np.array(p, dtype=float) * vs_arr).tolist() for p in voxel_path]
            else:
                phys_path = [src_pos.tolist(), tgt_pos.tolist()]
        else:
            phys_path = [src_pos.tolist(), tgt_pos.tolist()]
        tags = {}
        if bridges is not None:
            found = bridges.bridge(
                src_pos, tgt_pos, preferred=phys_path if voxel_path else None
            )
            if found is None:
                continue
            phys_path, tags = found[0].tolist(), bridges.attributes(found[1])

        length = _physical_path_length(phys_path)
        if length <= 0:
            length = float(np.linalg.norm(tgt_pos - src_pos))
        add_edge_safe(
            G,
            src,
            tgt,
            length=length,
            voxels=phys_path,
            reconnected=True,
            orphan_reconnect=True,
            validated=bool(validate_reconnections and skeleton_data is not None),
            **tags,
        )
        added_edges_per_node[src] = added_edges_per_node.get(src, 0) + 1
        reconnect_count += 1

    if bridges is not None:
        bridges.log_summary()
    if debug and reconnect_count > 0:
        logger.info("Reconnected %d orphan/dangling node edge(s)", reconnect_count)
    return G
