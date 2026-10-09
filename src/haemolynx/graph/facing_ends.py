"""Join two dead ends that point at each other across a gap in the segmentation.

A vessel the segmentation broke -- a dim stretch, a few voxels classed as
background -- comes out of graph building as two dead ends, one either side of
the gap, each heading for the other. Every other reconnect refuses to cross more
than a dropout of background (``bridge_max_background_gap_um``), because two
vessels with tissue between them must never be joined. Two ends that point at
each other, across one stretch of background and nothing else, are the one case
where crossing more is the vessel and not the tissue.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from haemolynx.preprocessing.bridge_mask_support import MaskSupport, _pieces, _sample_step

from ._helpers import EdgeSampleIndex, densify_polyline, duplicates_existing_vessel, edge_sample_points
from .build import GAP_BRIDGE_TANGENT_LENGTH_UM, MIN_GAP_BRIDGE_END_LENGTH_UM
from .cartwheel_guard import _point_along_polyline

logger = logging.getLogger(__name__)

#: Longest gap two facing dead ends are joined across, in microns.
DEFAULT_FACING_MAX_GAP_UM = 10.0

#: How far a join may turn from either end's own heading, in degrees. Of the
#: E14.5 stack's dead-end pairs within 10 um, half were siblings off one
#: junction and a fifth lay side by side; the 15 pointing at each other all
#: fell within this.
FACING_MAX_TURN_DEG = 60.0


def _heading(G: nx.MultiGraph, tip: Any) -> tuple[Optional[np.ndarray], float]:
    """``(direction, length)``: the unit vector out of a dead end, read over
    the last ``GAP_BRIDGE_TANGENT_LENGTH_UM`` of its vessel, and the vessel's
    length back to the first junction -- through any degree-2 nodes, since
    before the last merge a vessel is often a chain of short edges."""
    pieces, previous, current = [], None, tip
    while True:
        onward = [n for n in G.neighbors(current) if n != previous]
        if not onward:
            break
        following = onward[0]
        data = min(G[current][following].values(), key=lambda d: d.get("length", 0.0) or 0.0)
        positions = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (current, following)}
        pieces.append(edge_sample_points(current, following, data, positions))
        previous, current = current, following
        if G.degree(current) != 2 or current == tip:
            break
    if not pieces:
        return None, 0.0
    path = np.vstack(pieces)
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    inward = _point_along_polyline(path, GAP_BRIDGE_TANGENT_LENGTH_UM) - path[0]
    norm = float(np.linalg.norm(inward))
    return (None if norm <= 0.0 else -inward / norm), length


def _background(path: np.ndarray, support: MaskSupport) -> tuple[int, float]:
    """How many separate stretches of background a path crosses, and their
    length in microns all told."""
    mids, weights = _pieces(path, _sample_step(support.voxel_size_zyx))
    outside = ~np.asarray(support.inside(mids), dtype=bool)
    stretches = int(np.count_nonzero(outside[1:] & ~outside[:-1]) + int(outside[0])) if len(outside) else 0
    return stretches, float(weights[outside].sum())


#: Why :func:`facing_dead_end_pairs` says a pair is not joined, in the order
#: :func:`join_facing_dead_ends` asks.
REFUSED_SHORT_END = f"an end shorter than {MIN_GAP_BRIDGE_END_LENGTH_UM:g} um"
REFUSED_GAP_LONGER_THAN_ENDS = "gap longer than the two ends together"
REFUSED_TURN = "turns too far from an end's heading"
REFUSED_GAP_TOO_WIDE = "gap wider than the limit"
REFUSED_STRETCHES = "crosses more than one stretch of background"
REFUSED_BESIDE_VESSEL = "runs beside a vessel in the same lumen"


@dataclass(frozen=True)
class FacingPair:
    """Two dead ends within reach of each other, and what the join made of them."""

    a: Any
    b: Any
    gap_um: float
    #: The larger of the two ends' turns onto the straight join, in degrees.
    turn_deg: float
    #: Why the pair is not joined (one of the ``REFUSED_*``); None when
    #: :func:`join_facing_dead_ends` would join it.
    refused: Optional[str]


def _tips_and_headings(G: nx.MultiGraph):
    tips = [n for n in G.nodes if G.degree(n) == 1 and "pos" in G.nodes[n]]
    positions = np.array([G.nodes[n]["pos"] for n in tips], dtype=float).reshape(-1, 3)
    return tips, positions, {n: _heading(G, n) for n in tips}


def _candidate_pairs(tips, positions, headings, *, search_gap_um, max_gap_um, max_turn_deg):
    """``(gap, i, j, turn_deg, refused)`` for each pair of tips within
    *search_gap_um* that both have a heading, judged by the geometric tests
    alone (refused None: they pass)."""
    found = []
    if len(tips) < 2:
        return found
    cosine = float(np.cos(np.radians(max_turn_deg)))
    for i, j in cKDTree(positions).query_pairs(float(search_gap_um), output_type="ndarray"):
        (out_i, length_i), (out_j, length_j) = headings[tips[i]], headings[tips[j]]
        if out_i is None or out_j is None:
            continue
        chord = positions[j] - positions[i]
        gap = float(np.linalg.norm(chord))
        if gap <= 0.0:
            continue
        cos_i, cos_j = float(chord @ out_i / gap), float(-chord @ out_j / gap)
        turn = float(np.degrees(np.arccos(np.clip(min(cos_i, cos_j), -1.0, 1.0))))
        if min(length_i, length_j) < MIN_GAP_BRIDGE_END_LENGTH_UM:
            refused = REFUSED_SHORT_END
        elif gap > length_i + length_j:
            refused = REFUSED_GAP_LONGER_THAN_ENDS
        elif cos_i < cosine or cos_j < cosine:
            refused = REFUSED_TURN
        elif search_gap_um > max_gap_um and gap > max_gap_um:
            refused = REFUSED_GAP_TOO_WIDE
        else:
            refused = None
        found.append((gap, int(i), int(j), turn, refused))
    return found


def facing_dead_end_pairs(
    G: nx.MultiGraph,
    support: MaskSupport,
    *,
    max_gap_um: float = DEFAULT_FACING_MAX_GAP_UM,
    max_turn_deg: float = FACING_MAX_TURN_DEG,
    search_gap_um: Optional[float] = None,
) -> list[FacingPair]:
    """Every pair of dead ends within *search_gap_um* (default *max_gap_um*)
    of each other, closest first, with why :func:`join_facing_dead_ends`
    would not join it -- read-only. On a graph the join has already run on,
    every pair listed is one it left open."""
    tips, positions, headings = _tips_and_headings(G)
    search = float(max_gap_um if search_gap_um is None else search_gap_um)
    if search <= 0:
        return []
    index = None
    pairs = []
    for gap, i, j, turn, refused in sorted(_candidate_pairs(
        tips, positions, headings, search_gap_um=search, max_gap_um=max_gap_um, max_turn_deg=max_turn_deg,
    )):
        if refused is None:
            path = densify_polyline(np.vstack([positions[i], positions[j]]), max_step_um=1.0)
            if _background(path, support)[0] > 1:
                refused = REFUSED_STRETCHES
            else:
                if index is None:
                    index = EdgeSampleIndex(G)
                if duplicates_existing_vessel(path, G, support.inside, support.radius, index=index):
                    refused = REFUSED_BESIDE_VESSEL
        pairs.append(FacingPair(tips[i], tips[j], gap, turn, refused))
    return pairs


def join_facing_dead_ends(
    G: nx.MultiGraph,
    support: MaskSupport,
    *,
    max_gap_um: float = DEFAULT_FACING_MAX_GAP_UM,
    max_turn_deg: float = FACING_MAX_TURN_DEG,
) -> nx.MultiGraph:
    """Join, in place, each pair of dead ends pointing at each other across a
    gap of at most *max_gap_um*.

    A pair is joined by the straight line between them when that line leaves
    each end within *max_turn_deg* of the way its vessel was heading, each
    vessel is at least ``MIN_GAP_BRIDGE_END_LENGTH_UM`` long, and the gap is
    no longer than the two vessels together. The line may cross background --
    that is the gap -- but in one stretch only, so it never jumps a third
    vessel, and it must not run beside a vessel already in the same lumen.
    Closest pairs go first, and each end joins once. Joins carry
    ``reconnected=True``, ``bridge_kind="facing_ends"`` and
    ``bridge_background_um``. :func:`facing_dead_end_pairs` lists the pairs
    and why each is or is not joined.
    """
    if max_gap_um <= 0:
        return G
    tips, positions, headings = _tips_and_headings(G)
    if len(tips) < 2:
        return G
    pairs = [
        (gap, i, j)
        for gap, i, j, _turn, refused in _candidate_pairs(
            tips, positions, headings,
            search_gap_um=max_gap_um, max_gap_um=max_gap_um, max_turn_deg=max_turn_deg,
        )
        if refused is None
    ]

    index = None
    joined = refused_stretches = refused_duplicate = 0
    for gap, i, j in sorted(pairs):
        a, b = tips[i], tips[j]
        if G.degree(a) != 1 or G.degree(b) != 1 or G.has_edge(a, b):
            continue
        path = densify_polyline(np.vstack([positions[i], positions[j]]), max_step_um=1.0)
        stretches, background_um = _background(path, support)
        if stretches > 1:
            refused_stretches += 1
            continue
        if index is None:
            index = EdgeSampleIndex(G)
        if duplicates_existing_vessel(path, G, support.inside, support.radius, index=index):
            refused_duplicate += 1
            continue
        G.add_edge(
            a, b,
            length=gap, voxels=path.tolist(), reconnected=True,
            bridge_kind="facing_ends", bridge_background_um=background_um,
        )
        index.add(path)
        joined += 1
    if joined or refused_stretches or refused_duplicate:
        logger.info(
            "Joined %d pair(s) of dead ends pointing at each other across a gap of up to "
            "%.3g um; refused %d crossing more than one stretch of background and %d "
            "beside a vessel in the same lumen",
            joined, float(max_gap_um), refused_stretches, refused_duplicate,
        )
    return G


__all__ = [
    "DEFAULT_FACING_MAX_GAP_UM",
    "FACING_MAX_TURN_DEG",
    "FacingPair",
    "REFUSED_BESIDE_VESSEL",
    "REFUSED_GAP_LONGER_THAN_ENDS",
    "REFUSED_GAP_TOO_WIDE",
    "REFUSED_SHORT_END",
    "REFUSED_STRETCHES",
    "REFUSED_TURN",
    "facing_dead_end_pairs",
    "join_facing_dead_ends",
]
