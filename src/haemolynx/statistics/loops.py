"""Loop (anastomosis) structure and its scale.

The cyclomatic number already counts independent loops; this measures their
*size*: for every vessel, the shortest loop that contains it -- the vessel
plus the shortest detour between its two ends that avoids it. A short loop
means local, capillary-scale redundancy; a long one means the only
alternative route is a distant collateral; no loop at all (a bridge) means
no alternative. Summarised as a loop-length distribution, overall and by
vessel type, which shows at what scale the network's redundancy sits.

(Hierarchical loop *nesting* in the Katifori/Modes sense needs a planar
embedding, which a 3D network does not have; the per-vessel loop scale is
the 3D-applicable answer to the same question.)
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence

import networkx as nx
import numpy as np

from ._route_graph import (
    _clear_edge_attributes,
    _effective_weighting,
    _iter_edges,
    _route_graph,
    _set_edge,
)
from ._sampling import REPRODUCIBILITY_SEED
from .bifurcation import branch_order_category

__all__ = ["LOOP_EDGE_ATTRIBUTES", "compute_loop_hierarchy"]

LOOP_EDGE_ATTRIBUTES: tuple[str, ...] = ("loop_length_um",)
#: ``fast`` mode measures at most this many vessel pairs (seeded sample); the
#: rest get NaN. Each is one bidirectional shortest-path search.
DEFAULT_FAST_MAX_PAIRS = 20000


def compute_loop_hierarchy(
    G: nx.Graph,
    *,
    statistics_mode: str = "fast",
    image_dimensions: Optional[Sequence[int]] = None,
    voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
    max_pairs_fast: int = DEFAULT_FAST_MAX_PAIRS,
) -> Dict[str, Any]:
    """Shortest loop through each vessel, and the loop-length distribution.

    Length-weighted (falls back to vessel count without lengths). Writes
    ``loop_length_um`` onto every edge -- NaN on a bridge (no loop), and on
    vessels left out of ``fast`` mode's sample. A parallel vessel between
    the same two junctions forms a loop with its partner.
    """
    _clear_edge_attributes(G, LOOP_EDGE_ATTRIBUTES)
    effective, label = _effective_weighting(G, "length")
    routes = _route_graph(G, effective)
    H = routes.graph
    unit = "um" if effective == "length" else "vessels"
    bridges = {frozenset(e) for e in nx.bridges(H)}

    pairs = list(routes.members)
    sampled = pairs
    method = "exact"
    if statistics_mode == "fast" and len(pairs) > max_pairs_fast:
        sampled = random.Random(REPRODUCIBILITY_SEED).sample(pairs, max_pairs_fast)
        method = f"sampled {max_pairs_fast} of {len(pairs)} junction pairs"
    sampled_set = set(sampled)

    loop_length: dict = {}
    for pair in pairs:
        members = routes.members[pair]
        if pair not in sampled_set:
            for u, v, key, _w in members:
                loop_length[(u, v, key)] = math.nan
            continue
        u, v = tuple(pair)
        detour = math.inf
        if pair not in bridges:
            attrs = dict(H[u][v])
            H.remove_edge(u, v)
            try:
                detour = nx.bidirectional_dijkstra(H, u, v, weight="route_weight")[0]
            except nx.NetworkXNoPath:
                detour = math.inf
            finally:
                H.add_edge(u, v, **attrs)
        weights = sorted(w for *_uvk, w in members)
        for mu, mv, mkey, w in members:
            others = list(weights)
            others.remove(w)
            best_other = others[0] if others else math.inf
            around = min(best_other, detour)
            loop_length[(mu, mv, mkey)] = w + around if math.isfinite(around) else math.nan

    categories: dict = defaultdict(list)
    for u, v, key, data in _iter_edges(G):
        if u == v:
            value = float(data.get("length", 1.0)) if effective == "length" else 1.0
        else:
            value = loop_length.get((u, v, key), math.nan)
        _set_edge(G, u, v, key, "loop_length_um", value)
        categories[branch_order_category(data.get("branch_order"))].append(value)

    values = np.asarray([x for vals in categories.values() for x in vals], dtype=float)
    measured = values[np.isfinite(values)]
    total = values.size
    evaluated = (
        total
        if method == "exact"
        else sum(len(routes.members[p]) for p in sampled) + nx.number_of_selfloops(G)
    )
    no_loop = int(np.sum(~np.isfinite(values))) - (total - evaluated)
    result: Dict[str, Any] = {
        "Loop Length Weighting": label,
        "Loop Method": method,
        "Vessels On No Loop": no_loop,
        "Vessels On No Loop Fraction": no_loop / evaluated if evaluated else 0.0,
    }
    if measured.size:
        for name, q in (("P10", 10), ("P25", 25), ("Median", 50), ("P75", 75), ("P90", 90)):
            result[f"Loop Length {name} ({unit})"] = float(np.percentile(measured, q))
    result["Median Loop Length By Vessel Type"] = {
        category: (
            float(np.median(np.asarray(vals)[np.isfinite(vals)]))
            if np.isfinite(vals).any()
            else "N/A (no vessel of this type is on a loop)"
        )
        for category, vals in categories.items()
    }
    if image_dimensions is not None:
        volume_mm3 = float(np.prod([d * s for d, s in zip(image_dimensions, voxel_size)])) / 1e9
        independent = (
            G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(H)
        )
        result["Independent Loops Per mm^3"] = (
            independent / volume_mm3 if volume_mm3 > 0 else "N/A (no imaged volume)"
        )
    return result
