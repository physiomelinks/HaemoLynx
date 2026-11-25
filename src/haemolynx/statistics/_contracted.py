"""The network as the statistics measure it: each zero-resistance bridge's two
ends one node (:func:`haemolynx.graph.thick_vessel_junctions.contract_bridges`),
and where each vessel whose end moved onto a thick vessel's centreline really
ends -- its own centreline end, where its bridge began.

A vessel's straight-line span is read between its own two ends: measured to
the centreline node its end became, a vessel's length (which leaves the
bridge out) could be shorter than its span, a tortuosity below 1.
"""
from __future__ import annotations

from typing import Any, Optional

import networkx as nx
import numpy as np

from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE, contract_bridges

#: Per vessel whose end a contraction moved, keyed by ``id`` of its edge data
#: (which the contracted graph shares): the positions of its own two ends.
OwnEnds = dict


def contracted_for_statistics(G: nx.Graph) -> "tuple[nx.Graph, dict, OwnEnds]":
    """``(graph, became, own ends)``: *G* with its bridges contracted, the
    node each of *G*'s nodes became, and :data:`OwnEnds`. *G* itself, with
    nothing moved, when it has no bridge."""
    edges = (
        [(u, v, d) for u, v, _k, d in G.edges(keys=True, data=True)]
        if G.is_multigraph()
        else list(G.edges(data=True))
    )
    if not any(d.get(IS_ZERO_RESISTANCE) for *_e, d in edges):
        return G, {}, {}
    contracted, became = contract_bridges(G)
    own_ends: OwnEnds = {}
    for u, v, data in edges:
        if data.get(IS_ZERO_RESISTANCE) or (became[u] == u and became[v] == v):
            continue
        ends = [G.nodes[n].get("pos") for n in (u, v)]
        if all(end is not None for end in ends):
            own_ends[id(data)] = tuple(np.asarray(end, dtype=float) for end in ends)
    return contracted, became, own_ends


def vessel_span_um(
    u: Any, v: Any, data: dict, node_positions: Optional[dict], own_ends: Optional[OwnEnds]
) -> Optional[float]:
    """The straight distance (um) between a vessel's own two ends: its
    :data:`OwnEnds` where a contraction moved one, else its nodes' positions;
    None without positions."""
    ends = (own_ends or {}).get(id(data))
    if ends is None:
        if not node_positions or u not in node_positions or v not in node_positions:
            return None
        ends = (node_positions[u], node_positions[v])
    return float(np.linalg.norm(np.asarray(ends[0], dtype=float) - np.asarray(ends[1], dtype=float)))
