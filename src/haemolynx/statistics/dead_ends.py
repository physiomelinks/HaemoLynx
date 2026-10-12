"""The statistics CSV's "Dead Ends" section (PLAN_dead_ends.md, Phase 7).

What each dead end is (``graph.dead_ends``): how many of each kind, how dense
the interior ones are, how long their chains run, how much of the network the
image faces cut, and -- once the network is solved -- how much of it carries
no flow, and why (``graph.FLOW_STATE``).

Dead ends that leave the image are open in the tissue: they are counted, but
never in a dead-end density. The density counts only the interior ones that
could be the tissue's own blind ends (``blind``, ``unknown``, ``broken``,
``stops_short``), per mm of vessel and per mm^3 of tissue when the tissue was
measured; a vessel the image cut has a censored length, and the share of the
network on such chains is reported so the length statistics can be read with
it (decision D5).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import networkx as nx
import numpy as np

#: The kinds an interior dead-end density counts: everything that could be
#: the tissue's own end -- not the image's, not an artefact, not an island.
DENSITY_KINDS = ("blind", "unknown", "broken", "stops_short")

_KIND_ROWS = {
    "leaves_image": "Dead Ends Leaving the Image",
    "stops_short": "Dead Ends the Graph Stopped Short of (mask runs on)",
    "broken": "Dead Ends of Vessels Broken by the Segmentation",
    "artefact": "Dead Ends That Are Artefacts",
    "blind": "Blind Ends (marked by hand)",
    "unknown": "Interior Dead Ends, Blind or Broken (unknown)",
    "island_end": "Ends of Pieces Without a Junction",
}

_FLOW_STATE_ROWS = {
    "solved": "Vessel Length Carrying Flow (microns)",
    "blind_end": "Vessel Length Behind Interior Dead Ends, No Flow (microns)",
    "closed_end": "Vessel Length Behind Closed Image-Face Ends, No Flow (microns)",
    "no_boundary": "Vessel Length Without Inlet and Outlet, No Flow (microns)",
}


def _length(data: dict) -> float:
    length = data.get("length")
    return float(length) if length is not None and np.isfinite(float(length)) else 0.0


def compute_dead_end_statistics(
    G: nx.Graph,
    *,
    tissue_volume_um3: Optional[float] = None,
) -> Dict[str, Any]:
    """``{"Dead Ends": {row: value}}`` for *G*: see the module docstring.

    Kinds are read off the nodes (``dead_end_kind``, written when the network
    was built); a dead end without one -- a graph from before, or one a later
    step made -- is judged then by the geometric part of the definition."""
    from haemolynx.graph.dead_ends import (
        DEAD_END_KIND,
        DEAD_END_KINDS,
        LEAVES_IMAGE,
        chain_length_um,
        chain_to_junction,
        classify_dead_ends,
    )
    from haemolynx.graph.prune import FLOW_STATE

    multi = G if G.is_multigraph() else nx.MultiGraph(G)
    tips = [n for n in multi if multi.degree(n) == 1]
    unmarked = [n for n in tips if multi.nodes[n].get(DEAD_END_KIND) is None]
    judged = classify_dead_ends(multi, nodes=unmarked) if unmarked else {}
    kinds = {n: (multi.nodes[n].get(DEAD_END_KIND) or getattr(judged.get(n), "kind", "unknown")) for n in tips}

    rows: Dict[str, Any] = {"Dead Ends": len(tips)}
    for kind in DEAD_END_KINDS:
        rows[_KIND_ROWS[kind]] = sum(1 for k in kinds.values() if k == kind)

    total_um = sum(_length(d) for *_e, d in multi.edges(data=True))
    interior = [n for n, k in kinds.items() if k in DENSITY_KINDS]
    if total_um > 0:
        rows["Interior Dead Ends per mm of Vessel (1/mm)"] = len(interior) / (total_um / 1000.0)
    if tissue_volume_um3:
        rows["Interior Dead Ends per mm^3 of Tissue (1/mm^3)"] = len(interior) / (float(tissue_volume_um3) / 1e9)

    chains = {n: chain_to_junction(multi, n) for n in tips}
    lengths = np.asarray([
        chain_length_um(multi, chains[n][1]) for n in interior if multi.degree(chains[n][0][-1]) > 1
    ], dtype=float)
    if len(lengths):
        rows["Interior Dead-End Chain Length, median (microns)"] = float(np.median(lengths))
        rows["Interior Dead-End Chain Length, 10th percentile (microns)"] = float(np.percentile(lengths, 10))
        rows["Interior Dead-End Chain Length, 90th percentile (microns)"] = float(np.percentile(lengths, 90))

    cut = {
        tuple(sorted((a, b), key=str)) + (k,)
        for n, kind in kinds.items() if kind == LEAVES_IMAGE for a, b, k in chains[n][1]
    }
    cut_um = sum(_length(multi.edges[u, v, k]) for u, v, k in cut if multi.has_edge(u, v, k))
    rows["Vessel Length on Chains Cut by the Image Faces (microns)"] = cut_um
    if total_um > 0:
        rows["Share of Vessel Length Cut by the Image Faces (fraction)"] = cut_um / total_um

    islands = [c for c in nx.connected_components(multi) if all(multi.degree(n) <= 2 for n in c) and len(c) > 1]
    rows["Pieces Without a Junction"] = len(islands)
    rows["Length of Pieces Without a Junction (microns)"] = sum(
        _length(d) for c in islands for *_e, d in multi.subgraph(c).edges(data=True)
    )

    states = {}
    for *_e, d in multi.edges(data=True):
        state = d.get(FLOW_STATE)
        if state is not None:
            states[state] = states.get(state, 0.0) + _length(d)
    for state, label in _FLOW_STATE_ROWS.items():
        if states:
            rows[label] = states.get(state, 0.0)
    return {"Dead Ends": rows}


__all__ = ["DENSITY_KINDS", "compute_dead_end_statistics"]
