"""The dead-end census and the flow coverage of a graph (PLAN_dead_ends.md, Phase 0).

Pure functions over one graph, so a test can pin them on a fixture and the
graph-artefacts report can print them for every side and step:

- :func:`census` -- every dead end by kind (``haemolynx.graph.dead_ends``), by
  its distance to an image face and whether the mask runs on to it, by depth,
  how long its chain is (in microns and in radii of the vessel it leaves), and
  what lies straight ahead of it within 5, 10 and 15 um: a vessel's side,
  another dead end's tip, or an image face;
- :func:`made_by_step` -- which topology step first left each final dead end;
- :func:`collapse_changes` -- what one step (cluster collapse) did to the dead
  ends it was handed: kept, joined to something, or removed;
- :func:`flow_coverage` -- for a solved graph, how much of the centreline
  carries flow and why the rest does not, and how much would with every dead
  end leaving the image opened as a boundary (topology alone: an upper bound).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Optional, Sequence

import networkx as nx
import numpy as np

#: The face bands the census counts tips in, microns (upper edges).
FACE_BANDS_UM = (2.0, 5.0, 10.0, 20.0)
#: How far ahead of a tip another centreline is looked for, microns.
AHEAD_BANDS_UM = (5.0, 10.0, 15.0)
DEPTH_SLAB_UM = 20.0
#: What can lie straight ahead of a tip, nearest first wins.
AHEAD_SIDE = "a vessel's side"
AHEAD_TIP = "another dead end's tip"
AHEAD_FACE = "an image face"
AHEAD_THINGS = (AHEAD_SIDE, AHEAD_TIP, AHEAD_FACE)
#: Another dead end's tip this close behind the nearest centreline point
#: ahead is what lies ahead: the nearest point is often on its own chain.
AHEAD_TIP_SLACK_UM = 2.0



def _band(distance: Optional[float]) -> str:
    if distance is None:
        return "unknown"
    lower = 0.0
    for upper in FACE_BANDS_UM:
        if distance <= upper:
            return f"{lower:g}-{upper:g}"
        lower = upper
    return f">{FACE_BANDS_UM[-1]:g}"


def census(
    G: nx.MultiGraph,
    *,
    support=None,
    extent_um: Optional[Sequence[float]] = None,
    stub_radius_multiple: float = 3.0,
) -> dict[str, Any]:
    """Every dead end of *G*, counted the ways the plan's section 2 counts them.

    *support* (a ``MaskSupport``) lets the mask kinds be told; *extent_um*
    is the image's extent (the graph's own when None).
    """
    from haemolynx.graph.dead_ends import DEAD_END_KINDS, classify_dead_ends, graph_image_extent_um

    ends = classify_dead_ends(G, extent_um=extent_um, support=support, stub_radius_multiple=stub_radius_multiple)
    # What lies ahead is read further than the definition looks (its 10 um),
    # with the kinds kept as the definition gives them.
    wide = classify_dead_ends(
        G, extent_um=extent_um, support=support, stub_radius_multiple=stub_radius_multiple,
        ahead_reach_um=max(AHEAD_BANDS_UM),
    )
    extent = np.asarray(extent_um, dtype=float) if extent_um is not None else graph_image_extent_um(G)
    by_kind = Counter(end.kind for end in ends.values())
    by_band: dict[str, Counter] = {}
    by_axis = Counter()
    for end in ends.values():
        by_band.setdefault(_band(end.face_distance_um), Counter())["reaches face" if end.reaches_face else "does not"] += 1
        if end.face_axis is not None and end.face_distance_um is not None and end.face_distance_um <= FACE_BANDS_UM[-1]:
            by_axis["zyx"[end.face_axis]] += 1
    chains = [e for e in ends.values() if e.interior and e.kind != "island_end"]
    lengths = np.asarray([e.chain_length_um for e in chains], dtype=float)
    in_radii = np.asarray(
        [e.chain_length_um / e.junction_radius_um for e in chains if e.junction_radius_um], dtype=float
    )
    depth = Counter(int(e.tip_um[0] // DEPTH_SLAB_UM) for e in ends.values() if e.interior)
    ahead = {
        f"within {limit:g} um": sum(
            1 for tip, e in ends.items() if e.interior and wide[tip].ahead_um is not None and wide[tip].ahead_um <= limit
        )
        for limit in AHEAD_BANDS_UM
    }
    first_ahead = {tip: _first_thing_ahead(wide[tip], wide, extent) for tip, e in ends.items()
                   if e.interior and e.kind != "island_end"}
    what_ahead = {
        f"within {limit:g} um": {
            thing: sum(1 for found in first_ahead.values() if found is not None and found[0] == thing and found[1] <= limit)
            for thing in AHEAD_THINGS
        }
        for limit in AHEAD_BANDS_UM
    }
    return {
        "dead_ends": len(ends),
        "by_kind": {kind: int(by_kind.get(kind, 0)) for kind in DEAD_END_KINDS},
        "by_face_band": {band: dict(counts) for band, counts in sorted(by_band.items())},
        "near_face_by_axis": dict(by_axis),
        "interior_chain_length_um": (
            {f"p{q}": float(np.percentile(lengths, q)) for q in (10, 25, 50, 75, 90)} if len(lengths) else {}
        ),
        "interior_chain_length_radii": (
            {f"p{q}": float(np.percentile(in_radii, q)) for q in (10, 25, 50, 75, 90)} if len(in_radii) else {}
        ),
        "interior_by_depth_slab": {
            f"{int(slab * DEPTH_SLAB_UM)}-{int((slab + 1) * DEPTH_SLAB_UM)}": int(n) for slab, n in sorted(depth.items())
        },
        "interior_with_a_vessel_ahead": ahead,
        "interior_first_thing_ahead": what_ahead,
        "ends": ends,
    }


def _first_thing_ahead(end, ends: Mapping[Any, Any], extent: Optional[np.ndarray]) -> Optional[tuple[str, float]]:
    """``(what, distance)`` lying first straight ahead of dead end *end*
    (within the definition's turn limit): the side of a vessel, another dead
    end's tip, or the image face its heading runs into; None with no heading
    or nothing within the census' reach."""
    from haemolynx.graph.dead_ends import AHEAD_MAX_TURN_DEG, _distance_to_box_along

    if end.heading is None:
        return None
    reach = max(AHEAD_BANDS_UM)
    tip = np.asarray(end.tip_um, dtype=float)
    heading = np.asarray(end.heading, dtype=float)
    found: list[tuple[float, str]] = []
    if end.ahead_um is not None:
        found.append((float(end.ahead_um), AHEAD_SIDE))
    cosine = float(np.cos(np.radians(AHEAD_MAX_TURN_DEG)))
    tips = []
    for other, other_end in ends.items():
        if other == end.tip or other_end is end:
            continue
        chord = np.asarray(other_end.tip_um, dtype=float) - tip
        gap = float(np.linalg.norm(chord))
        if 1e-6 < gap <= reach and float(chord @ heading) / gap >= cosine:
            tips.append(gap)
    if tips:
        nearest_tip = min(tips)
        if end.ahead_um is None or nearest_tip <= float(end.ahead_um) + AHEAD_TIP_SLACK_UM:
            found = [(nearest_tip, AHEAD_TIP)]
    if extent is not None:
        face = _distance_to_box_along(tip, heading, np.asarray(extent, dtype=float))
        if face is not None and face <= reach:
            found.append((float(face), AHEAD_FACE))
    if not found:
        return None
    distance, thing = min(found)
    return thing, distance


def collapse_changes(before: nx.MultiGraph, after: nx.MultiGraph) -> dict[str, int]:
    """What cluster collapse (inconsistency 8) did to each dead end of
    *before*, the graph it was handed, seen in *after*, the graph it left:
    ``kept`` (still a dead end), ``removed`` (merged into the junction its
    own chain hangs from, so the chain went as a self-loop) or ``joined``
    (merged with another vessel's node, or another node merged into it).
    Collapse keeps each cluster's representative's node ID and moves it, so
    a tip that is gone is read as merged into the node of *after* nearest
    where it was."""
    from scipy.spatial import cKDTree

    from haemolynx.graph.dead_ends import chain_to_junction

    counts = {"kept": 0, "joined": 0, "removed": 0}
    tips = [n for n in before if before.degree(n) == 1 and before.nodes[n].get("pos") is not None]
    placed = [n for n in after if after.nodes[n].get("pos") is not None]
    if not tips or not placed:
        counts["removed"] += len(tips)
        return counts
    tree = cKDTree(np.vstack([np.asarray(after.nodes[n]["pos"], dtype=float) for n in placed]))
    for tip in tips:
        if tip in after:
            counts["kept" if after.degree(tip) == 1 else "joined"] += 1
            continue
        nodes, _edges = chain_to_junction(before, tip)
        own = nodes[-1]
        _distance, index = tree.query(np.asarray(before.nodes[tip]["pos"], dtype=float))
        counts["removed" if placed[int(index)] == own else "joined"] += 1
    return counts


def made_by_step(
    final: nx.MultiGraph,
    steps: Iterable[tuple[str, nx.MultiGraph]],
    *,
    tolerance_um: float = 1.0,
) -> dict[str, int]:
    """For each dead end of *final*, the first of *steps* (``(label, graph)``,
    in order) in which a dead end already sat within *tolerance_um* of it:
    how many each step left. Matched by position, since node IDs change from
    step to step; one no step had is counted under ``"after the steps"``."""
    from scipy.spatial import cKDTree

    tips = [n for n in final if final.degree(n) == 1 and final.nodes[n].get("pos") is not None]
    if not tips:
        return {}
    remaining = {n: np.asarray(final.nodes[n]["pos"], dtype=float) for n in tips}
    counts: dict[str, int] = {}
    for label, G in steps:
        step_tips = [np.asarray(G.nodes[n]["pos"], dtype=float) for n in G if G.degree(n) == 1 and G.nodes[n].get("pos") is not None]
        if not step_tips or not remaining:
            counts.setdefault(label, 0)
            continue
        tree = cKDTree(np.vstack(step_tips))
        found = [n for n, p in remaining.items() if tree.query(p, distance_upper_bound=tolerance_um)[0] <= tolerance_um]
        counts[label] = len(found)
        for n in found:
            remaining.pop(n)
    counts["after the steps"] = len(remaining)
    return counts


def _length(G: nx.MultiGraph, edges: Iterable[tuple]) -> float:
    return float(sum(float(G.edges[e].get("length") or 0.0) for e in edges))


def _through(G: nx.MultiGraph, boundary: set) -> set:
    """Edges on a path between two different nodes of *boundary*: in a
    biconnected block with a virtual node joined to every boundary node."""
    V = object()
    H = nx.Graph()
    H.add_edges_from((u, v) for u, v in G.edges() if u != v)
    H.add_edges_from((V, n) for n in boundary if n in G)
    keep: set = set()
    for block in nx.biconnected_component_edges(H):
        pairs = {frozenset(e) for e in block}
        if len(pairs) > 1 and any(V in p for p in pairs):
            keep |= pairs
    return {(u, v, k) for u, v, k in G.edges(keys=True) if u != v and frozenset((u, v)) in keep}


def flow_coverage(
    G: nx.MultiGraph,
    *,
    inlets: Sequence[Any] = (),
    outlets: Sequence[Any] = (),
    extent_um: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """How much of a solved *G*'s centreline carries flow, and why the rest does not.

    Solved means ``FLOW_SOLVED`` is True. The rest is split by what it hangs
    on: a piece that touches the flowing network and reaches a dead end
    leaving the image is "behind a closed face end"; one that touches it and
    reaches only interior dead ends (or loops back on itself) is "behind
    interior dead ends"; one that does not touch it at all is "no boundary".
    The what-if opens every dead end leaving the image (the kinds on the
    nodes, or distance alone) as a boundary beside *inlets* and *outlets*.
    """
    from haemolynx.graph import FLOW_SOLVED
    from haemolynx.graph.dead_ends import graph_image_extent_um, node_leaves_image

    extent = np.asarray(extent_um, dtype=float) if extent_um is not None else graph_image_extent_um(G)
    edges = list(G.edges(keys=True))
    total = _length(G, edges)
    solved = [e for e in edges if G.edges[e].get(FLOW_SOLVED) is True]
    flowing_nodes = {n for e in solved for n in e[:2]}
    rest = G.edge_subgraph([e for e in edges if G.edges[e].get(FLOW_SOLVED) is not True])
    reasons = Counter()
    for piece in nx.connected_components(rest):
        sub = rest.subgraph(piece)
        length = _length(sub, sub.edges(keys=True))
        if not piece & flowing_nodes:
            reasons["no boundary"] += length
        elif any(G.degree(n) == 1 and node_leaves_image(G, n, extent) for n in piece):
            reasons["behind a closed face end"] += length
        else:
            reasons["behind interior dead ends"] += length
    open_ends = [n for n in G if G.degree(n) == 1 and node_leaves_image(G, n, extent)]
    boundary = set(inlets) | set(outlets)
    designated = _through(G, boundary) if boundary else set()
    opened = _through(G, boundary | set(open_ends))
    face_distance = {}
    if extent is not None:
        for n in boundary:
            if n in G and G.nodes[n].get("pos") is not None:
                p = np.asarray(G.nodes[n]["pos"], dtype=float)
                face_distance[n] = float(np.min(np.minimum(p, extent - p)))
    return {
        "centreline_um": total,
        "solved_um": _length(G, solved),
        "solved_share": _length(G, solved) / total if total else 0.0,
        "unsolved_um_by_reason": {k: float(v) for k, v in reasons.items()},
        "boundary_face_distance_um": face_distance,
        "open_ends": len(open_ends),
        "through_designated_um": _length(G, designated),
        "through_with_open_ends_um": _length(G, opened),
        "through_with_open_ends_share": _length(G, opened) / total if total else 0.0,
    }


def census_markdown(rows: Mapping[str, Mapping[str, Any]]) -> str:
    """One table per section of :func:`census`, one column per side."""
    names = list(rows)
    lines: list[str] = []

    def table(title: str, keys: list[str], get) -> None:
        if not keys:
            return
        lines.extend([f"**{title}**", "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)])
        for key in keys:
            lines.append(f"| {key} | " + " | ".join(str(get(rows[n], key)) for n in names) + " |")
        lines.append("")

    table("Dead ends", ["all"], lambda r, k: r["dead_ends"])
    table("By kind", list(next(iter(rows.values()))["by_kind"]), lambda r, k: r["by_kind"].get(k, 0))
    bands = sorted({b for r in rows.values() for b in r["by_face_band"]})
    table("By distance to a face (mask reaches the face / does not)", bands,
          lambda r, k: "{} / {}".format(r["by_face_band"].get(k, {}).get("reaches face", 0),
                                        r["by_face_band"].get(k, {}).get("does not", 0)))
    slabs = sorted({s for r in rows.values() for s in r["interior_by_depth_slab"]}, key=lambda s: int(s.split("-")[0]))
    table("Interior dead ends by depth (z, um)", slabs, lambda r, k: r["interior_by_depth_slab"].get(k, 0))
    table("Interior chain length (um)", ["p10", "p25", "p50", "p75", "p90"],
          lambda r, k: f"{r['interior_chain_length_um'].get(k, float('nan')):.1f}")
    table("Interior chain length, in radii of the vessel it leaves", ["p10", "p25", "p50", "p75", "p90"],
          lambda r, k: f"{r['interior_chain_length_radii'].get(k, float('nan')):.1f}")
    ahead = list(next(iter(rows.values()))["interior_with_a_vessel_ahead"])
    table("Interior dead ends with a vessel straight ahead", ahead, lambda r, k: r["interior_with_a_vessel_ahead"][k])
    table("What lies first straight ahead of an interior dead end (vessel side / dead-end tip / face)", ahead,
          lambda r, k: " / ".join(str(r["interior_first_thing_ahead"][k][thing]) for thing in AHEAD_THINGS))
    return "\n".join(lines)
