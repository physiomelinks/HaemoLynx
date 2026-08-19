"""Transit time along flow-directed paths, for H2 §2.4.

§2.4 asks for the transit time of blood from the arterial inlet to the distal ends of the
capillaries lying within the TH-positive glomus boundaries, and for the PO2 depletion along
those paths.

**Reported as a ratio, not an absolute.** Two reasons, and they compound. S15 puts an absolute
flow quantity under a ±45% floor from calibre alone, and S22 changed apparent viscosity by a
factor of three or four without moving any within-specimen ratio. Separately, the pressure,
viscosity and length units in this pipeline are not reconciled to one system, so the magnitude
returned here is in arbitrary units and only ratios of it mean anything. Both problems have the
same answer: compare transit time to one set of terminals against transit time to another,
computed identically, and the shared error divides out.
"""
from __future__ import annotations

import heapq
from typing import Any, Iterable

import numpy as np


def _edge_diameter(data) -> float | None:
    d = data.get("assigned_diameter_um", data.get("fwhm_diameter_um"))
    return float(d) if d is not None and float(d) > 0 else None


def edge_transit_times(G) -> dict:
    """Time for blood to traverse each edge: lumen volume over volumetric flow.

    ``tau = pi * (d/2)^2 * L / Q``. Quadratic in diameter, where resistance is quartic, so
    this is a different sensitivity to calibre than the flow solve has and inherits its own
    share of the ±45% floor.

    An edge carrying no flow gets ``inf`` rather than a large number: blood that does not move
    does not arrive, and a finite stand-in would propagate as a merely slow path.
    """
    missing = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
               if _edge_diameter(d) is None]
    if missing:
        raise ValueError(
            f"{len(missing)} edges have no usable diameter, for example {missing[:3]}. "
            f"Transit time is lumen volume over flow, so a fabricated calibre would produce a "
            f"fabricated transit time quadratically."
        )

    out = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        d = _edge_diameter(data)
        length = float(data.get("length", 0.0))
        flow = abs(float(data.get("flow_abs", 0.0)))
        volume = np.pi * (d / 2.0) ** 2 * length
        out[(u, v, key)] = (volume / flow) if flow > 0 else float("inf")
    return out


def transit_time_from_inlets(G, inlets: Iterable[Any]) -> dict:
    """Shortest accumulated transit time from any inlet to every node.

    Traversal follows the solved flow direction, not adjacency: an edge carrying blood away
    from a node cannot deliver blood to it, and ignoring that would report a transit time along
    a route no blood takes.

    Dijkstra rather than a topological pass, because the flow directions come from a numerical
    solve and can contain a small cycle. A topological sort would raise on those; this returns
    the same answer where the directions are acyclic and does not hang where they are not.

    Every node appears in the result. Unreachable ones carry ``inf`` rather than being absent,
    so a caller cannot read a missing key as zero.
    """
    tau = edge_transit_times(G)
    inlets = list(inlets)

    # Direct each edge by its solved flow sign. flow_signed > 0 means u -> v as stored.
    outgoing: dict = {n: [] for n in G.nodes()}
    for u, v, key, data in G.edges(keys=True, data=True):
        signed = float(data.get("flow_signed", data.get("flow_abs", 0.0)))
        cost = tau[(u, v, key)]
        if signed >= 0:
            outgoing[u].append((v, cost))
        else:
            outgoing[v].append((u, cost))

    best = {n: float("inf") for n in G.nodes()}
    queue = []
    for n in inlets:
        if n in best:
            best[n] = 0.0
            heapq.heappush(queue, (0.0, id(n), n))

    while queue:
        cost, _tie, node = heapq.heappop(queue)
        if cost > best[node]:
            continue
        for nxt, step in outgoing.get(node, ()):
            if not np.isfinite(step):
                continue
            candidate = cost + step
            if candidate < best[nxt]:
                best[nxt] = candidate
                heapq.heappush(queue, (candidate, id(nxt), nxt))
    return best
