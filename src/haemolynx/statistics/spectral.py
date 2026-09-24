"""Algebraic connectivity (Fiedler value) and the Fiedler split.

The second-smallest eigenvalue of the conductance-weighted graph Laplacian,
lambda_2, is zero for a disconnected network and grows the more redundantly
connected it is -- one number for how hard the network is to cut in two,
and how fast anything diffusing through it mixes. Its eigenvector (the
Fiedler vector) gives the network's most natural two-way split: the
cheapest cut relative to the size of the two halves.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import networkx as nx
import numpy as np

from ._route_graph import (
    _clear_edge_attributes,
    _edge_conductance,
    _effective_weighting,
    _iter_edges,
    _set_edge,
    _unavailable,
)
from ._sampling import REPRODUCIBILITY_SEED

__all__ = ["SPECTRAL_EDGE_ATTRIBUTES", "compute_algebraic_connectivity"]

SPECTRAL_EDGE_ATTRIBUTES: tuple[str, ...] = ("fiedler_value", "fiedler_side")


def compute_algebraic_connectivity(G: nx.Graph, *, weighting: str = "length") -> Dict[str, Any]:
    """lambda_2 of the largest connected component's weighted Laplacian,
    also normalised by its mean weighted degree (comparable across networks
    of different size and units), and the Fiedler split.

    Edge weights are conductances under *weighting* (one per vessel,
    1/length, or real conductance), rescaled to at most 1 for the
    eigensolver and scaled back. Writes ``fiedler_value`` (mean of the two
    ends' Fiedler-vector entries) and ``fiedler_side`` (``A``/``B``, or
    ``cut`` for a vessel crossing the split; ``outside`` off the largest
    component) onto every edge.
    """
    _clear_edge_attributes(G, SPECTRAL_EDGE_ATTRIBUTES)
    effective, label = _effective_weighting(G, weighting)
    S = nx.Graph()
    for u, v, _key, data in _iter_edges(G):
        if u == v:
            continue
        g = _edge_conductance(data, effective)
        if S.has_edge(u, v):
            S[u][v]["weight"] += g
        else:
            S.add_edge(u, v, weight=g)
    if S.number_of_nodes() < 2:
        return _unavailable("Algebraic Connectivity", "fewer than two connected nodes", label)
    largest = S.subgraph(max(nx.connected_components(S), key=len)).copy()
    n = largest.number_of_nodes()
    if n < 2:
        return _unavailable("Algebraic Connectivity", "fewer than two connected nodes", label)
    scale = max(d["weight"] for _u, _v, d in largest.edges(data=True))
    for _u, _v, d in largest.edges(data=True):
        d["weight"] /= scale

    method = "tracemin_lu" if n > 2 else "lobpcg"
    if n == 2:
        u, v = list(largest.nodes)
        lambda2 = 2.0 * largest[u][v]["weight"]
        fiedler = {u: -1 / math.sqrt(2), v: 1 / math.sqrt(2)}
    else:
        lambda2 = float(
            nx.algebraic_connectivity(largest, weight="weight", method=method, seed=REPRODUCIBILITY_SEED)
        )
        vector = nx.fiedler_vector(largest, weight="weight", method=method, seed=REPRODUCIBILITY_SEED)
        fiedler = dict(zip(largest.nodes, vector))
    # Fix the eigenvector's arbitrary sign so side A is always the larger half.
    if sum(1 for x in fiedler.values() if x < 0) < sum(1 for x in fiedler.values() if x > 0):
        fiedler = {k: -x for k, x in fiedler.items()}

    mean_degree = 2.0 * sum(d["weight"] for _u, _v, d in largest.edges(data=True)) / n
    cut = 0
    side_a = side_b = 0
    for u, v, key, _data in _iter_edges(G):
        if u in fiedler and v in fiedler:
            fu, fv = fiedler[u], fiedler[v]
            value = 0.5 * (fu + fv)
            side = "cut" if (fu < 0) != (fv < 0) else ("A" if fu < 0 else "B")
        else:
            value, side = math.nan, "outside"
        cut += side == "cut"
        side_a += side == "A"
        side_b += side == "B"
        _set_edge(G, u, v, key, "fiedler_value", float(value))
        _set_edge(G, u, v, key, "fiedler_side", side)

    return {
        "Algebraic Connectivity Route Weighting": label,
        "Algebraic Connectivity lambda_2": lambda2 * scale,
        # lambda_2 / mean weighted degree: unit-free, comparable across networks.
        "Normalised Algebraic Connectivity": (
            lambda2 / mean_degree if mean_degree > 0 else math.nan
        ),
        "Algebraic Connectivity Component Nodes": n,
        "Fiedler Split Vessel Counts [A, B]": (side_a, side_b),
        "Fiedler Cut Vessel Count": cut,
    }
