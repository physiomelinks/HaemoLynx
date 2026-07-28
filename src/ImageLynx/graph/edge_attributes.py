"""Canonical vascular-graph edge attributes and their units.

Kept dependency-free so both ``_helpers`` and ``validate`` can import it.
"""
from __future__ import annotations

from typing import Any

import networkx as nx

#: Edge attributes carried by a vascular graph, with their units.
EDGE_ATTRIBUTE_UNITS = {
    "length": "um",
    "resistance": "Pa.s/m^3",
    "conductance": "m^3/(Pa.s)",
}

#: Removed in favour of the explicit attributes above. ``weight`` meant physical
#: length at graph-build time and conductance after haemodynamics ran, so any
#: consumer reading it got whichever the last writer happened to mean.
FORBIDDEN_EDGE_ATTRIBUTES = ("weight",)


def assert_no_forbidden_edge_attributes(G: Any, *, context: str = "") -> None:
    """Raise if any edge carries a removed, ambiguous attribute such as ``weight``.

    ``weight`` was overloaded: graph construction stored physical length in it
    and haemodynamics later overwrote it with conductance, so statistics read
    conductances back as microns. Use ``length``, ``resistance`` and
    ``conductance`` instead — see :data:`EDGE_ATTRIBUTE_UNITS`.
    """
    if not isinstance(G, nx.Graph):
        return
    edges = G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
    for item in edges:
        data = item[-1]
        for name in FORBIDDEN_EDGE_ATTRIBUTES:
            if name in data:
                where = f" in {context}" if context else ""
                raise ValueError(
                    f"Edge {item[:-1]} carries the removed '{name}' attribute{where}. "
                    f"'{name}' was ambiguous — it held physical length before "
                    "haemodynamics and conductance afterwards. Use the explicit "
                    + ", ".join(f"'{k}' ({v})" for k, v in EDGE_ATTRIBUTE_UNITS.items())
                    + " attributes instead."
                )
