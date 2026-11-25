"""Widen or narrow every arteriole by one percentage, and re-solve for it.

The simplest question to ask of a finished network: what happens to it if the
arterioles dilate? One percentage for the whole arteriole tree, nothing else
touched -- so whatever changes in the answer is that dilation and not a second
edit riding along with it.

Unlike a pericyte constriction, this is **whole-branch** scaling: every edge
whose ``branch_order`` names an arteriole (``Art…``) has its diameter moved by
the same factor. No focal constriction sites are placed.

Two things carry a diameter and both have to move together. The branch-order
table is what :meth:`PoiseuilleModel.set_poiseuille_resistances` reads, but a
run that measured its diameters from the image has a ``fwhm_diameter_um`` on
each edge, and ``prefer_edge_fwhm_diameter`` makes that per-edge value win --
so scaling the table alone would leave every measured arteriole exactly as it
was, silently, on precisely the runs whose diameters are real.

The user-facing setting is a **percentage change** (``+10`` → 1.10×,
``−20`` → 0.80×). Internally the scale factor is what the resistance model
multiplies by; :func:`percent_change_to_scale` is the conversion.

The baseline is left alone: the graph comes back as a copy, because a
perturbation is a question about the network and not an edit to it.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

import networkx as nx

from .poiseuille import PoiseuilleModel, scale_stored_edge_diameters

logger = logging.getLogger(__name__)

__all__ = [
    "ARTERIOLE_PREFIX",
    "LARGE_ARTERIOLE_PREFIX",
    "is_arteriole_branch_order",
    "percent_change_to_scale",
    "scale_arteriole_diameters",
]

#: How `build_diameter_by_branch_order` labels an arteriole: ``Art1``, ``Art2``
#: -- not zero-padded, unlike the capillaries' ``B01``. So an arteriole is
#: named by prefix rather than by a fixed-width pattern.
ARTERIOLE_PREFIX = "Art"

#: A large arteriole kept in the network (graph.large_vessel_network) is
#: still an arteriole -- just upstream and wider -- so a "scale every
#: arteriole" perturbation should reach it too, not leave the one segment
#: with the most influence on the whole tree's resistance unperturbed. Its
#: own literal prefix, not a match against ARTERIOLE_PREFIX: "Large_Art1"
#: does not start with "Art".
LARGE_ARTERIOLE_PREFIX = "Large_Art"


def is_arteriole_branch_order(branch_order: Any) -> bool:
    """Whether *branch_order* names an arteriole (small or large) rather than
    a capillary or venule."""
    label = str(branch_order)
    return label.startswith(ARTERIOLE_PREFIX) or label.startswith(LARGE_ARTERIOLE_PREFIX)


def percent_change_to_scale(percent: float) -> float:
    """Convert a percentage diameter change to a multiplicative scale factor.

    ``+10`` means 10% wider (1.10×); ``−20`` means 20% narrower (0.80×);
    ``0`` leaves diameters unchanged (1.0×). A change of ``−100`` or below
    would make a non-positive diameter, which is refused.
    """
    percent = float(percent)
    scale = 1.0 + (percent / 100.0)
    if not scale > 0:
        raise ValueError(
            f"arteriole diameter change of {percent}% gives scale {scale}, "
            "which is not > 0. A change must be greater than -100%."
        )
    return scale


def scale_arteriole_diameters(
    graph: nx.MultiGraph,
    diameter_by_branch_order: Mapping[str, float],
    scale: float,
    *,
    model: PoiseuilleModel,
    prefer_edge_fwhm_diameter: bool = True,
) -> tuple[nx.MultiGraph, dict[str, float], dict[str, Any]]:
    """A copy of *graph* with every arteriole scaled by *scale*, re-solved.

    *scale* is a multiplicative factor (use :func:`percent_change_to_scale` when
    the caller has a percentage). *model* is required rather than defaulted
    because a resistance is only comparable with the baseline's when the same
    viscosity law produced it.

    Returns the new graph, the diameter table it was solved with, and a summary
    of what moved.
    """
    scale = float(scale)
    if not scale > 0:
        raise ValueError(
            f"scale must be > 0, got {scale}. A factor of 1.0 leaves the "
            "arterioles as they are."
        )

    scaled_table = {
        branch_order: (
            float(diameter) * scale
            if is_arteriole_branch_order(branch_order)
            else float(diameter)
        )
        for branch_order, diameter in (diameter_by_branch_order or {}).items()
    }

    scaled = graph.copy()
    arteriole_edges = 0
    edges_measured = 0
    for _u, _v, _key, data in scaled.edges(keys=True, data=True):
        if not is_arteriole_branch_order(data.get("branch_order")):
            continue
        arteriole_edges += 1
        if scale_stored_edge_diameters(data, scale):
            edges_measured += 1

    scaled, results = model.set_poiseuille_resistances(
        scaled,
        scaled_table,
        prefer_edge_fwhm_diameter=prefer_edge_fwhm_diameter,
    )

    summary = {
        "scale": scale,
        "branch_orders_scaled": tuple(
            sorted(order for order in scaled_table if is_arteriole_branch_order(order))
        ),
        "arteriole_edges": arteriole_edges,
        "edges_with_measured_diameter_scaled": edges_measured,
        "resistances": results,
    }
    logger.info(
        f"Arteriole diameters scaled by {scale}: {arteriole_edges} arteriole "
        f"edge(s), {edges_measured} of them carrying a measured diameter"
    )
    return scaled, scaled_table, summary
