"""The constriction model: how narrowings change an edge's resistance.

A constricted vessel is a tube whose diameter dips from its passive ``d1`` to a
constricted ``d2`` around each constriction site and ramps back, so its
resistance is the Poiseuille resistance per unit length integrated along it.
Every pericyte model in this package shares that. They differ only in *where*
the sites are, which is what a :class:`ConstrictionSites` object decides:

* :mod:`ImageLynx.haemodynamics.pericyte_mask` puts one site at each pericyte
  found in a segmented mask,
* :mod:`ImageLynx.haemodynamics.probability` puts them at a fixed spacing and
  activates each one with a probability.

Both then hand their sites to :func:`apply_constriction_sites`, which is the
only place edge resistance is computed from a constriction.

Units
-----
Lengths and diameters are in micrometres. Note that the resistance integrated
here uses the *uncalibrated* viscosity ``1 / d^1.647``, not the SI-pinned law in
:class:`~ImageLynx.haemodynamics.poiseuille.PoiseuilleModel`, so these
resistances are in model units and are not comparable in magnitude with that
class's Pa.s/m^3. That predates this module and is preserved deliberately;
reconciling the two laws is a separate change.
"""
from __future__ import annotations

from typing import Any, Iterable, Protocol

import networkx as nx
import numpy as np

from .poiseuille import set_edge_resistance

#: Exponent of the capillary viscosity power law, as used by the constriction
#: integral. Kept separate from the calibrated constants in ``poiseuille`` to
#: make the divergence between the two visible rather than accidental.
VISCOSITY_DIAMETER_EXPONENT = 1.647


# --- Which edges and which sites -------------------------------------------


def is_capillary_branch_order(branch_order: str | None) -> bool:
    """Return True for capillary labels (B01, B02, ...)."""
    if branch_order is None:
        return False
    label = str(branch_order).strip()
    return label.startswith("B")


def select_active_pericyte_indices(
    total_pericytes: int,
    constriction_probability: float,
    *,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Randomly select pericyte indices that are active for constriction.

    Parameters
    ----------
    total_pericytes:
        Number of pericytes available.
    constriction_probability:
        Activation probability in [0, 1]. Example: 0.8 means 80% expected active.
    rng:
        Optional random generator. If omitted, uses a fresh default RNG so each
        pipeline run naturally produces a different cohort.
    """
    if total_pericytes < 0:
        raise ValueError(f"total_pericytes must be >= 0, got {total_pericytes}.")
    if not (0.0 <= float(constriction_probability) <= 1.0):
        raise ValueError(
            "constriction_probability must be in [0, 1], "
            f"got {constriction_probability}."
        )
    if total_pericytes == 0:
        return []
    generator = rng if rng is not None else np.random.default_rng()
    active_mask = generator.random(total_pericytes) < float(constriction_probability)
    return np.flatnonzero(active_mask).astype(int).tolist()


def validate_active_pericyte_indices(
    active_pericyte_indices: Iterable[int] | None,
    *,
    total_pericytes: int,
) -> list[int]:
    """Validate and normalize a caller-supplied active cohort."""
    if active_pericyte_indices is None:
        return []
    out: list[int] = []
    for idx in active_pericyte_indices:
        idx_int = int(idx)
        if idx_int < 0 or idx_int >= int(total_pericytes):
            raise ValueError(
                f"Active pericyte index {idx_int} outside valid range "
                f"[0, {int(total_pericytes) - 1}]."
            )
        out.append(idx_int)
    return sorted(set(out))


# --- Parameter checks -------------------------------------------------------


def require_positive_constriction_length(constriction_length: float) -> None:
    if constriction_length <= 0:
        raise ValueError(
            f"constriction_length must be > 0, got {constriction_length}."
        )


def require_enough_integration_points(num_integration_points: int) -> None:
    if num_integration_points < 3:
        raise ValueError(
            f"num_integration_points must be >= 3, got {num_integration_points}."
        )


# --- Diameters --------------------------------------------------------------


def resolve_edge_diameters(
    *,
    edge_data: dict[str, Any],
    branch_order: str,
    diameter_by_branch_order: dict,
    constriction_factor_by_branch_order: dict[str, float] | None,
    prefer_edge_fwhm_baseline: bool,
) -> tuple[float, float, bool]:
    """Return ``(d1, d2, used_fwhm_baseline)`` for one edge.

    ``diameter_by_branch_order`` maps a branch order either to a passive
    diameter, in which case ``d2`` comes from
    ``constriction_factor_by_branch_order``, or to an explicit
    ``{"d1": ..., "d2": ...}`` pair. With ``prefer_edge_fwhm_baseline`` the
    edge's own measured ``fwhm_diameter_um`` supersedes the table as ``d1``, and
    the table supplies only the fallback for edges that were never measured.
    """
    used_fwhm_baseline = False
    if prefer_edge_fwhm_baseline:
        spec = diameter_by_branch_order.get(branch_order)
        if spec is None:
            raise ValueError(
                f"No fallback baseline diameter for branch_order '{branch_order}'."
            )
        if isinstance(spec, dict):
            raise ValueError(
                "With prefer_edge_fwhm_baseline=True, diameter_by_branch_order must "
                f"map '{branch_order}' to a numeric fallback baseline diameter."
            )
        d1 = float(spec)
        fwhm_diameter = edge_data.get("fwhm_diameter_um")
        if fwhm_diameter is not None and float(fwhm_diameter) > 0:
            d1 = float(fwhm_diameter)
            used_fwhm_baseline = True
    else:
        spec = diameter_by_branch_order.get(branch_order)
        if spec is None:
            raise ValueError(f"No diameter mapping for branch_order '{branch_order}'.")
        if isinstance(spec, dict):
            if "d1" not in spec or "d2" not in spec:
                raise ValueError(
                    f"Invalid diameter mapping for '{branch_order}'. "
                    "Expected keys d1 and d2."
                )
            return float(spec["d1"]), float(spec["d2"]), used_fwhm_baseline
        d1 = float(spec)

    factor = None
    if constriction_factor_by_branch_order is not None:
        factor = constriction_factor_by_branch_order.get(branch_order)
    if factor is None:
        raise ValueError(
            f"No constriction factor for branch_order '{branch_order}'."
        )
    return d1, d1 * float(factor), used_fwhm_baseline


def diameter_at_position(
    position: float,
    d1: float,
    d2: float,
    constriction_centers: list[float],
    constriction_length: float,
) -> float:
    """Diameter at one arc-length position, given the constriction sites.

    Each site holds ``d2`` over a plateau a quarter of ``constriction_length``
    either side of its center and ramps linearly back to ``d1`` over the next
    quarter. Overlapping sites take the narrowest diameter.
    """
    if not constriction_centers or constriction_length <= 0:
        return float(d1)

    half_window = float(constriction_length) / 2.0
    ramp_width = float(constriction_length) / 4.0
    plateau_half = float(constriction_length) / 4.0

    diameter = float(d1)
    for center in constriction_centers:
        distance_from_center = abs(float(position) - float(center))
        if distance_from_center >= half_window:
            continue
        if distance_from_center <= plateau_half:
            local_diameter = float(d2)
        else:
            if ramp_width <= 0:
                local_diameter = float(d1)
            else:
                alpha = (distance_from_center - plateau_half) / ramp_width
                local_diameter = float(d2 + (d1 - d2) * alpha)
        diameter = min(diameter, local_diameter)
    return diameter


def integrated_resistance(
    *,
    length: float,
    d1: float,
    d2: float,
    constriction_centers: list[float],
    constriction_length: float,
    num_points: int,
) -> float:
    """Trapezoidal integral of resistance per unit length along one edge."""
    if length <= 0:
        return float("inf")
    positions = np.linspace(0.0, float(length), int(num_points))
    diameters = np.asarray(
        [
            diameter_at_position(
                position=pos,
                d1=d1,
                d2=d2,
                constriction_centers=constriction_centers,
                constriction_length=constriction_length,
            )
            for pos in positions
        ],
        dtype=float,
    )
    diameters = np.clip(diameters, a_min=1e-9, a_max=None)
    viscosity = 1.0 / (diameters ** VISCOSITY_DIAMETER_EXPONENT)
    resistance_per_length = (128.0 * viscosity) / (np.pi * (diameters ** 4))
    integ = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integ(resistance_per_length, x=positions))


# --- Applying a set of sites to a graph -------------------------------------


class ConstrictionSites(Protocol):
    """Where the constrictions on each edge are.

    Implementations decide the sites however they like — from a mask, from a
    spacing rule, from a fixed cohort — and report whatever they want the run
    summary to say about that choice.
    """

    def centers_for_edge(
        self,
        u: Any,
        v: Any,
        key: Any,
        edge_data: dict[str, Any],
        *,
        length: float,
    ) -> list[float]:
        """Arc-length positions, in microns from the edge's first point."""

    def summary(self) -> dict[str, Any]:
        """Counts and settings to merge into the run summary, read after the run."""


def apply_constriction_sites(
    graph: nx.MultiGraph,
    sites: ConstrictionSites,
    *,
    diameter_by_branch_order: dict,
    constriction_factor_by_branch_order: dict[str, float] | None,
    prefer_edge_fwhm_baseline: bool,
    constriction_length: float,
    num_integration_points: int,
) -> tuple[nx.MultiGraph, dict[str, Any]]:
    """Set ``resistance``/``conductance`` on every edge from its constrictions.

    Each edge also records the sites it was given, as
    ``pericyte_count_assigned`` and ``pericyte_centers_um``, so a later
    visualization or diagnostic can show where the model put them.
    """
    results: dict[str, Any] = {"edges_set": 0, "used_fwhm_baseline": 0}

    for u, v, key, edge_data in graph.edges(keys=True, data=True):
        branch_order = edge_data.get("branch_order")
        if branch_order is None:
            raise ValueError(
                f"Edge ({u}, {v}, {key}) missing required 'branch_order' attribute."
            )
        length = edge_data.get("length")
        if length is None or float(length) <= 0:
            raise ValueError(
                f"Edge ({u}, {v}, {key}) has invalid length: {length}."
            )
        d1, d2, used_fwhm = resolve_edge_diameters(
            edge_data=edge_data,
            branch_order=str(branch_order),
            diameter_by_branch_order=diameter_by_branch_order,
            constriction_factor_by_branch_order=constriction_factor_by_branch_order,
            prefer_edge_fwhm_baseline=bool(prefer_edge_fwhm_baseline),
        )
        if used_fwhm:
            results["used_fwhm_baseline"] += 1
        if d1 <= 0 or d2 <= 0:
            raise ValueError(
                f"Edge ({u}, {v}, {key}) has non-positive diameters d1={d1}, d2={d2}."
            )

        centers = sites.centers_for_edge(u, v, key, edge_data, length=float(length))
        total_resistance = integrated_resistance(
            length=float(length),
            d1=float(d1),
            d2=float(d2),
            constriction_centers=centers,
            constriction_length=float(constriction_length),
            num_points=int(num_integration_points),
        )
        set_edge_resistance(graph[u][v][key], float(total_resistance))
        graph[u][v][key]["pericyte_count_assigned"] = int(len(centers))
        graph[u][v][key]["pericyte_centers_um"] = [float(s) for s in centers]
        results["edges_set"] += 1

    results.update(sites.summary())
    return graph, results
