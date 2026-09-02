"""The constriction model: how narrowings change an edge's resistance.

A constricted vessel is a tube whose diameter dips from its passive ``d1`` to a
constricted ``d2`` around each constriction site and ramps back, so its
resistance is the Poiseuille resistance per unit length integrated along it.
Every pericyte model in this package shares that. They differ only in *where*
the sites are, which is what a :class:`ConstrictionSites` object decides:

* :mod:`haemolynx.haemodynamics.pericyte_mask` puts one site at each pericyte
  found in a segmented mask,
* :mod:`haemolynx.haemodynamics.probability` puts them at a fixed spacing and
  activates each one with a probability.

Both then hand their sites to :func:`apply_constriction_sites`, which is the
only place edge resistance is computed from a constriction.

Units
-----
Lengths and diameters are in micrometres, as everywhere else in the package.
Resistance comes back in Pa.s/m^3 and conductance in m^3/(Pa.s), the same units
:class:`~haemolynx.haemodynamics.poiseuille.PoiseuilleModel` produces, so a
constricted edge and a uniform one are directly comparable: an edge with no
sites on it gets exactly
:meth:`~haemolynx.haemodynamics.poiseuille.PoiseuilleModel.resistance_of_uniform_segment`.

Viscosity comes from :mod:`haemolynx.haemodynamics.viscosity` and is therefore
whichever law the run selected — ``viscosity_law``, ``haematocrit`` and
``diameter_basis`` are threaded down to here, because a constricted vessel is
the narrowest one in the tree and so the one the law matters most for. This
module used to integrate a dimensionless ``1 / d^1.647`` instead, which ignored
those settings and left the result in arbitrary units.
"""
from __future__ import annotations

from typing import Any, Iterable, Protocol

import networkx as nx
import numpy as np

from .poiseuille import set_edge_resistance
from .viscosity import DEFAULT_HAEMATOCRIT, viscosity_for

#: Micrometres per metre. Diameters and lengths arrive in um and the resistance
#: is wanted in SI, so the integral converts once, here.
UM_PER_M = 1.0e6


# --- Which edges and which sites -------------------------------------------


def is_capillary_branch_order(branch_order: str | None) -> bool:
    """Return True for capillary labels (B01, B02, ...)."""
    if branch_order is None:
        return False
    label = str(branch_order).strip()
    return label.startswith("B")


def resolve_generator(
    rng: np.random.Generator | None,
    seed: int | None,
) -> np.random.Generator:
    """Return the generator the pericyte cohort is drawn from.

    An explicit generator always wins, so a caller driving several draws from
    one stream keeps that stream. Otherwise it is built from *seed*;
    ``seed=None`` means fresh entropy, i.e. a different cohort every run.
    """
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def select_active_pericyte_indices(
    total_pericytes: int,
    constriction_probability: float,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> list[int]:
    """Randomly select pericyte indices that are active for constriction.

    Parameters
    ----------
    total_pericytes:
        Number of pericytes available.
    constriction_probability:
        Activation probability in [0, 1]. Example: 0.8 means 80% expected active.
    rng:
        Optional random generator, used as-is when given. Takes precedence over
        *seed*.
    seed:
        Seed for the generator built when *rng* is omitted. ``None`` draws from
        fresh entropy, so the cohort differs on every call.
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
    generator = resolve_generator(rng, seed)
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
    default_constriction_factor: float = 1.0,
) -> tuple[float, float, bool]:
    """Return ``(d1, d2, used_fwhm_baseline)`` for one edge.

    ``diameter_by_branch_order`` maps a branch order either to a passive
    diameter, in which case ``d2`` comes from the effective constriction
    factor, or to an explicit ``{"d1": ..., "d2": ...}`` pair. With
    ``prefer_edge_fwhm_baseline`` the edge's own measured ``fwhm_diameter_um``
    supersedes the table as ``d1``, and the table supplies only the fallback
    for edges that were never measured.

    The effective factor for an order is the map entry when present; otherwise
    ``default_constriction_factor``. Map values **replace** the default for that
    order only (they are not multiplied by it).
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

    if (
        constriction_factor_by_branch_order is not None
        and branch_order in constriction_factor_by_branch_order
    ):
        factor = float(constriction_factor_by_branch_order[branch_order])
    else:
        factor = float(default_constriction_factor)
    return d1, d1 * factor, used_fwhm_baseline


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


def _viscosity_profile(
    diameters: np.ndarray,
    *,
    viscosity_law: str,
    haematocrit: float,
    diameter_basis: str,
) -> np.ndarray:
    """Apparent viscosity (Pa.s) at each diameter, by the configured law.

    The laws are scalar functions, so they are evaluated once per *distinct*
    diameter rather than once per integration point: an edge with no
    constriction on it is one call, not a thousand.
    """
    unique_diameters, inverse = np.unique(diameters, return_inverse=True)
    viscosities = np.asarray(
        [
            viscosity_for(
                float(diameter),
                law=viscosity_law,
                haematocrit=float(haematocrit),
                diameter_basis=diameter_basis,
            )
            for diameter in unique_diameters
        ],
        dtype=float,
    )
    return viscosities[inverse]


def integrated_resistance(
    *,
    length: float,
    d1: float,
    d2: float,
    constriction_centers: list[float],
    constriction_length: float,
    num_points: int,
    viscosity_law: str = "pries",
    haematocrit: float = DEFAULT_HAEMATOCRIT,
    diameter_basis: str = "plasma_column",
) -> float:
    """Resistance (Pa.s/m^3) of one edge, integrated along its constrictions.

    Trapezoidal integral of the Poiseuille resistance per unit length,
    ``128 * mu(d) / (pi * d^4)``, with *mu* from the configured viscosity law
    and the integration done in metres so the result is SI. ``length``, ``d1``,
    ``d2`` and ``constriction_length`` are in micrometres.

    With no sites this reduces exactly to the uniform Poiseuille resistance, so
    it is comparable with
    :meth:`~haemolynx.haemodynamics.poiseuille.PoiseuilleModel.resistance_of_uniform_segment`.
    """
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
    viscosity = _viscosity_profile(
        diameters,
        viscosity_law=viscosity_law,
        haematocrit=haematocrit,
        diameter_basis=diameter_basis,
    )
    diameters_m = diameters / UM_PER_M
    resistance_per_length = (128.0 * viscosity) / (np.pi * (diameters_m ** 4))
    integ = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integ(resistance_per_length, x=positions / UM_PER_M))


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
    viscosity_law: str = "pries",
    haematocrit: float = DEFAULT_HAEMATOCRIT,
    diameter_basis: str = "plasma_column",
) -> tuple[nx.MultiGraph, dict[str, Any]]:
    """Set ``resistance``/``conductance`` on every edge from its constrictions.

    Resistances are in Pa.s/m^3, from the viscosity law named by
    ``viscosity_law``/``haematocrit``/``diameter_basis``. They are not
    comparable across laws, so the caller records which one ran.

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
            viscosity_law=viscosity_law,
            haematocrit=float(haematocrit),
            diameter_basis=diameter_basis,
        )
        set_edge_resistance(graph[u][v][key], float(total_resistance))
        graph[u][v][key]["pericyte_count_assigned"] = int(len(centers))
        graph[u][v][key]["pericyte_centers_um"] = [float(s) for s in centers]
        results["edges_set"] += 1

    results.update(sites.summary())
    return graph, results
