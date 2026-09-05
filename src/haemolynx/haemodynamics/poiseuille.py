"""Poiseuille law: viscosity, resistance, conductance."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


def build_diameter_by_branch_order(
    *,
    all_diams_const: bool,
    max_branch_order: int = 51,
    default_diameter: float = 4.0,
    manual_capillary_diameter_by_branch_order: dict[str, float] | None = None,
    manual_arteriole_diameter_by_branch_order: dict[str, float] | None = None,
    manual_venule_diameter_by_branch_order: dict[str, float] | None = None,
    manual_large_arteriole_diameter_by_branch_order: dict[str, float] | None = None,
    manual_large_venule_diameter_by_branch_order: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build diameter mapping for Bxx, Artx, Venx, Large_Artx, Large_Venx labels."""
    if max_branch_order < 1:
        raise ValueError(
            f"max_branch_order must be >= 1, got {max_branch_order}."
        )
    if default_diameter <= 0:
        raise ValueError(
            f"default_diameter must be positive, got {default_diameter}."
        )

    capillary_overrides = manual_capillary_diameter_by_branch_order or {}
    arteriole_overrides = manual_arteriole_diameter_by_branch_order or {}
    venule_overrides = manual_venule_diameter_by_branch_order or {}
    large_arteriole_overrides = manual_large_arteriole_diameter_by_branch_order or {}
    large_venule_overrides = manual_large_venule_diameter_by_branch_order or {}

    diameter_by_branch_order: dict[str, float] = {}
    if all_diams_const:
        for i in range(1, max_branch_order + 1):
            diameter_by_branch_order[f"B{i:02d}"] = default_diameter
            diameter_by_branch_order[f"Art{i}"] = default_diameter
            diameter_by_branch_order[f"Ven{i}"] = default_diameter
            diameter_by_branch_order[f"Large_Art{i}"] = default_diameter
            diameter_by_branch_order[f"Large_Ven{i}"] = default_diameter
        return diameter_by_branch_order

    for i in range(1, max_branch_order + 1):
        key = f"B{i:02d}"
        diameter_by_branch_order[key] = capillary_overrides.get(
            key,
            default_diameter,
        )

    default_small_vessel_diameter = diameter_by_branch_order.get(
        "B01",
        default_diameter,
    )
    for i in range(1, max_branch_order + 1):
        art_key = f"Art{i}"
        ven_key = f"Ven{i}"
        diameter_by_branch_order[art_key] = arteriole_overrides.get(
            art_key,
            default_small_vessel_diameter,
        )
        diameter_by_branch_order[ven_key] = venule_overrides.get(
            ven_key,
            default_small_vessel_diameter,
        )

    # Large_Art/Large_Ven fall back to default_diameter directly, not the
    # capillary-derived default_small_vessel_diameter the small Art/Ven loop
    # above uses: a large vessel is by definition bigger than a capillary,
    # so that fallback would be actively wrong here, not just approximate.
    # In practice this fallback rarely fires -- a real run resolves these
    # from per-edge FWHM measurement, same as the other branch orders.
    for i in range(1, max_branch_order + 1):
        large_art_key = f"Large_Art{i}"
        large_ven_key = f"Large_Ven{i}"
        diameter_by_branch_order[large_art_key] = large_arteriole_overrides.get(
            large_art_key,
            default_diameter,
        )
        diameter_by_branch_order[large_ven_key] = large_venule_overrides.get(
            large_ven_key,
            default_diameter,
        )
    return diameter_by_branch_order


#: How an edge's modelled ``diameter_um`` was chosen. ``measured`` is a FWHM
#: fit; ``table`` is the branch-order lookup; ``override`` is a human value.
DIAMETER_SOURCE_MEASURED = "measured"
DIAMETER_SOURCE_TABLE = "table"
DIAMETER_SOURCE_OVERRIDE = "override"
DIAMETER_SOURCES = frozenset(
    {
        DIAMETER_SOURCE_MEASURED,
        DIAMETER_SOURCE_TABLE,
        DIAMETER_SOURCE_OVERRIDE,
    }
)

_KEPT_DIAMETER_SOURCES = frozenset(
    {DIAMETER_SOURCE_MEASURED, DIAMETER_SOURCE_OVERRIDE}
)


def positive_diameter_um(value: object) -> float | None:
    """A finite positive diameter in micrometres, or ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


def clear_edge_resistances(G: nx.MultiGraph) -> None:
    """Drop ``resistance`` and ``conductance`` so diameters can exist alone."""
    for _u, _v, _key, data in G.edges(keys=True, data=True):
        data.pop("resistance", None)
        data.pop("conductance", None)


def set_edge_diameter_override(data: dict, diameter_um: float) -> None:
    """Record a human diameter on one edge; it wins over FWHM and the table."""
    diameter = positive_diameter_um(diameter_um)
    if diameter is None:
        raise ValueError(
            f"Override diameter must be finite and positive, got {diameter_um}."
        )
    data["diameter_um"] = diameter
    data["diameter_source"] = DIAMETER_SOURCE_OVERRIDE


def scale_stored_edge_diameters(data: dict, scale: float) -> bool:
    """Multiply ``fwhm_diameter_um`` and ``diameter_um`` when they are set.

    Returns whether either attribute moved, so a caller can count scaled edges.
    """
    factor = float(scale)
    moved = False
    for attr in ("fwhm_diameter_um", "diameter_um"):
        diameter = positive_diameter_um(data.get(attr))
        if diameter is None:
            continue
        data[attr] = diameter * factor
        moved = True
    return moved


def stamp_edge_diameters(
    G: nx.MultiGraph,
    diameter_by_branch_order: dict | None,
    *,
    keep_existing: bool = False,
) -> dict[str, int]:
    """Write ``diameter_um`` and ``diameter_source`` on every edge that can.

    When *keep_existing* is True, measured and override edges stay as they are
    (so a resume does not wipe approvals). Otherwise FWHM, when present, wins,
    then the branch-order table.
    """
    table = diameter_by_branch_order or {}
    counts = {"measured": 0, "table": 0, "override": 0, "unset": 0}
    for _u, _v, _key, data in G.edges(keys=True, data=True):
        source = data.get("diameter_source")
        if keep_existing and source in _KEPT_DIAMETER_SOURCES:
            kept = positive_diameter_um(data.get("diameter_um"))
            if kept is None and source == DIAMETER_SOURCE_MEASURED:
                kept = positive_diameter_um(data.get("fwhm_diameter_um"))
                if kept is not None:
                    data["diameter_um"] = kept
            if kept is not None:
                counts[str(source)] += 1
                continue
        if keep_existing:
            measured = positive_diameter_um(data.get("fwhm_diameter_um"))
            if measured is not None:
                data["diameter_um"] = measured
                data["diameter_source"] = DIAMETER_SOURCE_MEASURED
                counts["measured"] += 1
                continue

        measured = positive_diameter_um(data.get("fwhm_diameter_um"))
        if measured is not None:
            data["diameter_um"] = measured
            data["diameter_source"] = DIAMETER_SOURCE_MEASURED
            counts["measured"] += 1
            continue
        table_diameter = positive_diameter_um(table.get(data.get("branch_order")))
        if table_diameter is not None:
            data["diameter_um"] = table_diameter
            data["diameter_source"] = DIAMETER_SOURCE_TABLE
            counts["table"] += 1
            continue
        counts["unset"] += 1
    return counts


# --- Viscosity model -------------------------------------------------------
#
# Which law supplies the viscosity is a setting, because it changes every
# resistance a run produces and the laws do not agree. They live in
# `viscosity.py` with their validity ranges; the names are re-exported here
# because this is where callers have always found them.
from .viscosity import (  # noqa: E402  (re-export, must follow the docstring)
    CAPILLARY_REGIME_MAX_DIAMETER_UM,
    DEFAULT_HAEMATOCRIT,
    DIAMETER_BASES,
    LARGE_VESSEL_VISCOSITY_PA_S,
    PLACEHOLDER_REGIME_MAX_DIAMETER_UM,
    PLASMA_VISCOSITY_PA_S,
    PRIES_MAX_DIAMETER_UM,
    PRIES_MIN_DIAMETER_UM,
    REFERENCE_DIAMETER_UM,
    REFERENCE_VISCOSITY_PA_S,
    VISCOSITY_DIAMETER_EXPONENT,
    VISCOSITY_LAWS,
    PlaceholderViscosityWarning,
    describe_law,
    validity_range_um,
    viscosity_for,
)

UM_PER_M = 1.0e6


def set_edge_resistance(edge_data: dict, resistance: float) -> None:
    """Store both ``resistance`` and ``conductance`` on one edge's attribute dict.

    The two are always written together so no call site has to derive one from
    the other, and neither can go stale relative to the other.

    ``resistance`` is in Pa.s/m^3; ``conductance`` is its reciprocal.
    """
    value = float(resistance)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"Edge resistance must be finite and positive, got {resistance}."
        )
    edge_data["resistance"] = value
    edge_data["conductance"] = 1.0 / value


class PoiseuilleModel:
    """Encapsulates Poiseuille computations and constriction settings.

    Units
    -----
    Diameters and lengths are in micrometres (um). Viscosity is in Pa.s.
    Resistance is returned in Pa.s/m^3 and conductance in m^3/(Pa.s), so a
    pressure drop in Pa divided by a resistance yields a flow in m^3/s.
    """

    def __init__(
        self,
        constriction_length: float,
        constriction_spacing: float,
        viscosity_law: str = "pries",
        haematocrit: float = DEFAULT_HAEMATOCRIT,
        diameter_basis: str = "plasma_column",
    ) -> None:
        if viscosity_law not in VISCOSITY_LAWS:
            raise ValueError(
                f"Unknown viscosity_law {viscosity_law!r}. Fix: choose one of "
                f"{', '.join(VISCOSITY_LAWS)}."
            )
        if diameter_basis not in DIAMETER_BASES:
            raise ValueError(
                f"Unknown diameter_basis {diameter_basis!r}. Fix: choose one "
                f"of {', '.join(DIAMETER_BASES)}."
            )
        self.constriction_length = constriction_length
        self.constriction_spacing = constriction_spacing
        #: Which viscosity law this model applies. Resistances are not
        #: comparable across laws, so a run records it.
        self.viscosity_law = viscosity_law
        self.haematocrit = float(haematocrit)
        #: What the graph's diameters mean; see `viscosity.DIAMETER_BASES`.
        self.diameter_basis = diameter_basis

    def get_diameter_at_position(
        self, position: float, length: float, d1: float, d2: float
    ) -> float:
        """Diameter at position along vessel. Periodic constriction pattern."""
        if length <= 0:
            return d1
        phase = position % self.constriction_spacing
        if phase < self.constriction_length:
            # Ramp: 0-10 d1->d2, 10-30 d2, 30-40 d2->d1
            if phase < 10:
                return d1 + (d2 - d1) * (phase / 10)
            if phase < 30:
                return d2
            return d2 + (d1 - d2) * ((phase - 30) / 10)
        return d1

    def calculate_viscosity(self, diameter: float) -> float:
        """Apparent blood viscosity in Pa.s for a vessel of *diameter* um.

        Which law answers is `self.viscosity_law`; see
        :mod:`haemolynx.haemodynamics.viscosity` for what each covers.
        """
        return viscosity_for(
            diameter,
            law=self.viscosity_law,
            haematocrit=self.haematocrit,
            diameter_basis=self.diameter_basis,
        )

    def describe_viscosity_law(self) -> str:
        """The law and its range, for a run's metadata."""
        return describe_law(
            self.viscosity_law, self.haematocrit, self.diameter_basis
        )

    def resistance_of_uniform_segment(self, length: float, diameter: float) -> float:
        """Poiseuille resistance (Pa.s/m^3) of a straight uniform segment.

        ``length`` and ``diameter`` are in micrometres.
        """
        if length <= 0:
            return float("inf")
        viscosity = self.calculate_viscosity(diameter)
        length_m = length / UM_PER_M
        diameter_m = diameter / UM_PER_M
        return (128.0 * viscosity * length_m) / (np.pi * diameter_m ** 4)

    def resistance_integrand(
        self, position: float, length: float, d1: float, d2: float
    ) -> float:
        """Resistance per unit length (Pa.s/m^4) at *position* um along the vessel."""
        diameter = self.get_diameter_at_position(position, length, d1, d2)
        viscosity = self.calculate_viscosity(diameter)
        diameter_m = diameter / UM_PER_M
        return (128.0 * viscosity) / (np.pi * diameter_m ** 4)

    def calculate_integrated_resistance(
        self, length: float, d1: float, d2: float, num_points: int = 1000
    ) -> float:
        """Total resistance (Pa.s/m^3) by trapezoidal integration along the vessel."""
        if length <= 0:
            return float("inf")
        positions = np.linspace(0, length, num_points)
        resistances = [
            self.resistance_integrand(pos, length, d1, d2) for pos in positions
        ]
        dx = (length / (num_points - 1) if num_points > 1 else length) / UM_PER_M
        integ = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return float(integ(resistances, dx=dx))

    def set_poiseuille_resistances(
        self,
        G: nx.MultiGraph,
        diameter_by_branch_order: dict,
        *,
        prefer_edge_fwhm_diameter: bool = False,
    ) -> tuple[nx.MultiGraph, dict]:
        """
        Set edge resistance and conductance from Poiseuille's law.
        resistance = (128 * viscosity * length) / (π * diameter^4)
        Where viscosity is the apparent blood viscosity at that diameter under
        this model's ``viscosity_law`` (see
        :mod:`haemolynx.haemodynamics.viscosity`), so resistances are not
        comparable between runs that chose different laws.

        Parameters:
        -----------
        G : networkx.MultiGraph
            The multigraph with branch_order and length attributes
        diameter_by_branch_order : dict
            Dictionary mapping branch order strings to diameter values in micrometers (μm)
            e.g., {'BO1': 10.0, 'BO2': 8.0, 'BO3': 6.0}
        prefer_edge_fwhm_diameter : bool
            If True, use each edge's ``fwhm_diameter_um`` (when set and positive) instead
            of the branch-order table.

        Returns:
        --------
        dict : Summary of resistance assignments
        """
        import numpy as np

        PI = np.pi
        results = {
            'edges_set': 0,
            'missing_branch_order': [],
            'missing_length': [],
            'unknown_branch_order': [],
            'invalid_length': [],
            'invalid_diameter': [],
            'viscosity_calculations': {},  # Track viscosity for each diameter
            'used_fwhm_edge_diameter': 0,
        }

        logger.info("=== Poiseuille Resistance Calculation (Branch Order Based) ===")
        logger.info("Formula: resistance = (128 * viscosity * length) / (pi * diameter^4)")
        # The law is a setting, so the log has to name the one that actually
        # ran: reading a fixed formula here made every run look identical.
        logger.info(f"Viscosity: {self.describe_viscosity_law()}")
        logger.info("Units: diameter and length in um; resistance in Pa.s/m^3")

        # Pre-calculate viscosities for each diameter to avoid redundant calculations
        diameter_viscosity_map = {}
        for branch_order, diameter in diameter_by_branch_order.items():
            if diameter <= 0:
                logger.warning(f"Invalid diameter {diameter} for {branch_order}")
                continue
            viscosity = self.calculate_viscosity(diameter)
            diameter_viscosity_map[diameter] = viscosity
            results['viscosity_calculations'][branch_order] = {
                'diameter': diameter,
                'viscosity': viscosity
            }
            logger.info(
                f"{branch_order}: diameter={diameter}um, calculated viscosity={viscosity:.6f}"
            )

        for u, v, key, data in G.edges(keys=True, data=True):
            # Check for branch order
            branch_order = data.get('branch_order', None)
            if branch_order is None:
                results['missing_branch_order'].append((u, v, key))
                continue

            # Check for length
            length = data.get('length', None)
            if length is None:
                results['missing_length'].append((u, v, key))
                continue

            if length <= 0:
                results['invalid_length'].append((u, v, key, length))
                continue

            # Modelled diameter stamped at assign_diameters, else FWHM, else table.
            diameter = positive_diameter_um(data.get("diameter_um"))
            used_fwhm = data.get("diameter_source") == DIAMETER_SOURCE_MEASURED
            if diameter is None and prefer_edge_fwhm_diameter:
                diameter = positive_diameter_um(data.get("fwhm_diameter_um"))
                used_fwhm = diameter is not None
            if diameter is None:
                diameter = diameter_by_branch_order.get(branch_order, None)
                used_fwhm = False
            if used_fwhm:
                results['used_fwhm_edge_diameter'] += 1
            if diameter is None:
                results['unknown_branch_order'].append((u, v, key, branch_order))
                continue

            if diameter <= 0:
                results['invalid_diameter'].append((u, v, key, branch_order, diameter))
                continue

            # Get pre-calculated viscosity for this diameter
            viscosity = diameter_viscosity_map.get(diameter, None)
            if viscosity is None:
                # Fallback calculation if not in map
                viscosity = self.calculate_viscosity(diameter)

            resistance = self.resistance_of_uniform_segment(length, diameter)
            set_edge_resistance(G[u][v][key], resistance)

            results['edges_set'] += 1

            logger.debug(f"Edge ({u}, {v}, {key}): {branch_order}, "
                        f"diameter={diameter}um, length={length:.3f}um, "
                        f"viscosity={viscosity:.3e} Pa.s, "
                        f"resistance={resistance:.3e} Pa.s/m^3")

        # Log summary
        logger.info("=== Summary ===")
        logger.info(f"Edges assigned resistance: {results['edges_set']}")
        if prefer_edge_fwhm_diameter:
            logger.info(
                f"Edges using per-edge fwhm_diameter_um: "
                f"{results.get('used_fwhm_edge_diameter', 0)}"
            )
        # Every count below is an edge that was skipped, so it left the graph
        # without a resistance: a problem the run recovered from, not progress.
        if results['missing_branch_order']:
            logger.warning(
                f"Edges missing branch_order: {len(results['missing_branch_order'])}"
            )
        if results['missing_length']:
            logger.warning(f"Edges missing length: {len(results['missing_length'])}")
        if results['unknown_branch_order']:
            logger.warning(
                f"Edges with unknown branch_order: {len(results['unknown_branch_order'])}"
            )
        if results['invalid_length']:
            logger.warning(f"Edges with invalid length: {len(results['invalid_length'])}")
        if results['invalid_diameter']:
            logger.warning(
                f"Edges with invalid diameter: {len(results['invalid_diameter'])}"
            )

        return G, results

    def set_poiseuille_resistances_with_constrictions(
        self,
        G: nx.MultiGraph,
        diameter_by_branch_order: dict,
        *,
        prefer_edge_fwhm_baseline: bool = False,
        constriction_factor_by_branch_order: dict[str, float] | None = None,
    ) -> tuple[nx.MultiGraph, dict]:
        """Set edge resistance/conductance by integrating resistance along constrictions.

        Parameters
        ----------
        diameter_by_branch_order :
            If ``prefer_edge_fwhm_baseline`` is False: maps ``branch_order`` to
            ``{"d1": float, "d2": float}`` (passive and constricted diameters in µm).

            If True: maps ``branch_order`` to a **scalar** fallback diameter (µm) used
            only when an edge has no positive ``fwhm_diameter_um``.
        prefer_edge_fwhm_baseline :
            When True, per edge ``d1 = fwhm_diameter_um`` if set and positive, else the
            scalar fallback for that ``branch_order``; ``d2 = d1 * factor`` where
            ``factor`` comes from ``constriction_factor_by_branch_order[branch_order]``.
        constriction_factor_by_branch_order :
            Required when ``prefer_edge_fwhm_baseline`` is True: multiplier applied to
            baseline ``d1`` to obtain ``d2`` (same role as d2/d1 in the manual pipeline).
        """
        results = {
            "edges_set": 0,
            "missing_branch_order": [],
            "missing_length": [],
            "unknown_branch_order": [],
            "invalid_length": [],
            "invalid_diameter": [],
            "used_fwhm_baseline": 0,
        }
        if prefer_edge_fwhm_baseline:
            if constriction_factor_by_branch_order is None:
                raise ValueError(
                    "constriction_factor_by_branch_order is required when "
                    "prefer_edge_fwhm_baseline=True."
                )
        constr_map = constriction_factor_by_branch_order

        for u, v, key, data in G.edges(keys=True, data=True):
            branch_order = data.get("branch_order")
            if branch_order is None:
                raise ValueError(
                    f"Edge ({u}, {v}, {key}) missing required 'branch_order' attribute."
                )
            length = data.get("length")
            if length is None:
                raise ValueError(
                    f"Edge ({u}, {v}, {key}) missing required 'length' attribute."
                )
            if length <= 0:
                raise ValueError(
                    f"Edge ({u}, {v}, {key}) has non-positive length: {length}."
                )

            if prefer_edge_fwhm_baseline:
                spec = diameter_by_branch_order.get(branch_order)
                if spec is None:
                    raise ValueError(
                        f"Edge ({u}, {v}, {key}) has unknown branch_order '{branch_order}'. "
                        "No matching entry in diameter_by_branch_order (fallback diameters)."
                    )
                if isinstance(spec, dict):
                    raise ValueError(
                        "When prefer_edge_fwhm_baseline=True, diameter_by_branch_order "
                        f"must map to numeric fallbacks, not dict for '{branch_order}'."
                    )
                fallback_d1 = float(spec)
                fwhm_d = data.get("fwhm_diameter_um")
                if fwhm_d is not None and float(fwhm_d) > 0:
                    d1 = float(fwhm_d)
                    results["used_fwhm_baseline"] += 1
                else:
                    d1 = fallback_d1
                factor = constr_map.get(branch_order) if constr_map is not None else None
                if factor is None:
                    raise ValueError(
                        f"No constriction factor for branch_order '{branch_order}'."
                    )
                d2 = d1 * float(factor)
            else:
                diameters = diameter_by_branch_order.get(branch_order)
                if diameters is None:
                    raise ValueError(
                        f"Edge ({u}, {v}, {key}) has unknown branch_order '{branch_order}'. "
                        "No matching entry in diameter_by_branch_order."
                    )
                if not isinstance(diameters, dict) or "d1" not in diameters or "d2" not in diameters:
                    raise ValueError(
                        f"Invalid diameter mapping for branch_order '{branch_order}'. "
                        "Expected dict containing 'd1' and 'd2'."
                    )
                d1, d2 = diameters["d1"], diameters["d2"]

            if d1 <= 0 or d2 <= 0:
                raise ValueError(
                    f"Invalid non-positive diameters for edge ({u}, {v}, {key}) "
                    f"branch_order '{branch_order}': d1={d1}, d2={d2}."
                )
            try:
                total_resistance = self.calculate_integrated_resistance(length, d1, d2)
                set_edge_resistance(G[u][v][key], total_resistance)
                results["edges_set"] += 1
            except Exception as e:
                raise ValueError(f"Resistance calculation failed for edge ({u}, {v}, {key}): {e}")
        return G, results

    def set_poiseuille_edge_resistances(
        self,
        G: nx.MultiGraph,
        custom_edges,
        edge_diameter: float,
    ) -> dict:
        """Override resistance/conductance on specific edges using a fixed diameter.

        Both attributes are always written, so there is no longer a flag choosing
        between storing resistance and storing its inverse (see issue #12).
        """
        results = {
            "updated": [],
            "not_found": [],
            "no_length": [],
            "invalid_diameter": [],
        }
        if edge_diameter <= 0:
            results["invalid_diameter"].append(edge_diameter)
            return G, results
        edge_pairs = custom_edges.keys() if isinstance(custom_edges, dict) else custom_edges
        for edge_pair in edge_pairs:
            u, v = edge_pair
            edges_found = []
            if G.has_edge(u, v):
                for key in G[u][v]:
                    edges_found.append((u, v, key))
            elif u != v and G.has_edge(v, u):
                for key in G[v][u]:
                    edges_found.append((v, u, key))
            if not edges_found:
                results["not_found"].append(edge_pair)
                continue
            for u_actual, v_actual, key in edges_found:
                edge_data = G[u_actual][v_actual][key]
                vessel_length = edge_data.get("length")
                if vessel_length is None:
                    results["no_length"].append((u_actual, v_actual, key))
                    continue
                if vessel_length <= 0:
                    continue
                new_resistance = self.resistance_of_uniform_segment(
                    vessel_length, edge_diameter
                )
                set_edge_resistance(G[u_actual][v_actual][key], new_resistance)
                results["updated"].append(
                    {
                        "edge": (u_actual, v_actual, key),
                        "vessel_length": vessel_length,
                        "resistance": new_resistance,
                        "conductance": 1.0 / new_resistance,
                    }
                )
        return G, results
