"""Poiseuille law: viscosity, resistance, conductance."""
import logging

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
) -> dict[str, float]:
    """Build diameter mapping for Bxx, Artx, and Venx branch-order labels."""
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

    diameter_by_branch_order: dict[str, float] = {}
    if all_diams_const:
        for i in range(1, max_branch_order + 1):
            diameter_by_branch_order[f"B{i:02d}"] = default_diameter
            diameter_by_branch_order[f"Art{i}"] = default_diameter
            diameter_by_branch_order[f"Ven{i}"] = default_diameter
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
    return diameter_by_branch_order


class PoiseuilleModel:
    """Encapsulates Poiseuille computations and constriction settings."""

    def __init__(self, constriction_length: float, constriction_spacing: float) -> None:
        self.constriction_length = constriction_length
        self.constriction_spacing = constriction_spacing

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

    @staticmethod
    def calculate_viscosity(diameter: float) -> float:
        """μ = 1 / diameter^1.647"""
        return 1.0 / (diameter ** 1.647)

    def resistance_integrand(
        self, position: float, length: float, d1: float, d2: float
    ) -> float:
        """Resistance per unit length = (128 * viscosity) / (π * diameter^4)."""
        diameter = self.get_diameter_at_position(position, length, d1, d2)
        viscosity = self.calculate_viscosity(diameter)
        return (128.0 * viscosity) / (np.pi * diameter ** 4)

    def calculate_integrated_resistance(
        self, length: float, d1: float, d2: float, num_points: int = 1000
    ) -> float:
        """Total resistance by trapezoidal integration."""
        if length <= 0:
            return float("inf")
        positions = np.linspace(0, length, num_points)
        resistances = [
            self.resistance_integrand(pos, length, d1, d2) for pos in positions
        ]
        dx = length / (num_points - 1) if num_points > 1 else length
        integ = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return float(integ(resistances, dx=dx))

    def set_poiseuille_weights(
        self,
        G: nx.MultiGraph,
        diameter_by_branch_order: dict,
        *,
        prefer_edge_fwhm_diameter: bool = False,
    ) -> tuple[nx.MultiGraph, dict]:
        """
        Set edge weights using the inverse of Poiseuille's law with calculated viscosity.
        Weight = (π * diameter^4) / (128 * viscosity * length)
        Where viscosity = 1 / diameter^1.647

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
        dict : Summary of weight assignments
        """
        import numpy as np

        PI = np.pi
        results = {
            'weights_set': 0,
            'missing_branch_order': [],
            'missing_length': [],
            'unknown_branch_order': [],
            'invalid_length': [],
            'invalid_diameter': [],
            'viscosity_calculations': {},  # Track viscosity for each diameter
            'used_fwhm_edge_diameter': 0,
        }

        print(f"=== Poiseuille Weight Calculation (Branch Order Based) ===")
        print(f"Formula: Weight = (π * diameter^4) / (128 * viscosity * length)")
        print(f"Viscosity calculation: μ = 1 / diameter^1.647")
        print(f"Units: diameter and length in micrometers (μm)")
        print()

        # Pre-calculate viscosities for each diameter to avoid redundant calculations
        diameter_viscosity_map = {}
        for branch_order, diameter in diameter_by_branch_order.items():
            if diameter <= 0:
                print(f"Warning: Invalid diameter {diameter} for {branch_order}")
                continue
            viscosity = 1.0 / (diameter ** 1.647)
            diameter_viscosity_map[diameter] = viscosity
            results['viscosity_calculations'][branch_order] = {
                'diameter': diameter,
                'viscosity': viscosity
            }
            print(f"{branch_order}: diameter={diameter}μm, calculated viscosity={viscosity:.6f}")

        print()

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

            # Get diameter for this branch order (or per-edge FWHM measurement)
            diameter = None
            if prefer_edge_fwhm_diameter:
                fwhm_d = data.get("fwhm_diameter_um")
                if fwhm_d is not None and float(fwhm_d) > 0:
                    diameter = float(fwhm_d)
                    results['used_fwhm_edge_diameter'] += 1
            if diameter is None:
                diameter = diameter_by_branch_order.get(branch_order, None)
            if diameter is None:
                results['unknown_branch_order'].append((u, v, key, branch_order))
                continue

            if diameter <= 0:
                results['invalid_diameter'].append((u, v, key, branch_order, diameter))
                continue

            # Persist the exact diameter assigned for this edge so downstream
            # exports/tests can validate the pipeline's chosen vessel size directly.
            G[u][v][key]["assigned_diameter_um"] = float(diameter)

            # Get pre-calculated viscosity for this diameter
            viscosity = diameter_viscosity_map.get(diameter, None)
            if viscosity is None:
                # Fallback calculation if not in map
                viscosity = 1.0 / (diameter ** 1.647)

            # Calculate weight using inverse Poiseuille's law
            # Weight = (π * diameter^4) / (128 * viscosity * length)
            weight = (PI * diameter**4) / (128.0 * viscosity * length)

            # Store old weight for comparison
            old_weight = data.get('weight', None)

            # Set new weight
            G[u][v][key]['weight'] = weight

            results['weights_set'] += 1

            logger.debug(f"Edge ({u}, {v}, {key}): {branch_order}, "
                        f"diameter={diameter}μm, length={length:.3f}μm, "
                        f"viscosity={viscosity:.6f}, weight={weight:.6f}")

        # Print summary
        print(f"=== Summary ===")
        print(f"Weights successfully set: {results['weights_set']}")
        if prefer_edge_fwhm_diameter:
            print(
                f"Edges using per-edge fwhm_diameter_um: "
                f"{results.get('used_fwhm_edge_diameter', 0)}"
            )
        if results['missing_branch_order']:
            print(f"Edges missing branch_order: {len(results['missing_branch_order'])}")
        if results['missing_length']:
            print(f"Edges missing length: {len(results['missing_length'])}")
        if results['unknown_branch_order']:
            print(f"Edges with unknown branch_order: {len(results['unknown_branch_order'])}")
        if results['invalid_length']:
            print(f"Edges with invalid length: {len(results['invalid_length'])}")
        if results['invalid_diameter']:
            print(f"Edges with invalid diameter: {len(results['invalid_diameter'])}")

        return G, results

    def set_poiseuille_weights_with_constrictions(
        self,
        G: nx.MultiGraph,
        diameter_by_branch_order: dict,
        *,
        prefer_edge_fwhm_baseline: bool = False,
        constriction_factor_by_branch_order: dict[str, float] | None = None,
    ) -> tuple[nx.MultiGraph, dict]:
        """Set edge weights = 1/resistance using integrated resistance with constrictions.

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
            "weights_set": 0,
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
                weight = 1.0 / total_resistance
                G[u][v][key]["weight"] = weight
                results["weights_set"] += 1
            except Exception as e:
                raise ValueError(f"Resistance calculation failed for edge ({u}, {v}, {key}): {e}")
        return G, results

    def set_poiseuille_edge_weights(
        self,
        G: nx.MultiGraph,
        custom_edges,
        edge_diameter: float,
        use_resistance: bool = True,
    ) -> dict:
        """Set weights for specified edges. use_resistance=True -> weight=resistance."""
        results = {
            "updated": [],
            "not_found": [],
            "no_length": [],
            "invalid_diameter": [],
        }
        if edge_diameter <= 0:
            results["invalid_diameter"].append(edge_diameter)
            return G, results
        viscosity = self.calculate_viscosity(edge_diameter)
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
                if use_resistance:
                    new_weight = (128.0 * viscosity * vessel_length) / (
                        np.pi * edge_diameter ** 4
                    )
                else:
                    new_weight = (np.pi * edge_diameter ** 4) / (
                        128.0 * viscosity * vessel_length
                    )
                G[u_actual][v_actual][key]["weight"] = new_weight
                results["updated"].append(
                    {
                        "edge": (u_actual, v_actual, key),
                        "vessel_length": vessel_length,
                        "new_weight": new_weight,
                    }
                )
        return G, results
