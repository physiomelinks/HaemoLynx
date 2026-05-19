"""Poiseuille law: viscosity, resistance, conductance."""
import logging

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

class PoiseuilleModel:
    """Encapsulates Poiseuille computations and constriction settings."""

    def __init__(self, constriction_length: float, constriction_spacing: float, mode: str = "periodic", constant_radius_um: float = 5.0) -> None:
        self.constriction_length = constriction_length
        self.constriction_spacing = constriction_spacing
        self.mode = mode
        self.constant_radius_um = constant_radius_um

    def get_diameter_at_position(
        self, position: float, length: float, d1: float, d2: float
    ) -> float:
        """Diameter at position along vessel. Supports periodic, sphincter, or constant_radius patterns."""
        if self.mode == "constant_radius":
            return self.constant_radius_um * 2.0
            
        if length <= 0:
            return d1
            
        if self.mode == "sphincter":
            # Sphincter model: a single constriction exactly at the origin of the vessel
            # 0 to 1/4 length: ramp down, 1/4 to 3/4: hold d2, 3/4 to full: ramp up
            if position > self.constriction_length:
                return d1
            
            # Divide the sphincter length into sections
            ramp_len = self.constriction_length * 0.25
            if position < ramp_len:
                return d1 + (d2 - d1) * (position / ramp_len)
            if position < self.constriction_length - ramp_len:
                return d2
            return d2 + (d1 - d2) * ((position - (self.constriction_length - ramp_len)) / ramp_len)
            
        else:
            # Periodic model (default)
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

    def set_poiseuille_resistances(
        self,
        G: nx.MultiGraph,
        diameter_by_branch_order: dict,
        *,
        prefer_edge_fwhm_diameter: bool = False,
    ) -> tuple[nx.MultiGraph, dict]:
        """
        Set edge resistances using Poiseuille's law with calculated viscosity.
        Resistance = (128 * viscosity * length) / (π * diameter^4)
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
        dict : Summary of resistance assignments
        """
        import numpy as np
        
        PI = np.pi
        results = {
            'resistances_set': 0,
            'missing_branch_order': [],
            'missing_length': [],
            'unknown_branch_order': [],
            'invalid_length': [],
            'invalid_diameter': [],
            'viscosity_calculations': {},  # Track viscosity for each diameter
            'used_fwhm_edge_diameter': 0,
        }
        
        print(f"=== Poiseuille Resistance Calculation (Branch Order Based) ===")
        print(f"Formula: Resistance = (128 * viscosity * length) / (π * diameter^4)")
        print(f"Viscosity calculation: μ = 1 / diameter^1.647")
        print(f"Units: diameter and length in micrometers (μm)")
        print()
        
        # Pre-calculate viscosities for each diameter to avoid redundant calculations
        diameter_viscosity_map = {}
        for branch_order, val in diameter_by_branch_order.items():
            # Support both float and dict {"d1": ..., "d2": ...}
            diameter = val["d1"] if isinstance(val, dict) else val
            
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
                val = diameter_by_branch_order.get(branch_order)
                if val is None:
                    # Try fallback to DEFAULT
                    val = diameter_by_branch_order.get("DEFAULT")
                
                if val is not None:
                    diameter = val["d1"] if isinstance(val, dict) else val
            
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
                viscosity = 1.0 / (diameter ** 1.647)
            
            # Calculate resistance using Poiseuille's law
            # Resistance = (128 * viscosity * length) / (π * diameter^4)
            resistance = (128.0 * viscosity * length) / (PI * diameter**4)
            
            G[u][v][key]['resistance'] = resistance
            G[u][v][key]['assigned_diameter_um'] = diameter
            
            results['resistances_set'] += 1
            
            logger.debug(f"Edge ({u}, {v}, {key}): {branch_order}, "
                        f"diameter={diameter}μm, length={length:.3f}μm, "
                        f"viscosity={viscosity:.6f}, resistance={resistance:.6f}")
        
        # Print summary
        print(f"=== Summary ===")
        print(f"Resistances successfully set: {results['resistances_set']}")
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

    def set_poiseuille_resistances_with_constrictions(
        self, G: nx.MultiGraph, diameter_by_branch_order: dict,
        *,
        prefer_edge_fwhm_baseline: bool = False,
    ) -> dict:
        """Set edge resistances using integrated resistance with constrictions."""
        results = {
            "resistances_set": 0,
            "missing_branch_order": [],
            "missing_length": [],
            "unknown_branch_order": [],
            "invalid_length": [],
            "invalid_diameter": [],
            "used_fwhm_baseline": 0,
        }
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
            diameters = diameter_by_branch_order.get(branch_order)
            if diameters is None:
                # Try fallback to DEFAULT key if available
                diameters = diameter_by_branch_order.get("DEFAULT")
                if diameters is None:
                    raise ValueError(
                        f"Edge ({u}, {v}, {key}) has unknown branch_order '{branch_order}' "
                        "and no 'DEFAULT' entry in diameter_by_branch_order."
                    )
            if not isinstance(diameters, dict) or "d1" not in diameters or "d2" not in diameters:
                raise ValueError(
                    f"Invalid diameter mapping for branch_order '{branch_order}'. "
                    "Expected dict containing 'd1' and 'd2'."
                )
                
            d1_dict, d2_dict = diameters["d1"], diameters["d2"]
            if d1_dict <= 0 or d2_dict <= 0:
                raise ValueError(
                    f"Invalid non-positive diameters for branch_order '{branch_order}': "
                    f"d1={d1_dict}, d2={d2_dict}."
                )
                
            # If prefer_edge_fwhm_baseline is True, we grab d1 from the image measurement.
            # We then scale d2 by the exact same ratio defined in the dictionary.
            d1 = d1_dict
            d2 = d2_dict
            
            if prefer_edge_fwhm_baseline:
                fwhm_d = data.get("fwhm_diameter_um")
                if fwhm_d is not None and float(fwhm_d) > 0:
                    d1 = float(fwhm_d)
                    constriction_ratio = d2_dict / d1_dict
                    d2 = d1 * constriction_ratio
                    results["used_fwhm_baseline"] += 1
            
            try:
                total_resistance = self.calculate_integrated_resistance(length, d1, d2)
                G[u][v][key]["resistance"] = total_resistance
                results["resistances_set"] += 1
            except Exception as e:
                raise ValueError(f"Resistance calculation failed for edge ({u}, {v}, {key}): {e}")
        return G, results

    def set_poiseuille_edge_resistances(
        self,
        G: nx.MultiGraph,
        custom_edges,
        edge_diameter: float,
        use_resistance: bool = True,
    ) -> dict:
        """Set resistances for specified edges."""
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
                
                # Resistance = (128 * viscosity * length) / (π * diameter^4)
                new_resistance = (128.0 * viscosity * vessel_length) / (
                    np.pi * edge_diameter ** 4
                )
                
                G[u_actual][v_actual][key]["resistance"] = new_resistance
                G[u_actual][v_actual][key]["assigned_diameter_um"] = edge_diameter
                results["updated"].append(
                    {
                        "edge": (u_actual, v_actual, key),
                        "vessel_length": vessel_length,
                        "new_resistance": new_resistance,
                    }
                )
        return G, results

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
