"""Poiseuille law: viscosity, resistance, conductance."""
import logging

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

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

    def set_poiseuille_weights_with_constrictions(
        self, G: nx.MultiGraph, diameter_by_branch_order: dict
    ) -> dict:
        """Set edge weights = 1/resistance using integrated resistance with constrictions."""
        results = {
            "weights_set": 0,
            "missing_branch_order": [],
            "missing_length": [],
            "unknown_branch_order": [],
            "invalid_length": [],
            "invalid_diameter": [],
        }
        for u, v, key, data in G.edges(keys=True, data=True):
            branch_order = data.get("branch_order")
            if branch_order is None:
                results["missing_branch_order"].append((u, v, key))
                continue
            length = data.get("length")
            if length is None:
                results["missing_length"].append((u, v, key))
                continue
            if length <= 0:
                results["invalid_length"].append((u, v, key, length))
                continue
            diameters = diameter_by_branch_order.get(branch_order)
            if diameters is None:
                results["unknown_branch_order"].append((u, v, key, branch_order))
                continue
            if not isinstance(diameters, dict) or "d1" not in diameters or "d2" not in diameters:
                results["invalid_diameter"].append((u, v, key, branch_order))
                continue
            d1, d2 = diameters["d1"], diameters["d2"]
            if d1 <= 0 or d2 <= 0:
                results["invalid_diameter"].append((u, v, key, branch_order))
                continue
            try:
                total_resistance = self.calculate_integrated_resistance(length, d1, d2)
                weight = 1.0 / total_resistance
                G[u][v][key]["weight"] = weight
                results["weights_set"] += 1
            except Exception as e:
                logger.warning("Resistance calc failed for (%s,%s,%s): %s", u, v, key, e)
        return results

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
            return results
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
        return results
