"""Pericyte dilation and inlet-pressure sweeps over a vascular network.

Repeatedly re-solves one network while dilating its vessels and varying the
inlet pressure, producing the flow and equivalent-resistance curves used to
compare pericyte tone between conditions.

Kept here rather than in an example because both the whole-network solve and
the sweep are generally useful, and because a numerical result belongs
somewhere it can be tested.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx
import numpy as np

from .poiseuille import PoiseuilleModel
from .resistance import (
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
)

logger = logging.getLogger(__name__)

#: Columns of the sweep CSV, in order.
SWEEP_COLUMNS = (
    "dilation_percent",
    "dilation_factor",
    "inlet_pressure_pa",
    "outlet_pressure_pa",
    "total_inlet_flow",
    "total_outlet_flow",
    "flow_balance_error",
    "equivalent_resistance",
)


def solve_pressure_and_boundary_flow(
    conductance: np.ndarray,
    node_list: list[int],
    *,
    inlet_p_bc: float,
    outlet_p_bc: float,
    inlet_nodes: list[int],
    outlet_nodes: list[int],
) -> dict[str, Any]:
    """Solve nodal pressures and total flow through the boundary node sets.

    Returns the pressure field, the flow summed over the inlets and over the
    outlets, and the network's equivalent resistance.
    """
    if not inlet_nodes:
        raise ValueError("inlet_nodes cannot be empty for flow/resistance sweep.")
    if not outlet_nodes:
        raise ValueError("outlet_nodes cannot be empty for flow/resistance sweep.")

    n_nodes = conductance.shape[0]
    if conductance.ndim != 2 or conductance.shape[1] != n_nodes:
        raise ValueError("conductance must be a square matrix.")
    if len(node_list) != n_nodes:
        raise ValueError("node_list length must match conductance matrix dimensions.")

    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    missing_start = [n for n in inlet_nodes if n not in node_to_idx]
    missing_out = [n for n in outlet_nodes if n not in node_to_idx]
    if missing_start or missing_out:
        raise ValueError(
            "Boundary nodes missing from node_list "
            f"(missing_inlet={missing_start}, missing_output={missing_out})."
        )

    laplacian = calc_laplacian_from_conductance_matrix(conductance)
    pressure = np.zeros(n_nodes, dtype=float)
    bc_idx_to_p: dict[int, float] = {}
    for node_id in inlet_nodes:
        bc_idx_to_p[node_to_idx[node_id]] = float(inlet_p_bc)
    for node_id in outlet_nodes:
        idx = node_to_idx[node_id]
        existing = bc_idx_to_p.get(idx)
        if existing is not None and not np.isclose(existing, float(outlet_p_bc)):
            raise ValueError(
                f"Node {node_id} receives conflicting BC pressures "
                f"{existing} and {outlet_p_bc}."
            )
        bc_idx_to_p[idx] = float(outlet_p_bc)

    known_idx = np.array(sorted(bc_idx_to_p.keys()), dtype=int)
    pressure[known_idx] = np.array([bc_idx_to_p[idx] for idx in known_idx], dtype=float)
    unknown_idx = np.array(
        sorted(set(range(n_nodes)).difference(set(known_idx))), dtype=int
    )
    if unknown_idx.size:
        l_uu = laplacian[np.ix_(unknown_idx, unknown_idx)]
        l_uk = laplacian[np.ix_(unknown_idx, known_idx)]
        rhs = -l_uk @ pressure[known_idx]
        try:
            pressure[unknown_idx] = np.linalg.solve(l_uu, rhs)
        except np.linalg.LinAlgError:
            pressure[unknown_idx] = np.linalg.lstsq(l_uu, rhs, rcond=None)[0]

    def _boundary_flow(nodes: Iterable[int]) -> float:
        total = 0.0
        for node_id in nodes:
            i = node_to_idx[node_id]
            total += float(np.sum(conductance[i, :] * (pressure[i] - pressure)))
        return total

    total_inlet_flow = _boundary_flow(inlet_nodes)
    total_outlet_flow = _boundary_flow(outlet_nodes)

    pressure_drop = float(inlet_p_bc - outlet_p_bc)
    # Exact zero only: flows are in m^3/s and physiologically ~1e-14, so any
    # absolute tolerance would swallow every real result.
    equivalent_resistance = (
        np.inf if total_inlet_flow == 0.0 else pressure_drop / total_inlet_flow
    )

    return {
        "pressure": pressure,
        "total_inlet_flow": float(total_inlet_flow),
        "total_outlet_flow": float(total_outlet_flow),
        "equivalent_resistance": float(equivalent_resistance),
    }


def dilate_graph_diameters(
    G: nx.MultiGraph, dilation_factor: float
) -> nx.MultiGraph:
    """A copy of *G* with every measured FWHM diameter scaled."""
    dilated = G.copy()
    for _, _, _, edge_data in dilated.edges(keys=True, data=True):
        measured = edge_data.get("fwhm_diameter_um")
        if measured is not None and float(measured) > 0:
            edge_data["fwhm_diameter_um"] = float(measured) * dilation_factor
    return dilated


def run_pericyte_dilation_pressure_sweep(
    G: nx.MultiGraph,
    settings: Mapping[str, Any],
    *,
    inlet_nodes: list[int],
    outlet_nodes: list[int],
    output_dir: Path | str,
) -> dict[str, Any]:
    """Sweep pericyte dilation against inlet pressure, writing a CSV of curves.

    Reads its sweep ranges from *settings* — ``pericyte_dilation_{min,max,step}_percent``
    and ``inlet_pressure_{min,max,step}_pa`` — along with the diameter table,
    outlet pressure and constriction geometry, so the caller states the network
    and where to write and nothing else.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    poiseuille_model = PoiseuilleModel(
        constriction_length=float(settings.get("constriction_length_um", 40.0)),
        constriction_spacing=float(settings.get("constriction_spacing_um", 100.0)),
    )
    diameter_by_branch_order = settings["diameter_by_branch_order"]
    custom_edges = settings.get("custom_edges") or []
    outlet_pressure_pa = float(settings["outlet_p_bc"])

    dilation_values = range(
        int(settings["pericyte_dilation_min_percent"]),
        int(settings["pericyte_dilation_max_percent"]) + 1,
        int(settings["pericyte_dilation_step_percent"]),
    )
    inlet_pressures = range(
        int(settings["inlet_pressure_min_pa"]),
        int(settings["inlet_pressure_max_pa"]) + 1,
        int(settings["inlet_pressure_step_pa"]),
    )

    results: list[dict[str, Any]] = []
    for dilation_percent in dilation_values:
        dilation_factor = 1.0 + (float(dilation_percent) / 100.0)
        dilated = dilate_graph_diameters(G, dilation_factor)

        dilated, _ = poiseuille_model.set_poiseuille_resistances(
            dilated,
            {
                branch_order: float(diameter_um) * dilation_factor
                for branch_order, diameter_um in diameter_by_branch_order.items()
            },
            prefer_edge_fwhm_diameter=True,
        )
        if custom_edges:
            dilated, _ = poiseuille_model.set_poiseuille_edge_resistances(
                dilated,
                custom_edges,
                edge_diameter=float(settings.get("custom_edge_diameter", 6.0))
                * dilation_factor,
            )

        conductance, node_list = build_conductance_matrix_from_graph(dilated)
        for inlet_pressure_pa in inlet_pressures:
            solved = solve_pressure_and_boundary_flow(
                conductance,
                node_list,
                inlet_p_bc=float(inlet_pressure_pa),
                outlet_p_bc=outlet_pressure_pa,
                inlet_nodes=inlet_nodes,
                outlet_nodes=outlet_nodes,
            )
            results.append(
                {
                    "dilation_percent": int(dilation_percent),
                    "dilation_factor": float(dilation_factor),
                    "inlet_pressure_pa": int(inlet_pressure_pa),
                    "outlet_pressure_pa": outlet_pressure_pa,
                    "total_inlet_flow": solved["total_inlet_flow"],
                    "total_outlet_flow": solved["total_outlet_flow"],
                    "flow_balance_error": (
                        solved["total_inlet_flow"] + solved["total_outlet_flow"]
                    ),
                    "equivalent_resistance": solved["equivalent_resistance"],
                }
            )

    csv_path = write_sweep_csv(results, output_dir / "pericyte_dilation_pressure_sweep.csv")
    logger.info(
        f"Pericyte dilation sweep: {len(results)} points "
        f"({len(list(dilation_values))} dilations x {len(list(inlet_pressures))} pressures) "
        f"-> {csv_path}"
    )
    return {"results": results, "csv_path": str(csv_path)}


def write_sweep_csv(results: list[Mapping[str, Any]], csv_path: Path | str) -> Path:
    """Write sweep *results* to *csv_path*, one row per sweep point."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SWEEP_COLUMNS))
        writer.writeheader()
        writer.writerows(results)
    return csv_path
