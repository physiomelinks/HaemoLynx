"""Widen or narrow every capillary by one percentage, and re-solve for it.

Passive whole-branch scaling: every edge whose ``branch_order`` names a
capillary (``B01``, ``B02``, …) moves by the same factor. No focal
constriction sites are placed — unlike a pericyte tone change.

Two things carry a diameter and both have to move together. The branch-order
table is what :meth:`PoiseuilleModel.set_poiseuille_resistances` reads, but a
run that measured its diameters from the image has a ``fwhm_diameter_um`` on
each edge, and ``prefer_edge_fwhm_diameter`` makes that per-edge value win —
so scaling the table alone would leave every measured capillary exactly as it
was, silently, on precisely the runs whose diameters are real.

The sweep reuses the CSV / pressure-solve helpers from
:mod:`haemolynx.haemodynamics.pericyte_sweep` without changing pericyte or
arteriole paths.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from .arteriole import percent_change_to_scale
from .constriction import is_capillary_branch_order
from .poiseuille import PoiseuilleModel, scale_stored_edge_diameters
from .resistance import build_conductance_matrix_from_graph
from .sweep_flows import build_sweep_flow_grid, record_flows_after_solve

logger = logging.getLogger(__name__)

__all__ = [
    "is_capillary_branch_order",
    "percent_change_to_scale",
    "scale_capillary_diameters",
    "run_capillary_dilation_pressure_sweep",
]


def scale_capillary_diameters(
    graph: nx.MultiGraph,
    diameter_by_branch_order: Mapping[str, float],
    scale: float,
    *,
    model: PoiseuilleModel,
    prefer_edge_fwhm_diameter: bool = True,
) -> tuple[nx.MultiGraph, dict[str, float], dict[str, Any]]:
    """A copy of *graph* with every capillary scaled by *scale*, re-solved.

    *scale* is a multiplicative factor (use :func:`percent_change_to_scale`
    when the caller has a percentage). Arterioles and venules stay put; no
    focal constriction attributes are written.
    """
    scale = float(scale)
    if not scale > 0:
        raise ValueError(
            f"scale must be > 0, got {scale}. A factor of 1.0 leaves the "
            "capillary diameters as they are."
        )

    scaled_table = {
        branch_order: (
            float(diameter) * scale
            if is_capillary_branch_order(branch_order)
            else float(diameter)
        )
        for branch_order, diameter in (diameter_by_branch_order or {}).items()
    }

    scaled = graph.copy()
    capillary_edges = 0
    edges_measured = 0
    for _u, _v, _key, data in scaled.edges(keys=True, data=True):
        if not is_capillary_branch_order(data.get("branch_order")):
            continue
        capillary_edges += 1
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
            sorted(order for order in scaled_table if is_capillary_branch_order(order))
        ),
        "capillary_edges": capillary_edges,
        "edges_with_measured_diameter_scaled": edges_measured,
        "resistances": results,
    }
    logger.info(
        f"Capillary diameters scaled by {scale}: {capillary_edges} capillary "
        f"edge(s), {edges_measured} of them carrying a measured diameter"
    )
    return scaled, scaled_table, summary


def _capillary_dilation_percents(
    settings: Mapping[str, Any], *, sweep: bool
) -> Sequence[int]:
    """Percents to scale capillaries by, or a single 0% when pressure-only."""
    if not sweep:
        return (0,)
    return tuple(
        range(
            int(settings["capillary_dilation_min_percent"]),
            int(settings["capillary_dilation_max_percent"]) + 1,
            int(settings["capillary_dilation_step_percent"]),
        )
    )


def run_capillary_dilation_pressure_sweep(
    G: nx.MultiGraph,
    settings: Mapping[str, Any],
    *,
    inlet_nodes: list[int],
    outlet_nodes: list[int],
    output_dir: Path | str,
    sweep_dilation: bool = True,
    sweep_pressure: bool = True,
) -> dict[str, Any]:
    """Sweep capillary whole-branch diameter and/or inlet pressure.

    Passive dilation: every capillary (``B…``) scales together via
    :func:`scale_capillary_diameters`. Arterioles and venules stay put; no
    focal constriction sites are placed.

    *sweep_dilation* and *sweep_pressure* choose which axes move; both True is
    the combined capillary×pressure sweep.
    """
    from .pericyte_sweep import (
        _inlet_pressures,
        solve_pressure_and_boundary_flow,
        write_sweep_csv,
    )

    if not sweep_dilation and not sweep_pressure:
        raise ValueError(
            "A sweep must vary capillary dilation, inlet pressure, or both; "
            "got neither."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    poiseuille_model = PoiseuilleModel(
        constriction_length=float(settings.get("constriction_length_um", 40.0)),
        constriction_spacing=float(settings.get("constriction_spacing_um", 100.0)),
        viscosity_law=settings.get("viscosity_law", "pries"),
        haematocrit=float(settings.get("haematocrit", 0.45)),
        diameter_basis=settings.get("diameter_basis", "plasma_column"),
    )
    diameter_by_branch_order = settings["diameter_by_branch_order"]
    prefer_measured = bool(settings.get("use_fwhm_edge_diameters", True))
    outlet_pressure_pa = float(settings["outlet_p_bc"])

    dilation_values = _capillary_dilation_percents(settings, sweep=sweep_dilation)
    inlet_pressures = _inlet_pressures(settings, sweep=sweep_pressure)

    results: list[dict[str, Any]] = []
    recorded_flows: list[dict[str, np.ndarray]] = []
    last_node_list: list[int] = []
    for dilation_percent in dilation_values:
        scale = percent_change_to_scale(float(dilation_percent))
        scaled, _table, _summary = scale_capillary_diameters(
            G,
            diameter_by_branch_order,
            scale,
            model=poiseuille_model,
            prefer_edge_fwhm_diameter=prefer_measured,
        )
        conductance, node_list = build_conductance_matrix_from_graph(scaled)
        last_node_list = list(node_list)
        for inlet_pressure_pa in inlet_pressures:
            solved = solve_pressure_and_boundary_flow(
                conductance,
                node_list,
                inlet_p_bc=float(inlet_pressure_pa),
                outlet_p_bc=outlet_pressure_pa,
                inlet_nodes=inlet_nodes,
                outlet_nodes=outlet_nodes,
            )
            recorded_flows.append(
                record_flows_after_solve(scaled, node_list, solved["pressure"])
            )
            results.append(
                {
                    "dilation_percent": int(dilation_percent),
                    "dilation_factor": float(scale),
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

    if sweep_dilation and sweep_pressure:
        csv_name = "capillary_dilation_pressure_sweep.csv"
        label = "Capillary dilation x pressure sweep"
        axis_names = ("dilation_percent", "inlet_pressure_pa")
        axis_values = {
            "dilation_percent": dilation_values,
            "inlet_pressure_pa": inlet_pressures,
        }
    elif sweep_dilation:
        csv_name = "capillary_dilation_sweep.csv"
        label = "Capillary dilation sweep"
        axis_names = ("dilation_percent",)
        axis_values = {"dilation_percent": dilation_values}
    else:
        csv_name = "inlet_pressure_sweep.csv"
        label = "Inlet pressure sweep"
        axis_names = ("inlet_pressure_pa",)
        axis_values = {"inlet_pressure_pa": inlet_pressures}

    csv_path = write_sweep_csv(results, output_dir / csv_name)
    sweep_flows = build_sweep_flow_grid(
        axis_names=axis_names,
        axis_values=axis_values,
        recorded=recorded_flows,
        node_list=last_node_list,
    )
    logger.info(
        f"{label}: {len(results)} points "
        f"({len(dilation_values)} dilations x {len(inlet_pressures)} pressures) "
        f"-> {csv_path}"
    )
    return {
        "results": results,
        "csv_path": str(csv_path),
        "sweep_flows": sweep_flows,
    }
