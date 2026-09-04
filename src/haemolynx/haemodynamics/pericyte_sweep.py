"""Pericyte dilation and inlet-pressure sweeps over a vascular network.

Repeatedly re-solves one network while dilating its vessels and/or varying the
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
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from haemolynx.io.axis_order import CANONICAL_AXIS_ORDER

from .constriction_strategy import set_resistances_for_constriction_strategy
from .poiseuille import PoiseuilleModel, scale_stored_edge_diameters
from .resistance import (
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
)
from .sweep_flows import build_sweep_flow_grid, record_flows_after_solve

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
        scale_stored_edge_diameters(edge_data, dilation_factor)
    return dilated


def _dilation_percents(settings: Mapping[str, Any], *, sweep: bool) -> Sequence[int]:
    """Percents to dilate by, or a single 0% when the sweep is pressure-only."""
    if not sweep:
        return (0,)
    return tuple(
        range(
            int(settings["pericyte_dilation_min_percent"]),
            int(settings["pericyte_dilation_max_percent"]) + 1,
            int(settings["pericyte_dilation_step_percent"]),
        )
    )


def _inlet_pressures(settings: Mapping[str, Any], *, sweep: bool) -> Sequence[int]:
    """Inlet pressures to solve at, or the run's fixed ``inlet_p_bc`` alone."""
    if not sweep:
        return (int(round(float(settings["inlet_p_bc"]))),)
    return tuple(
        range(
            int(settings["inlet_pressure_min_pa"]),
            int(settings["inlet_pressure_max_pa"]) + 1,
            int(settings["inlet_pressure_step_pa"]),
        )
    )


def _apply_sweep_resistances(
    G: nx.MultiGraph,
    settings: Mapping[str, Any],
    *,
    scaled_diameters: dict[str, float],
    dilation_factor: float,
    sweep_dilation: bool,
    poiseuille_model: PoiseuilleModel,
) -> nx.MultiGraph:
    """Resistances for one sweep grid point.

    When *sweep_dilation* is True (``pericyte_dilation_sweep`` /
    ``pressure_and_pericyte_sweep``), diameters are already dilated on *G* and
    resistances go through :func:`set_resistances_for_constriction_strategy`
    so entry length, spacing and probability settings actually change the
    numbers. Pressure-only sweeps keep uniform Poiseuille and do not place
    focal constrictions.

    Order for dilation sweeps: dilate baseline diameters first, then place and
    apply constrictions on those dilated diameters. Length/spacing/probability
    stay fixed across the grid; only dilation % (and optionally pressure) move.
    """
    if sweep_dilation:
        # Same strategy path as ``pericyte_diameter_change`` — always called
        # here when dilation is swept. ``do_pericyte_construction`` is forced
        # False on every merge and does not gate this typed pericyte path.
        configured_probability = settings.get("pericyte_constriction_probability")
        G, _strategy, _strategy_results = set_resistances_for_constriction_strategy(
            G,
            diameter_by_branch_order=scaled_diameters,
            constriction_factor_by_branch_order=settings.get(
                "constriction_by_branch_order"
            ),
            use_pericyte_mask_constriction=bool(
                settings.get("use_pericyte_mask_constriction", False)
            ),
            use_probabilistic_constriction=bool(
                settings.get("use_probabilistic_pericyte_constriction", False)
            ),
            prefer_edge_fwhm_baseline=bool(
                settings.get("use_fwhm_edge_diameters", False)
            ),
            constriction_length=float(settings.get("constriction_length_um", 40.0)),
            constriction_spacing=float(settings.get("constriction_spacing_um", 100.0)),
            viscosity_law=settings.get("viscosity_law", "pries"),
            haematocrit=float(settings.get("haematocrit", 0.45)),
            diameter_basis=settings.get("diameter_basis", "plasma_column"),
            constriction_probability=(
                1.0
                if configured_probability is None
                else float(configured_probability)
            ),
            default_constriction_factor=float(
                settings.get("pericyte_constriction_factor", 1.0)
            ),
            pericyte_mask_path=settings.get("pericyte_mask_path"),
            pericyte_mask_h5_dataset_name=settings.get(
                "pericyte_mask_h5_dataset_name"
            ),
            max_assignment_distance_um=settings.get(
                "pericyte_max_assignment_distance_um", 3.0
            ),
            min_pericyte_diameter_um=settings.get("pericyte_min_diameter_um", 5.0),
            max_pericyte_diameter_um=settings.get("pericyte_max_diameter_um", 12.0),
            axis_order=settings.get("image_axis_order", CANONICAL_AXIS_ORDER),
            seed=settings.get("pericyte_constriction_seed"),
        )
    else:
        G, _ = poiseuille_model.set_poiseuille_resistances(
            G,
            scaled_diameters,
            prefer_edge_fwhm_diameter=True,
        )

    custom_edges = settings.get("custom_edges") or []
    if custom_edges:
        G, _ = poiseuille_model.set_poiseuille_edge_resistances(
            G,
            custom_edges,
            edge_diameter=float(settings.get("custom_edge_diameter", 6.0))
            * dilation_factor,
        )
    return G


def run_pericyte_dilation_pressure_sweep(
    G: nx.MultiGraph,
    settings: Mapping[str, Any],
    *,
    inlet_nodes: list[int],
    outlet_nodes: list[int],
    output_dir: Path | str,
    sweep_dilation: bool = True,
    sweep_pressure: bool = True,
) -> dict[str, Any]:
    """Sweep dilation and/or inlet pressure, writing a CSV of curves.

    *sweep_dilation* and *sweep_pressure* choose which axes move:

    * both True — the historical combined sweep (dilation × pressure)
    * dilation only — pericyte tone at the run's fixed ``inlet_p_bc``
    * pressure only — inlet pressure on the undilated network

    When dilation is swept, each grid point dilates diameters then applies the
    existing constriction strategy (periodic / probabilistic / mask) using the
    merged *settings*, so ``constriction_length_um``, ``constriction_spacing_um``
    and probability flags change the resistances. Pressure-only sweeps stay on
    uniform Poiseuille.
    """
    if not sweep_dilation and not sweep_pressure:
        raise ValueError(
            "A sweep must vary dilation, inlet pressure, or both; got neither."
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
    outlet_pressure_pa = float(settings["outlet_p_bc"])

    dilation_values = _dilation_percents(settings, sweep=sweep_dilation)
    inlet_pressures = _inlet_pressures(settings, sweep=sweep_pressure)

    results: list[dict[str, Any]] = []
    recorded_flows: list[dict[str, np.ndarray]] = []
    last_node_list: list[int] = []
    for dilation_percent in dilation_values:
        dilation_factor = 1.0 + (float(dilation_percent) / 100.0)
        dilated = dilate_graph_diameters(G, dilation_factor)
        scaled_diameters = {
            branch_order: float(diameter_um) * dilation_factor
            for branch_order, diameter_um in diameter_by_branch_order.items()
        }
        dilated = _apply_sweep_resistances(
            dilated,
            settings,
            scaled_diameters=scaled_diameters,
            dilation_factor=dilation_factor,
            sweep_dilation=sweep_dilation,
            poiseuille_model=poiseuille_model,
        )

        conductance, node_list = build_conductance_matrix_from_graph(dilated)
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
                record_flows_after_solve(dilated, node_list, solved["pressure"])
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

    if sweep_dilation and sweep_pressure:
        csv_name = "pericyte_dilation_pressure_sweep.csv"
        label = "Pericyte dilation x pressure sweep"
        axis_names = ("dilation_percent", "inlet_pressure_pa")
        axis_values = {
            "dilation_percent": dilation_values,
            "inlet_pressure_pa": inlet_pressures,
        }
    elif sweep_dilation:
        csv_name = "pericyte_dilation_sweep.csv"
        label = "Pericyte dilation sweep"
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


def write_sweep_csv(results: list[Mapping[str, Any]], csv_path: Path | str) -> Path:
    """Write sweep *results* to *csv_path*, one row per sweep point."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SWEEP_COLUMNS))
        writer.writeheader()
        writer.writerows(results)
    return csv_path


def _arteriole_dilation_percents(
    settings: Mapping[str, Any], *, sweep: bool
) -> Sequence[int]:
    """Percents to scale arterioles by, or a single 0% when pressure-only."""
    if not sweep:
        return (0,)
    return tuple(
        range(
            int(settings["arteriole_dilation_min_percent"]),
            int(settings["arteriole_dilation_max_percent"]) + 1,
            int(settings["arteriole_dilation_step_percent"]),
        )
    )


def run_arteriole_dilation_pressure_sweep(
    G: nx.MultiGraph,
    settings: Mapping[str, Any],
    *,
    inlet_nodes: list[int],
    outlet_nodes: list[int],
    output_dir: Path | str,
    sweep_dilation: bool = True,
    sweep_pressure: bool = True,
) -> dict[str, Any]:
    """Sweep arteriole whole-branch diameter and/or inlet pressure.

    Unlike :func:`run_pericyte_dilation_pressure_sweep`, dilation here scales
    **only arteriole** branch orders (table + ``fwhm_diameter_um``), via
    :func:`~haemolynx.haemodynamics.arteriole.scale_arteriole_diameters` —
    capillaries and venules stay put, and no focal constriction sites are
    placed.

    *sweep_dilation* and *sweep_pressure* choose which axes move; both True is
    the combined arteriole×pressure sweep.
    """
    if not sweep_dilation and not sweep_pressure:
        raise ValueError(
            "A sweep must vary arteriole dilation, inlet pressure, or both; "
            "got neither."
        )

    from .arteriole import percent_change_to_scale, scale_arteriole_diameters

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

    dilation_values = _arteriole_dilation_percents(settings, sweep=sweep_dilation)
    inlet_pressures = _inlet_pressures(settings, sweep=sweep_pressure)

    results: list[dict[str, Any]] = []
    recorded_flows: list[dict[str, np.ndarray]] = []
    last_node_list: list[int] = []
    for dilation_percent in dilation_values:
        scale = percent_change_to_scale(float(dilation_percent))
        scaled, _table, _summary = scale_arteriole_diameters(
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
        csv_name = "arteriole_dilation_pressure_sweep.csv"
        label = "Arteriole dilation x pressure sweep"
        axis_names = ("dilation_percent", "inlet_pressure_pa")
        axis_values = {
            "dilation_percent": dilation_values,
            "inlet_pressure_pa": inlet_pressures,
        }
    elif sweep_dilation:
        csv_name = "arteriole_dilation_sweep.csv"
        label = "Arteriole dilation sweep"
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


# stages.py still imports the capillary sweep from this module; the
# implementation lives in capillary.py. Lazy-safe: capillary only imports
# back into this module inside the sweep function.
from .capillary import run_capillary_dilation_pressure_sweep  # noqa: E402,F401
