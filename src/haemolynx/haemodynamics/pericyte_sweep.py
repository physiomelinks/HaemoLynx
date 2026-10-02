"""Pericyte dilation and inlet-pressure sweeps over a vascular network.

Repeatedly re-solves one network while dilating its vessels and/or varying the
inlet pressure, producing the flow and equivalent-resistance curves used to
compare pericyte tone between conditions.

Kept here rather than in an example because both the whole-network solve and
the sweep are generally useful, and because a numerical result belongs
somewhere it can be tested.

Every sweep grid point is solved once (:func:`solve_sweep_point`): flow is
linear in the pressure drop, so one solve at a unit drop gives every inlet
pressure by scaling (:class:`UnitPressureSolve`), and the discharge
haematocrit -- which depends on how flow divides at each junction, not on its
size -- is iterated to its fixed point for that point's geometry, as a single
re-solve does, when ``haematocrit_model`` is ``distributed_iterative``.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp

from .apply import apply_final_resistance_overrides
from .constriction_strategy import (
    constriction_strategy_kwargs,
    set_resistances_for_constriction_strategy,
)
from .haematocrit_distribution import JUNCTION_RULE_NO_SEPARATION, iterate_flow_and_haematocrit
from .poiseuille import PoiseuilleModel, scale_stored_edge_diameters
from .resistance import (
    FLOW_SOLVER_DENSE,
    boundary_flow,
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
    reachable_through_conductances,
    reachable_unknown_node_indices,
    solve_reduced_laplacian_sparse,
)
from .sweep_flows import (
    PericyteSites,
    build_sweep_flow_grid,
    edge_pericyte_sites,
    record_flows_after_solve,
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
    return unit_pressure_solve(
        conductance, node_list, inlet_nodes=inlet_nodes, outlet_nodes=outlet_nodes
    ).at(inlet_p_bc, outlet_p_bc)


@dataclass(frozen=True)
class UnitPressureSolve:
    """A network solved once, at inlets 1 Pa above the outlets.

    Flow is linear in the pressure drop and the Laplacian's rows sum to zero,
    so at inlet pressure ``p_in`` and outlet pressure ``p_out`` every node the
    boundaries reach sits at ``p_out + (p_in - p_out) * pressure`` and every
    flow scales by ``p_in - p_out``: :meth:`at` gives the same answer as a
    fresh solve at those pressures, to rounding, for the cost of a multiply.
    A node with no conductive path to a boundary stays at 0 Pa, as a solve
    leaves it.
    """

    #: Pressure at a unit drop: 1 at the inlets, 0 at the outlets and at every
    #: node the boundaries do not reach.
    pressure: np.ndarray
    #: The nodes whose pressure the boundaries set.
    reached: np.ndarray
    #: Total flow through the inlets, and through the outlets, per Pa of drop.
    inlet_flow: float
    outlet_flow: float
    #: Whether any conductive path joins an inlet to an outlet.
    connected: bool
    #: Nodes listed as both an inlet and an outlet.
    both_ends: tuple

    def at(self, inlet_p_bc: float, outlet_p_bc: float) -> dict[str, Any]:
        """Pressures, boundary flows and equivalent resistance at these pressures."""
        inlet_p_bc, outlet_p_bc = float(inlet_p_bc), float(outlet_p_bc)
        drop = inlet_p_bc - outlet_p_bc
        if self.both_ends and not np.isclose(inlet_p_bc, outlet_p_bc):
            raise ValueError(
                f"Node {self.both_ends[0]} receives conflicting BC pressures "
                f"{inlet_p_bc} and {outlet_p_bc}."
            )
        if drop != 0.0 and not self.connected:
            raise ValueError(
                "Inlet and outlet boundary nodes are not connected by "
                "conductance-carrying edges; cannot solve for flow. Check that "
                "the inlet and outlet land on the same conductive part of the "
                "network."
            )
        pressure = np.where(self.reached, outlet_p_bc + drop * self.pressure, 0.0)
        total_inlet_flow = drop * self.inlet_flow
        # Exact zero only: flows are in m^3/s and physiologically ~1e-14, so any
        # absolute tolerance would swallow every real result.
        equivalent_resistance = (
            np.inf if total_inlet_flow == 0.0 else drop / total_inlet_flow
        )
        return {
            "pressure": pressure,
            "total_inlet_flow": float(total_inlet_flow),
            "total_outlet_flow": float(drop * self.outlet_flow),
            "equivalent_resistance": float(equivalent_resistance),
        }


def unit_pressure_solve(
    conductance: np.ndarray,
    node_list: list[int],
    *,
    inlet_nodes: list[int],
    outlet_nodes: list[int],
) -> UnitPressureSolve:
    """Solve *conductance* once at a unit pressure drop -- see
    :class:`UnitPressureSolve` for how that answers every other drop.

    A ``scipy.sparse`` *conductance* (``haemodynamics_solver="sparse"``) is
    solved by sparse LU, to the same pressures to rounding."""
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
        bc_idx_to_p[node_to_idx[node_id]] = 1.0
    both_ends: list[int] = []
    for node_id in outlet_nodes:
        idx = node_to_idx[node_id]
        if bc_idx_to_p.get(idx) == 1.0:
            # Refused by `at` unless the two pressures are equal, when the node
            # sits at the one pressure either way.
            both_ends.append(node_id)
        bc_idx_to_p[idx] = 0.0

    known_idx = np.array(sorted(bc_idx_to_p.keys()), dtype=int)
    pressure[known_idx] = np.array([bc_idx_to_p[idx] for idx in known_idx], dtype=float)

    # Same policy as solve_flow_from_conductance_matrix: a node with no
    # conductive path to any boundary node gives the reduced Laplacian a
    # zero row, and solving for it anyway forced the lstsq fallback below to
    # degrade every node's pressure, not just the disconnected ones.
    adjacency = conductance > 0
    inlet_idx = np.array([node_to_idx[n] for n in inlet_nodes], dtype=int)
    outlet_idx = np.array([node_to_idx[n] for n in outlet_nodes], dtype=int)
    connected = bool(np.any(reachable_through_conductances(adjacency, inlet_idx)[outlet_idx]))
    unknown_idx, stranded_conductive = reachable_unknown_node_indices(
        conductance, known_idx
    )
    if stranded_conductive:
        logger.warning(
            f"{stranded_conductive} node(s) carry conductances but have no "
            "conductive path to any boundary node; their pressure stays 0 "
            "and their edges carry zero flow."
        )
    if unknown_idx.size and sp.issparse(laplacian):
        pressure[unknown_idx] = solve_reduced_laplacian_sparse(
            laplacian, pressure, unknown_idx, known_idx
        )
    elif unknown_idx.size:
        l_uu = laplacian[np.ix_(unknown_idx, unknown_idx)]
        l_uk = laplacian[np.ix_(unknown_idx, known_idx)]
        rhs = -l_uk @ pressure[known_idx]
        try:
            pressure[unknown_idx] = np.linalg.solve(l_uu, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Conductance network Laplacian is singular and cannot be "
                "solved. Check that inlet and outlet boundary nodes lie on "
                "the same connected, conductance-carrying part of the "
                "network and that boundary pressures differ."
            ) from exc

    def _boundary_flow(nodes: Iterable[int]) -> float:
        if sp.issparse(conductance):
            return boundary_flow(conductance, pressure, [node_to_idx[n] for n in nodes])
        total = 0.0
        for node_id in nodes:
            i = node_to_idx[node_id]
            total += float(np.sum(conductance[i, :] * (pressure[i] - pressure)))
        return total

    reached = np.zeros(n_nodes, dtype=bool)
    reached[known_idx] = True
    reached[unknown_idx] = True
    return UnitPressureSolve(
        pressure=pressure,
        reached=reached,
        inlet_flow=_boundary_flow(inlet_nodes),
        outlet_flow=_boundary_flow(outlet_nodes),
        connected=connected,
        both_ends=tuple(both_ends),
    )


def solve_sweep_point(
    G: nx.MultiGraph,
    settings: Mapping[str, Any],
    *,
    recompute_resistances: Callable[[], None],
    inlet_nodes: list[int],
    outlet_nodes: list[int],
) -> tuple[UnitPressureSolve, list[int], dict[str, Any] | None]:
    """One sweep grid point, solved the way a single re-solve is.

    *G* carries the point's own resistances, and *recompute_resistances* redoes
    them in place from each edge's ``discharge_haematocrit``. When
    ``haematocrit_model`` is ``distributed_iterative``, flow and haematocrit
    are iterated to their fixed point first
    (:func:`~haemolynx.haemodynamics.haematocrit_distribution.iterate_flow_and_haematocrit`),
    at the run's own boundary pressures: the haematocrit follows how flow
    divides at each junction, which scaling the pressure drop leaves alone,
    so the one fixed point serves every inlet pressure. Then one unit solve,
    from the resistances that fixed point left, with the run's
    ``haemodynamics_solver``.

    Returns the unit solve, its node order, and the haematocrit iteration's
    report (``None`` without one).
    """
    solver = settings.get("haemodynamics_solver") or FLOW_SOLVER_DENSE
    iterated = None
    if settings.get("haematocrit_model") == "distributed_iterative":
        outlet_p = float(settings["outlet_p_bc"])
        inlet_p = float(settings["inlet_p_bc"])
        iterated = iterate_flow_and_haematocrit(
            G,
            recompute_resistances=recompute_resistances,
            inlet_haematocrit=float(settings["haematocrit"]),
            # Any drop gives the same fixed point; a zero one gives no flow to
            # divide.
            inlet_p_bc=inlet_p if inlet_p != outlet_p else outlet_p + 1.0,
            outlet_p_bc=outlet_p,
            inlet_nodes=list(inlet_nodes),
            outlet_nodes=list(outlet_nodes),
            max_iterations=int(settings.get("haematocrit_distribution_max_iterations", 20)),
            tolerance=float(settings.get("haematocrit_distribution_tolerance", 0.01)),
            junction_rule=settings.get("haematocrit_junction_rule") or JUNCTION_RULE_NO_SEPARATION,
            solver=solver,
            # The unit solve just below is that final solve.
            final_solve=False,
        )
    conductance, node_list = build_conductance_matrix_from_graph(G, solver=solver)
    unit = unit_pressure_solve(
        conductance, list(node_list), inlet_nodes=inlet_nodes, outlet_nodes=outlet_nodes
    )
    return unit, list(node_list), iterated


def sweep_rows_at_pressures(
    unit: UnitPressureSolve,
    G: nx.MultiGraph,
    node_list: list[int],
    inlet_pressures: Sequence[float],
    outlet_pressure_pa: float,
    fixed: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    """One CSV row, and the per-vessel flows, for each inlet pressure of a grid
    point solved by :func:`solve_sweep_point`; *fixed* are the row's own
    geometry columns."""
    rows: list[dict[str, Any]] = []
    flows: list[dict[str, np.ndarray]] = []
    for inlet_pressure_pa in inlet_pressures:
        solved = unit.at(float(inlet_pressure_pa), outlet_pressure_pa)
        flows.append(record_flows_after_solve(G, node_list, solved["pressure"]))
        rows.append(
            {
                **fixed,
                "inlet_pressure_pa": inlet_pressure_pa,
                "outlet_pressure_pa": outlet_pressure_pa,
                "total_inlet_flow": solved["total_inlet_flow"],
                "total_outlet_flow": solved["total_outlet_flow"],
                "flow_balance_error": (
                    solved["total_inlet_flow"] + solved["total_outlet_flow"]
                ),
                "equivalent_resistance": solved["equivalent_resistance"],
            }
        )
    return rows, flows


def haematocrit_sweep_report(iterations: Sequence[dict[str, Any] | None]) -> dict[str, Any] | None:
    """How the grid points' haematocrit iterations went, or ``None`` when the
    sweep did not iterate it; warns about any that did not converge."""
    ran = [item for item in iterations if item is not None]
    if not ran:
        return None
    not_converged = sum(1 for item in ran if not item.get("converged"))
    if not_converged:
        logger.warning(
            f"Sweep: the discharge haematocrit did not converge at {not_converged} of "
            f"{len(ran)} grid point(s) within haematocrit_distribution_max_iterations; "
            "their flows use the last iteration's."
        )
    return {
        "points": len(ran),
        "not_converged": not_converged,
        "max_iterations_used": max(int(item.get("iterations", 0)) for item in ran),
    }


def dilate_graph_diameters(
    G: nx.MultiGraph, dilation_factor: float
) -> nx.MultiGraph:
    """A copy of *G* with every measured FWHM diameter scaled."""
    dilated = G.copy()
    for _, _, _, edge_data in dilated.edges(keys=True, data=True):
        scale_stored_edge_diameters(edge_data, dilation_factor)
    return dilated


def inclusive_int_range(
    settings: Mapping[str, Any], min_key: str, max_key: str, step_key: str
) -> tuple[int, ...]:
    """``min..max`` inclusive in steps, refusing an inverted range by name
    rather than sweeping zero points and failing later."""
    start, stop, step = (int(settings[key]) for key in (min_key, max_key, step_key))
    if stop < start:
        raise ValueError(f"{max_key} ({stop}) is below {min_key} ({start}); the sweep would be empty.")
    if step <= 0:
        raise ValueError(f"{step_key} must be positive, got {step}.")
    return tuple(range(start, stop + 1, step))


def _dilation_percents(settings: Mapping[str, Any], *, sweep: bool) -> Sequence[int]:
    """Percents to dilate by, or a single 0% when the sweep is pressure-only."""
    if not sweep:
        return (0,)
    return inclusive_int_range(
        settings,
        "pericyte_dilation_min_percent",
        "pericyte_dilation_max_percent",
        "pericyte_dilation_step_percent",
    )


def _inlet_pressures(settings: Mapping[str, Any], *, sweep: bool) -> Sequence[float]:
    """Inlet pressures to solve at, or the run's fixed ``inlet_p_bc`` alone --
    exactly, so a sweep's fixed-pressure points match the baseline."""
    if not sweep:
        return (float(settings["inlet_p_bc"]),)
    return inclusive_int_range(
        settings, "inlet_pressure_min_pa", "inlet_pressure_max_pa", "inlet_pressure_step_pa"
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
        G, _strategy, _strategy_results = set_resistances_for_constriction_strategy(
            G,
            **constriction_strategy_kwargs(
                settings, diameter_by_branch_order=scaled_diameters
            ),
        )
    else:
        G, _ = poiseuille_model.set_poiseuille_resistances(
            G,
            scaled_diameters,
            prefer_edge_fwhm_diameter=True,
        )

    apply_baseline_overrides(G, settings, poiseuille_model, custom_edge_diameter_scale=dilation_factor)
    return G


def apply_baseline_overrides(
    G: nx.MultiGraph,
    settings: Mapping[str, Any],
    poiseuille_model: PoiseuilleModel,
    *,
    custom_edge_diameter_scale: float = 1.0,
) -> None:
    """The baseline's own last resistance writes (custom edges, thick-vessel
    bridges), redone on a perturbed graph so those edges match the baseline."""
    apply_final_resistance_overrides(
        G,
        poiseuille_model,
        custom_edges=settings.get("custom_edges"),
        custom_edge_diameter=settings.get("custom_edge_diameter"),
        custom_edge_diameter_scale=custom_edge_diameter_scale,
    )


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
    recorded_sites: list[PericyteSites] = []
    iterations: list[dict[str, Any] | None] = []
    last_node_list: list[int] = []
    for dilation_percent in dilation_values:
        dilation_factor = 1.0 + (float(dilation_percent) / 100.0)
        dilated = dilate_graph_diameters(G, dilation_factor)
        scaled_diameters = {
            branch_order: float(diameter_um) * dilation_factor
            for branch_order, diameter_um in diameter_by_branch_order.items()
        }

        def recompute_resistances(
            graph=dilated, scaled=scaled_diameters, factor=dilation_factor
        ) -> None:
            _apply_sweep_resistances(
                graph,
                settings,
                scaled_diameters=scaled,
                dilation_factor=factor,
                sweep_dilation=sweep_dilation,
                poiseuille_model=poiseuille_model,
            )

        recompute_resistances()
        unit, node_list, iterated = solve_sweep_point(
            dilated,
            settings,
            recompute_resistances=recompute_resistances,
            inlet_nodes=inlet_nodes,
            outlet_nodes=outlet_nodes,
        )
        iterations.append(iterated)
        last_node_list = node_list
        rows, flows = sweep_rows_at_pressures(
            unit,
            dilated,
            node_list,
            inlet_pressures,
            outlet_pressure_pa,
            {"dilation_percent": int(dilation_percent), "dilation_factor": float(dilation_factor)},
        )
        results.extend(rows)
        recorded_flows.extend(flows)
        if sweep_dilation:
            # Only a dilation sweep places constrictions; every inlet pressure
            # at this dilation shares its sites.
            recorded_sites.extend([edge_pericyte_sites(dilated)] * len(flows))

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
        pericyte_sites=recorded_sites if sweep_dilation else None,
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
        "haematocrit_distribution": haematocrit_sweep_report(iterations),
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
    return inclusive_int_range(
        settings,
        "arteriole_dilation_min_percent",
        "arteriole_dilation_max_percent",
        "arteriole_dilation_step_percent",
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
    iterations: list[dict[str, Any] | None] = []
    last_node_list: list[int] = []
    for dilation_percent in dilation_values:
        scale = percent_change_to_scale(float(dilation_percent))
        scaled, scaled_table, _summary = scale_arteriole_diameters(
            G,
            diameter_by_branch_order,
            scale,
            model=poiseuille_model,
            prefer_edge_fwhm_diameter=prefer_measured,
        )
        apply_baseline_overrides(scaled, settings, poiseuille_model)

        def recompute_resistances(graph=scaled, table=scaled_table) -> None:
            # The diameters are already scaled on the edges; this only moves
            # resistance, from the edges' current discharge_haematocrit.
            poiseuille_model.set_poiseuille_resistances(
                graph, table, prefer_edge_fwhm_diameter=prefer_measured
            )
            apply_baseline_overrides(graph, settings, poiseuille_model)

        unit, node_list, iterated = solve_sweep_point(
            scaled,
            settings,
            recompute_resistances=recompute_resistances,
            inlet_nodes=inlet_nodes,
            outlet_nodes=outlet_nodes,
        )
        iterations.append(iterated)
        last_node_list = node_list
        rows, flows = sweep_rows_at_pressures(
            unit,
            scaled,
            node_list,
            inlet_pressures,
            outlet_pressure_pa,
            {"dilation_percent": int(dilation_percent), "dilation_factor": float(scale)},
        )
        results.extend(rows)
        recorded_flows.extend(flows)

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
        "haematocrit_distribution": haematocrit_sweep_report(iterations),
    }


# stages.py still imports the capillary sweep from this module; the
# implementation lives in capillary.py. Lazy-safe: capillary only imports
# back into this module inside the sweep function.
from .capillary import run_capillary_dilation_pressure_sweep  # noqa: E402,F401
