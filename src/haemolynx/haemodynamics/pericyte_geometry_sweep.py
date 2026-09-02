"""Pericyte geometry-axis sweeps at fixed tone and fixed inlet pressure.

Two complementary sweeps over focal-constriction geometry:

* ``pericyte_spacing_sweep`` — vary ``constriction_spacing_um``; length fixed
* ``pericyte_length_sweep`` — vary ``constriction_length_um``; spacing fixed

Both keep a single fixed ``pericyte_geometry_dilation_percent`` (same diameter
scale as ``dilate_graph_diameters`` / the pericyte dilation sweep) and solve at
the run's ``inlet_p_bc``. Resistances at each grid point go through
:func:`~haemolynx.haemodynamics.constriction_strategy.set_resistances_for_constriction_strategy`.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import networkx as nx
import numpy as np

from haemolynx.io.axis_order import CANONICAL_AXIS_ORDER

from .constriction_strategy import set_resistances_for_constriction_strategy
from .pericyte_sweep import dilate_graph_diameters, solve_pressure_and_boundary_flow
from .resistance import build_conductance_matrix_from_graph
from .sweep_flows import build_sweep_flow_grid, record_flows_after_solve

logger = logging.getLogger(__name__)

#: Columns of a geometry-axis sweep CSV, in order.
GEOMETRY_SWEEP_COLUMNS = (
    "constriction_spacing_um",
    "constriction_length_um",
    "dilation_percent",
    "dilation_factor",
    "inlet_pressure_pa",
    "outlet_pressure_pa",
    "total_inlet_flow",
    "total_outlet_flow",
    "flow_balance_error",
    "equivalent_resistance",
)

GeometryAxis = Literal["spacing", "length"]


def _um_axis_values(
    settings: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    step_key: str,
    axis_label: str,
) -> Sequence[float]:
    """Inclusive float range for a geometry axis (spacing or length)."""
    start = float(settings[min_key])
    stop = float(settings[max_key])
    step = float(settings[step_key])
    if start <= 0 or stop <= 0 or step <= 0:
        raise ValueError(
            f"{axis_label} sweep requires positive min/max/step; "
            f"got min={start}, max={stop}, step={step}."
        )
    if stop < start:
        raise ValueError(
            f"{axis_label} sweep max ({stop}) is below min ({start})."
        )
    values: list[float] = []
    index = 0
    while True:
        value = start + index * step
        if value > stop + 1e-9:
            break
        values.append(float(value))
        index += 1
        if index > 100_000:
            raise ValueError(
                f"{axis_label} sweep produced too many points; check step={step}."
            )
    if not values:
        raise ValueError(f"{axis_label} sweep produced no points.")
    return tuple(values)


def _apply_focal_constrictions(
    G: nx.MultiGraph,
    settings: Mapping[str, Any],
    *,
    scaled_diameters: Mapping[str, float],
    constriction_length: float,
    constriction_spacing: float,
) -> nx.MultiGraph:
    """Place and apply focal constrictions for one geometry grid point."""
    configured_probability = settings.get("pericyte_constriction_probability")
    G, _strategy, _results = set_resistances_for_constriction_strategy(
        G,
        diameter_by_branch_order=dict(scaled_diameters),
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
        constriction_length=float(constriction_length),
        constriction_spacing=float(constriction_spacing),
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
    return G


def write_geometry_sweep_csv(
    results: list[Mapping[str, Any]], csv_path: Path | str
) -> Path:
    """Write geometry-sweep *results* to *csv_path*, one row per point."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GEOMETRY_SWEEP_COLUMNS))
        writer.writeheader()
        writer.writerows(results)
    return csv_path


def run_pericyte_geometry_sweep(
    G: nx.MultiGraph,
    settings: Mapping[str, Any],
    *,
    inlet_nodes: list[int],
    outlet_nodes: list[int],
    output_dir: Path | str,
    sweep_axis: GeometryAxis,
) -> dict[str, Any]:
    """Sweep spacing or constriction length at fixed dilation and pressure.

    *sweep_axis* ``"spacing"`` reads ``constriction_spacing_{min,max,step}_um``
    and holds ``constriction_length_um`` fixed. ``"length"`` does the reverse.

    Fixed tone is ``pericyte_geometry_dilation_percent``: the same
    ``dilation_factor = 1 + percent/100`` used by
    :func:`~haemolynx.haemodynamics.pericyte_sweep.dilate_graph_diameters`.
    Inlet pressure is the run's ``inlet_p_bc`` (not a pressure sweep).
    """
    if sweep_axis not in ("spacing", "length"):
        raise ValueError(
            f"sweep_axis must be 'spacing' or 'length', got {sweep_axis!r}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dilation_percent = int(settings["pericyte_geometry_dilation_percent"])
    from .arteriole import percent_change_to_scale

    dilation_factor = percent_change_to_scale(float(dilation_percent))
    inlet_pressure_pa = int(round(float(settings["inlet_p_bc"])))
    outlet_pressure_pa = float(settings["outlet_p_bc"])
    diameter_by_branch_order = settings["diameter_by_branch_order"]
    scaled_diameters = {
        branch_order: float(diameter_um) * dilation_factor
        for branch_order, diameter_um in diameter_by_branch_order.items()
    }
    dilated_base = dilate_graph_diameters(G, dilation_factor)

    if sweep_axis == "spacing":
        axis_values = _um_axis_values(
            settings,
            min_key="constriction_spacing_min_um",
            max_key="constriction_spacing_max_um",
            step_key="constriction_spacing_step_um",
            axis_label="Spacing",
        )
        fixed_length = float(settings["constriction_length_um"])
        if fixed_length <= 0:
            raise ValueError(
                f"constriction_length_um must be > 0, got {fixed_length}."
            )
        csv_name = "pericyte_spacing_sweep.csv"
        label = "Pericyte spacing sweep"
    else:
        axis_values = _um_axis_values(
            settings,
            min_key="constriction_length_min_um",
            max_key="constriction_length_max_um",
            step_key="constriction_length_step_um",
            axis_label="Length",
        )
        fixed_spacing = float(settings["constriction_spacing_um"])
        if fixed_spacing <= 0:
            raise ValueError(
                f"constriction_spacing_um must be > 0, got {fixed_spacing}."
            )
        csv_name = "pericyte_length_sweep.csv"
        label = "Pericyte length sweep"

    results: list[dict[str, Any]] = []
    recorded_flows: list[dict[str, np.ndarray]] = []
    last_node_list: list[int] = []
    for axis_value in axis_values:
        if sweep_axis == "spacing":
            spacing = float(axis_value)
            length = fixed_length
        else:
            length = float(axis_value)
            spacing = fixed_spacing

        step_graph = dilated_base.copy()
        step_graph = _apply_focal_constrictions(
            step_graph,
            settings,
            scaled_diameters=scaled_diameters,
            constriction_length=length,
            constriction_spacing=spacing,
        )
        conductance, node_list = build_conductance_matrix_from_graph(step_graph)
        last_node_list = list(node_list)
        solved = solve_pressure_and_boundary_flow(
            conductance,
            node_list,
            inlet_p_bc=float(inlet_pressure_pa),
            outlet_p_bc=outlet_pressure_pa,
            inlet_nodes=inlet_nodes,
            outlet_nodes=outlet_nodes,
        )
        recorded_flows.append(
            record_flows_after_solve(step_graph, node_list, solved["pressure"])
        )
        results.append(
            {
                "constriction_spacing_um": spacing,
                "constriction_length_um": length,
                "dilation_percent": dilation_percent,
                "dilation_factor": float(dilation_factor),
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

    if sweep_axis == "spacing":
        flow_axis_name = "constriction_spacing_um"
    else:
        flow_axis_name = "constriction_length_um"
    sweep_flows = build_sweep_flow_grid(
        axis_names=(flow_axis_name,),
        axis_values={flow_axis_name: axis_values},
        recorded=recorded_flows,
        node_list=last_node_list,
    )

    csv_path = write_geometry_sweep_csv(results, output_dir / csv_name)
    logger.info(
        f"{label}: {len(results)} points at dilation={dilation_percent}% "
        f"and inlet_p_bc={inlet_pressure_pa} Pa -> {csv_path}"
    )
    return {
        "results": results,
        "csv_path": str(csv_path),
        "sweep_flows": sweep_flows,
    }
