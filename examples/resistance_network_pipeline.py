#!/usr/bin/env python3
"""ImageLynx main pipeline package."""
import logging
import sys
import inspect
import ast
import pickle
import json
import time
from datetime import datetime
import importlib.util
from pathlib import Path
from skan import csr
import tifffile
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Ensure package and sibling example modules are importable.
root_dir = Path(__file__).resolve().parents[1]
examples_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

SETTINGS_FILE_PATH = examples_dir / "resistance_pipeline_settings.py"


from ImageLynx import graph, haemodynamics, io, preprocessing, statistics, visualization
from ImageLynx.haemodynamics import arteriole_comparison as arteriole_comparison_haemodynamics
from ImageLynx.haemodynamics import capillary_comparison as capillary_comparison_haemodynamics
from ImageLynx.haemodynamics import pericyte_comparison as pericyte_comparison_haemodynamics
from ImageLynx.haemodynamics import pericyte_mask as pericyte_mask_haemodynamics
from ImageLynx.haemodynamics import probability as probability_haemodynamics
from ImageLynx.io.voxel_validation import resolve_voxel_size_xyz
from preflight import run_preflight_checklist
from resistance_pipeline_settings import *  # noqa: F403
from cached_clean_large_masks import (
    load_cleaned_mask_cache,
    save_cleaned_mask_cache,
)
from settings_persistence import (
    persist_automated_io_assignment_to_settings_file,
    persist_small_vessel_boundary_assignment_to_settings_file,
)
from wizard import run_interactive_setup_wizard


def _solve_pressure_and_boundary_flow(
    conductance: np.ndarray,
    node_list: list[int],
    input_p_bc: float,
    output_p_bc: float,
    starting_nodes: list[int],
    output_nodes: list[int],
) -> dict[str, object]:
    """Solve pressure field and aggregate source/sink flow for BC node sets."""
    if not starting_nodes:
        raise ValueError("starting_nodes cannot be empty for Alice sweep.")
    if not output_nodes:
        raise ValueError("output_nodes cannot be empty for Alice sweep.")

    n_nodes = conductance.shape[0]
    if conductance.ndim != 2 or conductance.shape[1] != n_nodes:
        raise ValueError("conductance must be a square matrix.")
    if len(node_list) != n_nodes:
        raise ValueError("node_list length must match conductance matrix dimensions.")

    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    missing_start = [n for n in starting_nodes if n not in node_to_idx]
    missing_out = [n for n in output_nodes if n not in node_to_idx]
    if missing_start or missing_out:
        raise ValueError(
            "Boundary nodes missing from node_list "
            f"(missing_starting={missing_start}, missing_output={missing_out})."
        )

    laplacian = haemodynamics.calc_laplacian_from_conductance_matrix(conductance)
    pressure = np.zeros(n_nodes, dtype=float)
    bc_idx_to_p: dict[int, float] = {}
    for node_id in starting_nodes:
        bc_idx_to_p[node_to_idx[node_id]] = float(input_p_bc)
    for node_id in output_nodes:
        idx = node_to_idx[node_id]
        existing = bc_idx_to_p.get(idx)
        if existing is not None and not np.isclose(existing, float(output_p_bc)):
            raise ValueError(
                f"Node {node_id} receives conflicting BC pressures {existing} and {output_p_bc}."
            )
        bc_idx_to_p[idx] = float(output_p_bc)

    known_idx = np.array(sorted(bc_idx_to_p.keys()), dtype=int)
    pressure[known_idx] = np.array([bc_idx_to_p[idx] for idx in known_idx], dtype=float)
    unknown_idx = np.array(sorted(set(range(n_nodes)).difference(set(known_idx))), dtype=int)
    if unknown_idx.size:
        l_uu = laplacian[np.ix_(unknown_idx, unknown_idx)]
        l_uk = laplacian[np.ix_(unknown_idx, known_idx)]
        rhs = -l_uk @ pressure[known_idx]
        try:
            pressure[unknown_idx] = np.linalg.solve(l_uu, rhs)
        except np.linalg.LinAlgError:
            pressure[unknown_idx] = np.linalg.lstsq(l_uu, rhs, rcond=None)[0]

    total_inlet_flow = 0.0
    for node_id in starting_nodes:
        i = node_to_idx[node_id]
        total_inlet_flow += float(np.sum(conductance[i, :] * (pressure[i] - pressure)))

    total_outlet_flow = 0.0
    for node_id in output_nodes:
        i = node_to_idx[node_id]
        total_outlet_flow += float(np.sum(conductance[i, :] * (pressure[i] - pressure)))

    pressure_drop = float(input_p_bc - output_p_bc)
    equivalent_resistance = np.inf if np.isclose(total_inlet_flow, 0.0) else pressure_drop / total_inlet_flow
    return {
        "pressure": pressure,
        "total_inlet_flow": float(total_inlet_flow),
        "total_outlet_flow": float(total_outlet_flow),
        "equivalent_resistance": float(equivalent_resistance),
    }


def _check_boundary_coordinate_unit_consistency(
    G: nx.MultiGraph,
    *,
    coordinate_sets: dict[str, list[tuple[float, float, float]]],
    mode: str = "warn",
    max_fraction_of_graph_diagonal: float = 0.25,
) -> None:
    """Warn/error if manual boundary coordinates look inconsistent with node units."""
    mode_norm = str(mode).strip().lower()
    if mode_norm == "off":
        return
    if mode_norm not in {"warn", "error"}:
        raise ValueError(
            "boundary coordinate unit-check mode must be one of: off, warn, error. "
            f"Got {mode!r}."
        )
    node_pos = nx.get_node_attributes(G, "pos")
    if not node_pos:
        return
    node_points = np.asarray(list(node_pos.values()), dtype=float)
    if node_points.ndim != 2 or node_points.shape[1] != 3 or node_points.shape[0] == 0:
        return
    span = np.ptp(node_points, axis=0)
    graph_diag = float(np.linalg.norm(span))
    if not np.isfinite(graph_diag) or graph_diag <= 1e-12:
        return
    threshold = float(max_fraction_of_graph_diagonal) * graph_diag
    if threshold <= 0:
        return
    problems: list[str] = []
    for role, coords in coordinate_sets.items():
        if not coords:
            continue
        pts = np.asarray(coords, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            continue
        dists = np.linalg.norm(node_points[None, :, :] - pts[:, None, :], axis=2)
        nearest = np.min(dists, axis=1)
        median_nearest = float(np.median(nearest))
        if median_nearest > threshold:
            problems.append(
                f"{role}: median nearest-node distance={median_nearest:.3f} "
                f"(threshold={threshold:.3f}, graph_diag={graph_diag:.3f})"
            )
    if not problems:
        return
    message = (
        "Boundary coordinate/unit consistency check indicates manual boundary "
        "coordinates may be in a different unit system than graph node positions. "
        + " | ".join(problems)
    )
    if mode_norm == "error":
        raise ValueError(message)
    print(f"Warning: {message}")


def _load_alicepaper_module():
    """Load AlicePaper.py from repository root."""
    module_path = root_dir / "AlicePaper.py"
    if not module_path.exists():
        raise FileNotFoundError(f"AlicePaper module not found at: {module_path}")
    spec = importlib.util.spec_from_file_location("AlicePaper", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec for: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_alice_pericyte_dilation_pressure_sweep(
    graph_with_branch_orders: nx.MultiGraph,
    *,
    diameter_by_branch_order: dict,
    starting_nodes: list[int],
    output_nodes: list[int],
    output_p_bc: float,
    output_dir: Path,
    custom_edges_for_sweep: list | None = None,
    constriction_length_um: float = ALICE_CONSTRICTION_LENGTH_UM,
    constriction_spacing_um: float = ALICE_CONSTRICTION_SPACING_UM,
    min_dilation_percent: int = 1,
    max_dilation_percent: int = 30,
    dilation_step_percent: int = 1,
    min_inlet_pressure_pa: int = 4500,
    max_inlet_pressure_pa: int = 6000,
    inlet_pressure_step_pa: int = 500,
    run_passive_capillary_diameter_beforeafter: bool = True,
    run_arteriole_dilation_sweep_plots: bool = True,
    run_passive_capillary_dilation_sweep_plots: bool = True,
    arteriole_sweep_min_dilation_percent: int = 1,
    arteriole_sweep_max_dilation_percent: int = 30,
    arteriole_sweep_dilation_step_percent: int = 1,
    arteriole_sweep_min_inlet_pressure_pa: int = 4500,
    arteriole_sweep_max_inlet_pressure_pa: int = 6000,
    arteriole_sweep_inlet_pressure_step_pa: int = 500,
    passive_capillary_sweep_min_dilation_percent: int = 1,
    passive_capillary_sweep_max_dilation_percent: int = 30,
    passive_capillary_sweep_dilation_step_percent: int = 1,
    passive_capillary_sweep_min_inlet_pressure_pa: int = 4500,
    passive_capillary_sweep_max_inlet_pressure_pa: int = 6000,
    passive_capillary_sweep_inlet_pressure_step_pa: int = 500,
    capillary_passive_dilation_percent: float | None = None,
    run_passive_arteriole_diameter_beforeafter: bool = False,
    arteriole_passive_diameter_delta_um: float = 0.0,
    run_pericyte_spacing_sweep_plots: bool = False,
    pericyte_spacing_sweep_min_um: int = 30,
    pericyte_spacing_sweep_max_um: int = 120,
    pericyte_spacing_sweep_step_um: int = 10,
    pericyte_spacing_sweep_min_inlet_pressure_pa: int = 4500,
    pericyte_spacing_sweep_max_inlet_pressure_pa: int = 6000,
    pericyte_spacing_sweep_inlet_pressure_step_pa: int = 500,
    pericyte_spacing_sweep_percent: float = 0.0,
    run_pericyte_spacing_beforeafter: bool = False,
    pericyte_beforeafter_percent: float = -20.0,
    pericyte_spacing_delta_um: float = 0.0,
) -> dict[str, object]:
    """Sweep pericyte dilation and inlet pressure; compute flow/resistance curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    custom_edges_for_sweep = [] if custom_edges_for_sweep is None else list(custom_edges_for_sweep)

    poiseuille_model = haemodynamics.PoiseuilleModel(
        constriction_length=constriction_length_um,
        constriction_spacing=constriction_spacing_um,
    )
    def _build_sweep_values(
        *,
        min_value: int,
        max_value: int,
        step_value: int,
        name: str,
    ) -> list[int]:
        step = int(step_value)
        if step <= 0:
            raise ValueError(f"{name} step must be > 0, got {step_value}.")
        start = int(min_value)
        stop = int(max_value)
        if start <= stop:
            return list(range(start, stop + 1, step))
        return list(range(start, stop - 1, -step))

    dilation_values = _build_sweep_values(
        min_value=int(min_dilation_percent),
        max_value=int(max_dilation_percent),
        step_value=int(dilation_step_percent),
        name="pericyte sweep dilation",
    )
    inlet_pressures = _build_sweep_values(
        min_value=int(min_inlet_pressure_pa),
        max_value=int(max_inlet_pressure_pa),
        step_value=int(inlet_pressure_step_pa),
        name="pericyte sweep inlet pressure",
    )

    for dilation_percent in dilation_values:
        dilation_factor = 1.0 + (float(dilation_percent) / 100.0)
        if dilation_factor <= 0.0:
            raise ValueError(
                "alice_pericyte_dilation_percent produces non-positive factor; "
                f"got percent={dilation_percent}."
            )
        G_sweep = graph_with_branch_orders.copy()
        pericyte_factor_by_branch_order = {
            branch_order: (
                dilation_factor
                if str(branch_order).startswith("B")
                else 1.0
            )
            for branch_order in diameter_by_branch_order.keys()
        }
        G_sweep, _ = poiseuille_model.set_poiseuille_resistances_with_constrictions(
            G_sweep,
            diameter_by_branch_order,
            prefer_edge_fwhm_baseline=True,
            constriction_factor_by_branch_order=pericyte_factor_by_branch_order,
        )
        if custom_edges_for_sweep:
            G_sweep, _ = poiseuille_model.set_poiseuille_edge_weights(
                G_sweep,
                custom_edges_for_sweep,
                edge_diameter=6.0 * dilation_factor,
                use_resistance=False,
            )

        conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G_sweep)
        for inlet_pressure_pa in inlet_pressures:
            solve_result = _solve_pressure_and_boundary_flow(
                conductance=conductance,
                node_list=node_list,
                input_p_bc=float(inlet_pressure_pa),
                output_p_bc=float(output_p_bc),
                starting_nodes=starting_nodes,
                output_nodes=output_nodes,
            )
            results.append(
                {
                    "dilation_percent": int(dilation_percent),
                    "dilation_factor": float(dilation_factor),
                    "inlet_pressure_pa": int(inlet_pressure_pa),
                    "outlet_pressure_pa": float(output_p_bc),
                    "total_inlet_flow": float(solve_result["total_inlet_flow"]),
                    "total_outlet_flow": float(solve_result["total_outlet_flow"]),
                    "flow_balance_error": float(
                        float(solve_result["total_inlet_flow"])
                        + float(solve_result["total_outlet_flow"])
                    ),
                    "equivalent_resistance": float(solve_result["equivalent_resistance"]),
                }
            )

    sweep_csv_path = output_dir / "alice_pericyte_dilation_pressure_sweep.csv"
    header = [
        "dilation_percent",
        "dilation_factor",
        "inlet_pressure_pa",
        "outlet_pressure_pa",
        "total_inlet_flow",
        "total_outlet_flow",
        "flow_balance_error",
        "equivalent_resistance",
    ]
    with sweep_csv_path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved Alice pressure+dilation sweep CSV to: {sweep_csv_path}")

    alicepaper = _load_alicepaper_module()
    plot_outputs = alicepaper.graph(results, output_dir=output_dir)
    arteriole_sweep_plot_outputs = None
    passive_capillary_sweep_plot_outputs = None
    pericyte_spacing_sweep_plot_outputs = None

    def _run_selective_dilation_sweep(
        *,
        branch_prefix: str,
        record_key: str,
        dilation_values_local: list[int],
        inlet_pressures_local: list[int],
    ) -> list[dict[str, object]]:
        selective_results: list[dict[str, object]] = []
        for dilation_percent in dilation_values_local:
            dilation_factor = 1.0 + (float(dilation_percent) / 100.0)
            G_sweep = graph_with_branch_orders.copy()
            for _, _, _, edge_data in G_sweep.edges(keys=True, data=True):
                branch_order = str(edge_data.get("branch_order", ""))
                if not branch_order.startswith(branch_prefix):
                    continue
                fwhm_d = edge_data.get("fwhm_diameter_um")
                if fwhm_d is not None and float(fwhm_d) > 0:
                    edge_data["fwhm_diameter_um"] = float(fwhm_d) * dilation_factor

            scaled_diameter_by_branch_order: dict[object, float] = {}
            for branch_order, diameter_um in diameter_by_branch_order.items():
                bo = str(branch_order)
                if bo.startswith(branch_prefix):
                    scaled_diameter_by_branch_order[branch_order] = (
                        float(diameter_um) * dilation_factor
                    )
                else:
                    scaled_diameter_by_branch_order[branch_order] = float(diameter_um)

            G_sweep, _ = poiseuille_model.set_poiseuille_resistances(
                G_sweep,
                scaled_diameter_by_branch_order,
                prefer_edge_fwhm_diameter=True,
            )
            if custom_edges_for_sweep:
                G_sweep, _ = poiseuille_model.set_poiseuille_edge_weights(
                    G_sweep,
                    custom_edges_for_sweep,
                    edge_diameter=6.0 * dilation_factor,
                    use_resistance=False,
                )

            conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G_sweep)
            for inlet_pressure_pa in inlet_pressures_local:
                solve_result = _solve_pressure_and_boundary_flow(
                    conductance=conductance,
                    node_list=node_list,
                    input_p_bc=float(inlet_pressure_pa),
                    output_p_bc=float(output_p_bc),
                    starting_nodes=starting_nodes,
                    output_nodes=output_nodes,
                )
                selective_results.append(
                    {
                        record_key: int(dilation_percent),
                        "dilation_percent": int(dilation_percent),
                        "dilation_factor": float(dilation_factor),
                        "inlet_pressure_pa": int(inlet_pressure_pa),
                        "outlet_pressure_pa": float(output_p_bc),
                        "total_inlet_flow": float(solve_result["total_inlet_flow"]),
                        "total_outlet_flow": float(solve_result["total_outlet_flow"]),
                        "flow_balance_error": float(
                            float(solve_result["total_inlet_flow"])
                            + float(solve_result["total_outlet_flow"])
                        ),
                        "equivalent_resistance": float(solve_result["equivalent_resistance"]),
                    }
                )
        return selective_results

    if run_arteriole_dilation_sweep_plots:
        arteriole_dilation_values = _build_sweep_values(
            min_value=int(arteriole_sweep_min_dilation_percent),
            max_value=int(arteriole_sweep_max_dilation_percent),
            step_value=int(arteriole_sweep_dilation_step_percent),
            name="arteriole sweep dilation",
        )
        arteriole_inlet_pressures = _build_sweep_values(
            min_value=int(arteriole_sweep_min_inlet_pressure_pa),
            max_value=int(arteriole_sweep_max_inlet_pressure_pa),
            step_value=int(arteriole_sweep_inlet_pressure_step_pa),
            name="arteriole sweep inlet pressure",
        )
        arteriole_records = _run_selective_dilation_sweep(
            branch_prefix="Art",
            record_key="arteriole_dilation_percent",
            dilation_values_local=arteriole_dilation_values,
            inlet_pressures_local=arteriole_inlet_pressures,
        )
        arteriole_sweep_csv_path = output_dir / "alice_arteriole_dilation_pressure_sweep.csv"
        with arteriole_sweep_csv_path.open("w", encoding="utf-8", newline="") as handle:
            import csv

            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "arteriole_dilation_percent",
                    "dilation_percent",
                    "dilation_factor",
                    "inlet_pressure_pa",
                    "outlet_pressure_pa",
                    "total_inlet_flow",
                    "total_outlet_flow",
                    "flow_balance_error",
                    "equivalent_resistance",
                ],
            )
            writer.writeheader()
            writer.writerows(arteriole_records)
        arteriole_sweep_plot_outputs = alicepaper.graph_arteriole_dilation(
            arteriole_records,
            output_dir=output_dir,
        )

    if run_passive_capillary_dilation_sweep_plots:
        passive_capillary_dilation_values = _build_sweep_values(
            min_value=int(passive_capillary_sweep_min_dilation_percent),
            max_value=int(passive_capillary_sweep_max_dilation_percent),
            step_value=int(passive_capillary_sweep_dilation_step_percent),
            name="passive capillary sweep dilation",
        )
        passive_capillary_inlet_pressures = _build_sweep_values(
            min_value=int(passive_capillary_sweep_min_inlet_pressure_pa),
            max_value=int(passive_capillary_sweep_max_inlet_pressure_pa),
            step_value=int(passive_capillary_sweep_inlet_pressure_step_pa),
            name="passive capillary sweep inlet pressure",
        )
        capillary_records = _run_selective_dilation_sweep(
            branch_prefix="B",
            record_key="capillary_dilation_percent",
            dilation_values_local=passive_capillary_dilation_values,
            inlet_pressures_local=passive_capillary_inlet_pressures,
        )
        capillary_sweep_csv_path = (
            output_dir / "alice_passive_capillary_dilation_pressure_sweep.csv"
        )
        with capillary_sweep_csv_path.open("w", encoding="utf-8", newline="") as handle:
            import csv

            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "capillary_dilation_percent",
                    "dilation_percent",
                    "dilation_factor",
                    "inlet_pressure_pa",
                    "outlet_pressure_pa",
                    "total_inlet_flow",
                    "total_outlet_flow",
                    "flow_balance_error",
                    "equivalent_resistance",
                ],
            )
            writer.writeheader()
            writer.writerows(capillary_records)
        passive_capillary_sweep_plot_outputs = alicepaper.graph_passive_capillary_dilation(
            capillary_records,
            output_dir=output_dir,
        )
    if run_pericyte_spacing_sweep_plots:
        pericyte_spacing_values = _build_sweep_values(
            min_value=int(pericyte_spacing_sweep_min_um),
            max_value=int(pericyte_spacing_sweep_max_um),
            step_value=int(pericyte_spacing_sweep_step_um),
            name="pericyte spacing sweep",
        )
        pericyte_spacing_inlet_pressures = _build_sweep_values(
            min_value=int(pericyte_spacing_sweep_min_inlet_pressure_pa),
            max_value=int(pericyte_spacing_sweep_max_inlet_pressure_pa),
            step_value=int(pericyte_spacing_sweep_inlet_pressure_step_pa),
            name="pericyte spacing sweep inlet pressure",
        )
        pericyte_spacing_factor = 1.0 + (float(pericyte_spacing_sweep_percent) / 100.0)
        if pericyte_spacing_factor <= 0.0:
            raise ValueError(
                "pericyte_spacing_sweep_percent produces non-positive factor; "
                f"got percent={pericyte_spacing_sweep_percent}."
            )
        pericyte_factor_by_branch_order = {
            branch_order: (
                pericyte_spacing_factor
                if str(branch_order).startswith("B")
                else 1.0
            )
            for branch_order in diameter_by_branch_order.keys()
        }
        pericyte_spacing_records: list[dict[str, object]] = []
        for spacing_um in pericyte_spacing_values:
            spacing_model = haemodynamics.PoiseuilleModel(
                constriction_length=float(constriction_length_um),
                constriction_spacing=float(spacing_um),
            )
            G_sweep = graph_with_branch_orders.copy()
            G_sweep, _ = spacing_model.set_poiseuille_resistances_with_constrictions(
                G_sweep,
                diameter_by_branch_order,
                prefer_edge_fwhm_baseline=True,
                constriction_factor_by_branch_order=pericyte_factor_by_branch_order,
            )
            conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G_sweep)
            for inlet_pressure_pa in pericyte_spacing_inlet_pressures:
                solve_result = _solve_pressure_and_boundary_flow(
                    conductance=conductance,
                    node_list=node_list,
                    input_p_bc=float(inlet_pressure_pa),
                    output_p_bc=float(output_p_bc),
                    starting_nodes=starting_nodes,
                    output_nodes=output_nodes,
                )
                pericyte_spacing_records.append(
                    {
                        "pericyte_spacing_um": float(spacing_um),
                        "pericyte_percent": float(pericyte_spacing_sweep_percent),
                        "inlet_pressure_pa": int(inlet_pressure_pa),
                        "outlet_pressure_pa": float(output_p_bc),
                        "total_inlet_flow": float(solve_result["total_inlet_flow"]),
                        "total_outlet_flow": float(solve_result["total_outlet_flow"]),
                        "flow_balance_error": float(
                            float(solve_result["total_inlet_flow"])
                            + float(solve_result["total_outlet_flow"])
                        ),
                        "equivalent_resistance": float(solve_result["equivalent_resistance"]),
                    }
                )
        pericyte_spacing_sweep_csv_path = output_dir / "alice_pericyte_spacing_pressure_sweep.csv"
        with pericyte_spacing_sweep_csv_path.open("w", encoding="utf-8", newline="") as handle:
            import csv

            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "pericyte_spacing_um",
                    "pericyte_percent",
                    "inlet_pressure_pa",
                    "outlet_pressure_pa",
                    "total_inlet_flow",
                    "total_outlet_flow",
                    "flow_balance_error",
                    "equivalent_resistance",
                ],
            )
            writer.writeheader()
            writer.writerows(pericyte_spacing_records)
        pericyte_spacing_sweep_plot_outputs = alicepaper.graph_pericyte_spacing(
            pericyte_spacing_records,
            output_dir=output_dir,
        )
    else:
        print(
            "Skipping Alice pericyte-spacing sweep plots "
            "(run_pericyte_spacing_sweep_plots=False)."
        )
    capillary_flow_change_outputs = None
    arteriole_flow_change_outputs = None
    pericyte_spacing_beforeafter_outputs = None
    if run_passive_capillary_diameter_beforeafter:
        capillary_percent = (
            float(max_dilation_percent)
            if capillary_passive_dilation_percent is None
            else float(capillary_passive_dilation_percent)
        )
        capillary_flow_change_outputs = alicepaper.passive_capillary_diameter_beforeafter(
            graph_with_branch_orders=graph_with_branch_orders,
            diameter_by_branch_order=diameter_by_branch_order,
            poiseuille_model=poiseuille_model,
            solve_pressure_and_boundary_flow=_solve_pressure_and_boundary_flow,
            starting_nodes=starting_nodes,
            output_nodes=output_nodes,
            inlet_pressures_pa=inlet_pressures,
            output_p_bc=output_p_bc,
            capillary_dilation_percent=capillary_percent,
            output_dir=output_dir,
        )
    if run_passive_arteriole_diameter_beforeafter:
        arteriole_flow_change_outputs = alicepaper.passive_arteriole_diameter_beforeafter(
            graph_with_branch_orders=graph_with_branch_orders,
            diameter_by_branch_order=diameter_by_branch_order,
            poiseuille_model=poiseuille_model,
            solve_pressure_and_boundary_flow=_solve_pressure_and_boundary_flow,
            starting_nodes=starting_nodes,
            output_nodes=output_nodes,
            inlet_pressures_pa=inlet_pressures,
            output_p_bc=output_p_bc,
            arteriole_diameter_delta_um=float(arteriole_passive_diameter_delta_um),
            output_dir=output_dir,
        )
    if run_pericyte_spacing_beforeafter:
        pericyte_spacing_beforeafter_outputs = (
            alicepaper.pericyte_constriction_dilation_spacing_beforeafter(
                graph_with_branch_orders=graph_with_branch_orders,
                diameter_by_branch_order=diameter_by_branch_order,
                solve_pressure_and_boundary_flow=_solve_pressure_and_boundary_flow,
                starting_nodes=starting_nodes,
                output_nodes=output_nodes,
                inlet_pressures_pa=inlet_pressures,
                output_p_bc=output_p_bc,
                pericyte_percent=float(pericyte_beforeafter_percent),
                pericyte_spacing_delta_um=float(pericyte_spacing_delta_um),
                output_dir=output_dir,
                constriction_length_um=float(constriction_length_um),
                baseline_constriction_spacing_um=float(constriction_spacing_um),
            )
        )
    print(f"Alice paper curve plots saved to: {plot_outputs}")
    print("Alice pericyte-dilation sweep plot mode: mode=d1d2")
    if capillary_flow_change_outputs is not None:
        print(
            "Alice capillary-only passive dilation flow-change plot saved to: "
            f"{capillary_flow_change_outputs}"
        )
    if arteriole_sweep_plot_outputs is not None:
        print(
            "Alice arteriole-only dilation sweep plots saved to: "
            f"{arteriole_sweep_plot_outputs}"
        )
        print("Alice arteriole-only dilation sweep plot mode: mode=passive")
    if passive_capillary_sweep_plot_outputs is not None:
        print(
            "Alice passive capillary-only dilation sweep plots saved to: "
            f"{passive_capillary_sweep_plot_outputs}"
        )
        print("Alice passive capillary-only dilation sweep plot mode: mode=passive")
    if pericyte_spacing_sweep_plot_outputs is not None:
        print(
            "Alice pericyte-spacing sweep plots saved to: "
            f"{pericyte_spacing_sweep_plot_outputs}"
        )
        print("Alice pericyte-spacing sweep plot mode: mode=d1d2")
    if arteriole_flow_change_outputs is not None:
        print(
            "Alice arteriole-only passive diameter change flow-change plot saved to: "
            f"{arteriole_flow_change_outputs}"
        )
        print("Alice arteriole passive before/after plot mode: mode=passive")
    if pericyte_spacing_beforeafter_outputs is not None:
        print(
            "Alice pericyte before/after spacing comparison plots saved to: "
            f"{pericyte_spacing_beforeafter_outputs}"
        )
        print("Alice pericyte spacing before/after plot mode: mode=d1d2")
    return {
        "results": results,
        "csv_path": str(sweep_csv_path),
        "plot_outputs": plot_outputs,
        "arteriole_sweep_plot_outputs": arteriole_sweep_plot_outputs,
        "passive_capillary_sweep_plot_outputs": passive_capillary_sweep_plot_outputs,
        "pericyte_spacing_sweep_plot_outputs": pericyte_spacing_sweep_plot_outputs,
        "capillary_flow_change_outputs": capillary_flow_change_outputs,
        "arteriole_flow_change_outputs": arteriole_flow_change_outputs,
        "pericyte_spacing_beforeafter_outputs": pericyte_spacing_beforeafter_outputs,
    }


def image_to_model_pipeline(image_path=INPUT_PATH,
                            use_ilastik_segmentation=USE_ILASTIK_SEGMENTATION,
                            ilastik_unsegmented_image_path=ILASTIK_UNSEGMENTED_IMAGE_PATH,
                            ilastik_classifier_path=ILASTIK_CLASSIFIER_PATH,
                            ilastik_executable=ILASTIK_EXECUTABLE,
                            ilastik_output_dir=ILASTIK_OUTPUT_DIR,
                            ilastik_output_suffix=ILASTIK_OUTPUT_SUFFIX,
                            voxel_size_override_xyz_px_per_um=VOXEL_SIZE_OVERRIDE_XYZ_PX_PER_UM,
                            voxel_size_policy=VOXEL_SIZE_POLICY,
                            use_large_vessel_masks=USE_LARGE_VESSEL_MASKS,
                            use_ilastik_large_vessel_segmentation=USE_ILASTIK_LARGE_VESSEL_SEGMENTATION,
                            large_vessel_mask_dilation_microns=LARGE_VESSEL_MASK_DILATION_MICRONS,
                            large_vessel_min_component_volume_um3=LARGE_VESSEL_MIN_COMPONENT_VOLUME_UM3,
                            large_vessel_remove_small_opposite_attached_components=LARGE_VESSEL_REMOVE_SMALL_OPPOSITE_ATTACHED_COMPONENTS,
                            large_vessel_opposite_attached_max_component_volume_um3=LARGE_VESSEL_OPPOSITE_ATTACHED_MAX_COMPONENT_VOLUME_UM3,
                            large_vessel_opposite_attached_max_distance_microns=LARGE_VESSEL_OPPOSITE_ATTACHED_MAX_DISTANCE_MICRONS,
                            use_small_vessel_masks_for_boundary_assignment=USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT,
                            use_ilastik_small_vessel_segmentation=USE_ILASTIK_SMALL_VESSEL_SEGMENTATION,
                            small_vessel_mask_min_overlap_fraction=SMALL_VESSEL_MASK_MIN_OVERLAP_FRACTION,
                            small_vessel_mask_dilation_microns=SMALL_VESSEL_MASK_DILATION_MICRONS,
                            small_vessel_min_component_volume_um3=SMALL_VESSEL_MIN_COMPONENT_VOLUME_UM3,
                            small_vessel_boundary_assignment_fast_mode=SMALL_VESSEL_BOUNDARY_ASSIGNMENT_FAST_MODE,
                            small_vessel_boundary_assignment_apply_overlap_cleanup_in_normal_mode=SMALL_VESSEL_BOUNDARY_ASSIGNMENT_APPLY_OVERLAP_CLEANUP_IN_NORMAL_MODE,
                            small_vessel_boundary_assignment_enable_overlap_cleanup=SMALL_VESSEL_BOUNDARY_ASSIGNMENT_ENABLE_OVERLAP_CLEANUP,
                            small_vessel_overlap_parallel_workers=SMALL_VESSEL_OVERLAP_PARALLEL_WORKERS,
                            small_vessel_mask_continuity_enable=SMALL_VESSEL_MASK_CONTINUITY_ENABLE,
                            small_vessel_mask_continuity_allow_small_to_large=SMALL_VESSEL_MASK_CONTINUITY_ALLOW_SMALL_TO_LARGE,
                            small_vessel_mask_continuity_allow_small_to_small=SMALL_VESSEL_MASK_CONTINUITY_ALLOW_SMALL_TO_SMALL,
                            small_vessel_mask_continuity_enforce_cylinder_only=SMALL_VESSEL_MASK_CONTINUITY_ENFORCE_CYLINDER_ONLY,
                            small_vessel_mask_continuity_min_cylindricality=SMALL_VESSEL_MASK_CONTINUITY_MIN_CYLINDRICALITY,
                            small_vessel_mask_continuity_max_axis_angle_degrees=SMALL_VESSEL_MASK_CONTINUITY_MAX_AXIS_ANGLE_DEGREES,
                            small_vessel_mask_continuity_min_facing_cosine=SMALL_VESSEL_MASK_CONTINUITY_MIN_FACING_COSINE,
                            small_vessel_mask_continuity_max_radius_ratio=SMALL_VESSEL_MASK_CONTINUITY_MAX_RADIUS_RATIO,
                            small_vessel_mask_continuity_max_bridge_distance_microns=SMALL_VESSEL_MASK_CONTINUITY_MAX_BRIDGE_DISTANCE_MICRONS,
                            small_vessel_mask_continuity_corridor_max_distance_microns=SMALL_VESSEL_MASK_CONTINUITY_CORRIDOR_MAX_DISTANCE_MICRONS,
                            small_vessel_mask_continuity_opposite_exclusion_distance_microns=SMALL_VESSEL_MASK_CONTINUITY_OPPOSITE_EXCLUSION_DISTANCE_MICRONS,
                            small_vessel_tangential_redefinition_enable=SMALL_VESSEL_TANGENTIAL_REDEFINITION_ENABLE,
                            small_vessel_tangential_redefinition_max_contact_distance_microns=SMALL_VESSEL_TANGENTIAL_REDEFINITION_MAX_CONTACT_DISTANCE_MICRONS,
                            small_vessel_tangential_redefinition_touch_distance_microns=SMALL_VESSEL_TANGENTIAL_REDEFINITION_TOUCH_DISTANCE_MICRONS,
                            small_vessel_tangential_redefinition_tangency_cosine_max=SMALL_VESSEL_TANGENTIAL_REDEFINITION_TANGENCY_COSINE_MAX,
                            small_vessel_tangential_redefinition_margin=SMALL_VESSEL_TANGENTIAL_REDEFINITION_MARGIN,
                            small_vessel_tangential_redefinition_parallel_workers=SMALL_VESSEL_TANGENTIAL_REDEFINITION_PARALLEL_WORKERS,
                            use_gpu_mask_continuity_acceleration=USE_GPU_MASK_CONTINUITY_ACCELERATION,
                            small_vessel_sandwich_reassign_enable=SMALL_VESSEL_SANDWICH_REASSIGN_ENABLE,
                            small_vessel_sandwich_reassign_max_endpoint_distance_microns=SMALL_VESSEL_SANDWICH_REASSIGN_MAX_ENDPOINT_DISTANCE_MICRONS,
                            small_vessel_sandwich_reassign_min_facing_cosine=SMALL_VESSEL_SANDWICH_REASSIGN_MIN_FACING_COSINE,
                            small_vessel_sandwich_reassign_max_axis_angle_degrees=SMALL_VESSEL_SANDWICH_REASSIGN_MAX_AXIS_ANGLE_DEGREES,
                            small_vessel_3d_volume_downsample_stride=SMALL_VESSEL_3D_VOLUME_DOWNSAMPLE_STRIDE,
                            small_vessel_boundary_fallback_to_hop_distance=SMALL_VESSEL_BOUNDARY_FALLBACK_TO_HOP_DISTANCE,
                            small_vessel_boundary_fallback_hop_distance=SMALL_VESSEL_BOUNDARY_FALLBACK_HOP_DISTANCE,
                            write_small_vessel_boundary_labelling_3d_html=WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML,
                            automated_vessel_assignment=AUTOMATED_VESSEL_ASSIGNMENT,
                            automated_vessel_assignment_fast_mode=AUTOMATED_VESSEL_ASSIGNMENT_FAST_MODE,
                            automated_vessel_assignment_apply_overlap_cleanup_in_normal_mode=AUTOMATED_VESSEL_ASSIGNMENT_APPLY_OVERLAP_CLEANUP_IN_NORMAL_MODE,
                            automated_vessel_assignment_enable_overlap_cleanup=AUTOMATED_VESSEL_ASSIGNMENT_ENABLE_OVERLAP_CLEANUP,
                            automated_vessel_overlap_parallel_workers=AUTOMATED_VESSEL_OVERLAP_PARALLEL_WORKERS,
                            automated_vessel_assignment_use_legacy_mode=AUTOMATED_VESSEL_ASSIGNMENT_USE_LEGACY_MODE,
                            automated_vessel_confidence_margin=AUTOMATED_VESSEL_CONFIDENCE_MARGIN,
                            automated_vessel_min_confidence=AUTOMATED_VESSEL_MIN_CONFIDENCE,
                            automated_vessel_topology_penalty=AUTOMATED_VESSEL_TOPOLOGY_PENALTY,
                            automated_vessel_quality_max_overlap_fraction=AUTOMATED_VESSEL_QUALITY_MAX_OVERLAP_FRACTION,
                            automated_vessel_quality_min_terminal_coverage=AUTOMATED_VESSEL_QUALITY_MIN_TERMINAL_COVERAGE,
                            automated_vessel_quality_max_component_count=AUTOMATED_VESSEL_QUALITY_MAX_COMPONENT_COUNT,
                            automated_vessel_conservative_max_dilation_microns=AUTOMATED_VESSEL_CONSERVATIVE_MAX_DILATION_MICRONS,
                            large_vessel_3d_volume_downsample_stride=LARGE_VESSEL_3D_VOLUME_DOWNSAMPLE_STRIDE,
                            write_fast_mode_preassignment_large_vessel_debug_3d_html=WRITE_FAST_MODE_PREASSIGNMENT_LARGE_VESSEL_DEBUG_3D_HTML,
                            auto_persist_automated_io_assignment_to_settings=AUTO_PERSIST_AUTOMATED_IO_ASSIGNMENT_TO_SETTINGS,
                            auto_persist_small_vessel_boundary_assignment_to_settings=AUTO_PERSIST_SMALL_VESSEL_BOUNDARY_ASSIGNMENT_TO_SETTINGS,
                            large_arteriole_mask_path=LARGE_ARTERIOLE_MASK_PATH,
                            large_venule_mask_path=LARGE_VENULE_MASK_PATH,
                            small_arteriole_mask_path=SMALL_ARTERIOLE_MASK_PATH,
                            small_venule_mask_path=SMALL_VENULE_MASK_PATH,
                            ilastik_unsegmented_arteriole_image_path=ILASTIK_UNSEGMENTED_ARTERIOLE_IMAGE_PATH,
                            ilastik_unsegmented_venule_image_path=ILASTIK_UNSEGMENTED_VENULE_IMAGE_PATH,
                            ilastik_arteriole_classifier_path=ILASTIK_ARTERIOLE_CLASSIFIER_PATH,
                            ilastik_venule_classifier_path=ILASTIK_VENULE_CLASSIFIER_PATH,
                            ilastik_unsegmented_small_arteriole_image_path=ILASTIK_UNSEGMENTED_SMALL_ARTERIOLE_IMAGE_PATH,
                            ilastik_unsegmented_small_venule_image_path=ILASTIK_UNSEGMENTED_SMALL_VENULE_IMAGE_PATH,
                            ilastik_small_arteriole_classifier_path=ILASTIK_SMALL_ARTERIOLE_CLASSIFIER_PATH,
                            ilastik_small_venule_classifier_path=ILASTIK_SMALL_VENULE_CLASSIFIER_PATH,
                            diameter_by_branch_order=DIAMETER_BY_BRANCH_ORDER,
                            constriction_by_branch_order=CONSTRICTION_BY_BRANCH_ORDER,
                            do_pericyte_constriction=DO_PERICYTE_CONSTRUCTION,
                            use_pericyte_mask_constriction=USE_PERICYTE_MASK_CONSTRICTION,
                            pericyte_mask_path=PERICYTE_MASK_PATH,
                            pericyte_mask_h5_dataset_name=PERICYTE_MASK_H5_DATASET_NAME,
                            pericyte_max_assignment_distance_um=PERICYTE_MAX_ASSIGNMENT_DISTANCE_UM,
                            pericyte_min_diameter_um=PERICYTE_MIN_DIAMETER_UM,
                            pericyte_max_diameter_um=PERICYTE_MAX_DIAMETER_UM,
                            use_probabilistic_pericyte_constriction=USE_PROBABILISTIC_PERICYTE_CONSTRICTION,
                            pericyte_constriction_probability=PERICYTE_CONSTRICTION_PROBABILITY,
                            pericyte_constriction_length_um=PERICYTE_CONSTRICTION_LENGTH_UM,
                            pericyte_constriction_spacing_um=PERICYTE_CONSTRICTION_SPACING_UM,
                            run_pericyte_resistance_comparison=RUN_PERICYTE_RESISTANCE_COMPARISON,
                            pericyte_comparison_baseline_value=PERICYTE_COMPARISON_BASELINE_VALUE,
                            pericyte_comparison_constricted_value=PERICYTE_COMPARISON_CONSTRICTED_VALUE,
                            run_arteriole_resistance_comparison=RUN_ARTERIOLE_RESISTANCE_COMPARISON,
                            arteriole_comparison_baseline_value=ARTERIOLE_COMPARISON_BASELINE_VALUE,
                            arteriole_comparison_dilated_value=ARTERIOLE_COMPARISON_DILATED_VALUE,
                            arteriole_comparison_branch_prefix=ARTERIOLE_COMPARISON_BRANCH_PREFIX,
                            arteriole_comparison_use_constriction_integrator=ARTERIOLE_COMPARISON_USE_CONSTRICTION_INTEGRATOR,
                            run_capillary_resistance_comparison=RUN_CAPILLARY_RESISTANCE_COMPARISON,
                            capillary_comparison_baseline_value=CAPILLARY_COMPARISON_BASELINE_VALUE,
                            capillary_comparison_dilated_value=CAPILLARY_COMPARISON_DILATED_VALUE,
                            capillary_comparison_branch_prefix=CAPILLARY_COMPARISON_BRANCH_PREFIX,
                            capillary_comparison_use_constriction_integrator=CAPILLARY_COMPARISON_USE_CONSTRICTION_INTEGRATOR,
                            reuse_comparison_pericyte_cohort_for_main_run=REUSE_COMPARISON_PERICYTE_COHORT_FOR_MAIN_RUN,
                            run_alice_paper_sweep=RUN_ALICE_PAPER_SWEEP,
                            alice_paper_output_dir=ALICE_PAPER_OUTPUT_DIR,
                            alice_pericyte_dilation_min_percent=ALICE_PERICYTE_DILATION_MIN_PERCENT,
                            alice_pericyte_dilation_max_percent=ALICE_PERICYTE_DILATION_MAX_PERCENT,
                            alice_pericyte_dilation_step_percent=ALICE_PERICYTE_DILATION_STEP_PERCENT,
                            alice_inlet_pressure_min_pa=ALICE_INLET_PRESSURE_MIN_PA,
                            alice_inlet_pressure_max_pa=ALICE_INLET_PRESSURE_MAX_PA,
                            alice_inlet_pressure_step_pa=ALICE_INLET_PRESSURE_STEP_PA,
                            run_alice_arteriole_dilation_sweep_plots=RUN_ALICE_ARTERIOLE_DILATION_SWEEP_PLOTS,
                            alice_arteriole_sweep_min_dilation_percent=ALICE_ARTERIOLE_SWEEP_MIN_DILATION_PERCENT,
                            alice_arteriole_sweep_max_dilation_percent=ALICE_ARTERIOLE_SWEEP_MAX_DILATION_PERCENT,
                            alice_arteriole_sweep_dilation_step_percent=ALICE_ARTERIOLE_SWEEP_DILATION_STEP_PERCENT,
                            alice_arteriole_sweep_min_inlet_pressure_pa=ALICE_ARTERIOLE_SWEEP_MIN_INLET_PRESSURE_PA,
                            alice_arteriole_sweep_max_inlet_pressure_pa=ALICE_ARTERIOLE_SWEEP_MAX_INLET_PRESSURE_PA,
                            alice_arteriole_sweep_inlet_pressure_step_pa=ALICE_ARTERIOLE_SWEEP_INLET_PRESSURE_STEP_PA,
                            run_alice_passive_capillary_dilation_sweep_plots=RUN_ALICE_PASSIVE_CAPILLARY_DILATION_SWEEP_PLOTS,
                            alice_passive_capillary_sweep_min_dilation_percent=ALICE_PASSIVE_CAPILLARY_SWEEP_MIN_DILATION_PERCENT,
                            alice_passive_capillary_sweep_max_dilation_percent=ALICE_PASSIVE_CAPILLARY_SWEEP_MAX_DILATION_PERCENT,
                            alice_passive_capillary_sweep_dilation_step_percent=ALICE_PASSIVE_CAPILLARY_SWEEP_DILATION_STEP_PERCENT,
                            alice_passive_capillary_sweep_min_inlet_pressure_pa=ALICE_PASSIVE_CAPILLARY_SWEEP_MIN_INLET_PRESSURE_PA,
                            alice_passive_capillary_sweep_max_inlet_pressure_pa=ALICE_PASSIVE_CAPILLARY_SWEEP_MAX_INLET_PRESSURE_PA,
                            alice_passive_capillary_sweep_inlet_pressure_step_pa=ALICE_PASSIVE_CAPILLARY_SWEEP_INLET_PRESSURE_STEP_PA,
                            alice_constriction_length_um=ALICE_CONSTRICTION_LENGTH_UM,
                            alice_constriction_spacing_um=ALICE_CONSTRICTION_SPACING_UM,
                            run_alice_passive_capillary_diameter_beforeafter=RUN_ALICE_PASSIVE_CAPILLARY_DIAMETER_BEFOREAFTER,
                            alice_passive_capillary_diameter_beforeafter_percent=ALICE_PASSIVE_CAPILLARY_DIAMETER_BEFOREAFTER_PERCENT,
                            run_alice_passive_arteriole_diameter_beforeafter=RUN_ALICE_PASSIVE_ARTERIOLE_DIAMETER_BEFOREAFTER,
                            alice_passive_arteriole_diameter_beforeafter_delta_um=ALICE_PASSIVE_ARTERIOLE_DIAMETER_BEFOREAFTER_DELTA_UM,
                            run_alice_pericyte_spacing_sweep_plots=RUN_ALICE_PERICYTE_SPACING_SWEEP_PLOTS,
                            alice_pericyte_spacing_sweep_min_um=ALICE_PERICYTE_SPACING_SWEEP_MIN_UM,
                            alice_pericyte_spacing_sweep_max_um=ALICE_PERICYTE_SPACING_SWEEP_MAX_UM,
                            alice_pericyte_spacing_sweep_step_um=ALICE_PERICYTE_SPACING_SWEEP_STEP_UM,
                            alice_pericyte_spacing_sweep_min_inlet_pressure_pa=ALICE_PERICYTE_SPACING_SWEEP_MIN_INLET_PRESSURE_PA,
                            alice_pericyte_spacing_sweep_max_inlet_pressure_pa=ALICE_PERICYTE_SPACING_SWEEP_MAX_INLET_PRESSURE_PA,
                            alice_pericyte_spacing_sweep_inlet_pressure_step_pa=ALICE_PERICYTE_SPACING_SWEEP_INLET_PRESSURE_STEP_PA,
                            alice_pericyte_spacing_sweep_percent=ALICE_PERICYTE_SPACING_SWEEP_PERCENT,
                            run_alice_pericyte_spacing_beforeafter=RUN_ALICE_PERICYTE_SPACING_BEFOREAFTER,
                            alice_pericyte_beforeafter_percent=ALICE_PERICYTE_BEFOREAFTER_PERCENT,
                            alice_pericyte_spacing_delta_um=ALICE_PERICYTE_SPACING_DELTA_UM,
                            alice_custom_edges_for_sweep=ALICE_CUSTOM_EDGES_FOR_SWEEP,
                            plot_dir=BASE_PLOT_DIR,
                            verbose_logging=VERBOSE_LOGGING,
                            do_skeletonize=DO_SKELETONIZE,
                            do_graph_building=DO_GRAPH_BUILDING,
                            run_haemodynamics=RUN_HAEMODYNAMICS,
                            do_equiv_resistance_calculation=DO_EQUIV_RESISTANCE_CALCULATION,
                            min_branch_length=MIN_BRANCH_LENGTH,
                            min_stub_length=MIN_STUB_LENGTH,
                            cluster_collapse_distance=CLUSTER_COLLAPSE_DISTANCE,
                            remove_graph_elements_in_volumes=REMOVE_GRAPH_ELEMENTS_IN_VOLUMES,
                            remove_disconnected_io_components_after_final_assignment=False,
                            endpoint_snap_max_distance=3.0,
                            endpoint_alignment_tolerance=1e-6,
                            fail_on_unaligned_edge_endpoints=False,
                            endpoint_snap_reference_mode="skeleton",
                            endpoint_skeleton_search_radius_voxels=10,
                            graph_element_removal_volumes=GRAPH_ELEMENT_REMOVAL_VOLUMES,
                            vtk_output_prefix=VTK_OUTPUT_PREFIX,
                            skeleton_closing_radius=SKELETON_CLOSING_RADIUS,
                            skeleton_bridge_gap_size=SKELETON_BRIDGE_GAP_SIZE,
                            skeleton_min_branch_length=SKELETON_MIN_BRANCH_LENGTH,
                            skeleton_max_bridge_distance=SKELETON_MAX_BRIDGE_DISTANCE,
                            skeleton_component_connectivity=SKELETON_COMPONENT_CONNECTIVITY,
                            skeleton_min_component_percent=SKELETON_MIN_COMPONENT_PERCENT,
                            graph_reconnect_threshold=GRAPH_RECONNECT_THRESHOLD,
                            final_orphan_reconnect_threshold=FINAL_ORPHAN_RECONNECT_THRESHOLD,
                            smoothing_options=None,
                            smoothing_method="bspline",
                            preserve_single_path_geometry=True,
                            single_path_geometry_max_tortuosity=1.08,
                            single_path_geometry_max_rms_distance_vox=0.35,
                            single_path_geometry_max_distance_vox=0.8,
                            starting_node_selection_method=STARTING_NODE_SELECTION_METHOD,
                            output_node_selection_method=OUTPUT_NODE_SELECTION_METHOD,
                            arteriole_boundary_selection_method=ARTERIOLE_BOUNDARY_SELECTION_METHOD,
                            venule_boundary_selection_method=VENULE_BOUNDARY_SELECTION_METHOD,
                            starting_node_coordinate_order=STARTING_NODE_COORDINATE_ORDER,
                            output_node_coordinate_order=OUTPUT_NODE_COORDINATE_ORDER,
                            arteriole_boundary_coordinate_order=ARTERIOLE_BOUNDARY_COORDINATE_ORDER,
                            venule_boundary_coordinate_order=VENULE_BOUNDARY_COORDINATE_ORDER,
                            starting_node_coordinates=STARTING_NODE_COORDINATES,
                            output_node_coordinates=OUTPUT_NODE_COORDINATES,
                            arteriole_boundary_node_coordinates=ARTERIOLE_BOUNDARY_NODE_COORDINATES,
                            venule_boundary_node_coordinates=VENULE_BOUNDARY_NODE_COORDINATES,
                            starting_node_volumes=STARTING_NODE_VOLUMES,
                            output_node_volumes=OUTPUT_NODE_VOLUMES,
                            arteriole_boundary_node_volumes=ARTERIOLE_BOUNDARY_NODE_VOLUMES,
                            venule_boundary_node_volumes=VENULE_BOUNDARY_NODE_VOLUMES,
                            boundary_coordinate_unit_check_mode=BOUNDARY_COORDINATE_UNIT_CHECK_MODE,
                            boundary_coordinate_unit_check_max_fraction_of_diagonal=BOUNDARY_COORDINATE_UNIT_CHECK_MAX_FRACTION_OF_DIAGONAL,
                            strict_branch_order_assignment=STRICT_BRANCH_ORDER_ASSIGNMENT,
                            starting_nodes=STARTING_NODES, 
                            output_nodes=OUTPUT_NODES, 
                            arteriole_boundary_nodes=ARTERIOLE_BOUNDARY_NODES,
                            venule_boundary_nodes=VENULE_BOUNDARY_NODES,
                            input_p_bc=INPUT_P_BC, 
                            output_p_bc=OUTPUT_P_BC, 
                            visualize_results=VISUALIZE_RESULTS, 
                            interactive_plots=INTERACTIVE_PLOTS,
                            show_plots_in_ide=SHOW_PLOTS_IN_IDE,
                            ide_plot_mode=IDE_PLOT_MODE,
                            hold_ide_plots_open=HOLD_IDE_PLOTS_OPEN,
                            final_render_mode=FINAL_RENDER_MODE,
                            visualize_vtk=VISUALIZE_VTK,
                            measurement_3d_to_cell_mask=MEASUREMENT_3D_TO_CELL_MASK,
                            cell_mask_path=CELL_MASK_PATH,
                            cell_mask_h5_dataset_name=CELL_MASK_H5_DATASET_NAME,
                            measurement_3d_vessel_mask_path=MEASUREMENT_3D_VESSEL_MASK_PATH,
                            measurement_3d_vessel_mask_h5_dataset_name=MEASUREMENT_3D_VESSEL_MASK_H5_DATASET_NAME,
                            measurement_3d_reference_image_path=MEASUREMENT_3D_REFERENCE_IMAGE_PATH,
                            measurement_3d_reference_h5_dataset_name=MEASUREMENT_3D_REFERENCE_H5_DATASET_NAME,
                            statistics_mode=STATISTICS_MODE,
                            use_fwhm_edge_diameters=USE_FWHM_EDGE_DIAMETERS,
                            fwhm_raw_tiff_path=FWHM_RAW_TIFF_PATH,
                            fwhm_sample_spacing_along_edge_um=FWHM_SAMPLE_SPACING_ALONG_EDGE_UM,
                            fwhm_transverse_profile_step_um=FWHM_TRANSVERSE_PROFILE_STEP_UM,
                            fwhm_transverse_half_extent_um=FWHM_TRANSVERSE_HALF_EXTENT_UM,
                            fwhm_diameter_guess_um=FWHM_DIAMETER_GUESS_UM,
                            fwhm_min_total_extent_multiplier=FWHM_MIN_TOTAL_EXTENT_MULTIPLIER,
                            fwhm_background_label=FWHM_BACKGROUND_LABEL,
                            fwhm_junction_label=FWHM_JUNCTION_LABEL,
                            fwhm_allow_junction_crossing=FWHM_ALLOW_JUNCTION_CROSSING,
                            fwhm_profile_baseline_mode=FWHM_PROFILE_BASELINE_MODE,
                            fwhm_profile_baseline_wing_fraction=FWHM_PROFILE_BASELINE_WING_FRACTION,
                            fwhm_constrain_fitted_baseline=FWHM_CONSTRAIN_FITTED_BASELINE,
                            fwhm_baseline_constraint_half_width_ptp=FWHM_BASELINE_CONSTRAINT_HALF_WIDTH_PTP,
                            fwhm_clip_profile_to_single_vessel=FWHM_CLIP_PROFILE_TO_SINGLE_VESSEL,
                            fwhm_clip_min_drop_fraction_of_center=FWHM_CLIP_MIN_DROP_FRACTION_OF_CENTER,
                            fwhm_clip_re_rise_fraction_of_center=FWHM_CLIP_RE_RISE_FRACTION_OF_CENTER,
                            fwhm_branch_endpoint_exclusion_um=FWHM_BRANCH_ENDPOINT_EXCLUSION_UM,
                            fwhm_terminal_endpoint_exclusion_um=FWHM_TERMINAL_ENDPOINT_EXCLUSION_UM,
                            fwhm_junction_proximity_exclusion_um=FWHM_JUNCTION_PROXIMITY_EXCLUSION_UM,
                            fwhm_enforce_same_edge_locality=FWHM_ENFORCE_SAME_EDGE_LOCALITY,
                            fwhm_same_edge_arc_window_um=FWHM_SAME_EDGE_ARC_WINDOW_UM,
                            fwhm_same_edge_arc_window_multiplier=FWHM_SAME_EDGE_ARC_WINDOW_MULTIPLIER,
                            fwhm_same_edge_arc_window_min_um=FWHM_SAME_EDGE_ARC_WINDOW_MIN_UM,
                            fwhm_cap_half_extent_by_nonlocal_same_edge_distance=FWHM_CAP_HALF_EXTENT_BY_NONLOCAL_SAME_EDGE_DISTANCE,
                            fwhm_nonlocal_same_edge_arc_separation_um=FWHM_NONLOCAL_SAME_EDGE_ARC_SEPARATION_UM,
                            fwhm_nonlocal_same_edge_half_extent_factor=FWHM_NONLOCAL_SAME_EDGE_HALF_EXTENT_FACTOR,
                            fwhm_reject_samples_with_center_offset=FWHM_REJECT_SAMPLES_WITH_CENTER_OFFSET,
                            fwhm_max_fit_center_offset_um=FWHM_MAX_FIT_CENTER_OFFSET_UM,
                            fwhm_reject_samples_with_low_fit_r2=FWHM_REJECT_SAMPLES_WITH_LOW_FIT_R2,
                            fwhm_min_fit_r2=FWHM_MIN_FIT_R2,
                            fwhm_edge_parallel_workers=FWHM_EDGE_PARALLEL_WORKERS,
                            fwhm_edge_parallel_batch_size=FWHM_EDGE_PARALLEL_BATCH_SIZE,
                            fwhm_min_valid_cross_section_span_um=FWHM_MIN_VALID_CROSS_SECTION_SPAN_UM,
                            fwhm_min_valid_profile_count_per_edge=FWHM_MIN_VALID_PROFILE_COUNT_PER_EDGE,
                            fwhm_diameter_aggregation_trim_fraction=FWHM_DIAMETER_AGGREGATION_TRIM_FRACTION,
                            fwhm_diameter_bounds_mode=FWHM_DIAMETER_BOUNDS_MODE,
                            fwhm_diameter_bounds_by_vessel_class_um=FWHM_DIAMETER_BOUNDS_BY_VESSEL_CLASS_UM) -> None:
    image_path = Path(image_path)
    preconfigured_starting_nodes = [int(node_id) for node_id in starting_nodes]
    preconfigured_output_nodes = [int(node_id) for node_id in output_nodes]
    preconfigured_arteriole_boundary_nodes = [
        int(node_id) for node_id in arteriole_boundary_nodes
    ]
    preconfigured_venule_boundary_nodes = [
        int(node_id) for node_id in venule_boundary_nodes
    ]
    if use_ilastik_segmentation:
        unsegmented_image_path = Path(ilastik_unsegmented_image_path)
        unsegmented_image_path = io.resolve_image_path_with_optional_zip(unsegmented_image_path)
        if ilastik_classifier_path is None:
            raise ValueError(
                "ilastik_classifier_path must be set when use_ilastik_segmentation=True."
            )
        ilastik_output_dir = Path(ilastik_output_dir)
        ilastik_segmented_path = ilastik_output_dir / (
            f"{unsegmented_image_path.stem}_segmented{ilastik_output_suffix}"
        )
        print(f"Running ilastik segmentation for unsegmented image: {unsegmented_image_path}")
        image_path = io.run_ilastik_headless_segmentation(
            input_image_path=unsegmented_image_path,
            classifier_path=Path(ilastik_classifier_path),
            output_path=ilastik_segmented_path,
            ilastik_executable=ilastik_executable,
        )
        print(f"Using ilastik-segmented image: {image_path}")
    else:
        print(f"Using segmented input image: {image_path}")

    image_path = io.resolve_image_path_with_optional_zip(image_path)
    # get image format from image_path
    input_format = image_path.suffix[1:].lower()
    if input_format not in ["tif", "tiff", "h5"]:
        raise ValueError(f"Invalid image format: {input_format}")
    vtk_output_prefix = Path(vtk_output_prefix)
    output_dir = vtk_output_prefix.parent
    valid_final_render_modes = {"2d", "3d"}
    if final_render_mode not in valid_final_render_modes:
        raise ValueError(
            f"Invalid final_render_mode='{final_render_mode}'. "
            f"Choose one of {sorted(valid_final_render_modes)}."
        )

    logging.basicConfig(
        level=logging.DEBUG if verbose_logging else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # 1) Load image and skeletonize.
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    skeleton_path = output_dir / f"{image_path.stem}_skeleton.npy"
    voxel_meta_path = output_dir / f"{image_path.stem}_voxel_size.json"
    graph_path = output_dir / f"{image_path.stem}_graph.pkl"
    projection_path = plot_dir / "skeleton_projection.png"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True, exist_ok=True)

    if do_skeletonize:
        if input_format in {"tif", "tiff"}:
            (
                image,
                skeleton,
                voxel_size_x,
                voxel_size_y,
                voxel_size_z,
                voxel_meta_status,
            ) = io.load_and_skeletonize_3d_tif(
                image_path,
            )
            metadata_voxel_size = (
                float(voxel_size_x),
                float(voxel_size_y),
                float(voxel_size_z),
            )
        elif input_format == "h5":
            (
                image,
                skeleton,
                voxel_size_x,
                voxel_size_y,
                voxel_size_z,
                voxel_meta_status,
            ) = io.load_and_skeletonize_3d_h5(
                image_path,
            )
            metadata_voxel_size = (
                float(voxel_size_x),
                float(voxel_size_y),
                float(voxel_size_z),
            )
        else:
            raise ValueError("INPUT_FORMAT must be 'tif', 'tiff', or 'h5'.")
        voxel_size, voxel_size_source = resolve_voxel_size_xyz(
            metadata_voxel_size_xyz=metadata_voxel_size,
            metadata_status=voxel_meta_status,
            voxel_size_override_xyz_px_per_um=voxel_size_override_xyz_px_per_um,
            voxel_size_policy=voxel_size_policy,
        )
        print(
            "Voxel-size resolution: "
            f"source={voxel_size_source}, "
            f"metadata_status={voxel_meta_status.get('status')}, "
            f"metadata={metadata_voxel_size}, "
            f"final={voxel_size}"
        )
        
        preprocessing.print_skeleton_connectivity_stats(
            "raw",
            skeleton,
            component_connectivity=skeleton_component_connectivity,
        )
        visualization.visualize_skeleton(skeleton, save_path=plot_dir / "raw_skeleton.png")

        skeleton = preprocessing.preprocess_skeleton_for_graph(
            skeleton,
            min_branch_length=skeleton_min_branch_length,
            max_bridge_distance=skeleton_max_bridge_distance,
            component_connectivity=skeleton_component_connectivity,
            min_component_fraction=skeleton_min_component_percent / 100.0,
            closing_radius=skeleton_closing_radius,
            bridge_gap_size=skeleton_bridge_gap_size,
        )
        preprocessing.print_skeleton_connectivity_stats(
            "cleaned",
            skeleton,
            component_connectivity=skeleton_component_connectivity,
        )
        
        # save the skeleton
        np.save(skeleton_path, skeleton)
        voxel_meta_path.write_text(
            json.dumps(
                {
                    "voxel_size": voxel_size,
                    "voxel_size_source": voxel_size_source,
                    "voxel_metadata_status": voxel_meta_status,
                    "voxel_size_policy": voxel_size_policy,
                    "voxel_size_override_xyz_px_per_um": voxel_size_override_xyz_px_per_um,
                }
            )
        )
        print(f"Saved skeleton to: {skeleton_path}")
    else:
        # load the skeleton
        skeleton = np.load(skeleton_path)
        image = tifffile.imread(image_path)
        if voxel_meta_path.exists():
            cached_voxel_meta = json.loads(voxel_meta_path.read_text())
            metadata_voxel_size = tuple(cached_voxel_meta["voxel_size"])
            voxel_meta_status = cached_voxel_meta.get(
                "voxel_metadata_status",
                {"source": "cache", "status": "unknown"},
            )
        else:
            metadata_voxel_size = (1.0, 1.0, 1.0)
            voxel_meta_status = {"source": "none", "status": "missing"}
        voxel_size, voxel_size_source = resolve_voxel_size_xyz(
            metadata_voxel_size_xyz=metadata_voxel_size,
            metadata_status=voxel_meta_status,
            voxel_size_override_xyz_px_per_um=voxel_size_override_xyz_px_per_um,
            voxel_size_policy=voxel_size_policy,
        )
        print(f"Loaded skeleton from: {skeleton_path}")
        print(
            "Voxel-size resolution (from cache/default): "
            f"source={voxel_size_source}, "
            f"metadata_status={voxel_meta_status.get('status')}, "
            f"final={voxel_size}"
        )

    print("Visualizing skeleton projection...")
    visualization.visualize_skeleton(skeleton, save_path=projection_path)
    print("Skeleton projection saved.")
    mask_cache_path = output_dir / f"{image_path.stem}_cleaned_mask_cache.pkl"
    cached_mask_payload = load_cleaned_mask_cache(
        mask_cache_path,
        expected_image_shape_zyx=tuple(int(v) for v in image.shape),
    )
    if cached_mask_payload is not None:
        print(f"Loaded cleaned mask cache for visualization from: {mask_cache_path}")

    if use_ilastik_large_vessel_segmentation and not use_large_vessel_masks:
        raise ValueError(
            "use_ilastik_large_vessel_segmentation=True requires "
            "use_large_vessel_masks=True."
        )

    effective_large_arteriole_mask_path = large_arteriole_mask_path
    effective_large_venule_mask_path = large_venule_mask_path
    if not use_large_vessel_masks:
        # Keep the loader contract strict: disabled mode must not receive mask paths.
        effective_large_arteriole_mask_path = None
        effective_large_venule_mask_path = None
    if use_large_vessel_masks and use_ilastik_large_vessel_segmentation:
        if ilastik_unsegmented_arteriole_image_path is None:
            raise ValueError(
                "ilastik_unsegmented_arteriole_image_path must be set when "
                "use_ilastik_large_vessel_segmentation=True."
            )
        if ilastik_unsegmented_venule_image_path is None:
            raise ValueError(
                "ilastik_unsegmented_venule_image_path must be set when "
                "use_ilastik_large_vessel_segmentation=True."
            )
        if ilastik_arteriole_classifier_path is None:
            raise ValueError(
                "ilastik_arteriole_classifier_path must be set when "
                "use_ilastik_large_vessel_segmentation=True."
            )
        if ilastik_venule_classifier_path is None:
            raise ValueError(
                "ilastik_venule_classifier_path must be set when "
                "use_ilastik_large_vessel_segmentation=True."
            )

        ilastik_output_dir = Path(ilastik_output_dir)
        unsegmented_arteriole_image_path = io.resolve_image_path_with_optional_zip(
            Path(ilastik_unsegmented_arteriole_image_path)
        )
        unsegmented_venule_image_path = io.resolve_image_path_with_optional_zip(
            Path(ilastik_unsegmented_venule_image_path)
        )
        ilastik_segmented_arteriole_path = ilastik_output_dir / (
            f"{unsegmented_arteriole_image_path.stem}_segmented{ilastik_output_suffix}"
        )
        ilastik_segmented_venule_path = ilastik_output_dir / (
            f"{unsegmented_venule_image_path.stem}_segmented{ilastik_output_suffix}"
        )

        print(
            "Running ilastik segmentation for large arteriole image: "
            f"{unsegmented_arteriole_image_path}"
        )
        effective_large_arteriole_mask_path = io.run_ilastik_headless_segmentation(
            input_image_path=unsegmented_arteriole_image_path,
            classifier_path=Path(ilastik_arteriole_classifier_path),
            output_path=ilastik_segmented_arteriole_path,
            ilastik_executable=ilastik_executable,
        )
        print(
            "Running ilastik segmentation for large venule image: "
            f"{unsegmented_venule_image_path}"
        )
        effective_large_venule_mask_path = io.run_ilastik_headless_segmentation(
            input_image_path=unsegmented_venule_image_path,
            classifier_path=Path(ilastik_venule_classifier_path),
            output_path=ilastik_segmented_venule_path,
            ilastik_executable=ilastik_executable,
        )
        print(
            "Using ilastik-segmented large-vessel masks: "
            f"arteriole={effective_large_arteriole_mask_path}, "
            f"venule={effective_large_venule_mask_path}"
        )

    (
        large_arteriole_mask,
        large_venule_mask,
        large_arteriole_mask_voxel_size,
        large_venule_mask_voxel_size,
    ) = io.load_large_vessel_masks(
        enabled=use_large_vessel_masks,
        large_arteriole_mask_path=effective_large_arteriole_mask_path,
        large_venule_mask_path=effective_large_venule_mask_path,
    )
    if large_arteriole_mask is not None and large_venule_mask is not None:
        shape_mismatch = False
        if large_arteriole_mask.shape != image.shape:
            shape_mismatch = True
            print(
                "Warning: large_arteriole_mask shape does not match input image shape; "
                f"disabling large-vessel masks for this run "
                f"({large_arteriole_mask.shape} != {image.shape})."
            )
        if large_venule_mask.shape != image.shape:
            shape_mismatch = True
            print(
                "Warning: large_venule_mask shape does not match input image shape; "
                f"disabling large-vessel masks for this run "
                f"({large_venule_mask.shape} != {image.shape})."
            )
        if shape_mismatch:
            large_arteriole_mask = None
            large_venule_mask = None
            large_arteriole_mask_voxel_size = None
            large_venule_mask_voxel_size = None
    if large_arteriole_mask is not None and large_venule_mask is not None:
        print(
            "Loaded large-vessel masks: "
            f"arteriole={large_arteriole_mask.shape}, "
            f"venule={large_venule_mask.shape}"
        )
        print(
            "Large-vessel mask voxel sizes (x, y, z): "
            f"arteriole={large_arteriole_mask_voxel_size}, "
            f"venule={large_venule_mask_voxel_size}"
        )
        main_voxel_size_xyz = tuple(float(v) for v in voxel_size)
        arteriole_voxel_size_xyz = tuple(float(v) for v in large_arteriole_mask_voxel_size)
        venule_voxel_size_xyz = tuple(float(v) for v in large_venule_mask_voxel_size)
        voxel_match_main_vs_arteriole = np.allclose(
            main_voxel_size_xyz,
            arteriole_voxel_size_xyz,
            rtol=0.0,
            atol=0.0,
        )
        voxel_match_main_vs_venule = np.allclose(
            main_voxel_size_xyz,
            venule_voxel_size_xyz,
            rtol=0.0,
            atol=0.0,
        )
        voxel_match_arteriole_vs_venule = np.allclose(
            arteriole_voxel_size_xyz,
            venule_voxel_size_xyz,
            rtol=0.0,
            atol=0.0,
        )
        if not (
            voxel_match_main_vs_arteriole
            and voxel_match_main_vs_venule
            and voxel_match_arteriole_vs_venule
        ):
            if str(voxel_size_source).strip().lower() == "manual_override":
                print(
                    "Voxel-size mismatch detected for large-vessel masks while manual "
                    "override is active. Using manual override voxel units for large "
                    f"masks: main={main_voxel_size_xyz}, "
                    f"arteriole(original)={arteriole_voxel_size_xyz}, "
                    f"venule(original)={venule_voxel_size_xyz}."
                )
                large_arteriole_mask_voxel_size = tuple(main_voxel_size_xyz)
                large_venule_mask_voxel_size = tuple(main_voxel_size_xyz)
                arteriole_voxel_size_xyz = tuple(main_voxel_size_xyz)
                venule_voxel_size_xyz = tuple(main_voxel_size_xyz)
                voxel_match_main_vs_arteriole = True
                voxel_match_main_vs_venule = True
                voxel_match_arteriole_vs_venule = True
            else:
                error_message = (
                    "Voxel-size mismatch detected across input image and large-vessel masks. "
                    f"main={main_voxel_size_xyz}, "
                    f"arteriole={arteriole_voxel_size_xyz}, "
                    f"venule={venule_voxel_size_xyz}. "
                    "All three must match exactly in x, y, and z."
                )
                print(error_message)
                raise ValueError(error_message)
        print(
            "Voxel-size check passed. Arteriole and venule masks are aligned "
            "to the same physical voxel units as the main image."
        )
        if large_vessel_mask_dilation_microns > 0:
            print(
                "Configured automated large-vessel assignment dilation max: "
                f"{float(large_vessel_mask_dilation_microns):.3f} microns "
                "(applied as progressive 5-micron steps during assignment only)."
            )
    else:
        print("Large-vessel masks disabled; skipping arteriole/venule mask loading.")

    if (
        use_ilastik_small_vessel_segmentation
        and not use_small_vessel_masks_for_boundary_assignment
    ):
        raise ValueError(
            "use_ilastik_small_vessel_segmentation=True requires "
            "use_small_vessel_masks_for_boundary_assignment=True."
        )

    effective_small_arteriole_mask_path = small_arteriole_mask_path
    effective_small_venule_mask_path = small_venule_mask_path
    if not use_small_vessel_masks_for_boundary_assignment:
        # Keep the loader contract strict: disabled mode must not receive mask paths.
        effective_small_arteriole_mask_path = None
        effective_small_venule_mask_path = None
    if (
        use_small_vessel_masks_for_boundary_assignment
        and use_ilastik_small_vessel_segmentation
    ):
        if ilastik_unsegmented_small_arteriole_image_path is None:
            raise ValueError(
                "ilastik_unsegmented_small_arteriole_image_path must be set when "
                "use_ilastik_small_vessel_segmentation=True."
            )
        if ilastik_unsegmented_small_venule_image_path is None:
            raise ValueError(
                "ilastik_unsegmented_small_venule_image_path must be set when "
                "use_ilastik_small_vessel_segmentation=True."
            )
        if ilastik_small_arteriole_classifier_path is None:
            raise ValueError(
                "ilastik_small_arteriole_classifier_path must be set when "
                "use_ilastik_small_vessel_segmentation=True."
            )
        if ilastik_small_venule_classifier_path is None:
            raise ValueError(
                "ilastik_small_venule_classifier_path must be set when "
                "use_ilastik_small_vessel_segmentation=True."
            )

        ilastik_output_dir = Path(ilastik_output_dir)
        unsegmented_small_arteriole_image_path = io.resolve_image_path_with_optional_zip(
            Path(ilastik_unsegmented_small_arteriole_image_path)
        )
        unsegmented_small_venule_image_path = io.resolve_image_path_with_optional_zip(
            Path(ilastik_unsegmented_small_venule_image_path)
        )
        ilastik_segmented_small_arteriole_path = ilastik_output_dir / (
            f"{unsegmented_small_arteriole_image_path.stem}_segmented{ilastik_output_suffix}"
        )
        ilastik_segmented_small_venule_path = ilastik_output_dir / (
            f"{unsegmented_small_venule_image_path.stem}_segmented{ilastik_output_suffix}"
        )

        print(
            "Running ilastik segmentation for small arteriole image: "
            f"{unsegmented_small_arteriole_image_path}"
        )
        effective_small_arteriole_mask_path = io.run_ilastik_headless_segmentation(
            input_image_path=unsegmented_small_arteriole_image_path,
            classifier_path=Path(ilastik_small_arteriole_classifier_path),
            output_path=ilastik_segmented_small_arteriole_path,
            ilastik_executable=ilastik_executable,
        )
        print(
            "Running ilastik segmentation for small venule image: "
            f"{unsegmented_small_venule_image_path}"
        )
        effective_small_venule_mask_path = io.run_ilastik_headless_segmentation(
            input_image_path=unsegmented_small_venule_image_path,
            classifier_path=Path(ilastik_small_venule_classifier_path),
            output_path=ilastik_segmented_small_venule_path,
            ilastik_executable=ilastik_executable,
        )
        print(
            "Using ilastik-segmented small-vessel masks: "
            f"arteriole={effective_small_arteriole_mask_path}, "
            f"venule={effective_small_venule_mask_path}"
        )

    (
        small_arteriole_mask,
        small_venule_mask,
        small_arteriole_mask_voxel_size,
        small_venule_mask_voxel_size,
    ) = io.load_large_vessel_masks(
        enabled=use_small_vessel_masks_for_boundary_assignment,
        large_arteriole_mask_path=effective_small_arteriole_mask_path,
        large_venule_mask_path=effective_small_venule_mask_path,
    )
    if small_arteriole_mask is not None and small_venule_mask is not None:
        if small_arteriole_mask.shape != image.shape:
            raise ValueError(
                "small_arteriole_mask shape does not match input image shape: "
                f"{small_arteriole_mask.shape} != {image.shape}"
            )
        if small_venule_mask.shape != image.shape:
            raise ValueError(
                "small_venule_mask shape does not match input image shape: "
                f"{small_venule_mask.shape} != {image.shape}"
            )
        main_voxel_size_xyz = tuple(float(v) for v in voxel_size)
        small_arteriole_voxel_size_xyz = tuple(
            float(v) for v in small_arteriole_mask_voxel_size
        )
        small_venule_voxel_size_xyz = tuple(float(v) for v in small_venule_mask_voxel_size)
        if not (
            np.allclose(main_voxel_size_xyz, small_arteriole_voxel_size_xyz, rtol=0.0, atol=0.0)
            and np.allclose(main_voxel_size_xyz, small_venule_voxel_size_xyz, rtol=0.0, atol=0.0)
            and np.allclose(
                small_arteriole_voxel_size_xyz,
                small_venule_voxel_size_xyz,
                rtol=0.0,
                atol=0.0,
            )
        ):
            if str(voxel_size_source).strip().lower() == "manual_override":
                print(
                    "Voxel-size mismatch detected for small-vessel masks while manual "
                    "override is active. Using manual override voxel units for small "
                    f"masks: main={main_voxel_size_xyz}, "
                    f"small_arteriole(original)={small_arteriole_voxel_size_xyz}, "
                    f"small_venule(original)={small_venule_voxel_size_xyz}."
                )
                small_arteriole_mask_voxel_size = tuple(main_voxel_size_xyz)
                small_venule_mask_voxel_size = tuple(main_voxel_size_xyz)
                small_arteriole_voxel_size_xyz = tuple(main_voxel_size_xyz)
                small_venule_voxel_size_xyz = tuple(main_voxel_size_xyz)
            else:
                raise ValueError(
                    "Voxel-size mismatch detected across input image and small-vessel masks. "
                    f"main={main_voxel_size_xyz}, "
                    f"small_arteriole={small_arteriole_voxel_size_xyz}, "
                    f"small_venule={small_venule_voxel_size_xyz}. "
                    "All must match exactly in x, y, and z."
                )
        print(
            "Loaded small-vessel masks for boundary assignment: "
            f"arteriole={small_arteriole_mask.shape}, venule={small_venule_mask.shape}, "
            f"min_overlap_fraction={float(small_vessel_mask_min_overlap_fraction):.3f}"
        )
    else:
        if use_small_vessel_masks_for_boundary_assignment:
            print(
                "Small-vessel boundary assignment is enabled, but small-vessel masks "
                "were not loaded. Fallback/manual boundary selection may be used."
            )
        else:
            print(
                "Small-vessel-mask boundary assignment disabled; "
                "manual arteriole/venule boundary-node selection remains available."
            )

    large_viz_arteriole_mask = large_arteriole_mask
    large_viz_venule_mask = large_venule_mask
    small_viz_arteriole_mask = small_arteriole_mask
    small_viz_venule_mask = small_venule_mask
    small_changed_viz_arteriole_mask = None
    small_changed_viz_venule_mask = None
    if cached_mask_payload is not None:
        if large_viz_arteriole_mask is None or large_viz_venule_mask is None:
            cached_large_arteriole = cached_mask_payload.get("large_arteriole_mask")
            cached_large_venule = cached_mask_payload.get("large_venule_mask")
            if cached_large_arteriole is not None and cached_large_venule is not None:
                large_viz_arteriole_mask = np.asarray(cached_large_arteriole, dtype=bool)
                large_viz_venule_mask = np.asarray(cached_large_venule, dtype=bool)
                print(
                    "Using cached cleaned large-vessel masks for visualization "
                    "because live large-vessel masks are unavailable."
                )
        if small_viz_arteriole_mask is None or small_viz_venule_mask is None:
            cached_small_arteriole = cached_mask_payload.get("small_arteriole_mask")
            cached_small_venule = cached_mask_payload.get("small_venule_mask")
            if cached_small_arteriole is not None and cached_small_venule is not None:
                small_viz_arteriole_mask = np.asarray(cached_small_arteriole, dtype=bool)
                small_viz_venule_mask = np.asarray(cached_small_venule, dtype=bool)
                print(
                    "Using cached cleaned small-vessel masks for visualization "
                    "because live small-vessel masks are unavailable."
                )

    if do_graph_building:
        # 3) Convert skeleton to graph.
        print("Building skan Skeleton object...")
        sk = csr.Skeleton(skeleton)
        print(f"skan Skeleton built: {sk.n_paths} paths")

        print("Building graph (loop detection + segment extraction)...")
        G, voxel_loops, loop_edges = graph.build_graph_segment_skan_stitched_loops(
            sk,
            skeleton,
            debug=verbose_logging,
            voxel_size=voxel_size,
            reconnect_threshold=graph_reconnect_threshold,
        )
        visualization.save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "build_graph_segment_skan_stitched_loops",
        )
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=True,
            show_debug_dual_axes=True,
            save_path=plot_dir / "build_graph_segment_skan_stitched_loops.png",
        )
        G = graph.reconnect_secondary_loop_edges(
            G,
            skeleton,
            voxel_size=voxel_size,
            debug=verbose_logging,
        )
        visualization.save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "reconnect_secondary_loop_edges",
        )
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=True,
            show_debug_dual_axes=True,
            save_path=plot_dir / "reconnect_secondary_loop_edges.png",
        )
        
        G, _ = graph.optimise_graph_topology_fixed(
            G,
            voxel_loops,
            loop_edges,
            skeleton_data=skeleton,
            debug=verbose_logging,
            reconnect_threshold=graph_reconnect_threshold,
        )
        visualization.save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "optimise_graph_topology_fixed",
        )
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=True,
            show_debug_dual_axes=True,
            save_path=plot_dir / "optimise_graph_topology_fixed.png",
        )
        degree2_pass1_max_degree = 4
        degree2_pass2_max_degree = 8
        G = graph.smart_multigraph_degree2_removal(
            G,
            skeleton,
            max_degree=degree2_pass1_max_degree,
            debug=verbose_logging,
        )
        visualization.save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "smart_multigraph_degree2_removal_pass1",
        )
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=True,
            show_debug_dual_axes=True,
            save_path=plot_dir / "smart_multigraph_degree2_removal.png",
        )
        degree2_diag = graph.diagnose_degree2_nodes(
            G, max_degree=degree2_pass1_max_degree
        )
        print(graph.format_degree2_diagnostics_report(degree2_diag))

        G = graph.collapse_node_clusters(
            G,
            distance_threshold=cluster_collapse_distance,
            debug=verbose_logging,
        )
        visualization.save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "collapse_node_clusters",
        )
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=True,
            show_debug_dual_axes=True,
            save_path=plot_dir / "collapse_node_clusters.png",
        )

        # Collapsing clusters can create new degree-2 pass-through nodes;
        # run a second degree-2 cleanup pass with a higher threshold since
        # remaining degree-2 nodes typically neighbour high-degree junctions.
        G = graph.smart_multigraph_degree2_removal(
            G,
            skeleton,
            max_degree=degree2_pass2_max_degree,
            debug=verbose_logging,
        )
        visualization.save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "smart_multigraph_degree2_removal_post_collapse",
        )
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=True,
            show_debug_dual_axes=True,
            save_path=plot_dir / "smart_multigraph_degree2_removal_post_collapse.png",
        )

        G = graph.prune_vascular_stubs(G, debug=verbose_logging, min_stub_length=min_stub_length)
        visualization.save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "prune_vascular_stubs",
        )
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=True,
            show_debug_dual_axes=True,
            save_path=plot_dir / "prune_vascular_stubs.png",
        )
        degree2_diag = graph.diagnose_degree2_nodes(
            G, max_degree=degree2_pass2_max_degree
        )
        print(graph.format_degree2_diagnostics_report(degree2_diag))

        G = graph.smart_multigraph_degree2_removal(
            G,
            skeleton,
            max_degree=degree2_pass2_max_degree,
            debug=verbose_logging,
        )
        visualization.save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "smart_multigraph_degree2_removal_post_prune",
        )
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=True,
            show_debug_dual_axes=True,
            save_path=plot_dir / "smart_multigraph_degree2_removal_post_prune.png",
        )
        degree2_diag = graph.diagnose_degree2_nodes(
            G, max_degree=degree2_pass2_max_degree
        )
        print(graph.format_degree2_diagnostics_report(degree2_diag))

        # remove any nodes that are connected to themselves with no nodes in between
        G = graph.remove_edges_for_self_connected_nodes(G)
        visualization.save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "remove_edges_for_self_connected_nodes",
        )

        # Final topology repair:
        # 1) reconnect remaining orphan/dangling nodes only if a skeleton path
        #    validates the link, then
        # 2) remove any new degree-2 pass-through nodes that remain.
        G = graph.reconnect_orphan_and_dangling_nodes(
            G,
            skeleton_data=skeleton,
            reconnect_threshold=final_orphan_reconnect_threshold,
            include_degree1=True,
            max_new_edges_per_node=1,
            validate_reconnections=True,
            debug=verbose_logging,
        )
        visualization.save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "reconnect_orphan_and_dangling_nodes",
        )
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=True,
            show_debug_dual_axes=True,
            save_path=plot_dir / "reconnect_orphan_and_dangling_nodes.png",
        )

        G = graph.smart_multigraph_degree2_removal(
            G,
            skeleton,
            max_degree=4,
            debug=verbose_logging,
        )
        visualization.save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "smart_multigraph_degree2_removal_post_orphan_reconnect",
        )
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=True,
            show_debug_dual_axes=True,
            save_path=plot_dir / "smart_multigraph_degree2_removal_post_orphan_reconnect.png",
        )
        degree2_diag = graph.diagnose_degree2_nodes(
            G, max_degree=degree2_pass2_max_degree
        )
        print(graph.format_degree2_diagnostics_report(degree2_diag))

        # Remove tiny disconnected fragments represented by a single
        # edge with degree-1 terminal nodes on both endpoints.
        if G.number_of_edges() > 1:
            G, removed_terminal_terminal_edges = graph.remove_terminal_terminal_edges(
                G,
                debug=verbose_logging,
                return_removed_count=True,
            )
            print(
                "Removed "
                f"{removed_terminal_terminal_edges} edge(s) with degree-1 terminals on both ends."
            )
        else:
            # Keep single-edge graphs so one-vessel synthetic/integration cases
            # are not fully pruned before boundary assignment and export.
            removed_terminal_terminal_edges = 0
            print(
                "Skipped terminal-terminal edge removal for single-edge graph."
            )
        visualization.save_graph_snapshot(
            G, image, output_dir, plot_dir, image_path.stem,
            "remove_terminal_terminal_edges",
        )

        # Final cleanup: remove isolated nodes (degree 0) left after topology
        # operations so they do not propagate into boundary selection/statistics.
        nodes_before_isolated_cleanup = G.number_of_nodes()
        G = graph.remove_isolated_nodes(G)
        isolated_removed = nodes_before_isolated_cleanup - G.number_of_nodes()
        if isolated_removed:
            print(f"Removed {isolated_removed} isolated degree-0 node(s).")

        if bool(remove_graph_elements_in_volumes):
            G, removal_stats = graph.remove_graph_elements_in_volumes(
                G,
                graph_element_removal_volumes,
                remove_nodes=True,
                remove_edges=True,
                remove_isolated_nodes_after=True,
            )
            print(
                "Graph volume-removal applied after construction/optimisation: "
                f"volumes={int(removal_stats['configured_volume_count'])}, "
                f"removed_edges={int(removal_stats['removed_edges'])}, "
                f"removed_nodes={int(removal_stats['removed_nodes'])}, "
                f"removed_isolated_nodes={int(removal_stats['removed_isolated_nodes'])}."
            )
            visualization.save_graph_snapshot(
                G, image, output_dir, plot_dir, image_path.stem,
                "remove_graph_elements_in_volumes",
            )
            visualization.visualize_edges_and_nodes(
                image,
                G,
                label_nodes=True,
                show_debug_dual_axes=True,
                save_path=plot_dir / "remove_graph_elements_in_volumes.png",
            )

        smoothing_opts = dict(smoothing_options or {})
        if "method" not in smoothing_opts and smoothing_method is not None:
            smoothing_opts["method"] = smoothing_method
        smooth_stats = graph.smooth_graph_edge_centerlines_continuous(
            G,
            skeleton_data=skeleton,
            smoothing_options=smoothing_opts,
            voxel_size=voxel_size,
            chaikin_iterations=2,
            max_distance_vox=1.0,
            debug=verbose_logging,
        )
        print(f"Continuous centerline smoothing summary: {smooth_stats}")

        with graph_path.open("wb") as f:
            pickle.dump(G, f)
        print(f"Saved graph to: {graph_path}")
    else:
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Graph file not found at {graph_path}. "
                "Set DO_GRAPH_BUILDING=True to generate it first."
            )
        with graph_path.open("rb") as f:
            G = pickle.load(f)
        print(f"Loaded graph from: {graph_path}")
        if bool(remove_graph_elements_in_volumes):
            G, removal_stats = graph.remove_graph_elements_in_volumes(
                G,
                graph_element_removal_volumes,
                remove_nodes=True,
                remove_edges=True,
                remove_isolated_nodes_after=True,
            )
            print(
                "Graph volume-removal applied to loaded graph: "
                f"volumes={int(removal_stats['configured_volume_count'])}, "
                f"removed_edges={int(removal_stats['removed_edges'])}, "
                f"removed_nodes={int(removal_stats['removed_nodes'])}, "
                f"removed_isolated_nodes={int(removal_stats['removed_isolated_nodes'])}."
            )

    # Ensure every edge polyline is oriented/snap-aligned to its incident nodes.
    endpoint_repair_stats = graph.snap_edge_endpoints_to_node_positions(
        G,
        update_length=True,
        update_weight_if_matches_length=True,
        max_snap_distance=float(endpoint_snap_max_distance)
        if endpoint_snap_max_distance is not None
        else None,
        reference_skeleton_data=skeleton,
        voxel_size=voxel_size,
        prefer_skeleton_reference=str(endpoint_snap_reference_mode).strip().lower() == "skeleton",
        skeleton_search_radius_voxels=int(endpoint_skeleton_search_radius_voxels),
        atol=float(endpoint_alignment_tolerance),
    )
    print(
        "Endpoint-node alignment repair: "
        f"snapped={int(endpoint_repair_stats.get('edges_snapped', 0))}, "
        f"reoriented={int(endpoint_repair_stats.get('edges_reoriented', 0))}, "
        f"skeleton_ref_endpoints={int(endpoint_repair_stats.get('endpoints_using_skeleton_reference', 0))}, "
        f"skipped_distance={int(endpoint_repair_stats.get('edges_skipped_distance', 0))}, "
        f"edges_seen={int(endpoint_repair_stats.get('edges_seen', 0))}."
    )
    if bool(preserve_single_path_geometry):
        min_voxel_size = float(np.min(np.asarray(voxel_size, dtype=float)))
        straightening_stats = graph.regularize_isolated_near_straight_edges(
            G,
            max_component_edges=1,
            min_points=3,
            max_tortuosity=float(single_path_geometry_max_tortuosity),
            max_rms_distance=float(single_path_geometry_max_rms_distance_vox) * min_voxel_size,
            max_max_distance=float(single_path_geometry_max_distance_vox) * min_voxel_size,
        )
        print(
            "Single-path geometry regularization: "
            f"regularized={int(straightening_stats.get('regularized_edges', 0))}, "
            f"eligible={int(straightening_stats.get('eligible_edges', 0))}, "
            f"edges_seen={int(straightening_stats.get('edges_seen', 0))}, "
            f"components_seen={int(straightening_stats.get('components_seen', 0))}."
        )
    endpoint_alignment_diag = graph.diagnose_edge_endpoint_node_alignment(
        G,
        atol=float(endpoint_alignment_tolerance),
        sample_limit=20,
    )
    print(
        "Endpoint-node alignment check: "
        f"misaligned_edges={int(endpoint_alignment_diag.get('misaligned_edges', 0))}, "
        f"max_misalignment={float(endpoint_alignment_diag.get('max_misalignment', 0.0)):.6g}, "
        f"edges_missing_node_pos={int(endpoint_alignment_diag.get('edges_missing_node_pos', 0))}."
    )
    endpoint_skeleton_diag = graph.diagnose_edge_endpoint_skeleton_alignment(
        G,
        skeleton,
        voxel_size=voxel_size,
        max_search_radius_voxels=2,
        sample_limit=20,
    )
    print(
        "Endpoint-skeleton alignment check: "
        f"off_skeleton_endpoints={int(endpoint_skeleton_diag.get('off_skeleton_endpoints', 0))}, "
        f"max_distance_to_skeleton={float(endpoint_skeleton_diag.get('max_distance_to_skeleton', 0.0)):.6g}."
    )
    if bool(fail_on_unaligned_edge_endpoints) and (
        int(endpoint_alignment_diag.get("misaligned_edges", 0)) > 0
        or int(endpoint_alignment_diag.get("edges_missing_node_pos", 0)) > 0
        or int(endpoint_skeleton_diag.get("off_skeleton_endpoints", 0)) > 0
    ):
        raise ValueError(
            "Edge endpoint-node alignment check failed: "
            f"misaligned_edges={int(endpoint_alignment_diag.get('misaligned_edges', 0))}, "
            f"edges_missing_node_pos={int(endpoint_alignment_diag.get('edges_missing_node_pos', 0))}, "
            f"off_skeleton_endpoints={int(endpoint_skeleton_diag.get('off_skeleton_endpoints', 0))}, "
            f"sample_node_mismatch={endpoint_alignment_diag.get('sample_misaligned_edges', [])}, "
            f"sample_off_skeleton={endpoint_skeleton_diag.get('sample_off_skeleton_endpoints', [])}."
        )

    # Store physical voxel-unit metadata used for skeleton/graph geometry and mask alignment.
    G.graph["image_voxel_size_xyz"] = tuple(float(v) for v in voxel_size)
    if large_arteriole_mask is not None and large_venule_mask is not None:
        G.graph["large_arteriole_mask_voxel_size_xyz"] = tuple(
            float(v) for v in large_arteriole_mask_voxel_size
        )
        G.graph["large_venule_mask_voxel_size_xyz"] = tuple(
            float(v) for v in large_venule_mask_voxel_size
        )
    
    # Visualize final graph used for boundary-node verification.
    if final_render_mode == "3d":
        final_graph_3d_path = plot_dir / "final_graph_3d.html"
        visualization.visualize_3d_plotly(
            G,
            title="Final Graph (Interactive 3D)",
            save_html_path=str(final_graph_3d_path),
            show=show_plots_in_ide or interactive_plots,
        )
        print(f"Saved interactive 3D final graph to: {final_graph_3d_path}")
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=True,
            show_debug_dual_axes=True,
            save_path=plot_dir / "final_graph.png",
        )
        print(f"Saved final graph PNG to: {plot_dir / 'final_graph.png'}")
    else:
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=False,
            save_path=plot_dir / "final_graph.png",
            show_coordinates_degree_1=True,
        )

    auto_start_nodes: list[int] = []
    auto_output_nodes: list[int] = []
    resistance_node_pairs: list[tuple[int, int]] = []
    if automated_vessel_assignment:
        print("Starting automated input/output assignment from large vessel masks...")
        cleanup_enabled_for_large = bool(automated_vessel_assignment_enable_overlap_cleanup)
        use_legacy_large_vessel_assignment = bool(automated_vessel_assignment_use_legacy_mode)
        effective_assignment_fast_mode = bool(automated_vessel_assignment_fast_mode)
        effective_max_large_vessel_dilation_microns = float(large_vessel_mask_dilation_microns)
        apply_overlap_cleanup_prepass = bool(
            cleanup_enabled_for_large
            and (
                effective_assignment_fast_mode
                or automated_vessel_assignment_apply_overlap_cleanup_in_normal_mode
            )
        )
        if effective_assignment_fast_mode:
            if cleanup_enabled_for_large:
                print(
                    "Automated large-vessel assignment fast mode enabled: "
                    "removing overlap voxels from smaller overlapping components "
                    "before assignment."
                )
            else:
                print(
                    "Automated large-vessel assignment fast mode enabled, but overlap "
                    "cleanup is disabled by AUTOMATED_VESSEL_ASSIGNMENT_ENABLE_OVERLAP_CLEANUP=False."
                )
        elif apply_overlap_cleanup_prepass:
            print(
                "Automated large-vessel assignment: overlap cleanup pre-pass "
                "enabled in normal mode."
            )
        elif not cleanup_enabled_for_large:
            print("Automated large-vessel assignment: overlap cleanup pre-pass disabled.")
        if large_arteriole_mask is None or large_venule_mask is None:
            raise ValueError(
                "automated_vessel_assignment=True requires arteriole and venule masks. "
                "Set use_large_vessel_masks=True and provide mask paths."
            )
        assignment_large_arteriole_mask = large_arteriole_mask
        assignment_large_venule_mask = large_venule_mask
        if float(large_vessel_min_component_volume_um3) > 0:
            (
                assignment_large_arteriole_mask,
                assignment_large_venule_mask,
                large_component_volume_stats,
            ) = graph.remove_small_vessel_components_by_volume(
                assignment_large_arteriole_mask,
                assignment_large_venule_mask,
                voxel_size_xyz=tuple(float(v) for v in voxel_size),
                min_component_volume_um3=float(large_vessel_min_component_volume_um3),
            )
            arteriole_stats = large_component_volume_stats.get("arteriole") or {}
            venule_stats = large_component_volume_stats.get("venule") or {}
            print(
                "Large-vessel component-volume filtering: "
                f"threshold={float(large_vessel_min_component_volume_um3):.3f} um^3, "
                f"removed_components(arteriole={int(arteriole_stats.get('removed_component_count', 0))}, "
                f"venule={int(venule_stats.get('removed_component_count', 0))}), "
                f"removed_volume_um3(arteriole={float(arteriole_stats.get('removed_volume_um3', 0.0)):.3f}, "
                f"venule={float(venule_stats.get('removed_volume_um3', 0.0)):.3f})."
            )
        if bool(large_vessel_remove_small_opposite_attached_components):
            (
                assignment_large_arteriole_mask,
                assignment_large_venule_mask,
                opposite_attached_cleanup_stats,
            ) = graph.remove_small_opposite_attached_large_vessel_components(
                assignment_large_arteriole_mask,
                assignment_large_venule_mask,
                voxel_size_xyz=tuple(float(v) for v in voxel_size),
                max_component_volume_um3=float(
                    large_vessel_opposite_attached_max_component_volume_um3
                ),
                max_attach_distance_microns=float(
                    large_vessel_opposite_attached_max_distance_microns
                ),
            )
            oa_art = opposite_attached_cleanup_stats.get("arteriole") or {}
            oa_ven = opposite_attached_cleanup_stats.get("venule") or {}
            print(
                "Large-vessel opposite-attached tiny-component cleanup: "
                f"max_component_volume_um3={float(large_vessel_opposite_attached_max_component_volume_um3):.3f}, "
                f"max_attach_distance_microns={float(large_vessel_opposite_attached_max_distance_microns):.3f}, "
                f"removed_components(arteriole={int(oa_art.get('removed_component_count', 0))}, "
                f"venule={int(oa_ven.get('removed_component_count', 0))}), "
                f"near_opposite_components(arteriole={int(oa_art.get('near_opposite_component_count', 0))}, "
                f"venule={int(oa_ven.get('near_opposite_component_count', 0))}), "
                f"near_opposite_too_large(arteriole={int(oa_art.get('near_opposite_too_large_component_count', 0))}, "
                f"venule={int(oa_ven.get('near_opposite_too_large_component_count', 0))}), "
                f"candidate_components(arteriole={int(oa_art.get('candidate_component_count', 0))}, "
                f"venule={int(oa_ven.get('candidate_component_count', 0))}), "
                f"removed_volume_um3(arteriole={float(oa_art.get('removed_volume_um3', 0.0)):.3f}, "
                f"venule={float(oa_ven.get('removed_volume_um3', 0.0)):.3f}), "
                f"min_component_distance_microns(arteriole={float(oa_art.get('min_component_distance_microns', 0.0)):.3f}, "
                f"venule={float(oa_ven.get('min_component_distance_microns', 0.0)):.3f})."
            )
        if not use_legacy_large_vessel_assignment:
            quality_metrics = graph.assess_large_vessel_assignment_quality(
                G,
                large_arteriole_mask=assignment_large_arteriole_mask,
                large_venule_mask=assignment_large_venule_mask,
                voxel_size_xyz=tuple(float(v) for v in voxel_size),
                quality_max_overlap_fraction=float(automated_vessel_quality_max_overlap_fraction),
                quality_min_terminal_coverage=float(automated_vessel_quality_min_terminal_coverage),
                quality_max_component_count=int(automated_vessel_quality_max_component_count),
            )
            print(
                "Robust large-vessel quality gate: "
                f"overlap_fraction={float(quality_metrics['overlap_fraction']):.4f}, "
                f"terminal_coverage={float(quality_metrics['terminal_coverage']):.4f}, "
                f"component_count(arteriole={int(quality_metrics['arteriole_component_count'])}, "
                f"venule={int(quality_metrics['venule_component_count'])}), "
                f"poor_quality={bool(quality_metrics['poor_quality'])}."
            )
            if bool(quality_metrics["poor_quality"]):
                effective_assignment_fast_mode = False
                apply_overlap_cleanup_prepass = False
                effective_max_large_vessel_dilation_microns = min(
                    float(effective_max_large_vessel_dilation_microns),
                    float(automated_vessel_conservative_max_dilation_microns),
                )
                print(
                    "Robust large-vessel assignment switched to conservative mode: "
                    "disabling overlap-cleanup pre-pass and reducing max dilation "
                    f"to {float(effective_max_large_vessel_dilation_microns):.3f} microns."
                )
        if effective_assignment_fast_mode:
            if write_fast_mode_preassignment_large_vessel_debug_3d_html:
                pre_assignment_before_cleanup_html_path = (
                    plot_dir / "pre_assignment_large_vessel_masks_before_overlap_cleanup_3d.html"
                )
                print(
                    "Rendering fast-mode pre-assignment debug view (before overlap cleanup)..."
                )
                visualization.visualize_3d_plotly_large_vessel_assignment(
                    G,
                    large_arteriole_mask=assignment_large_arteriole_mask,
                    large_venule_mask=assignment_large_venule_mask,
                    input_nodes=[],
                    output_nodes=[],
                    voxel_size_xyz=tuple(float(v) for v in voxel_size),
                    volume_downsample_stride=int(large_vessel_3d_volume_downsample_stride),
                    title=(
                        "Pre-Assignment Debug View (Fast Mode, Before Overlap Cleanup): "
                        "Large-Vessel Masks + Graph"
                    ),
                    save_html_path=str(pre_assignment_before_cleanup_html_path),
                    show=False,
                )
                print(
                    "Saved fast-mode pre-assignment (before cleanup) large-vessel "
                    f"debug 3D visualization to: {pre_assignment_before_cleanup_html_path}"
                )
            cleaned_arteriole_mask, cleaned_venule_mask = (
                graph.exclude_smaller_overlapping_large_vessel_components(
                    assignment_large_arteriole_mask,
                    assignment_large_venule_mask,
                )
            )
            overlap_before_cleanup = int(
                np.count_nonzero(
                    np.logical_and(
                        assignment_large_arteriole_mask.astype(bool, copy=False),
                        assignment_large_venule_mask.astype(bool, copy=False),
                    )
                )
            )
            overlap_after_cleanup = int(
                np.count_nonzero(
                    np.logical_and(
                        cleaned_arteriole_mask.astype(bool, copy=False),
                        cleaned_venule_mask.astype(bool, copy=False),
                    )
                )
            )
            print(
                "Fast-mode overlap cleanup voxel counts: "
                f"before={overlap_before_cleanup}, after={overlap_after_cleanup}."
            )
            if cleaned_arteriole_mask is not None and cleaned_venule_mask is not None:
                assignment_large_arteriole_mask = cleaned_arteriole_mask
                assignment_large_venule_mask = cleaned_venule_mask
            if write_fast_mode_preassignment_large_vessel_debug_3d_html:
                pre_assignment_after_cleanup_html_path = (
                    plot_dir / "pre_assignment_large_vessel_masks_after_overlap_cleanup_3d.html"
                )
                print(
                    "Rendering fast-mode pre-assignment debug view (after overlap cleanup)..."
                )
                visualization.visualize_3d_plotly_large_vessel_assignment(
                    G,
                    large_arteriole_mask=assignment_large_arteriole_mask,
                    large_venule_mask=assignment_large_venule_mask,
                    input_nodes=[],
                    output_nodes=[],
                    voxel_size_xyz=tuple(float(v) for v in voxel_size),
                    volume_downsample_stride=int(large_vessel_3d_volume_downsample_stride),
                    title=(
                        "Pre-Assignment Debug View (Fast Mode, After Overlap Cleanup): "
                        "Large-Vessel Masks + Graph"
                    ),
                    save_html_path=str(pre_assignment_after_cleanup_html_path),
                    show=False,
                )
                print(
                    "Saved fast-mode pre-assignment (after cleanup) large-vessel "
                    f"debug 3D visualization to: {pre_assignment_after_cleanup_html_path}"
                )
        # Keep downstream visualization/cache in sync with masks used for assignment.
        large_viz_arteriole_mask = np.asarray(
            assignment_large_arteriole_mask,
            dtype=bool,
        )
        large_viz_venule_mask = np.asarray(
            assignment_large_venule_mask,
            dtype=bool,
        )
        if use_legacy_large_vessel_assignment:
            auto_start_nodes, auto_output_nodes = (
                graph.select_terminal_nodes_from_large_vessel_masks_progressive_dilation(
                    G,
                    large_arteriole_mask=assignment_large_arteriole_mask,
                    large_venule_mask=assignment_large_venule_mask,
                    voxel_size_xyz=tuple(float(v) for v in voxel_size),
                    max_dilation_microns=float(effective_max_large_vessel_dilation_microns),
                    dilation_step_microns=5.0,
                    allow_overlap=False,
                    exclude_smaller_overlapping_volumes=(
                        apply_overlap_cleanup_prepass and not effective_assignment_fast_mode
                    ),
                    overlap_parallel_workers=int(automated_vessel_overlap_parallel_workers),
                )
            )
        else:
            robust_assignment = (
                graph.select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence(
                    G,
                    large_arteriole_mask=assignment_large_arteriole_mask,
                    large_venule_mask=assignment_large_venule_mask,
                    voxel_size_xyz=tuple(float(v) for v in voxel_size),
                    max_dilation_microns=float(effective_max_large_vessel_dilation_microns),
                    dilation_step_microns=5.0,
                    confidence_margin=float(automated_vessel_confidence_margin),
                    minimum_confidence=float(automated_vessel_min_confidence),
                    topology_penalty=float(automated_vessel_topology_penalty),
                    quality_max_overlap_fraction=float(automated_vessel_quality_max_overlap_fraction),
                    quality_min_terminal_coverage=float(automated_vessel_quality_min_terminal_coverage),
                    quality_max_component_count=int(automated_vessel_quality_max_component_count),
                    conservative_max_dilation_microns=float(
                        automated_vessel_conservative_max_dilation_microns
                    ),
                )
            )
            auto_start_nodes = list(robust_assignment["input_nodes"])
            auto_output_nodes = list(robust_assignment["output_nodes"])
            unresolved_nodes = list(robust_assignment["unresolved_nodes"])
            if unresolved_nodes:
                print(
                    "Robust large-vessel assignment unresolved terminals (manual QC suggested): "
                    f"{unresolved_nodes}"
                )
            print(
                "Robust large-vessel assignment confidence mode: "
                f"conservative_mode={bool(robust_assignment['conservative_mode'])}, "
                f"effective_max_dilation={float(robust_assignment['effective_max_dilation_microns']):.3f} microns."
            )
        auto_start_nodes, auto_output_nodes, dropped_auto_start_nodes, dropped_auto_output_nodes = (
            graph.filter_io_nodes_to_terminal_degree1(
                G,
                auto_start_nodes,
                auto_output_nodes,
            )
        )
        if dropped_auto_start_nodes or dropped_auto_output_nodes:
            print(
                "Filtered non-terminal nodes before automated-assignment visualization: "
                f"dropped_inputs={dropped_auto_start_nodes}, "
                f"dropped_outputs={dropped_auto_output_nodes}."
            )
        if not auto_start_nodes:
            raise ValueError(
                "automated_vessel_assignment=True found no terminal nodes in the "
                "arteriole mask (after any configured dilation)."
            )
        if not auto_output_nodes:
            raise ValueError(
                "automated_vessel_assignment=True found no terminal nodes in the "
                "venule mask (after any configured dilation)."
            )
        starting_node_coordinates = [
            tuple(np.asarray(G.nodes[node_id]["pos"], dtype=float))
            for node_id in auto_start_nodes
        ]
        output_node_coordinates = [
            tuple(np.asarray(G.nodes[node_id]["pos"], dtype=float))
            for node_id in auto_output_nodes
        ]
        print(
            "Automated vessel assignment selected "
            f"{len(starting_node_coordinates)} input coordinates from arteriole-mask overlap "
            f"and {len(output_node_coordinates)} output coordinates from venule-mask overlap."
        )
    elif cached_mask_payload is not None:
        cached_large_arteriole = cached_mask_payload.get("large_arteriole_mask")
        cached_large_venule = cached_mask_payload.get("large_venule_mask")
        if cached_large_arteriole is not None and cached_large_venule is not None:
            large_viz_arteriole_mask = np.asarray(cached_large_arteriole, dtype=bool)
            large_viz_venule_mask = np.asarray(cached_large_venule, dtype=bool)
            print(
                "Automated assignment disabled: reusing cached cleaned large-vessel "
                "masks for final large-vessel assignment visualization."
            )

    starting_nodes[:] = []
    output_nodes[:] = []
    arteriole_boundary_nodes[:] = []
    venule_boundary_nodes[:] = []
    used_preconfigured_io_nodes = False
    if automated_vessel_assignment:
        # Use direct terminal-node overlap assignment from vessel masks.
        start_nodes = auto_start_nodes
        out_nodes = [node_id for node_id in auto_output_nodes if node_id not in set(start_nodes)]
    else:
        if preconfigured_starting_nodes or preconfigured_output_nodes:
            used_preconfigured_io_nodes = True
            start_nodes = list(preconfigured_starting_nodes)
            out_nodes = [
                node_id
                for node_id in preconfigured_output_nodes
                if node_id not in set(start_nodes)
            ]
            print(
                "Using preconfigured STARTING_NODES/OUTPUT_NODES from settings "
                "because automated_vessel_assignment=False."
            )
        else:
            def _normalize_boundary_coords_for_check(
                coords: list[tuple[float, float, float]] | list | tuple,
                order: str,
            ) -> list[tuple[float, float, float]]:
                out: list[tuple[float, float, float]] = []
                order_norm = str(order).strip().lower()
                for c in (coords or []):
                    arr = np.asarray(c, dtype=float).ravel()
                    if arr.shape != (3,):
                        continue
                    if order_norm == "zyx":
                        arr = arr[[2, 1, 0]]
                    out.append((float(arr[0]), float(arr[1]), float(arr[2])))
                return out

            _check_boundary_coordinate_unit_consistency(
                G,
                coordinate_sets={
                    "starting_nodes": _normalize_boundary_coords_for_check(
                        starting_node_coordinates,
                        starting_node_coordinate_order,
                    ),
                    "output_nodes": _normalize_boundary_coords_for_check(
                        output_node_coordinates,
                        output_node_coordinate_order,
                    ),
                    "arteriole_boundary_nodes": _normalize_boundary_coords_for_check(
                        arteriole_boundary_node_coordinates,
                        arteriole_boundary_coordinate_order,
                    ),
                    "venule_boundary_nodes": _normalize_boundary_coords_for_check(
                        venule_boundary_node_coordinates,
                        venule_boundary_coordinate_order,
                    ),
                },
                mode=str(boundary_coordinate_unit_check_mode),
                max_fraction_of_graph_diagonal=float(
                    boundary_coordinate_unit_check_max_fraction_of_diagonal
                ),
            )
            start_nodes = graph.select_boundary_nodes_by_method(
                G,
                image.shape,
                method=starting_node_selection_method,
                node_role="input",
                coordinates=starting_node_coordinates,
                volume_boxes=starting_node_volumes,
                coordinate_order=starting_node_coordinate_order,
            )
            out_nodes = graph.select_boundary_nodes_by_method(
                G,
                image.shape,
                method=output_node_selection_method,
                node_role="output",
                coordinates=output_node_coordinates,
                volume_boxes=output_node_volumes,
                exclude_nodes=start_nodes,
                coordinate_order=output_node_coordinate_order,
            )
    # Enforce terminal-only I/O assignment.
    start_nodes, out_nodes, dropped_start_nodes, dropped_out_nodes = (
        graph.filter_io_nodes_to_terminal_degree1(G, start_nodes, out_nodes)
    )
    if dropped_start_nodes or dropped_out_nodes:
        print(
            "Filtered non-terminal boundary nodes from I/O assignment: "
            f"dropped_inputs={dropped_start_nodes}, dropped_outputs={dropped_out_nodes}."
        )
    starting_nodes.extend(start_nodes)
    output_nodes.extend(out_nodes)
    invalid_start_nodes = [node_id for node_id in starting_nodes if int(G.degree(node_id)) != 1]
    invalid_output_nodes = [node_id for node_id in output_nodes if int(G.degree(node_id)) != 1]
    if invalid_start_nodes or invalid_output_nodes:
        raise ValueError(
            "I/O assignment contains non-terminal nodes after filtering: "
            f"invalid_inputs={invalid_start_nodes}, invalid_outputs={invalid_output_nodes}."
        )
    if bool(remove_disconnected_io_components_after_final_assignment):
        G, io_prune_stats = graph.remove_components_without_connected_io(
            G,
            starting_nodes,
            output_nodes,
        )
        if int(io_prune_stats["removed_components"]) > 0:
            starting_nodes[:] = [int(node_id) for node_id in starting_nodes if node_id in G.nodes]
            output_nodes[:] = [int(node_id) for node_id in output_nodes if node_id in G.nodes]
            arteriole_boundary_nodes[:] = [
                int(node_id) for node_id in arteriole_boundary_nodes if node_id in G.nodes
            ]
            venule_boundary_nodes[:] = [
                int(node_id) for node_id in venule_boundary_nodes if node_id in G.nodes
            ]
            print(
                "Removed disconnected graph component(s) lacking STARTING_NODES or "
                "OUTPUT_NODES after final assignment: "
                f"removed_components={int(io_prune_stats['removed_components'])}, "
                f"removed_nodes={int(io_prune_stats['removed_nodes'])}, "
                f"remaining_nodes={int(io_prune_stats['remaining_nodes'])}."
            )
            if not starting_nodes or not output_nodes:
                raise ValueError(
                    "After removing disconnected components without both STARTING_NODES "
                    "and OUTPUT_NODES, no valid boundary nodes remained."
                )
    if automated_vessel_assignment and auto_persist_automated_io_assignment_to_settings:
        persist_automated_io_assignment_to_settings_file(
            settings_file_path=SETTINGS_FILE_PATH,
            assigned_start_nodes=[int(node_id) for node_id in starting_nodes],
            assigned_output_nodes=[int(node_id) for node_id in output_nodes],
        )
    if automated_vessel_assignment:
        if large_viz_arteriole_mask is None or large_viz_venule_mask is None:
            raise ValueError(
                "automated_vessel_assignment=True requires large arteriole/venule masks "
                "for visualization."
            )
        has_small_viz_masks = (
            small_viz_arteriole_mask is not None and small_viz_venule_mask is not None
        )
        if not has_small_viz_masks:
            print(
                "Small-vessel volume overlay unavailable for final large-vessel assignment "
                "view; skipping small-vessel volume rendering."
            )
        final_assignment_html_path = plot_dir / "final_graph_large_vessel_assignment_3d.html"
        visualization.visualize_3d_plotly_large_vessel_assignment(
            G,
            large_arteriole_mask=large_viz_arteriole_mask,
            large_venule_mask=large_viz_venule_mask,
            small_arteriole_mask=small_viz_arteriole_mask,
            small_venule_mask=small_viz_venule_mask,
            input_nodes=list(starting_nodes),
            output_nodes=list(output_nodes),
            arteriole_boundary_nodes=list(arteriole_boundary_nodes),
            venule_boundary_nodes=list(venule_boundary_nodes),
            voxel_size_xyz=tuple(float(v) for v in voxel_size),
            volume_downsample_stride=int(large_vessel_3d_volume_downsample_stride),
            title="Final Graph with Automated Large+Small Vessel Assignment (3D)",
            save_html_path=str(final_assignment_html_path),
            show=show_plots_in_ide or interactive_plots,
        )
        print(
            "Saved final automated large-vessel assignment 3D visualization to: "
            f"{final_assignment_html_path}"
        )
    used_nodes = set(starting_nodes) | set(output_nodes)
    if arteriole_boundary_node_coordinates or arteriole_boundary_node_volumes:
        art_boundary = graph.select_boundary_nodes_by_method(
            G,
            image.shape,
            method=arteriole_boundary_selection_method,
            node_role="input",
            coordinates=arteriole_boundary_node_coordinates,
            volume_boxes=arteriole_boundary_node_volumes,
            terminal_only=False,
            exclude_nodes=list(used_nodes),
            coordinate_order=arteriole_boundary_coordinate_order,
        )
        arteriole_boundary_nodes.extend(art_boundary)
        used_nodes.update(arteriole_boundary_nodes)
    elif (
        preconfigured_arteriole_boundary_nodes
        and not use_small_vessel_masks_for_boundary_assignment
    ):
        filtered_art_boundary = [
            node_id for node_id in preconfigured_arteriole_boundary_nodes
            if node_id in G.nodes and node_id not in used_nodes
        ]
        arteriole_boundary_nodes.extend(filtered_art_boundary)
        used_nodes.update(arteriole_boundary_nodes)
        print(
            "Using preconfigured ARTERIOLE_BOUNDARY_NODES from settings "
            "because mask/coordinate boundary assignment was not requested."
        )

    if venule_boundary_node_coordinates or venule_boundary_node_volumes:
        ven_boundary = graph.select_boundary_nodes_by_method(
            G,
            image.shape,
            method=venule_boundary_selection_method,
            node_role="output",
            coordinates=venule_boundary_node_coordinates,
            volume_boxes=venule_boundary_node_volumes,
            terminal_only=False,
            exclude_nodes=list(used_nodes),
            coordinate_order=venule_boundary_coordinate_order,
        )
        venule_boundary_nodes.extend(ven_boundary)
    elif (
        preconfigured_venule_boundary_nodes
        and not use_small_vessel_masks_for_boundary_assignment
    ):
        filtered_ven_boundary = [
            node_id for node_id in preconfigured_venule_boundary_nodes
            if node_id in G.nodes and node_id not in used_nodes
        ]
        venule_boundary_nodes.extend(filtered_ven_boundary)
        print(
            "Using preconfigured VENULE_BOUNDARY_NODES from settings "
            "because mask/coordinate boundary assignment was not requested."
        )
    if use_small_vessel_masks_for_boundary_assignment:
        if small_arteriole_mask is None or small_venule_mask is None:
            raise ValueError(
                "use_small_vessel_masks_for_boundary_assignment=True requires "
                "small_arteriole_mask_path and small_venule_mask_path."
            )
        if arteriole_boundary_nodes or venule_boundary_nodes:
            print(
                "Small-vessel boundary assignment enabled: overriding existing "
                "arteriole/venule boundary node selections."
            )
        arteriole_boundary_nodes[:] = []
        venule_boundary_nodes[:] = []
        cleanup_enabled_for_small = bool(
            small_vessel_boundary_assignment_enable_overlap_cleanup
        )
        apply_small_overlap_cleanup_prepass = bool(
            cleanup_enabled_for_small
            and (
                small_vessel_boundary_assignment_fast_mode
                or small_vessel_boundary_assignment_apply_overlap_cleanup_in_normal_mode
            )
        )
        if small_vessel_boundary_assignment_fast_mode:
            if cleanup_enabled_for_small:
                print(
                    "Small-vessel boundary assignment fast mode enabled: "
                    "removing overlap voxels from smaller overlapping components "
                    "before edge classification."
                )
            else:
                print(
                    "Small-vessel boundary assignment fast mode enabled, but overlap "
                    "cleanup is disabled by SMALL_VESSEL_BOUNDARY_ASSIGNMENT_ENABLE_OVERLAP_CLEANUP=False."
                )
        elif apply_small_overlap_cleanup_prepass:
            print(
                "Small-vessel boundary assignment: overlap cleanup pre-pass "
                "enabled in normal mode."
            )
        elif not cleanup_enabled_for_small:
            print("Small-vessel boundary assignment: overlap cleanup pre-pass disabled.")
        assignment_small_arteriole_mask = small_arteriole_mask
        assignment_small_venule_mask = small_venule_mask
        original_assignment_small_arteriole_mask = np.asarray(
            assignment_small_arteriole_mask,
            dtype=bool,
        ).copy()
        original_assignment_small_venule_mask = np.asarray(
            assignment_small_venule_mask,
            dtype=bool,
        ).copy()
        if float(small_vessel_min_component_volume_um3) > 0:
            (
                assignment_small_arteriole_mask,
                assignment_small_venule_mask,
                small_component_volume_stats,
            ) = graph.remove_small_vessel_components_by_volume(
                assignment_small_arteriole_mask,
                assignment_small_venule_mask,
                voxel_size_xyz=tuple(float(v) for v in voxel_size),
                min_component_volume_um3=float(small_vessel_min_component_volume_um3),
            )
            small_arteriole_stats = small_component_volume_stats.get("arteriole") or {}
            small_venule_stats = small_component_volume_stats.get("venule") or {}
            print(
                "Small-vessel component-volume filtering: "
                f"threshold={float(small_vessel_min_component_volume_um3):.3f} um^3, "
                f"removed_components(arteriole={int(small_arteriole_stats.get('removed_component_count', 0))}, "
                f"venule={int(small_venule_stats.get('removed_component_count', 0))}), "
                f"removed_volume_um3(arteriole={float(small_arteriole_stats.get('removed_volume_um3', 0.0)):.3f}, "
                f"venule={float(small_venule_stats.get('removed_volume_um3', 0.0)):.3f})."
            )
        if apply_small_overlap_cleanup_prepass:
            cleaned_small_arteriole_mask, cleaned_small_venule_mask = (
                graph.exclude_smaller_overlapping_small_vessel_components(
                    assignment_small_arteriole_mask,
                    assignment_small_venule_mask,
                )
            )
            if (
                cleaned_small_arteriole_mask is not None
                and cleaned_small_venule_mask is not None
            ):
                assignment_small_arteriole_mask = cleaned_small_arteriole_mask
                assignment_small_venule_mask = cleaned_small_venule_mask
        if bool(small_vessel_tangential_redefinition_enable):
            redefinition_start_s = time.perf_counter()
            redefinition_result = graph.redefine_small_masks_from_large_tangential_contact(
                small_arteriole_mask=assignment_small_arteriole_mask,
                small_venule_mask=assignment_small_venule_mask,
                large_arteriole_mask=large_arteriole_mask,
                large_venule_mask=large_venule_mask,
                voxel_size_xyz=tuple(float(v) for v in voxel_size),
                enable_redefinition=True,
                max_contact_distance_microns=float(
                    small_vessel_tangential_redefinition_max_contact_distance_microns
                ),
                touch_distance_microns=float(
                    small_vessel_tangential_redefinition_touch_distance_microns
                ),
                tangency_cosine_max=float(
                    small_vessel_tangential_redefinition_tangency_cosine_max
                ),
                reassignment_margin=float(
                    small_vessel_tangential_redefinition_margin
                ),
                reassignment_parallel_workers=int(
                    small_vessel_tangential_redefinition_parallel_workers
                ),
                use_gpu_acceleration=bool(use_gpu_mask_continuity_acceleration),
                opposite_exclusion_distance_microns=float(
                    small_vessel_mask_continuity_opposite_exclusion_distance_microns
                ),
                enable_sandwiched_component_reassignment=bool(
                    small_vessel_sandwich_reassign_enable
                ),
                sandwiched_max_endpoint_distance_microns=float(
                    small_vessel_sandwich_reassign_max_endpoint_distance_microns
                ),
                sandwiched_min_facing_cosine=float(
                    small_vessel_sandwich_reassign_min_facing_cosine
                ),
                sandwiched_max_axis_angle_degrees=float(
                    small_vessel_sandwich_reassign_max_axis_angle_degrees
                ),
            )
            assignment_small_arteriole_mask = np.asarray(
                redefinition_result["small_arteriole_mask"], dtype=bool
            )
            assignment_small_venule_mask = np.asarray(
                redefinition_result["small_venule_mask"], dtype=bool
            )
            redefinition_stats = redefinition_result["stats"]
            print(
                "Small-vessel tangential redefinition applied: "
                f"reassigned_to_arteriole={int(redefinition_stats['reassigned_to_arteriole'])}, "
                f"reassigned_to_venule={int(redefinition_stats['reassigned_to_venule'])}, "
                f"sandwich_flips(art={int(redefinition_stats['sandwiched_flips_to_arteriole'])}, "
                f"ven={int(redefinition_stats['sandwiched_flips_to_venule'])}), "
                f"unresolved={int(redefinition_stats['unresolved_components'])}/"
                f"{int(redefinition_stats['component_count'])} components, "
                f"workers={int(small_vessel_tangential_redefinition_parallel_workers)}, "
                f"gpu(requested={bool(use_gpu_mask_continuity_acceleration)}, "
                f"available={bool(redefinition_stats.get('gpu_acceleration_available', False))}), "
                f"phase_timing_s(setup={float(redefinition_stats.get('setup_phase_elapsed_s', 0.0)):.1f}, "
                f"tangential={float(redefinition_stats.get('tangential_phase_elapsed_s', 0.0)):.1f}, "
                f"sandwiched={float(redefinition_stats.get('sandwiched_phase_elapsed_s', 0.0)):.1f}, "
                f"total={time.perf_counter() - redefinition_start_s:.1f})."
            )
        if bool(small_vessel_mask_continuity_enable):
            continuity_start_s = time.perf_counter()
            print("Starting small-vessel continuity bridging...")
            continuity_result = graph.enforce_small_vessel_mask_continuity(
                small_arteriole_mask=assignment_small_arteriole_mask,
                small_venule_mask=assignment_small_venule_mask,
                large_arteriole_mask=(
                    None if large_arteriole_mask is None else large_arteriole_mask
                ),
                large_venule_mask=(
                    None if large_venule_mask is None else large_venule_mask
                ),
                voxel_size_xyz=tuple(float(v) for v in voxel_size),
                enable_continuity=True,
                allow_small_to_large=bool(
                    small_vessel_mask_continuity_allow_small_to_large
                ),
                allow_small_to_small=bool(
                    small_vessel_mask_continuity_allow_small_to_small
                ),
                enforce_cylinder_only=bool(
                    small_vessel_mask_continuity_enforce_cylinder_only
                ),
                min_cylindricality=float(
                    small_vessel_mask_continuity_min_cylindricality
                ),
                max_axis_angle_degrees=float(
                    small_vessel_mask_continuity_max_axis_angle_degrees
                ),
                min_facing_cosine=float(
                    small_vessel_mask_continuity_min_facing_cosine
                ),
                max_radius_ratio=float(
                    small_vessel_mask_continuity_max_radius_ratio
                ),
                max_bridge_distance_microns=float(
                    small_vessel_mask_continuity_max_bridge_distance_microns
                ),
                corridor_max_distance_microns=float(
                    small_vessel_mask_continuity_corridor_max_distance_microns
                ),
                opposite_exclusion_distance_microns=float(
                    small_vessel_mask_continuity_opposite_exclusion_distance_microns
                ),
                use_gpu_acceleration=bool(use_gpu_mask_continuity_acceleration),
            )
            assignment_small_arteriole_mask = np.asarray(
                continuity_result["small_arteriole_mask"], dtype=bool
            )
            assignment_small_venule_mask = np.asarray(
                continuity_result["small_venule_mask"], dtype=bool
            )
            continuity_stats = continuity_result["stats"]
            print(
                "Small-vessel continuity bridging applied "
                f"(cylinder_only={bool(small_vessel_mask_continuity_enforce_cylinder_only)}): "
                f"arteriole accepted={int(continuity_stats['arteriole']['accepted_bridges'])}/"
                f"{int(continuity_stats['arteriole']['attempted_bridges'])} "
                f"(prefiltered={int(continuity_stats['arteriole'].get('prefiltered_out_count', 0))}, "
                f"roi={tuple(continuity_stats['arteriole'].get('roi_shape_zyx', (0, 0, 0)))}, "
                f"t_s=(build={float(continuity_stats['arteriole'].get('candidate_build_elapsed_s', 0.0)):.1f}, "
                f"setup={float(continuity_stats['arteriole'].get('distance_setup_elapsed_s', 0.0)):.1f}, "
                f"loop={float(continuity_stats['arteriole'].get('bridge_loop_elapsed_s', 0.0)):.1f})), "
                f"venule accepted={int(continuity_stats['venule']['accepted_bridges'])}/"
                f"{int(continuity_stats['venule']['attempted_bridges'])} "
                f"(prefiltered={int(continuity_stats['venule'].get('prefiltered_out_count', 0))}, "
                f"roi={tuple(continuity_stats['venule'].get('roi_shape_zyx', (0, 0, 0)))}, "
                f"t_s=(build={float(continuity_stats['venule'].get('candidate_build_elapsed_s', 0.0)):.1f}, "
                f"setup={float(continuity_stats['venule'].get('distance_setup_elapsed_s', 0.0)):.1f}, "
                f"loop={float(continuity_stats['venule'].get('bridge_loop_elapsed_s', 0.0)):.1f})), "
                f"elapsed={time.perf_counter() - continuity_start_s:.1f}s."
            )
        # Use updated small masks for downstream visualization/cache.
        small_viz_arteriole_mask = np.asarray(
            assignment_small_arteriole_mask,
            dtype=bool,
        )
        small_viz_venule_mask = np.asarray(
            assignment_small_venule_mask,
            dtype=bool,
        )
        if (
            original_assignment_small_arteriole_mask is not None
            and original_assignment_small_venule_mask is not None
        ):
            original_art = np.asarray(original_assignment_small_arteriole_mask, dtype=bool)
            original_ven = np.asarray(original_assignment_small_venule_mask, dtype=bool)
            updated_art = np.asarray(assignment_small_arteriole_mask, dtype=bool)
            updated_ven = np.asarray(assignment_small_venule_mask, dtype=bool)
            small_changed_viz_arteriole_mask = np.logical_xor(original_art, updated_art)
            small_changed_viz_venule_mask = np.logical_xor(original_ven, updated_ven)
            art_added = int(np.count_nonzero(updated_art & (~original_art)))
            art_removed = int(np.count_nonzero(original_art & (~updated_art)))
            ven_added = int(np.count_nonzero(updated_ven & (~original_ven)))
            ven_removed = int(np.count_nonzero(original_ven & (~updated_ven)))
            art_to_ven = int(np.count_nonzero(original_art & updated_ven))
            ven_to_art = int(np.count_nonzero(original_ven & updated_art))
            print(
                "Small-vessel mask update deltas (voxels): "
                f"arteriole(+{art_added}, -{art_removed}), "
                f"venule(+{ven_added}, -{ven_removed}), "
                f"reassigned(art->ven={art_to_ven}, ven->art={ven_to_art})."
            )
        if float(small_vessel_mask_dilation_microns) > 0:
            print(
                "Configured small-vessel boundary-assignment dilation max: "
                f"{float(small_vessel_mask_dilation_microns):.3f} microns "
                "(applied as progressive 5-micron steps during assignment only)."
            )
        boundary_infer_start_s = time.perf_counter()
        print("Starting small-vessel boundary inference...")
        inferred_boundary_results = graph.infer_boundary_nodes_from_small_vessel_masks_progressive_dilation(
            G,
            small_arteriole_mask=assignment_small_arteriole_mask,
            small_venule_mask=assignment_small_venule_mask,
            voxel_size_xyz=tuple(float(v) for v in voxel_size),
            max_dilation_microns=float(small_vessel_mask_dilation_microns),
            dilation_step_microns=5.0,
            minimum_overlap_fraction=float(small_vessel_mask_min_overlap_fraction),
            allow_overlap=False,
            exclude_smaller_overlapping_volumes=False,
            overlap_parallel_workers=int(small_vessel_overlap_parallel_workers),
        )
        arteriole_boundary_nodes[:] = list(
            inferred_boundary_results["arteriole_boundary_nodes"]
        )
        venule_boundary_nodes[:] = list(inferred_boundary_results["venule_boundary_nodes"])
        print(
            "Small-vessel mask boundary assignment selected "
            f"{len(arteriole_boundary_nodes)} arteriole boundary nodes and "
            f"{len(venule_boundary_nodes)} venule boundary nodes "
            f"(min_overlap_fraction={float(small_vessel_mask_min_overlap_fraction):.3f}, "
            f"elapsed={time.perf_counter() - boundary_infer_start_s:.1f}s)."
        )
        print(
            "Small-vessel mask edge labels: "
            f"arteriole_edges={inferred_boundary_results['arteriole_edge_count']}, "
            f"venule_edges={inferred_boundary_results['venule_edge_count']}, "
            f"overlap_edges={inferred_boundary_results['overlap_edge_count']}."
        )
        fallback_enabled = bool(small_vessel_boundary_fallback_to_hop_distance)
        fallback_hops = int(small_vessel_boundary_fallback_hop_distance)
        if fallback_enabled and fallback_hops < 1:
            raise ValueError(
                "small_vessel_boundary_fallback_hop_distance must be >= 1 when "
                "SMALL_VESSEL_BOUNDARY_FALLBACK_TO_HOP_DISTANCE=True."
            )
        exclude_io_nodes = set(starting_nodes) | set(output_nodes)
        (
            arteriole_seed_edges_covered,
            arteriole_uncovered_edge_count,
            arteriole_seed_edge_count,
        ) = graph.seed_edges_have_full_mask_coverage(
            G,
            list(starting_nodes),
            assignment_small_arteriole_mask,
        )
        (
            venule_seed_edges_covered,
            venule_uncovered_edge_count,
            venule_seed_edge_count,
        ) = graph.seed_edges_have_full_mask_coverage(
            G,
            list(output_nodes),
            assignment_small_venule_mask,
        )
        fallback_arteriole_needed = not arteriole_seed_edges_covered
        fallback_venule_needed = not venule_seed_edges_covered
        if fallback_enabled and (fallback_arteriole_needed or fallback_venule_needed):
            print(
                "Small-vessel mask boundary assignment missed immediate seed-edge mask "
                "coverage on at least one side; "
                f"applying hop-distance fallback ({fallback_hops} edges)."
            )
            if fallback_arteriole_needed:
                print(
                    "Arteriole immediate-edge mask coverage miss: "
                    f"{arteriole_uncovered_edge_count}/{arteriole_seed_edge_count} seed edges "
                    "without small-arteriole mask overlap."
                )
                fallback_art = graph.select_nodes_at_hop_distance(
                    G,
                    list(starting_nodes),
                    fallback_hops,
                    exclude_nodes=exclude_io_nodes,
                )
                arteriole_boundary_nodes[:] = fallback_art
                print(
                    "Fallback arteriole boundary nodes selected from STARTING_NODES: "
                    f"{len(arteriole_boundary_nodes)} nodes at {fallback_hops} hops."
                )
            if fallback_venule_needed:
                print(
                    "Venule immediate-edge mask coverage miss: "
                    f"{venule_uncovered_edge_count}/{venule_seed_edge_count} seed edges "
                    "without small-venule mask overlap."
                )
                fallback_ven = graph.select_nodes_at_hop_distance(
                    G,
                    list(output_nodes),
                    fallback_hops,
                    exclude_nodes=exclude_io_nodes,
                )
                venule_boundary_nodes[:] = fallback_ven
                print(
                    "Fallback venule boundary nodes selected from OUTPUT_NODES: "
                    f"{len(venule_boundary_nodes)} nodes at {fallback_hops} hops."
                )
        if auto_persist_small_vessel_boundary_assignment_to_settings:
            persist_small_vessel_boundary_assignment_to_settings_file(
                settings_file_path=SETTINGS_FILE_PATH,
                assigned_arteriole_boundary_nodes=[
                    int(node_id) for node_id in arteriole_boundary_nodes
                ],
                assigned_venule_boundary_nodes=[
                    int(node_id) for node_id in venule_boundary_nodes
                ],
            )
        if write_small_vessel_boundary_labelling_3d_html:
            boundary_html_original = (
                Path(plot_dir) / "small_vessel_mask_boundary_labelling_3d_original.html"
            )
            boundary_html_updated = (
                Path(plot_dir) / "small_vessel_mask_boundary_labelling_3d_updated.html"
            )
            Path(plot_dir).mkdir(parents=True, exist_ok=True)
            original_plot_graph = G
            original_plot_arteriole_boundary_nodes = list(arteriole_boundary_nodes)
            original_plot_venule_boundary_nodes = list(venule_boundary_nodes)
            # Recompute edge/node mask labels on a graph copy so the original-mask
            # HTML uses topology labels consistent with the original small masks.
            if (
                use_small_vessel_masks_for_boundary_assignment
                and original_assignment_small_arteriole_mask is not None
                and original_assignment_small_venule_mask is not None
            ):
                original_plot_graph = G.copy()
                original_plot_result = (
                    graph.infer_boundary_nodes_from_small_vessel_masks_progressive_dilation(
                        original_plot_graph,
                        small_arteriole_mask=original_assignment_small_arteriole_mask,
                        small_venule_mask=original_assignment_small_venule_mask,
                        voxel_size_xyz=tuple(float(v) for v in voxel_size),
                        max_dilation_microns=float(small_vessel_mask_dilation_microns),
                        dilation_step_microns=5.0,
                        minimum_overlap_fraction=float(
                            small_vessel_mask_min_overlap_fraction
                        ),
                        allow_overlap=False,
                        exclude_smaller_overlapping_volumes=False,
                        overlap_parallel_workers=int(small_vessel_overlap_parallel_workers),
                    )
                )
                original_plot_arteriole_boundary_nodes = list(
                    original_plot_result["arteriole_boundary_nodes"]
                )
                original_plot_venule_boundary_nodes = list(
                    original_plot_result["venule_boundary_nodes"]
                )
                print(
                    "Original-mask boundary 3D plotting labels: "
                    f"arteriole_edges={int(original_plot_result['arteriole_edge_count'])}, "
                    f"venule_edges={int(original_plot_result['venule_edge_count'])}, "
                    f"overlap_edges={int(original_plot_result['overlap_edge_count'])}."
                )
            ok_original = graph.write_small_vessel_mask_boundary_labelling_3d_html(
                original_plot_graph,
                small_arteriole_mask=original_assignment_small_arteriole_mask,
                small_venule_mask=original_assignment_small_venule_mask,
                large_arteriole_mask=large_viz_arteriole_mask,
                large_venule_mask=large_viz_venule_mask,
                arteriole_boundary_nodes=original_plot_arteriole_boundary_nodes,
                venule_boundary_nodes=original_plot_venule_boundary_nodes,
                voxel_size_xyz=tuple(float(v) for v in voxel_size),
                output_html_path=boundary_html_original,
                volume_downsample_stride=int(small_vessel_3d_volume_downsample_stride),
            )
            ok_updated = graph.write_small_vessel_mask_boundary_labelling_3d_html(
                G,
                small_arteriole_mask=small_viz_arteriole_mask,
                small_venule_mask=small_viz_venule_mask,
                large_arteriole_mask=large_viz_arteriole_mask,
                large_venule_mask=large_viz_venule_mask,
                arteriole_boundary_nodes=arteriole_boundary_nodes,
                venule_boundary_nodes=venule_boundary_nodes,
                voxel_size_xyz=tuple(float(v) for v in voxel_size),
                output_html_path=boundary_html_updated,
                volume_downsample_stride=int(small_vessel_3d_volume_downsample_stride),
            )
            if ok_original:
                print(
                    "Saved interactive 3D small-vessel boundary view "
                    f"(original masks): {boundary_html_original}"
                )
            else:
                print(
                    "Small-vessel boundary 3D original-mask HTML not written "
                    "(install plotly to enable)."
                )
            if ok_updated:
                print(
                    "Saved interactive 3D small-vessel boundary view "
                    f"(updated masks): {boundary_html_updated}"
                )
            else:
                print(
                    "Small-vessel boundary 3D updated-mask HTML not written "
                    "(install plotly to enable)."
                )
    elif (
        write_small_vessel_boundary_labelling_3d_html
        and small_viz_arteriole_mask is not None
        and small_viz_venule_mask is not None
        and (arteriole_boundary_nodes or venule_boundary_nodes)
    ):
        boundary_html = Path(plot_dir) / "small_vessel_mask_boundary_labelling_3d.html"
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        ok = graph.write_small_vessel_mask_boundary_labelling_3d_html(
            G,
            small_arteriole_mask=small_viz_arteriole_mask,
            small_venule_mask=small_viz_venule_mask,
            large_arteriole_mask=large_viz_arteriole_mask,
            large_venule_mask=large_viz_venule_mask,
            arteriole_boundary_nodes=arteriole_boundary_nodes,
            venule_boundary_nodes=venule_boundary_nodes,
            voxel_size_xyz=tuple(float(v) for v in voxel_size),
            output_html_path=boundary_html,
            volume_downsample_stride=int(small_vessel_3d_volume_downsample_stride),
        )
        if ok:
            print(
                "Saved interactive 3D small-vessel boundary view from "
                f"cached/loaded masks: {boundary_html}"
            )
        else:
            print(
                "Small-vessel boundary 3D HTML not written (install plotly to enable)."
            )
    cache_saved = save_cleaned_mask_cache(
        mask_cache_path,
        image_shape_zyx=tuple(int(v) for v in image.shape),
        large_arteriole_mask=large_viz_arteriole_mask,
        large_venule_mask=large_viz_venule_mask,
        small_arteriole_mask=small_viz_arteriole_mask,
        small_venule_mask=small_viz_venule_mask,
    )
    if cache_saved:
        print(f"Saved cleaned mask cache for visualization reuse to: {mask_cache_path}")
    else:
        print(
            "Skipped cleaned mask cache save because no large/small vessel volumes "
            "were available."
        )
    if automated_vessel_assignment:
        print(
            f"Selected {len(starting_nodes)} STARTING_NODES and {len(output_nodes)} "
            "OUTPUT_NODES directly from terminal-node overlap with vessel masks."
        )
    elif used_preconfigured_io_nodes:
        print(
            f"Selected {len(starting_nodes)} STARTING_NODES and {len(output_nodes)} "
            "OUTPUT_NODES from preconfigured settings node IDs."
        )
    else:
        print(
            f"Selected {len(starting_nodes)} STARTING_NODES and {len(output_nodes)} "
            "OUTPUT_NODES from manual coordinates."
        )
    print(f"Starting nodes are: {starting_nodes}")
    print(f"Output nodes are: {output_nodes}")
    print(f"Arteriole boundary nodes are: {arteriole_boundary_nodes}")
    print(f"Venule boundary nodes are: {venule_boundary_nodes}")

    if starting_nodes and output_nodes:
        resistance_node_pairs = haemodynamics.find_connected_start_output_pairs(
            G,
            starting_nodes,
            output_nodes,
        )
        if not resistance_node_pairs:
            raise ValueError(
                "No connected STARTING_NODES -> OUTPUT_NODES pairs found in the graph. "
                "Equivalent resistance requires connected node pairs."
            )
        resistance_node_pair = resistance_node_pairs[0]
        print(
            "Auto-selected default resistance node pair for single-pair APIs: "
            f"{resistance_node_pair}"
        )
        print(
            "Connected STARTING_NODES -> OUTPUT_NODES pairs for equivalent "
            f"resistance testing: {len(resistance_node_pairs)}"
        )
    else:
        if automated_vessel_assignment:
            raise ValueError(
                "No starting or output nodes found from terminal-node overlap with "
                "arteriole/venule masks."
            )
        raise ValueError(
            "No starting or output nodes found from manual input coordinates."
        )

    # 4) Add branch orders and hemodynamic edge weights.
    #HD note - eventually pericyte localisation should be able to be either determined by this manual method, or via loading in a segmented image of pericytes?
    #HD note - eventually add in probability of pericyte contraction?
    if starting_nodes:
        # Branch-order seeds follow mode-specific source selection:
        # - large-vessel seeds (start/output): automated when large-vessel automated
        #   assignment is active; otherwise settings fallback.
        # - small-vessel boundary seeds: automated when small-vessel mask boundary
        #   assignment is active; otherwise settings fallback.
        branch_starting_nodes = list(starting_nodes)
        branch_output_nodes = list(output_nodes)
        branch_arteriole_boundary_nodes = list(arteriole_boundary_nodes)
        branch_venule_boundary_nodes = list(venule_boundary_nodes)

        use_automated_large_branch_seeds = bool(
            automated_vessel_assignment and use_large_vessel_masks
        )
        use_automated_small_branch_seeds = bool(use_small_vessel_masks_for_boundary_assignment)

        if use_automated_large_branch_seeds:
            print(
                "Branch-order assignment: using automated large-vessel STARTING/OUTPUT "
                "nodes."
            )
        else:
            if preconfigured_starting_nodes:
                branch_starting_nodes = [
                    int(node_id)
                    for node_id in preconfigured_starting_nodes
                    if node_id in G.nodes
                ]
            if preconfigured_output_nodes:
                branch_output_nodes = [
                    int(node_id) for node_id in preconfigured_output_nodes if node_id in G.nodes
                ]
            print(
                "Branch-order assignment: using settings STARTING/OUTPUT nodes "
                f"(start={len(branch_starting_nodes)}, output={len(branch_output_nodes)})."
            )

        if use_automated_small_branch_seeds:
            print(
                "Branch-order assignment: using automated small-vessel boundary nodes."
            )
        else:
            if preconfigured_arteriole_boundary_nodes:
                branch_arteriole_boundary_nodes = [
                    int(node_id)
                    for node_id in preconfigured_arteriole_boundary_nodes
                    if node_id in G.nodes
                ]
            if preconfigured_venule_boundary_nodes:
                branch_venule_boundary_nodes = [
                    int(node_id)
                    for node_id in preconfigured_venule_boundary_nodes
                    if node_id in G.nodes
                ]
            print(
                "Branch-order assignment: using settings boundary nodes "
                f"(arteriole={len(branch_arteriole_boundary_nodes)}, "
                f"venule={len(branch_venule_boundary_nodes)})."
            )

        use_hierarchical_assignment = bool(
            branch_arteriole_boundary_nodes
            and branch_venule_boundary_nodes
            and branch_output_nodes
        )
        expects_hierarchical_assignment = bool(
            automated_vessel_assignment or use_small_vessel_masks_for_boundary_assignment
        )
        if (
            strict_branch_order_assignment
            and expects_hierarchical_assignment
            and not use_hierarchical_assignment
        ):
            raise ValueError(
                "Strict branch-order assignment is enabled, but hierarchical "
                "assignment prerequisites are missing. "
                f"Need non-empty output_nodes, arteriole_boundary_nodes, and "
                f"venule_boundary_nodes. Got counts: "
                f"output_nodes={len(branch_output_nodes)}, "
                f"arteriole_boundary_nodes={len(branch_arteriole_boundary_nodes)}, "
                f"venule_boundary_nodes={len(branch_venule_boundary_nodes)}. "
                "Fix mask inputs/thresholds or disable strict_branch_order_assignment."
            )
        if use_hierarchical_assignment:
            branch_assignment_results = graph.assign_hierarchical_branch_orders(
                G,
                starting_nodes=branch_starting_nodes,
                output_nodes=branch_output_nodes,
                arteriole_boundary_nodes=branch_arteriole_boundary_nodes,
                venule_boundary_nodes=branch_venule_boundary_nodes,
            )
            print(
                "Assigned hierarchical branch orders "
                "(Art*/Ven* first, then capillary B* from arteriole boundary)."
            )
            print(f"Branch assignment summary: {branch_assignment_results}")
        else:
            graph.assign_branch_orders(G, branch_starting_nodes)
            print(
                "Assigned capillary branch orders from STARTING_NODES only "
                "(no arteriole/venule boundary-node sets supplied)."
            )

        vessel_type_3d_path = plot_dir / "vessel_types_assigned_3d.html"
        visualization.visualize_3d_plotly_vessel_types(
            G,
            title="Assigned Vessel Types (Interactive 3D)",
            save_html_path=str(vessel_type_3d_path),
            show=False,
        )
        print(
            "Saved vessel-type 3D visualization after branch assignment to: "
            f"{vessel_type_3d_path}"
        )
        if large_viz_arteriole_mask is not None and large_viz_venule_mask is not None:
            has_small_viz_masks = (
                small_viz_arteriole_mask is not None and small_viz_venule_mask is not None
            )
            if not has_small_viz_masks:
                print(
                    "Small-vessel volume overlay unavailable for final vessel-type view; "
                    "skipping small-vessel volume rendering."
                )
            final_assignment_html_path = plot_dir / "final_graph_large_vessel_assignment_3d.html"
            visualization.visualize_3d_plotly_large_vessel_assignment(
                G,
                large_arteriole_mask=large_viz_arteriole_mask,
                large_venule_mask=large_viz_venule_mask,
                small_arteriole_mask=small_viz_arteriole_mask,
                small_venule_mask=small_viz_venule_mask,
                input_nodes=list(starting_nodes),
                output_nodes=list(output_nodes),
                arteriole_boundary_nodes=list(arteriole_boundary_nodes),
                venule_boundary_nodes=list(venule_boundary_nodes),
                voxel_size_xyz=tuple(float(v) for v in voxel_size),
                volume_downsample_stride=int(large_vessel_3d_volume_downsample_stride),
                title=(
                    "Final Graph with Automated Large+Small Vessel Assignment "
                    "(Vessel Types + Branch Orders, 3D)"
                ),
                save_html_path=str(final_assignment_html_path),
                show=show_plots_in_ide or interactive_plots,
            )
            print(
                "Saved final automated large-vessel assignment 3D visualization to: "
                f"{final_assignment_html_path}"
            )
            if (
                small_changed_viz_arteriole_mask is not None
                and small_changed_viz_venule_mask is not None
            ):
                changed_small_html_path = (
                    plot_dir / "final_graph_small_volume_changes_only_3d.html"
                )
                zero_large = np.zeros_like(large_viz_arteriole_mask, dtype=bool)
                visualization.visualize_3d_plotly_large_vessel_assignment(
                    G,
                    large_arteriole_mask=zero_large,
                    large_venule_mask=zero_large,
                    small_arteriole_mask=small_changed_viz_arteriole_mask,
                    small_venule_mask=small_changed_viz_venule_mask,
                    input_nodes=list(starting_nodes),
                    output_nodes=list(output_nodes),
                    arteriole_boundary_nodes=list(arteriole_boundary_nodes),
                    venule_boundary_nodes=list(venule_boundary_nodes),
                    voxel_size_xyz=tuple(float(v) for v in voxel_size),
                    volume_downsample_stride=int(large_vessel_3d_volume_downsample_stride),
                    title=(
                        "Final Graph with Changed Small Volumes Only "
                        "(Switching/Continuity/Reassignment, 3D)"
                    ),
                    save_html_path=str(changed_small_html_path),
                    show=False,
                )
                print(
                    "Saved changed-small-volume-only 3D visualization to: "
                    f"{changed_small_html_path}"
                )
        if not run_haemodynamics:
            print(
                "Haemodynamics disabled; skipping diameter fitting and "
                "Poiseuille conductance assignment."
            )
        if run_haemodynamics and use_fwhm_edge_diameters:
            if fwhm_raw_tiff_path is None:
                raise ValueError(
                    "use_fwhm_edge_diameters=True requires fwhm_raw_tiff_path."
                )
            raw_p = io.resolve_image_path_with_optional_zip(Path(fwhm_raw_tiff_path))
            voxel_sz = tuple(
                float(v) for v in G.graph.get("image_voxel_size_xyz", voxel_size)
            )
            fwhm_summary = haemodynamics.automated.measure_edge_diameters_fwhm_from_raw_tiff(
                G,
                raw_tiff_path=raw_p,
                voxel_size_xyz=voxel_sz,
                sample_spacing_along_edge_um=float(fwhm_sample_spacing_along_edge_um),
                transverse_profile_step_um=float(fwhm_transverse_profile_step_um),
                transverse_half_extent_um=float(fwhm_transverse_half_extent_um),
                diameter_guess_um=(
                    None
                    if fwhm_diameter_guess_um is None
                    else float(fwhm_diameter_guess_um)
                ),
                background_label=int(fwhm_background_label),
                junction_label=int(fwhm_junction_label),
                min_total_extent_multiplier=float(fwhm_min_total_extent_multiplier),
                profile_baseline_mode=fwhm_profile_baseline_mode,
                profile_baseline_wing_fraction=float(fwhm_profile_baseline_wing_fraction),
                constrain_fitted_baseline=bool(fwhm_constrain_fitted_baseline),
                allow_junction_crossing=bool(fwhm_allow_junction_crossing),
                baseline_constraint_half_width_ptp=float(
                    fwhm_baseline_constraint_half_width_ptp
                ),
                clip_profile_to_single_vessel=bool(fwhm_clip_profile_to_single_vessel),
                clip_min_drop_fraction_of_center=float(
                    fwhm_clip_min_drop_fraction_of_center
                ),
                clip_re_rise_fraction_of_center=float(
                    fwhm_clip_re_rise_fraction_of_center
                ),
                branch_endpoint_exclusion_um=float(
                    fwhm_branch_endpoint_exclusion_um
                ),
                terminal_endpoint_exclusion_um=float(
                    fwhm_terminal_endpoint_exclusion_um
                ),
                junction_proximity_exclusion_um=float(
                    fwhm_junction_proximity_exclusion_um
                ),
                enforce_same_edge_locality=bool(fwhm_enforce_same_edge_locality),
                same_edge_arc_window_um=(
                    None
                    if fwhm_same_edge_arc_window_um is None
                    else float(fwhm_same_edge_arc_window_um)
                ),
                same_edge_arc_window_multiplier=float(
                    fwhm_same_edge_arc_window_multiplier
                ),
                same_edge_arc_window_min_um=float(
                    fwhm_same_edge_arc_window_min_um
                ),
                cap_half_extent_by_nonlocal_same_edge_distance=bool(
                    fwhm_cap_half_extent_by_nonlocal_same_edge_distance
                ),
                nonlocal_same_edge_arc_separation_um=float(
                    fwhm_nonlocal_same_edge_arc_separation_um
                ),
                nonlocal_same_edge_half_extent_factor=float(
                    fwhm_nonlocal_same_edge_half_extent_factor
                ),
                reject_samples_with_center_offset=bool(
                    fwhm_reject_samples_with_center_offset
                ),
                max_fit_center_offset_um=float(
                    fwhm_max_fit_center_offset_um
                ),
                reject_samples_with_low_fit_r2=bool(
                    fwhm_reject_samples_with_low_fit_r2
                ),
                min_fit_r2=float(fwhm_min_fit_r2),
                edge_parallel_workers=(
                    None
                    if fwhm_edge_parallel_workers is None
                    else int(fwhm_edge_parallel_workers)
                ),
                edge_parallel_batch_size=int(fwhm_edge_parallel_batch_size),
                min_valid_cross_section_span_um=float(
                    fwhm_min_valid_cross_section_span_um
                ),
                min_valid_profile_count_per_edge=int(
                    fwhm_min_valid_profile_count_per_edge
                ),
                diameter_aggregation_trim_fraction=float(
                    fwhm_diameter_aggregation_trim_fraction
                ),
                diameter_bounds_mode=str(fwhm_diameter_bounds_mode),
                diameter_bounds_by_vessel_class_um=(
                    None
                    if fwhm_diameter_bounds_by_vessel_class_um is None
                    else {
                        str(k): (float(v[0]), float(v[1]))
                        for k, v in fwhm_diameter_bounds_by_vessel_class_um.items()
                    }
                ),
            )
            print(f"FWHM diameter measurement summary: {fwhm_summary}")
            if do_pericyte_constriction:
                print(
                    "Pericyte mode: passive diameter d1 from per-edge FWHM where available, "
                    "else DIAMETER_BY_BRANCH_ORDER; d2 = d1 * CONSTRICTION_BY_BRANCH_ORDER."
                )
        elif run_haemodynamics and not use_fwhm_edge_diameters:
            print(
                "Vessel diameters: manual mode (DIAMETER_BY_BRANCH_ORDER / "
                "set_poiseuille_resistances without per-edge FWHM)."
            )
        comparison_active_pericyte_indices: list[int] | None = None
        comparison_active_center_indices_by_edge: dict[str, list[int]] | None = None
        if run_haemodynamics and run_pericyte_resistance_comparison:
            comparison_csv_path = output_dir / f"{image_path.stem}_pericyte_resistance_comparison.csv"
            comparison_results = (
                pericyte_comparison_haemodynamics.compare_baseline_vs_pericyte_constriction(
                    G,
                    diameter_by_branch_order=diameter_by_branch_order,
                    constriction_factor_by_branch_order=constriction_by_branch_order,
                    resistance_node_pair=resistance_node_pair,
                    resistance_node_pairs=resistance_node_pairs,
                    output_csv_path=comparison_csv_path,
                    baseline_factor_value=float(pericyte_comparison_baseline_value),
                    constricted_factor_value=float(pericyte_comparison_constricted_value),
                    use_pericyte_mask_constriction=bool(use_pericyte_mask_constriction),
                    pericyte_mask_path=pericyte_mask_path,
                    pericyte_mask_h5_dataset_name=pericyte_mask_h5_dataset_name,
                    max_assignment_distance_um=(
                        None
                        if pericyte_max_assignment_distance_um is None
                        else float(pericyte_max_assignment_distance_um)
                    ),
                    min_pericyte_diameter_um=(
                        None
                        if pericyte_min_diameter_um is None
                        else float(pericyte_min_diameter_um)
                    ),
                    max_pericyte_diameter_um=(
                        None
                        if pericyte_max_diameter_um is None
                        else float(pericyte_max_diameter_um)
                    ),
                    prefer_edge_fwhm_baseline=bool(use_fwhm_edge_diameters),
                    constriction_length=float(pericyte_constriction_length_um),
                    constriction_spacing=float(pericyte_constriction_spacing_um),
                    use_probabilistic_pericyte_constriction=bool(
                        use_probabilistic_pericyte_constriction
                    ),
                    pericyte_constriction_probability=float(
                        pericyte_constriction_probability
                    ),
                    input_p_bc=float(input_p_bc),
                    output_p_bc=float(output_p_bc),
                )
            )
            if (
                reuse_comparison_pericyte_cohort_for_main_run
                and use_probabilistic_pericyte_constriction
            ):
                if use_pericyte_mask_constriction:
                    selected = comparison_results.get("active_pericyte_indices")
                    comparison_active_pericyte_indices = (
                        [int(idx) for idx in selected] if selected else []
                    )
                else:
                    selected_map = comparison_results.get("active_center_indices_by_edge")
                    if isinstance(selected_map, dict):
                        comparison_active_center_indices_by_edge = {
                            str(edge_id): [int(idx) for idx in idx_list]
                            for edge_id, idx_list in selected_map.items()
                        }
            print(
                "Pericyte resistance comparison complete: "
                f"baseline={comparison_results['baseline_resistance']:.6f}, "
                f"constricted={comparison_results['constricted_resistance']:.6f}, "
                f"delta={comparison_results['delta']:.6f}, "
                f"change={comparison_results['percent_change']:.3f}%."
            )
            print(
                "Saved pericyte resistance comparison CSV to: "
                f"{comparison_results['output_csv_path']}"
            )
            print(
                "Saved pericyte resistance before/after plot to: "
                f"{comparison_results['output_plot_path']}"
            )
            print(
                "Saved pericyte flow before/after plot to: "
                f"{comparison_results['output_flow_plot_path']}"
            )
            print("Pericyte resistance comparison plot mode: mode=d1d2")
        if run_haemodynamics and run_arteriole_resistance_comparison:
            arteriole_comparison_csv_path = (
                output_dir / f"{image_path.stem}_arteriole_resistance_comparison.csv"
            )
            arteriole_comparison_results = (
                arteriole_comparison_haemodynamics.compare_baseline_vs_arteriole_dilation(
                    G,
                    diameter_by_branch_order=diameter_by_branch_order,
                    resistance_node_pair=resistance_node_pair,
                    resistance_node_pairs=resistance_node_pairs,
                    output_csv_path=arteriole_comparison_csv_path,
                    baseline_factor_value=float(arteriole_comparison_baseline_value),
                    dilated_factor_value=float(arteriole_comparison_dilated_value),
                    arteriole_branch_prefix=str(arteriole_comparison_branch_prefix),
                    prefer_edge_fwhm_diameter=bool(use_fwhm_edge_diameters),
                    use_constriction_integrator=bool(
                        arteriole_comparison_use_constriction_integrator
                    ),
                    constriction_factor_by_branch_order=constriction_by_branch_order,
                    constriction_length=float(pericyte_constriction_length_um),
                    constriction_spacing=float(pericyte_constriction_spacing_um),
                    input_p_bc=float(input_p_bc),
                    output_p_bc=float(output_p_bc),
                )
            )
            print(
                "Arteriole resistance comparison complete: "
                f"baseline={arteriole_comparison_results['baseline_resistance']:.6f}, "
                f"dilated={arteriole_comparison_results['dilated_resistance']:.6f}, "
                f"delta={arteriole_comparison_results['delta']:.6f}, "
                f"change={arteriole_comparison_results['percent_change']:.3f}%."
            )
            print(
                "Saved arteriole resistance comparison CSV to: "
                f"{arteriole_comparison_results['output_csv_path']}"
            )
            print(
                "Saved arteriole resistance before/after plot to: "
                f"{arteriole_comparison_results['output_plot_path']}"
            )
            print(
                "Saved arteriole flow before/after plot to: "
                f"{arteriole_comparison_results['output_flow_plot_path']}"
            )
            print(
                "Arteriole resistance comparison plot mode: mode="
                f"{'d1d2' if bool(arteriole_comparison_use_constriction_integrator) else 'passive'}"
            )
        if run_haemodynamics and run_capillary_resistance_comparison:
            capillary_comparison_csv_path = (
                output_dir / f"{image_path.stem}_capillary_resistance_comparison.csv"
            )
            capillary_comparison_results = (
                capillary_comparison_haemodynamics.compare_baseline_vs_passive_capillary_dilation(
                    G,
                    diameter_by_branch_order=diameter_by_branch_order,
                    resistance_node_pair=resistance_node_pair,
                    resistance_node_pairs=resistance_node_pairs,
                    output_csv_path=capillary_comparison_csv_path,
                    baseline_factor_value=float(capillary_comparison_baseline_value),
                    dilated_factor_value=float(capillary_comparison_dilated_value),
                    capillary_branch_prefix=str(capillary_comparison_branch_prefix),
                    prefer_edge_fwhm_diameter=bool(use_fwhm_edge_diameters),
                    use_constriction_integrator=bool(
                        capillary_comparison_use_constriction_integrator
                    ),
                    constriction_factor_by_branch_order=(
                        constriction_by_branch_order
                        if bool(capillary_comparison_use_constriction_integrator)
                        else None
                    ),
                    constriction_length=float(pericyte_constriction_length_um),
                    constriction_spacing=float(pericyte_constriction_spacing_um),
                    input_p_bc=float(input_p_bc),
                    output_p_bc=float(output_p_bc),
                )
            )
            print(
                "Capillary resistance comparison complete: "
                f"baseline={capillary_comparison_results['baseline_resistance']:.6f}, "
                f"dilated={capillary_comparison_results['dilated_resistance']:.6f}, "
                f"delta={capillary_comparison_results['delta']:.6f}, "
                f"change={capillary_comparison_results['percent_change']:.3f}%."
            )
            print(
                "Saved capillary resistance comparison CSV to: "
                f"{capillary_comparison_results['output_csv_path']}"
            )
            print(
                "Saved capillary resistance before/after plot to: "
                f"{capillary_comparison_results['output_plot_path']}"
            )
            print(
                "Saved capillary flow before/after plot to: "
                f"{capillary_comparison_results['output_flow_plot_path']}"
            )
            print(
                "Capillary resistance comparison plot mode: mode="
                f"{'d1d2' if bool(capillary_comparison_use_constriction_integrator) else 'passive'}"
            )
        if run_haemodynamics:
            poiseuille_model = haemodynamics.PoiseuilleModel(
                constriction_length=float(pericyte_constriction_length_um),
                constriction_spacing=float(pericyte_constriction_spacing_um),
            )
            if do_pericyte_constriction:
                if use_pericyte_mask_constriction:
                    if pericyte_mask_path is None:
                        raise ValueError(
                            "pericyte_mask_path must be set when "
                            "use_pericyte_mask_constriction=True."
                        )
                    G, results = pericyte_mask_haemodynamics.set_poiseuille_resistances_with_pericyte_mask(
                        G,
                        diameter_by_branch_order=diameter_by_branch_order,
                        constriction_factor_by_branch_order=constriction_by_branch_order,
                        pericyte_mask_path=pericyte_mask_path,
                        pericyte_mask_h5_dataset_name=pericyte_mask_h5_dataset_name,
                        max_assignment_distance_um=(
                            None
                            if pericyte_max_assignment_distance_um is None
                            else float(pericyte_max_assignment_distance_um)
                        ),
                        min_pericyte_diameter_um=(
                            None
                            if pericyte_min_diameter_um is None
                            else float(pericyte_min_diameter_um)
                        ),
                        max_pericyte_diameter_um=(
                            None
                            if pericyte_max_diameter_um is None
                            else float(pericyte_max_diameter_um)
                        ),
                        prefer_edge_fwhm_baseline=bool(use_fwhm_edge_diameters),
                        constriction_length=float(pericyte_constriction_length_um),
                        use_probabilistic_constriction=bool(
                            use_probabilistic_pericyte_constriction
                        ),
                        constriction_probability=float(pericyte_constriction_probability),
                        active_pericyte_indices=(
                            comparison_active_pericyte_indices
                            if (
                                reuse_comparison_pericyte_cohort_for_main_run
                                and use_probabilistic_pericyte_constriction
                            )
                            else None
                        ),
                    )
                    print(
                        "Results from set_poiseuille_resistances_with_pericyte_mask "
                        f"(centroid-based d2 from mask): {results}"
                    )
                else:
                    if use_probabilistic_pericyte_constriction:
                        G, results = (
                            probability_haemodynamics
                            .set_poiseuille_resistances_with_probabilistic_periodic_constrictions(
                                G,
                                diameter_by_branch_order=diameter_by_branch_order,
                                constriction_factor_by_branch_order=constriction_by_branch_order,
                                prefer_edge_fwhm_baseline=bool(use_fwhm_edge_diameters),
                                constriction_length=float(pericyte_constriction_length_um),
                                constriction_spacing=float(pericyte_constriction_spacing_um),
                                constriction_probability=float(pericyte_constriction_probability),
                                active_center_indices_by_edge=(
                                    comparison_active_center_indices_by_edge
                                    if (
                                        reuse_comparison_pericyte_cohort_for_main_run
                                        and use_probabilistic_pericyte_constriction
                                    )
                                    else None
                                ),
                            )
                        )
                        print(
                            "Results from probabilistic periodic constrictions "
                            f"(active sites={results.get('active_periodic_pericyte_sites')}, "
                            f"total sites={results.get('total_periodic_pericyte_sites')}): "
                            f"{results}"
                        )
                    else:
                        if use_fwhm_edge_diameters:
                            G, results = poiseuille_model.set_poiseuille_resistances_with_constrictions(
                                G,
                                diameter_by_branch_order,
                                prefer_edge_fwhm_baseline=True,
                                constriction_factor_by_branch_order=constriction_by_branch_order,
                            )
                            print(
                                "Results from set_poiseuille_resistances_with_constrictions "
                                f"(FWHM baseline d1, constriction factors): {results}"
                            )
                        else:
                            diameter_by_branch_order_enhanced = {}
                            for branch_order, diameter in diameter_by_branch_order.items():
                                diameter_by_branch_order_enhanced[branch_order] = {
                                    "d1": diameter,
                                    "d2": diameter * constriction_by_branch_order[branch_order],
                                }

                            G, results = poiseuille_model.set_poiseuille_resistances_with_constrictions(
                                G,
                                diameter_by_branch_order_enhanced,
                            )
                            print(
                                f"Results from set_poiseuille_resistances_with_constrictions: {results}"
                            )
            else:
                G, results = poiseuille_model.set_poiseuille_resistances(
                    G,
                    diameter_by_branch_order,
                    prefer_edge_fwhm_diameter=bool(use_fwhm_edge_diameters),
                )
                _diam_mode = (
                    "per-edge FWHM (Gaussian fit) with branch-order fallback"
                    if use_fwhm_edge_diameters
                    else "branch-order table only"
                )
                print(f"Results from set_poiseuille_resistances ({_diam_mode}): {results}")

            G, results_2 = poiseuille_model.set_poiseuille_edge_resistances(
                G,
                custom_edges,
                edge_diameter=6.0,
                use_resistance=True,
            )

            print(f"Results from set_poiseuille_edge_resistances: {results_2}")
            # create list of resistances of all edges
            resistances = []
            skipped_missing_resistance = 0
            for u, v, key in G.edges(keys=True):
                resistance = G[u][v][key].get("resistance")
                if resistance is None:
                    skipped_missing_resistance += 1
                    continue
                resistances.append(resistance)

            if skipped_missing_resistance > 0:
                print(
                    "Skipped edges without branch-order resistance assignment: "
                    f"{skipped_missing_resistance}"
                )

    # 5) Export vessels/pericytes/nodes to VTK and optionally visualize in PyVista.
    # FA I have no idea if pericyte location is correct. AI did that part.
    # FA I don't fully understand how pericyte location is currently determined?
    vtk_export_payload: dict[str, object] = {}
    if run_haemodynamics and VTK_export:
        vtk_export = visualization.graph_to_vtk(G, vtk_output_prefix)
        vtk_export_payload = dict(vtk_export)
        print("\n=== VTK Export ===")
        print(f"  Vessels:   {vtk_export['vessels_path']}")
        print(f"  Pericytes: {vtk_export['pericytes_path']}")
        print(f"  Nodes:     {vtk_export['nodes_path']}")
        print(f"  Counts: vessels={vtk_export['vessel_line_count']}, "
          f"pericytes={vtk_export['pericyte_count']}, nodes={vtk_export['node_count']}")
    if run_haemodynamics and visualize_vtk and VTK_export:
        visualization.visualize_vtk_network(
            vtk_export["vessels_path"],
            vtk_export["pericytes_path"],
            vtk_export["nodes_path"],
            show_nodes=False,
        )
    if run_haemodynamics and visualize_vtk and not VTK_export:
        print("VTK visualization requested but VTK export is disabled. Set VTK_export=True to enable.")
    if run_haemodynamics and not visualize_vtk:
        print("VTK visualization skipped.") 
    # 6) Compute effective resistance between connected start/output node pairs.
    if run_haemodynamics:
        conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
        print(f"Conductance matrix built with shape {conductance.shape} and node_list length {len(node_list)}.")
    if run_haemodynamics and do_equiv_resistance_calculation:
        laplacian = haemodynamics.calc_laplacian_from_conductance_matrix(conductance)
        tested_pair_count = 0
        skipped_pair_count = 0
        for source_node, target_node in resistance_node_pairs:
            if source_node in node_to_idx and target_node in node_to_idx:
                two_point_resistance = haemodynamics.calc_two_point_from_laplacian_matrix_nodeID(
                    laplacian,
                    G,
                    source_node,
                    target_node,
                )
                print(
                    f"\nEffective resistance between nodes {source_node} and "
                    f"{target_node}: {two_point_resistance}"
                )
                tested_pair_count += 1
            else:
                print(
                    f"\nSkipped two-point resistance: nodes {(source_node, target_node)} "
                    "are not both present in the graph."
                )
                skipped_pair_count += 1
        print(
            "Two-point resistance testing summary: "
            f"tested_pairs={tested_pair_count}, skipped_pairs={skipped_pair_count}."
        )

    # 7) Compute and print vessel statistics.
    print("\nComputing vessel statistics...")
    if STATISTICS:
        valid_statistics_modes = {"fast", "full"}
        if statistics_mode not in valid_statistics_modes:
            raise ValueError(
                f"Invalid statistics_mode='{statistics_mode}'. "
                f"Choose one of {sorted(valid_statistics_modes)}."
            )
        node_positions = nx.get_node_attributes(G, "pos")
        stats = statistics.compute_comprehensive_vessel_statistics(
            G,
            node_positions=node_positions,
            image_dimensions=image.shape,
            statistics_mode=statistics_mode,
        )

        print("\n=== Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        stats_csv_path = output_dir / f"{image_path.stem}_statistics.csv"
        statistics.export_statistics_to_csv(stats, stats_csv_path)
        print(f"Saved statistics CSV to: {stats_csv_path}")

        branch_stats = statistics.compute_branch_order_statistics(
            G,
            node_positions=node_positions,
        )
        branch_stats_csv_path = output_dir / f"{image_path.stem}_branch_statistics.csv"
        statistics.export_branch_order_statistics_to_csv(
            branch_stats,
            branch_stats_csv_path,
        )
        print(f"Saved branch-order statistics CSV to: {branch_stats_csv_path}")

        if run_haemodynamics:
            weighted_measurements = statistics.compute_betweenness_and_community_measurements(G)
        else:
            weighted_measurements = {
                "edge_resistance": {
                    "Betweenness": {
                        "Betweenness Mean": "N/A (haemodynamics disabled)",
                        "Betweenness Max": "N/A (haemodynamics disabled)",
                        "Betweenness Top Nodes": "N/A (haemodynamics disabled)",
                        "Betweenness Method": "N/A (haemodynamics disabled)",
                    },
                    "Communities": {
                        "Community Count": "N/A (haemodynamics disabled)",
                        "Largest Community Size": "N/A (haemodynamics disabled)",
                        "Mean Community Size": "N/A (haemodynamics disabled)",
                        "Community Method": "N/A (haemodynamics disabled)",
                    },
                },
                "edge_length": {
                    "Betweenness": statistics.compute_weighted_betweenness_summary(
                        G,
                        source_attr="length",
                        inverse_source_attr=False,
                    ),
                    "Communities": statistics.compute_weighted_communities_summary(
                        G,
                        source_attr="length",
                        inverse_source_attr=False,
                    ),
                },
            }
        print("\n=== Weighted Betweenness and Communities ===")
        for model_name, model_results in weighted_measurements.items():
            print(f"  [{model_name}]")
            for metric_name, metric_values in model_results.items():
                print(f"    {metric_name}: {metric_values}")

        resistance_path = output_dir / f"{image_path.stem}_betweenness_communities_edge_resistance.json"
        resistance_path.write_text(
            json.dumps(weighted_measurements["edge_resistance"], indent=2)
        )
        length_path = output_dir / f"{image_path.stem}_betweenness_communities_edge_length.json"
        length_path.write_text(
            json.dumps(weighted_measurements["edge_length"], indent=2)
        )
        print(f"Saved resistance-weighted stats to: {resistance_path}")
        print(f"Saved edge-length stats to: {length_path}")
    else:
        print("Vessel statistics skipped.")

    # 8) Optional: nearest 3D distance from objects in a cell mask to vessel edge.
    if measurement_3d_to_cell_mask:
        if cell_mask_path is None:
            raise ValueError(
                "measurement_3d_to_cell_mask=True requires cell_mask_path."
            )
        distance_summary = statistics.run_3d_measurement_to_cell_mask(
            graph=G,
            cell_mask_path=Path(cell_mask_path),
            output_dir=output_dir,
            image_stem=image_path.stem,
            voxel_size_xyz=tuple(float(v) for v in voxel_size),
            vessel_mask_path=(
                None
                if measurement_3d_vessel_mask_path is None
                else Path(measurement_3d_vessel_mask_path)
            ),
            vessel_reference_image_path=(
                None
                if measurement_3d_reference_image_path is None
                else Path(measurement_3d_reference_image_path)
            ),
            cell_mask_h5_dataset_name=cell_mask_h5_dataset_name,
            vessel_mask_h5_dataset_name=measurement_3d_vessel_mask_h5_dataset_name,
            vessel_reference_h5_dataset_name=measurement_3d_reference_h5_dataset_name,
        )
        print(
            "3D cell-mask vessel-distance summary: "
            f"{distance_summary}"
        )
    else:
        print("3D cell-mask vessel-distance measurement skipped.")

    # 9) Also solve for flow throughout the network using the conductance matrix 
    # and the input and output pressures.
    if run_haemodynamics:
        print("\nSolving flow through the network...")
        flow, vtk_export = haemodynamics.solve_flow_from_conductance_matrix(
            conductance,
            node_list,
            input_p_bc,
            output_p_bc,
            starting_nodes,
            output_nodes,
            vtk_export_payload,
        )
        print("Flow through the network solved")
        print(f"Vtk file with flow data saved to: {vtk_export['vessels_path']}")
        flow_plotly_html_path = Path(plot_dir) / "flow_network_plotly.html"
        try:
            flow_html = visualization.write_flow_vtk_plotly_html(
                vtk_export["vessels_path"],
                flow_plotly_html_path,
                flow_field="flow_abs",
                title="Flow Network (Plotly)",
                show=show_plots_in_ide or interactive_plots,
            )
            print(f"Flow Plotly HTML saved to: {flow_html}")
        except Exception as exc:
            print(f"Skipped flow Plotly HTML export: {exc}")
        if run_alice_paper_sweep:
            print("\nRunning Alice pericyte-dilation x inlet-pressure sweep...")
            print(
                "Alice sweep toggles: "
                f"spacing_sweep={bool(run_alice_pericyte_spacing_sweep_plots)}, "
                f"spacing_beforeafter={bool(run_alice_pericyte_spacing_beforeafter)}, "
                f"output_dir={Path(alice_paper_output_dir)}"
            )
            sweep_outputs = _run_alice_pericyte_dilation_pressure_sweep(
                G,
                diameter_by_branch_order=diameter_by_branch_order,
                starting_nodes=starting_nodes,
                output_nodes=output_nodes,
                output_p_bc=float(output_p_bc),
                output_dir=Path(alice_paper_output_dir),
                custom_edges_for_sweep=list(alice_custom_edges_for_sweep),
                min_dilation_percent=int(alice_pericyte_dilation_min_percent),
                max_dilation_percent=int(alice_pericyte_dilation_max_percent),
                dilation_step_percent=int(alice_pericyte_dilation_step_percent),
                min_inlet_pressure_pa=int(alice_inlet_pressure_min_pa),
                max_inlet_pressure_pa=int(alice_inlet_pressure_max_pa),
                inlet_pressure_step_pa=int(alice_inlet_pressure_step_pa),
                constriction_length_um=float(alice_constriction_length_um),
                constriction_spacing_um=float(alice_constriction_spacing_um),
                run_passive_capillary_diameter_beforeafter=bool(
                    run_alice_passive_capillary_diameter_beforeafter
                ),
                run_arteriole_dilation_sweep_plots=bool(
                    run_alice_arteriole_dilation_sweep_plots
                ),
                arteriole_sweep_min_dilation_percent=int(
                    alice_arteriole_sweep_min_dilation_percent
                ),
                arteriole_sweep_max_dilation_percent=int(
                    alice_arteriole_sweep_max_dilation_percent
                ),
                arteriole_sweep_dilation_step_percent=int(
                    alice_arteriole_sweep_dilation_step_percent
                ),
                arteriole_sweep_min_inlet_pressure_pa=int(
                    alice_arteriole_sweep_min_inlet_pressure_pa
                ),
                arteriole_sweep_max_inlet_pressure_pa=int(
                    alice_arteriole_sweep_max_inlet_pressure_pa
                ),
                arteriole_sweep_inlet_pressure_step_pa=int(
                    alice_arteriole_sweep_inlet_pressure_step_pa
                ),
                run_passive_capillary_dilation_sweep_plots=bool(
                    run_alice_passive_capillary_dilation_sweep_plots
                ),
                passive_capillary_sweep_min_dilation_percent=int(
                    alice_passive_capillary_sweep_min_dilation_percent
                ),
                passive_capillary_sweep_max_dilation_percent=int(
                    alice_passive_capillary_sweep_max_dilation_percent
                ),
                passive_capillary_sweep_dilation_step_percent=int(
                    alice_passive_capillary_sweep_dilation_step_percent
                ),
                passive_capillary_sweep_min_inlet_pressure_pa=int(
                    alice_passive_capillary_sweep_min_inlet_pressure_pa
                ),
                passive_capillary_sweep_max_inlet_pressure_pa=int(
                    alice_passive_capillary_sweep_max_inlet_pressure_pa
                ),
                passive_capillary_sweep_inlet_pressure_step_pa=int(
                    alice_passive_capillary_sweep_inlet_pressure_step_pa
                ),
                capillary_passive_dilation_percent=float(
                    alice_passive_capillary_diameter_beforeafter_percent
                ),
                run_passive_arteriole_diameter_beforeafter=bool(
                    run_alice_passive_arteriole_diameter_beforeafter
                ),
                arteriole_passive_diameter_delta_um=float(
                    alice_passive_arteriole_diameter_beforeafter_delta_um
                ),
                run_pericyte_spacing_sweep_plots=bool(
                    run_alice_pericyte_spacing_sweep_plots
                ),
                pericyte_spacing_sweep_min_um=int(
                    alice_pericyte_spacing_sweep_min_um
                ),
                pericyte_spacing_sweep_max_um=int(
                    alice_pericyte_spacing_sweep_max_um
                ),
                pericyte_spacing_sweep_step_um=int(
                    alice_pericyte_spacing_sweep_step_um
                ),
                pericyte_spacing_sweep_min_inlet_pressure_pa=int(
                    alice_pericyte_spacing_sweep_min_inlet_pressure_pa
                ),
                pericyte_spacing_sweep_max_inlet_pressure_pa=int(
                    alice_pericyte_spacing_sweep_max_inlet_pressure_pa
                ),
                pericyte_spacing_sweep_inlet_pressure_step_pa=int(
                    alice_pericyte_spacing_sweep_inlet_pressure_step_pa
                ),
                pericyte_spacing_sweep_percent=float(
                    alice_pericyte_spacing_sweep_percent
                ),
                run_pericyte_spacing_beforeafter=bool(
                    run_alice_pericyte_spacing_beforeafter
                ),
                pericyte_beforeafter_percent=float(alice_pericyte_beforeafter_percent),
                pericyte_spacing_delta_um=float(alice_pericyte_spacing_delta_um),
            )
            print(
                "Completed Alice sweep: "
                f"rows={len(sweep_outputs['results'])}, "
                f"csv={sweep_outputs['csv_path']}"
            )
    else:
        print("Haemodynamics solve skipped (run_haemodynamics=False).")

    # 10) Optional matplotlib visualization.
    if visualize_results:
        print("\nGenerating visualizations...")
        valid_plot_modes = {"all", "final_only", "none"}
        if ide_plot_mode not in valid_plot_modes:
            raise ValueError(
                f"Invalid ide_plot_mode='{ide_plot_mode}'. "
                f"Choose one of {sorted(valid_plot_modes)}."
            )
        show_any_ide_plot = show_plots_in_ide and ide_plot_mode != "none"
        show_degree_plot = show_plots_in_ide and ide_plot_mode == "all"
        show_overlay_plot = show_any_ide_plot and final_render_mode == "2d"
        show_3d_plot = show_any_ide_plot and final_render_mode == "3d"
        show_branch_order_plot = show_plots_in_ide and ide_plot_mode == "all"
        visualization.plot_node_degree_distribution(
            G,
            save_path=None if interactive_plots else plot_dir / "node_degree_distribution.png",
            show=interactive_plots or show_degree_plot,
            show_after_save=show_degree_plot and not interactive_plots,
        )
        if final_render_mode == "3d":
            overlay_3d_path = None if interactive_plots else plot_dir / "edges_and_nodes_overlay_3d.html"
            visualization.visualize_3d_plotly(
                G,
                title="Edges and Nodes Overlay (Interactive 3D)",
                save_html_path=str(overlay_3d_path) if overlay_3d_path else None,
                show=interactive_plots or show_3d_plot,
            )
            if overlay_3d_path is not None:
                print(f"Saved interactive 3D overlay to: {overlay_3d_path}")
        else:
            visualization.visualize_edges_and_nodes(
                image,
                G,
                save_path=None if interactive_plots else plot_dir / "edges_and_nodes_overlay.png",
                show=interactive_plots or show_overlay_plot,
                show_after_save=show_overlay_plot and not interactive_plots,
            )
        #HD note - need visualisation of pericyte localisations (ie based upon constriction data)
        
        if starting_nodes:
            visualization.visualize_geometry_with_branch_orders(
                image,
                G,
                group_above=8,
                save_path=None if interactive_plots else plot_dir / "geometry_with_branch_orders.png",
                show=interactive_plots or show_branch_order_plot,
                show_after_save=show_branch_order_plot and not interactive_plots,
            )
        if (
            hold_ide_plots_open
            and show_any_ide_plot
            and not interactive_plots
            and plt.get_fignums()
        ):
            print("Holding plot windows open. Close them to finish the script.")
            plt.show(block=True)
    else:
        print("Matplotlib visualizations skipped.")


def _build_pipeline_kwargs_from_active_settings(plot_dir: Path) -> dict:
    """Build full pipeline kwargs from current module-level settings."""
    alias_to_settings = {
        "image_path": "INPUT_PATH",
        "do_pericyte_constriction": "DO_PERICYTE_CONSTRUCTION",
    }
    kwargs: dict = {}
    for param_name in inspect.signature(image_to_model_pipeline).parameters:
        if param_name == "plot_dir":
            kwargs[param_name] = plot_dir
            continue
        setting_name = alias_to_settings.get(param_name, param_name.upper())
        if setting_name in globals():
            kwargs[param_name] = globals()[setting_name]
    return kwargs


def _parse_cli_literal(value_text: str) -> object:
    lowered = value_text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(value_text)
    except (ValueError, SyntaxError):
        return value_text


def _coerce_pipeline_cli_value(param_name: str, value_text: str) -> object:
    value = _parse_cli_literal(value_text)
    if isinstance(value, str):
        if (
            param_name.endswith("_path")
            or param_name.endswith("_dir")
            or param_name.endswith("_prefix")
            or param_name == "plot_dir"
        ):
            return Path(value)
    return value


def _extract_pipeline_cli_overrides(cli_namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for param_name in inspect.signature(image_to_model_pipeline).parameters:
        dest = f"pipeline_arg__{param_name}"
        if not hasattr(cli_namespace, dest):
            continue
        raw_value = getattr(cli_namespace, dest)
        if raw_value is None:
            continue
        overrides[param_name] = _coerce_pipeline_cli_value(param_name, raw_value)
    return overrides


def _apply_standard_output_layout(
    pipeline_kwargs: dict[str, object],
    *,
    output_subdir: str = "nerve",
) -> dict[str, object]:
    """Route pipeline outputs to outputs/<subdir> and plots/<subdir>."""
    subdir_name = str(output_subdir).strip() or "nerve"
    outputs_dir = root_dir / "examples" / "outputs" / subdir_name
    plots_dir = root_dir / "examples" / "plots" / subdir_name
    segmentations_dir = outputs_dir / "segmentations"
    alice_dir = outputs_dir / "alice_paper"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    segmentations_dir.mkdir(parents=True, exist_ok=True)
    alice_dir.mkdir(parents=True, exist_ok=True)

    existing_vtk_prefix = Path(
        pipeline_kwargs.get("vtk_output_prefix", outputs_dir / "resistance_network")
    )
    vtk_prefix_name = existing_vtk_prefix.name or "resistance_network"

    pipeline_kwargs["plot_dir"] = plots_dir
    pipeline_kwargs["vtk_output_prefix"] = outputs_dir / vtk_prefix_name
    pipeline_kwargs["ilastik_output_dir"] = segmentations_dir
    pipeline_kwargs["alice_paper_output_dir"] = alice_dir
    return pipeline_kwargs


def _sanitize_config_stem(value: object) -> str:
    """Build filesystem-safe config filename stem from a path-like value."""
    if value is None:
        return "run"
    stem = Path(str(value)).stem.strip()
    if not stem:
        return "run"
    safe_stem = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in stem)
    safe_stem = safe_stem.strip("_")
    return safe_stem or "run"


def _build_auto_config_output_path(input_path: object) -> Path:
    """Create auto-save YAML path: <input_stem>_<YYYYMMDD_HHMMSS>.yaml."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _sanitize_config_stem(input_path)
    return Path(AUTO_SAVE_EFFECTIVE_CONFIG_DIR) / f"{stem}_{timestamp}.yaml"  # noqa: F405


if __name__ == "__main__":
    import argparse

    pipeline_signature = inspect.signature(image_to_model_pipeline)
    pipeline_param_names = set(pipeline_signature.parameters.keys())
    parser = argparse.ArgumentParser(description="Resistance network pipeline example.")
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(PRESET_DEFINITIONS.keys()),  # noqa: F405
        help="Preset profile for grouped settings.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to YAML config file with preset/settings/pipeline overrides. "
            "CLI overrides still take precedence."
        ),
    )
    parser.add_argument(
        "--save-config",
        type=Path,
        default=None,
        help="Write the effective resolved run configuration to a YAML file.",
    )
    parser.add_argument(
        "--wizard",
        action="store_true",
        help=(
            "Run interactive setup prompts for preset, image path, "
            "mask usage, and key toggles."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run preflight checklist and exit without executing the pipeline.",
    )
    parser.add_argument(
        "--set",
        dest="manual_setting_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Manual settings-file override (repeatable). "
            "Example: --set VERBOSE_LOGGING=True --set FWHM_RAW_TIFF_PATH='C:/data/raw.tif'"
        ),
    )
    parser.add_argument(
        "--run-small-vessel-boundary-labelling-tests",
        action="store_true",
        help="Run pytest on tests/test_small_vessel_mask_boundary_labelling.py and exit.",
    )
    parser.add_argument(
        "--use-fwhm-edge-diameters",
        action="store_true",
        help=(
            "Override USE_FWHM_EDGE_DIAMETERS: measure diameters from raw TIFF "
            "(Gaussian transverse fit). Requires --fwhm-raw-tiff unless "
            "FWHM_RAW_TIFF_PATH is set in this file."
        ),
    )
    parser.add_argument(
        "--fwhm-raw-tiff",
        type=Path,
        default=None,
        help="Path to raw single-channel TIFF for FWHM (overrides FWHM_RAW_TIFF_PATH).",
    )

    # Add dynamic CLI overrides for any image_to_model_pipeline(...) argument.
    existing_option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    for param_name in pipeline_signature.parameters:
        option_name = f"--{param_name.replace('_', '-')}"
        if option_name in existing_option_strings:
            continue
        parser.add_argument(
            option_name,
            dest=f"pipeline_arg__{param_name}",
            default=None,
            metavar="VALUE",
            help=(
                f"Override pipeline kwarg '{param_name}'. "
                "Use Python literals for lists/dicts/tuples and True/False/None."
            ),
        )

    cli = parser.parse_args()
    if cli.list_presets:
        print("Available presets:")
        for preset_name, description in list_presets().items():  # noqa: F405
            print(f"  - {preset_name}: {description}")
        raise SystemExit(0)
    if cli.run_small_vessel_boundary_labelling_tests:
        import pytest

        raise SystemExit(
            pytest.main([str(root_dir / "tests" / "test_small_vessel_mask_boundary_labelling.py"), "-q"])
        )

    config_preset_name: str | None = None
    config_setting_overrides: dict[str, object] = {}
    config_pipeline_overrides: dict[str, object] = {}
    effective_config_path: Path | None = cli.config
    if (
        effective_config_path is None
        and AUTO_LOAD_CONFIG_EACH_RUN  # noqa: F405
        and AUTO_LOAD_CONFIG_PATH is not None  # noqa: F405
    ):
        effective_config_path = Path(AUTO_LOAD_CONFIG_PATH)  # noqa: F405
    if (
        effective_config_path is None
        and AUTO_LOAD_CONFIG_EACH_RUN  # noqa: F405
        and AUTO_LOAD_CONFIG_PATH is None  # noqa: F405
    ):
        raise ValueError(
            "AUTO_LOAD_CONFIG_EACH_RUN=True but AUTO_LOAD_CONFIG_PATH is not set."
        )
    if effective_config_path is not None:
        loaded_config = load_config_yaml(  # noqa: F405
            config_path=effective_config_path,
            pipeline_param_names=pipeline_param_names,
        )
        config_preset_name = loaded_config["preset_name"]
        config_setting_overrides = dict(loaded_config["settings_overrides"])
        config_pipeline_overrides = dict(loaded_config["pipeline_overrides"])
        print(f"Loaded config from: {effective_config_path}")

    preset_name = cli.preset or config_preset_name or "none"
    manual_overrides: dict[str, object] = dict(config_setting_overrides)
    for override_text in cli.manual_setting_overrides:
        key, value = parse_cli_override(override_text)  # noqa: F405
        manual_overrides[key] = value
    if cli.use_fwhm_edge_diameters:
        manual_overrides["USE_FWHM_EDGE_DIAMETERS"] = True
    if cli.fwhm_raw_tiff is not None:
        manual_overrides["FWHM_RAW_TIFF_PATH"] = cli.fwhm_raw_tiff

    wizard_pipeline_overrides: dict[str, object] = {}
    if cli.wizard:
        wizard_results = run_interactive_setup_wizard(
            default_preset=preset_name,
            available_presets=sorted(PRESET_DEFINITIONS.keys()),  # noqa: F405
        )
        preset_name = wizard_results["preset_name"]
        manual_overrides.update(wizard_results["settings_overrides"])
        wizard_pipeline_overrides = dict(wizard_results["pipeline_overrides"])

    selected_settings = build_settings_for_preset(  # noqa: F405
        preset_name=preset_name,
        manual_overrides=manual_overrides,
    )
    apply_settings_to_namespace(selected_settings, globals())  # noqa: F405
    if manual_overrides:
        print(
            f"Applying preset '{preset_name}' with manual overrides: "
            f"{sorted(manual_overrides.keys())}"
        )
    else:
        print(f"Applying preset '{preset_name}'")

    plot_dir = Path(BASE_PLOT_DIR) / "nerve"  # noqa: F405
    pipeline_kwargs = _build_pipeline_kwargs_from_active_settings(plot_dir=plot_dir)
    pipeline_kwargs = _apply_standard_output_layout(
        pipeline_kwargs,
        output_subdir="nerve",
    )
    if config_pipeline_overrides:
        pipeline_kwargs.update(config_pipeline_overrides)
        print(
            "Applying config pipeline overrides: "
            f"{sorted(config_pipeline_overrides.keys())}"
        )
    if wizard_pipeline_overrides:
        pipeline_kwargs.update(wizard_pipeline_overrides)
        print(
            "Applying wizard pipeline overrides: "
            f"{sorted(wizard_pipeline_overrides.keys())}"
        )
    pipeline_cli_overrides = _extract_pipeline_cli_overrides(cli)
    if pipeline_cli_overrides:
        pipeline_kwargs.update(pipeline_cli_overrides)
        print(
            "Applying direct pipeline argument overrides: "
            f"{sorted(pipeline_cli_overrides.keys())}"
        )
    pipeline_kwargs = _apply_standard_output_layout(
        pipeline_kwargs,
        output_subdir="nerve",
    )
    effective_settings_snapshot = collect_current_settings_snapshot(globals())  # noqa: F405
    if cli.save_config is not None:
        saved_path = save_effective_config_yaml(  # noqa: F405
            output_path=cli.save_config,
            preset_name=preset_name,
            settings=effective_settings_snapshot,
            pipeline_kwargs=pipeline_kwargs,
        )
        print(f"Saved effective run config to: {saved_path}")
    if AUTO_SAVE_EFFECTIVE_CONFIG_EACH_RUN:  # noqa: F405
        auto_save_path = _build_auto_config_output_path(
            pipeline_kwargs.get("image_path", INPUT_PATH)  # noqa: F405
        )
        auto_saved_path = save_effective_config_yaml(  # noqa: F405
            output_path=auto_save_path,
            preset_name=preset_name,
            settings=effective_settings_snapshot,
            pipeline_kwargs=pipeline_kwargs,
        )
        print(f"Auto-saved effective run config to: {auto_saved_path}")
    preflight_report = run_preflight_checklist(pipeline_kwargs)
    if not preflight_report["ok"]:
        raise SystemExit(2)
    if cli.preflight_only:
        print("Preflight-only mode: exiting before pipeline execution.")
        raise SystemExit(0)
    image_to_model_pipeline(**pipeline_kwargs)
