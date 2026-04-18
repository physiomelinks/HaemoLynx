"""Utilities to compare baseline vs constricted pericyte haemodynamics."""
from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import networkx as nx

from .pericyte_mask import set_poiseuille_resistances_with_pericyte_mask
from .poiseuille import PoiseuilleModel
from .probability import set_poiseuille_resistances_with_probabilistic_periodic_constrictions
from .resistance import (
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
    calc_two_point_from_laplacian_matrix_nodeID,
)


def _absolute_factor_map(
    branch_orders: list[str],
    factor_value: float,
) -> dict[str, float]:
    return {
        str(branch_order): float(factor_value)
        for branch_order in branch_orders
    }


def _set_resistances_for_factor(
    graph: nx.MultiGraph,
    *,
    diameter_by_branch_order: dict,
    constriction_factor_by_branch_order: dict[str, float],
    factor_value: float,
    use_pericyte_mask_constriction: bool,
    pericyte_mask_path: str | Path | None,
    pericyte_mask_h5_dataset_name: str | None,
    prefer_edge_fwhm_baseline: bool,
    constriction_length: float,
    constriction_spacing: float,
    use_probabilistic_pericyte_constriction: bool,
    pericyte_constriction_probability: float,
    active_pericyte_indices: list[int] | None,
    active_center_indices_by_edge: dict[str, list[int]] | None,
    max_assignment_distance_um: float | None,
    min_pericyte_diameter_um: float | None,
    max_pericyte_diameter_um: float | None,
) -> tuple[nx.MultiGraph, dict[str, Any]]:
    """Apply edge resistances for a comparison scenario."""
    branch_orders = [str(bo) for bo in diameter_by_branch_order.keys()]
    factor_map = _absolute_factor_map(
        branch_orders=branch_orders,
        factor_value=factor_value,
    )
    if use_pericyte_mask_constriction:
        if pericyte_mask_path is None:
            raise ValueError(
                "pericyte_mask_path is required when use_pericyte_mask_constriction=True."
            )
        return set_poiseuille_resistances_with_pericyte_mask(
            graph,
            diameter_by_branch_order=diameter_by_branch_order,
            constriction_factor_by_branch_order=factor_map,
            pericyte_mask_path=pericyte_mask_path,
            pericyte_mask_h5_dataset_name=pericyte_mask_h5_dataset_name,
            prefer_edge_fwhm_baseline=prefer_edge_fwhm_baseline,
            constriction_length=constriction_length,
            use_probabilistic_constriction=bool(use_probabilistic_pericyte_constriction),
            constriction_probability=float(pericyte_constriction_probability),
            active_pericyte_indices=active_pericyte_indices,
            max_assignment_distance_um=max_assignment_distance_um,
            min_pericyte_diameter_um=min_pericyte_diameter_um,
            max_pericyte_diameter_um=max_pericyte_diameter_um,
        )

    if use_probabilistic_pericyte_constriction:
        return set_poiseuille_resistances_with_probabilistic_periodic_constrictions(
            graph,
            diameter_by_branch_order=diameter_by_branch_order,
            constriction_factor_by_branch_order=factor_map,
            prefer_edge_fwhm_baseline=prefer_edge_fwhm_baseline,
            constriction_length=float(constriction_length),
            constriction_spacing=float(constriction_spacing),
            constriction_probability=float(pericyte_constriction_probability),
            active_center_indices_by_edge=active_center_indices_by_edge,
        )

    poiseuille_model = PoiseuilleModel(
        constriction_length=float(constriction_length),
        constriction_spacing=float(constriction_spacing),
    )
    if prefer_edge_fwhm_baseline:
        return poiseuille_model.set_poiseuille_resistances_with_constrictions(
            graph,
            diameter_by_branch_order,
            prefer_edge_fwhm_baseline=True,
            constriction_factor_by_branch_order=factor_map,
        )

    enhanced_diameters: dict[str, dict[str, float]] = {}
    for branch_order, diameter in diameter_by_branch_order.items():
        enhanced_diameters[str(branch_order)] = {
            "d1": float(diameter),
            "d2": float(diameter) * float(factor_map[str(branch_order)]),
        }
    return poiseuille_model.set_poiseuille_resistances_with_constrictions(
        graph,
        enhanced_diameters,
    )


def _compute_two_point_resistance(
    graph: nx.MultiGraph,
    source_node: int,
    target_node: int,
) -> float:
    conductance, _ = build_conductance_matrix_from_graph(graph)
    laplacian = calc_laplacian_from_conductance_matrix(conductance)
    return float(
        calc_two_point_from_laplacian_matrix_nodeID(
            laplacian,
            graph,
            source_node,
            target_node,
        )
    )


def compare_baseline_vs_pericyte_constriction(
    graph: nx.MultiGraph,
    *,
    diameter_by_branch_order: dict,
    constriction_factor_by_branch_order: dict[str, float],
    resistance_node_pair: tuple[int, int],
    output_csv_path: str | Path,
    baseline_factor_value: float = 1.0,
    constricted_factor_value: float = 0.8,
    use_pericyte_mask_constriction: bool = False,
    pericyte_mask_path: str | Path | None = None,
    pericyte_mask_h5_dataset_name: str | None = None,
    prefer_edge_fwhm_baseline: bool = False,
    constriction_length: float = 40.0,
    constriction_spacing: float = 100.0,
    use_probabilistic_pericyte_constriction: bool = False,
    pericyte_constriction_probability: float = 1.0,
    max_assignment_distance_um: float | None = 3.0,
    min_pericyte_diameter_um: float | None = 5.0,
    max_pericyte_diameter_um: float | None = 12.0,
) -> dict[str, Any]:
    """Compare effective resistance at baseline vs constricted settings.

    Comparison factors are treated as absolute values (not scales). This means
    ``baseline_factor_value`` and ``constricted_factor_value`` override the
    non-comparison constriction magnitudes while comparison is running.

    Returns a summary dict and writes a human-readable CSV with one row per
    scenario plus a final delta row.
    """
    source_node, target_node = resistance_node_pair
    if source_node not in graph.nodes or target_node not in graph.nodes:
        raise ValueError(
            f"resistance_node_pair {resistance_node_pair} not present in graph nodes."
        )
    if not diameter_by_branch_order:
        raise ValueError("diameter_by_branch_order cannot be empty.")

    graph_baseline = deepcopy(graph)
    graph_constricted = deepcopy(graph)
    fixed_active_pericyte_indices: list[int] | None = None
    fixed_active_center_indices_by_edge: dict[str, list[int]] | None = None

    graph_baseline, baseline_weight_results = _set_resistances_for_factor(
        graph_baseline,
        diameter_by_branch_order=diameter_by_branch_order,
        constriction_factor_by_branch_order=constriction_factor_by_branch_order,
        factor_value=float(baseline_factor_value),
        use_pericyte_mask_constriction=bool(use_pericyte_mask_constriction),
        pericyte_mask_path=pericyte_mask_path,
        pericyte_mask_h5_dataset_name=pericyte_mask_h5_dataset_name,
        prefer_edge_fwhm_baseline=bool(prefer_edge_fwhm_baseline),
        constriction_length=float(constriction_length),
        constriction_spacing=float(constriction_spacing),
        use_probabilistic_pericyte_constriction=bool(use_probabilistic_pericyte_constriction),
        pericyte_constriction_probability=float(pericyte_constriction_probability),
        active_pericyte_indices=None,
        active_center_indices_by_edge=None,
        max_assignment_distance_um=max_assignment_distance_um,
        min_pericyte_diameter_um=min_pericyte_diameter_um,
        max_pericyte_diameter_um=max_pericyte_diameter_um,
    )
    if use_pericyte_mask_constriction and use_probabilistic_pericyte_constriction:
        selected = baseline_weight_results.get("active_pericyte_indices")
        fixed_active_pericyte_indices = [int(idx) for idx in selected] if selected else []
    if (not use_pericyte_mask_constriction) and use_probabilistic_pericyte_constriction:
        selected_map = baseline_weight_results.get("active_center_indices_by_edge")
        if isinstance(selected_map, dict):
            fixed_active_center_indices_by_edge = {
                str(edge_id): [int(idx) for idx in idx_list]
                for edge_id, idx_list in selected_map.items()
            }
    baseline_resistance = _compute_two_point_resistance(
        graph_baseline,
        source_node=source_node,
        target_node=target_node,
    )

    graph_constricted, constricted_weight_results = _set_resistances_for_factor(
        graph_constricted,
        diameter_by_branch_order=diameter_by_branch_order,
        constriction_factor_by_branch_order=constriction_factor_by_branch_order,
        factor_value=float(constricted_factor_value),
        use_pericyte_mask_constriction=bool(use_pericyte_mask_constriction),
        pericyte_mask_path=pericyte_mask_path,
        pericyte_mask_h5_dataset_name=pericyte_mask_h5_dataset_name,
        prefer_edge_fwhm_baseline=bool(prefer_edge_fwhm_baseline),
        constriction_length=float(constriction_length),
        constriction_spacing=float(constriction_spacing),
        use_probabilistic_pericyte_constriction=bool(use_probabilistic_pericyte_constriction),
        pericyte_constriction_probability=float(pericyte_constriction_probability),
        active_pericyte_indices=fixed_active_pericyte_indices,
        active_center_indices_by_edge=fixed_active_center_indices_by_edge,
        max_assignment_distance_um=max_assignment_distance_um,
        min_pericyte_diameter_um=min_pericyte_diameter_um,
        max_pericyte_diameter_um=max_pericyte_diameter_um,
    )
    constricted_resistance = _compute_two_point_resistance(
        graph_constricted,
        source_node=source_node,
        target_node=target_node,
    )

    delta = float(constricted_resistance - baseline_resistance)
    percent_change = (
        float((delta / baseline_resistance) * 100.0) if baseline_resistance != 0 else float("inf")
    )
    ratio = (
        float(constricted_resistance / baseline_resistance)
        if baseline_resistance != 0
        else float("inf")
    )

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "factor_value",
                "source_node",
                "target_node",
                "effective_resistance",
                "delta_vs_baseline",
                "percent_change_vs_baseline",
                "ratio_vs_baseline",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario": "baseline",
                "factor_value": float(baseline_factor_value),
                "source_node": int(source_node),
                "target_node": int(target_node),
                "effective_resistance": float(baseline_resistance),
                "delta_vs_baseline": 0.0,
                "percent_change_vs_baseline": 0.0,
                "ratio_vs_baseline": 1.0,
            }
        )
        writer.writerow(
            {
                "scenario": "constricted",
                "factor_value": float(constricted_factor_value),
                "source_node": int(source_node),
                "target_node": int(target_node),
                "effective_resistance": float(constricted_resistance),
                "delta_vs_baseline": float(delta),
                "percent_change_vs_baseline": float(percent_change),
                "ratio_vs_baseline": float(ratio),
            }
        )
        writer.writerow(
            {
                "scenario": "summary",
                "factor_value": "",
                "source_node": int(source_node),
                "target_node": int(target_node),
                "effective_resistance": "",
                "delta_vs_baseline": float(delta),
                "percent_change_vs_baseline": float(percent_change),
                "ratio_vs_baseline": float(ratio),
            }
        )

    return {
        "baseline_resistance": float(baseline_resistance),
        "constricted_resistance": float(constricted_resistance),
        "delta": float(delta),
        "percent_change": float(percent_change),
        "ratio": float(ratio),
        "baseline_factor_value": float(baseline_factor_value),
        "constricted_factor_value": float(constricted_factor_value),
        "output_csv_path": str(output_path),
        "baseline_weight_results": baseline_weight_results,
        "constricted_weight_results": constricted_weight_results,
        "active_pericyte_indices": fixed_active_pericyte_indices,
        "active_center_indices_by_edge": fixed_active_center_indices_by_edge,
    }
