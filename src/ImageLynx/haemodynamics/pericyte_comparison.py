"""Utilities to compare baseline vs constricted pericyte haemodynamics."""
from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

from .pericyte_mask import set_poiseuille_weights_with_pericyte_mask
from .poiseuille import PoiseuilleModel
from .probability import set_poiseuille_weights_with_probabilistic_periodic_constrictions
from .resistance import (
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
    calc_two_point_from_laplacian_matrix_nodeID,
    solve_pressure_and_boundary_flow,
)


def _absolute_factor_map(
    branch_orders: list[str],
    factor_value: float,
) -> dict[str, float]:
    return {
        str(branch_order): float(factor_value)
        for branch_order in branch_orders
    }


def _set_weights_for_factor(
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
    """Apply edge weights for a comparison scenario."""
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
        return set_poiseuille_weights_with_pericyte_mask(
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
        return set_poiseuille_weights_with_probabilistic_periodic_constrictions(
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
        return poiseuille_model.set_poiseuille_weights_with_constrictions(
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
    return poiseuille_model.set_poiseuille_weights_with_constrictions(
        graph,
        enhanced_diameters,
    )


def _resolve_resistance_pairs(
    graph: nx.MultiGraph,
    *,
    resistance_node_pair: tuple[int, int] | None,
    resistance_node_pairs: list[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    """Return validated resistance node pairs for comparison runs."""
    if resistance_node_pairs is not None and len(resistance_node_pairs) > 0:
        resolved_pairs = [(int(src), int(dst)) for src, dst in resistance_node_pairs]
    elif resistance_node_pair is not None:
        resolved_pairs = [(int(resistance_node_pair[0]), int(resistance_node_pair[1]))]
    else:
        raise ValueError(
            "Provide resistance_node_pair or non-empty resistance_node_pairs."
        )

    unique_pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for src, dst in resolved_pairs:
        pair = (int(src), int(dst))
        if pair in seen:
            continue
        seen.add(pair)
        unique_pairs.append(pair)

    for pair in unique_pairs:
        source_node, target_node = pair
        if source_node not in graph.nodes or target_node not in graph.nodes:
            raise ValueError(f"resistance pair {pair} not present in graph nodes.")

    return unique_pairs


def compare_baseline_vs_pericyte_constriction(
    graph: nx.MultiGraph,
    *,
    diameter_by_branch_order: dict,
    constriction_factor_by_branch_order: dict[str, float],
    resistance_node_pair: tuple[int, int] | None = None,
    resistance_node_pairs: list[tuple[int, int]] | None = None,
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
    input_p_bc: float = 5000.0,
    output_p_bc: float = 2000.0,
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
    resolved_pairs = _resolve_resistance_pairs(
        graph,
        resistance_node_pair=resistance_node_pair,
        resistance_node_pairs=resistance_node_pairs,
    )
    if not diameter_by_branch_order:
        raise ValueError("diameter_by_branch_order cannot be empty.")

    graph_baseline = deepcopy(graph)
    graph_constricted = deepcopy(graph)
    fixed_active_pericyte_indices: list[int] | None = None
    fixed_active_center_indices_by_edge: dict[str, list[int]] | None = None

    graph_baseline, baseline_weight_results = _set_weights_for_factor(
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
    baseline_conductance, baseline_node_list = build_conductance_matrix_from_graph(graph_baseline)
    baseline_laplacian = calc_laplacian_from_conductance_matrix(baseline_conductance)

    graph_constricted, constricted_weight_results = _set_weights_for_factor(
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
    constricted_conductance, constricted_node_list = build_conductance_matrix_from_graph(graph_constricted)
    constricted_laplacian = calc_laplacian_from_conductance_matrix(constricted_conductance)

    pair_results: list[dict[str, Any]] = []
    for source_node, target_node in resolved_pairs:
        baseline_resistance = float(
            calc_two_point_from_laplacian_matrix_nodeID(
                baseline_laplacian,
                graph_baseline,
                source_node,
                target_node,
            )
        )
        constricted_resistance = float(
            calc_two_point_from_laplacian_matrix_nodeID(
                constricted_laplacian,
                graph_constricted,
                source_node,
                target_node,
            )
        )
        delta = float(constricted_resistance - baseline_resistance)
        percent_change = (
            float((delta / baseline_resistance) * 100.0)
            if baseline_resistance != 0
            else float("inf")
        )
        ratio = (
            float(constricted_resistance / baseline_resistance)
            if baseline_resistance != 0
            else float("inf")
        )
        baseline_flow_result = solve_pressure_and_boundary_flow(
            conductance=baseline_conductance,
            node_list=baseline_node_list,
            input_p_bc=float(input_p_bc),
            output_p_bc=float(output_p_bc),
            starting_nodes=[int(source_node)],
            output_nodes=[int(target_node)],
        )
        constricted_flow_result = solve_pressure_and_boundary_flow(
            conductance=constricted_conductance,
            node_list=constricted_node_list,
            input_p_bc=float(input_p_bc),
            output_p_bc=float(output_p_bc),
            starting_nodes=[int(source_node)],
            output_nodes=[int(target_node)],
        )
        pair_results.append(
            {
                "source_node": int(source_node),
                "target_node": int(target_node),
                "baseline_resistance": float(baseline_resistance),
                "constricted_resistance": float(constricted_resistance),
                "delta": float(delta),
                "percent_change": float(percent_change),
                "ratio": float(ratio),
                "baseline_total_inlet_flow": float(baseline_flow_result["total_inlet_flow"]),
                "constricted_total_inlet_flow": float(
                    constricted_flow_result["total_inlet_flow"]
                ),
            }
        )

    aggregate_delta = float(sum(result["delta"] for result in pair_results))
    aggregate_percent_change = (
        float(sum(result["percent_change"] for result in pair_results) / len(pair_results))
        if pair_results
        else float("nan")
    )
    aggregate_ratio = (
        float(sum(result["ratio"] for result in pair_results) / len(pair_results))
        if pair_results
        else float("nan")
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
        for result in pair_results:
            writer.writerow(
                {
                    "scenario": "baseline",
                    "factor_value": float(baseline_factor_value),
                    "source_node": int(result["source_node"]),
                    "target_node": int(result["target_node"]),
                    "effective_resistance": float(result["baseline_resistance"]),
                    "delta_vs_baseline": 0.0,
                    "percent_change_vs_baseline": 0.0,
                    "ratio_vs_baseline": 1.0,
                }
            )
            writer.writerow(
                {
                    "scenario": "constricted",
                    "factor_value": float(constricted_factor_value),
                    "source_node": int(result["source_node"]),
                    "target_node": int(result["target_node"]),
                    "effective_resistance": float(result["constricted_resistance"]),
                    "delta_vs_baseline": float(result["delta"]),
                    "percent_change_vs_baseline": float(result["percent_change"]),
                    "ratio_vs_baseline": float(result["ratio"]),
                }
            )
        writer.writerow(
            {
                "scenario": "summary_all_pairs",
                "factor_value": "",
                "source_node": "",
                "target_node": "",
                "effective_resistance": "",
                "delta_vs_baseline": float(aggregate_delta),
                "percent_change_vs_baseline": float(aggregate_percent_change),
                "ratio_vs_baseline": float(aggregate_ratio),
            }
        )

    plot_path = output_path.with_name(f"{output_path.stem}_before_after_resistance.png")
    baseline_values = [float(result["baseline_resistance"]) for result in pair_results]
    constricted_values = [float(result["constricted_resistance"]) for result in pair_results]
    def _mean_sem(values: list[float]) -> tuple[float, float]:
        n = len(values)
        if n == 0:
            return 0.0, 0.0
        mean = float(sum(values) / n)
        if n < 2:
            return mean, 0.0
        variance = float(sum((v - mean) ** 2 for v in values) / (n - 1))
        sem = float((variance ** 0.5) / (n ** 0.5))
        return mean, sem
    baseline_mean, baseline_sem = _mean_sem(baseline_values)
    constricted_mean, constricted_sem = _mean_sem(constricted_values)
    input_change_percent = 0.0
    if baseline_factor_value != 0:
        input_change_percent = (
            (float(constricted_factor_value) - float(baseline_factor_value))
            / float(baseline_factor_value)
        ) * 100.0

    fig, ax = plt.subplots(figsize=(8, 5))
    x_baseline = 0.0
    x_after = 1.0
    for baseline_res, constricted_res in zip(baseline_values, constricted_values):
        ax.plot(
            [x_baseline, x_after],
            [baseline_res, constricted_res],
            color="tab:red",
            linewidth=1.8,
            alpha=0.9,
            zorder=1,
        )
    ax.scatter(
        [x_baseline] * len(baseline_values),
        baseline_values,
        color="tab:blue",
        s=90,
        alpha=0.9,
        zorder=3,
        label="Before",
    )
    ax.scatter(
        [x_after] * len(constricted_values),
        constricted_values,
        color="tab:red",
        s=90,
        alpha=0.9,
        zorder=3,
        label="After",
    )
    ax.bar(
        [x_baseline, x_after],
        [baseline_mean, constricted_mean],
        yerr=[baseline_sem, constricted_sem],
        width=0.35,
        color=["tab:blue", "tab:red"],
        alpha=0.25,
        ecolor="black",
        capsize=6,
        zorder=2,
        label="Mean ± SEM",
    )
    ax.set_xticks([x_baseline, x_after], labels=["Before", "After"])
    ax.set_ylabel("Effective resistance")
    ax.set_title(
        "Pericyte Comparison: Paired Before/After by Input-Output Pair\n"
        f"Input pericyte factor change: {input_change_percent:+.1f}%"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    flow_plot_path = output_path.with_name(f"{output_path.stem}_before_after_flow.png")
    baseline_flow_values = [
        float(result["baseline_total_inlet_flow"]) for result in pair_results
    ]
    constricted_flow_values = [
        float(result["constricted_total_inlet_flow"]) for result in pair_results
    ]
    baseline_flow_mean, baseline_flow_sem = _mean_sem(baseline_flow_values)
    constricted_flow_mean, constricted_flow_sem = _mean_sem(constricted_flow_values)

    fig_flow, ax_flow = plt.subplots(figsize=(8, 5))
    for baseline_flow, constricted_flow in zip(
        baseline_flow_values, constricted_flow_values
    ):
        ax_flow.plot(
            [x_baseline, x_after],
            [baseline_flow, constricted_flow],
            color="tab:red",
            linewidth=1.8,
            alpha=0.9,
            zorder=1,
        )
    ax_flow.scatter(
        [x_baseline] * len(baseline_flow_values),
        baseline_flow_values,
        color="tab:blue",
        s=90,
        alpha=0.9,
        zorder=3,
        label="Before",
    )
    ax_flow.scatter(
        [x_after] * len(constricted_flow_values),
        constricted_flow_values,
        color="tab:red",
        s=90,
        alpha=0.9,
        zorder=3,
        label="After",
    )
    ax_flow.bar(
        [x_baseline, x_after],
        [baseline_flow_mean, constricted_flow_mean],
        yerr=[baseline_flow_sem, constricted_flow_sem],
        width=0.35,
        color=["tab:blue", "tab:red"],
        alpha=0.25,
        ecolor="black",
        capsize=6,
        zorder=2,
        label="Mean ± SEM",
    )
    ax_flow.set_xticks([x_baseline, x_after], labels=["Before", "After"])
    ax_flow.set_ylabel("Total inlet flow")
    ax_flow.set_title(
        "Pericyte Comparison: Paired Before/After Flow by Input-Output Pair\n"
        f"Input pericyte factor change: {input_change_percent:+.1f}%"
    )
    ax_flow.grid(True, axis="y", alpha=0.3)
    ax_flow.legend()
    fig_flow.tight_layout()
    fig_flow.savefig(flow_plot_path, dpi=200)
    plt.close(fig_flow)

    first_result = pair_results[0]
    return {
        "baseline_resistance": float(first_result["baseline_resistance"]),
        "constricted_resistance": float(first_result["constricted_resistance"]),
        "delta": float(first_result["delta"]),
        "percent_change": float(first_result["percent_change"]),
        "ratio": float(first_result["ratio"]),
        "pair_results": pair_results,
        "pair_count": int(len(pair_results)),
        "aggregate_delta": float(aggregate_delta),
        "aggregate_percent_change": float(aggregate_percent_change),
        "aggregate_ratio": float(aggregate_ratio),
        "resistance_node_pairs": resolved_pairs,
        "baseline_factor_value": float(baseline_factor_value),
        "constricted_factor_value": float(constricted_factor_value),
        "output_csv_path": str(output_path),
        "output_plot_path": str(plot_path),
        "output_flow_plot_path": str(flow_plot_path),
        "baseline_weight_results": baseline_weight_results,
        "constricted_weight_results": constricted_weight_results,
        "active_pericyte_indices": fixed_active_pericyte_indices,
        "active_center_indices_by_edge": fixed_active_center_indices_by_edge,
    }
