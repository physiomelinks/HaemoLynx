"""Utilities to compare baseline vs passive capillary-dilated haemodynamics."""
from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

from .poiseuille import PoiseuilleModel
from .resistance import (
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
    calc_two_point_from_laplacian_matrix_nodeID,
    solve_pressure_and_boundary_flow,
)


def _resolve_resistance_pairs(
    graph: nx.MultiGraph,
    *,
    resistance_node_pair: tuple[int, int] | None,
    resistance_node_pairs: list[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
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


def _is_capillary_branch_order(branch_order: object, capillary_branch_prefix: str) -> bool:
    return str(branch_order).startswith(str(capillary_branch_prefix))


def _scaled_capillary_diameter_map(
    diameter_by_branch_order: dict,
    *,
    capillary_branch_prefix: str,
    factor_value: float,
) -> dict:
    factor = float(factor_value)
    if factor <= 0.0:
        raise ValueError(f"factor_value must be > 0, got {factor_value}.")
    out = {}
    for branch_order, diameter_um in diameter_by_branch_order.items():
        if _is_capillary_branch_order(branch_order, capillary_branch_prefix):
            out[branch_order] = float(diameter_um) * factor
        else:
            out[branch_order] = float(diameter_um)
    return out


def _scale_capillary_fwhm_edges_in_place(
    graph: nx.MultiGraph,
    *,
    capillary_branch_prefix: str,
    factor_value: float,
) -> None:
    factor = float(factor_value)
    if factor <= 0.0:
        raise ValueError(f"factor_value must be > 0, got {factor_value}.")
    for _, _, _, edge_data in graph.edges(keys=True, data=True):
        branch_order = edge_data.get("branch_order")
        if not _is_capillary_branch_order(branch_order, capillary_branch_prefix):
            continue
        fwhm_d = edge_data.get("fwhm_diameter_um")
        if fwhm_d is None:
            continue
        if float(fwhm_d) <= 0.0:
            continue
        edge_data["fwhm_diameter_um"] = float(fwhm_d) * factor


def compare_baseline_vs_passive_capillary_dilation(
    graph: nx.MultiGraph,
    *,
    diameter_by_branch_order: dict,
    resistance_node_pair: tuple[int, int] | None = None,
    resistance_node_pairs: list[tuple[int, int]] | None = None,
    output_csv_path: str | Path,
    baseline_factor_value: float = 1.0,
    dilated_factor_value: float = 1.2,
    capillary_branch_prefix: str = "B",
    prefer_edge_fwhm_diameter: bool = False,
    use_constriction_integrator: bool = False,
    constriction_factor_by_branch_order: dict[str, float] | None = None,
    input_p_bc: float = 5000.0,
    output_p_bc: float = 2000.0,
    constriction_length: float = 40.0,
    constriction_spacing: float = 100.0,
) -> dict[str, Any]:
    """Compare resistance at baseline vs passive capillary whole-vessel dilation."""
    resolved_pairs = _resolve_resistance_pairs(
        graph,
        resistance_node_pair=resistance_node_pair,
        resistance_node_pairs=resistance_node_pairs,
    )
    if not diameter_by_branch_order:
        raise ValueError("diameter_by_branch_order cannot be empty.")

    graph_baseline = deepcopy(graph)
    graph_dilated = deepcopy(graph)
    if bool(prefer_edge_fwhm_diameter):
        _scale_capillary_fwhm_edges_in_place(
            graph_baseline,
            capillary_branch_prefix=capillary_branch_prefix,
            factor_value=float(baseline_factor_value),
        )
        _scale_capillary_fwhm_edges_in_place(
            graph_dilated,
            capillary_branch_prefix=capillary_branch_prefix,
            factor_value=float(dilated_factor_value),
        )

    diameter_map_baseline = _scaled_capillary_diameter_map(
        diameter_by_branch_order,
        capillary_branch_prefix=capillary_branch_prefix,
        factor_value=float(baseline_factor_value),
    )
    diameter_map_dilated = _scaled_capillary_diameter_map(
        diameter_by_branch_order,
        capillary_branch_prefix=capillary_branch_prefix,
        factor_value=float(dilated_factor_value),
    )

    poiseuille_model = PoiseuilleModel(
        constriction_length=float(constriction_length),
        constriction_spacing=float(constriction_spacing),
    )
    if bool(use_constriction_integrator):
        base_factor_map = {} if constriction_factor_by_branch_order is None else {
            str(k): float(v) for k, v in constriction_factor_by_branch_order.items()
        }
        full_factor_map = {
            str(branch_order): float(base_factor_map.get(str(branch_order), 1.0))
            for branch_order in diameter_by_branch_order.keys()
        }
        if bool(prefer_edge_fwhm_diameter):
            graph_baseline, baseline_weight_results = (
                poiseuille_model.set_poiseuille_weights_with_constrictions(
                    graph_baseline,
                    diameter_map_baseline,
                    prefer_edge_fwhm_baseline=True,
                    constriction_factor_by_branch_order=full_factor_map,
                )
            )
            graph_dilated, dilated_weight_results = (
                poiseuille_model.set_poiseuille_weights_with_constrictions(
                    graph_dilated,
                    diameter_map_dilated,
                    prefer_edge_fwhm_baseline=True,
                    constriction_factor_by_branch_order=full_factor_map,
                )
            )
        else:
            baseline_enhanced = {
                str(branch_order): {
                    "d1": float(diameter_um),
                    "d2": float(diameter_um) * float(full_factor_map[str(branch_order)]),
                }
                for branch_order, diameter_um in diameter_map_baseline.items()
            }
            dilated_enhanced = {
                str(branch_order): {
                    "d1": float(diameter_um),
                    "d2": float(diameter_um) * float(full_factor_map[str(branch_order)]),
                }
                for branch_order, diameter_um in diameter_map_dilated.items()
            }
            graph_baseline, baseline_weight_results = (
                poiseuille_model.set_poiseuille_weights_with_constrictions(
                    graph_baseline,
                    baseline_enhanced,
                )
            )
            graph_dilated, dilated_weight_results = (
                poiseuille_model.set_poiseuille_weights_with_constrictions(
                    graph_dilated,
                    dilated_enhanced,
                )
            )
    else:
        graph_baseline, baseline_weight_results = poiseuille_model.set_poiseuille_weights(
            graph_baseline,
            diameter_map_baseline,
            prefer_edge_fwhm_diameter=bool(prefer_edge_fwhm_diameter),
        )
        graph_dilated, dilated_weight_results = poiseuille_model.set_poiseuille_weights(
            graph_dilated,
            diameter_map_dilated,
            prefer_edge_fwhm_diameter=bool(prefer_edge_fwhm_diameter),
        )

    baseline_conductance, baseline_node_list = build_conductance_matrix_from_graph(graph_baseline)
    baseline_laplacian = calc_laplacian_from_conductance_matrix(baseline_conductance)
    dilated_conductance, dilated_node_list = build_conductance_matrix_from_graph(graph_dilated)
    dilated_laplacian = calc_laplacian_from_conductance_matrix(dilated_conductance)

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
        dilated_resistance = float(
            calc_two_point_from_laplacian_matrix_nodeID(
                dilated_laplacian,
                graph_dilated,
                source_node,
                target_node,
            )
        )
        delta = float(dilated_resistance - baseline_resistance)
        percent_change = (
            float((delta / baseline_resistance) * 100.0)
            if baseline_resistance != 0
            else float("inf")
        )
        ratio = (
            float(dilated_resistance / baseline_resistance)
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
        dilated_flow_result = solve_pressure_and_boundary_flow(
            conductance=dilated_conductance,
            node_list=dilated_node_list,
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
                "dilated_resistance": float(dilated_resistance),
                "delta": float(delta),
                "percent_change": float(percent_change),
                "ratio": float(ratio),
                "baseline_total_inlet_flow": float(baseline_flow_result["total_inlet_flow"]),
                "dilated_total_inlet_flow": float(dilated_flow_result["total_inlet_flow"]),
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
                    "scenario": "dilated",
                    "factor_value": float(dilated_factor_value),
                    "source_node": int(result["source_node"]),
                    "target_node": int(result["target_node"]),
                    "effective_resistance": float(result["dilated_resistance"]),
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

    plot_path = output_path.with_name(f"{output_path.stem}_before_after_resistance.png")
    baseline_values = [float(result["baseline_resistance"]) for result in pair_results]
    dilated_values = [float(result["dilated_resistance"]) for result in pair_results]
    baseline_mean, baseline_sem = _mean_sem(baseline_values)
    dilated_mean, dilated_sem = _mean_sem(dilated_values)
    input_change_percent = 0.0
    if baseline_factor_value != 0:
        input_change_percent = (
            (float(dilated_factor_value) - float(baseline_factor_value))
            / float(baseline_factor_value)
        ) * 100.0

    fig, ax = plt.subplots(figsize=(8, 5))
    x_before = 0.0
    x_after = 1.0
    for baseline_res, dilated_res in zip(baseline_values, dilated_values):
        ax.plot(
            [x_before, x_after],
            [baseline_res, dilated_res],
            color="tab:red",
            linewidth=1.8,
            alpha=0.9,
            zorder=1,
        )
    ax.scatter(
        [x_before] * len(baseline_values),
        baseline_values,
        color="tab:blue",
        s=90,
        alpha=0.9,
        zorder=3,
        label="Before",
    )
    ax.scatter(
        [x_after] * len(dilated_values),
        dilated_values,
        color="tab:red",
        s=90,
        alpha=0.9,
        zorder=3,
        label="After",
    )
    ax.bar(
        [x_before, x_after],
        [baseline_mean, dilated_mean],
        yerr=[baseline_sem, dilated_sem],
        width=0.35,
        color=["tab:blue", "tab:red"],
        alpha=0.25,
        ecolor="black",
        capsize=6,
        zorder=2,
        label="Mean ± SEM",
    )
    ax.set_xticks([x_before, x_after], labels=["Before", "After"])
    ax.set_ylabel("Effective resistance")
    ax.set_title(
        "Capillary Comparison: Paired Before/After Resistance by Input-Output Pair\n"
        f"Input capillary diameter change: {input_change_percent:+.1f}%"
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
    dilated_flow_values = [
        float(result["dilated_total_inlet_flow"]) for result in pair_results
    ]
    baseline_flow_mean, baseline_flow_sem = _mean_sem(baseline_flow_values)
    dilated_flow_mean, dilated_flow_sem = _mean_sem(dilated_flow_values)

    fig_flow, ax_flow = plt.subplots(figsize=(8, 5))
    for baseline_flow, dilated_flow in zip(baseline_flow_values, dilated_flow_values):
        ax_flow.plot(
            [x_before, x_after],
            [baseline_flow, dilated_flow],
            color="tab:red",
            linewidth=1.8,
            alpha=0.9,
            zorder=1,
        )
    ax_flow.scatter(
        [x_before] * len(baseline_flow_values),
        baseline_flow_values,
        color="tab:blue",
        s=90,
        alpha=0.9,
        zorder=3,
        label="Before",
    )
    ax_flow.scatter(
        [x_after] * len(dilated_flow_values),
        dilated_flow_values,
        color="tab:red",
        s=90,
        alpha=0.9,
        zorder=3,
        label="After",
    )
    ax_flow.bar(
        [x_before, x_after],
        [baseline_flow_mean, dilated_flow_mean],
        yerr=[baseline_flow_sem, dilated_flow_sem],
        width=0.35,
        color=["tab:blue", "tab:red"],
        alpha=0.25,
        ecolor="black",
        capsize=6,
        zorder=2,
        label="Mean ± SEM",
    )
    ax_flow.set_xticks([x_before, x_after], labels=["Before", "After"])
    ax_flow.set_ylabel("Total inlet flow")
    ax_flow.set_title(
        "Capillary Comparison: Paired Before/After Flow by Input-Output Pair\n"
        f"Input capillary diameter change: {input_change_percent:+.1f}%"
    )
    ax_flow.grid(True, axis="y", alpha=0.3)
    ax_flow.legend()
    fig_flow.tight_layout()
    fig_flow.savefig(flow_plot_path, dpi=200)
    plt.close(fig_flow)

    first_result = pair_results[0]
    return {
        "baseline_resistance": float(first_result["baseline_resistance"]),
        "dilated_resistance": float(first_result["dilated_resistance"]),
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
        "dilated_factor_value": float(dilated_factor_value),
        "output_csv_path": str(output_path),
        "output_plot_path": str(plot_path),
        "output_flow_plot_path": str(flow_plot_path),
        "use_constriction_integrator": bool(use_constriction_integrator),
        "baseline_weight_results": baseline_weight_results,
        "dilated_weight_results": dilated_weight_results,
    }
