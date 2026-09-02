"""Compare baseline vs constricted pericyte haemodynamics.

Runs one constriction strategy twice over copies of the same graph — once at a
baseline constriction factor, once at a constricted one — and reports what that
did to the effective resistance between two nodes. Which strategy runs, and
with what settings, is :mod:`haemolynx.haemodynamics.constriction_strategy`;
this module only sets up the two scenarios and reports the difference.
"""
from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from haemolynx.io.axis_order import CANONICAL_AXIS_ORDER
from .constriction import resolve_generator
from .constriction_strategy import (
    set_resistances_for_constriction_strategy,
    uniform_constriction_factors,
)
from .resistance import (
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
    calc_two_point_from_laplacian_matrix_nodeID,
)
from .viscosity import DEFAULT_HAEMATOCRIT


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
    viscosity_law: str = "pries",
    haematocrit: float = DEFAULT_HAEMATOCRIT,
    diameter_basis: str = "plasma_column",
    max_assignment_distance_um: float | None = 3.0,
    min_pericyte_diameter_um: float | None = 5.0,
    max_pericyte_diameter_um: float | None = 12.0,
    axis_order: str = CANONICAL_AXIS_ORDER,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compare effective resistance at baseline vs constricted settings.

    Comparison factors are treated as absolute values (not scales). This means
    ``baseline_factor_value`` and ``constricted_factor_value`` override the
    non-comparison constriction magnitudes while comparison is running —
    including every entry of ``constriction_factor_by_branch_order``, which is
    accepted so a caller can pass its run settings through unchanged, and then
    superseded for the duration of the comparison.

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

    # One generator for both scenarios: the constricted arm reuses the cohort the
    # baseline arm drew, so what varies between them is the factor, not the draw.
    generator = resolve_generator(rng, seed)

    graph_baseline = deepcopy(graph)
    graph_constricted = deepcopy(graph)
    fixed_active_pericyte_indices: list[int] | None = None
    fixed_active_center_indices_by_edge: dict[str, list[int]] | None = None

    # Everything the two scenarios share. They differ only in their constriction
    # factor and, for a probabilistic model, in the second one reusing the
    # pericyte cohort the first one drew.
    scenario_settings: dict[str, Any] = {
        "diameter_by_branch_order": diameter_by_branch_order,
        "use_pericyte_mask_constriction": bool(use_pericyte_mask_constriction),
        "use_probabilistic_constriction": bool(use_probabilistic_pericyte_constriction),
        "prefer_edge_fwhm_baseline": bool(prefer_edge_fwhm_baseline),
        "constriction_length": float(constriction_length),
        "constriction_spacing": float(constriction_spacing),
        "constriction_probability": float(pericyte_constriction_probability),
        # The comparison's two arms must differ only in their constriction
        # factor, so the viscosity law travels with the rest of the settings.
        "viscosity_law": viscosity_law,
        "haematocrit": float(haematocrit),
        "diameter_basis": diameter_basis,
        "pericyte_mask_path": pericyte_mask_path,
        "pericyte_mask_h5_dataset_name": pericyte_mask_h5_dataset_name,
        "max_assignment_distance_um": max_assignment_distance_um,
        "min_pericyte_diameter_um": min_pericyte_diameter_um,
        "max_pericyte_diameter_um": max_pericyte_diameter_um,
        "axis_order": axis_order,
        "rng": generator,
    }

    graph_baseline, _strategy, baseline_resistance_results = (
        set_resistances_for_constriction_strategy(
            graph_baseline,
            constriction_factor_by_branch_order=uniform_constriction_factors(
                diameter_by_branch_order,
                float(baseline_factor_value),
            ),
            active_pericyte_indices=None,
            active_center_indices_by_edge=None,
            **scenario_settings,
        )
    )
    if use_pericyte_mask_constriction and use_probabilistic_pericyte_constriction:
        selected = baseline_resistance_results.get("active_pericyte_indices")
        fixed_active_pericyte_indices = [int(idx) for idx in selected] if selected else []
    if (not use_pericyte_mask_constriction) and use_probabilistic_pericyte_constriction:
        selected_map = baseline_resistance_results.get("active_center_indices_by_edge")
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

    graph_constricted, _strategy, constricted_resistance_results = (
        set_resistances_for_constriction_strategy(
            graph_constricted,
            constriction_factor_by_branch_order=uniform_constriction_factors(
                diameter_by_branch_order,
                float(constricted_factor_value),
            ),
            active_pericyte_indices=fixed_active_pericyte_indices,
            active_center_indices_by_edge=fixed_active_center_indices_by_edge,
            **scenario_settings,
        )
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
        "baseline_resistance_results": baseline_resistance_results,
        "constricted_resistance_results": constricted_resistance_results,
        "active_pericyte_indices": fixed_active_pericyte_indices,
        "active_center_indices_by_edge": fixed_active_center_indices_by_edge,
    }
