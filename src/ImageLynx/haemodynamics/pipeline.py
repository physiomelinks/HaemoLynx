"""High-level haemodynamics steps for vascular graphs."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from ImageLynx import io
from ImageLynx.haemodynamics import automated
from ImageLynx.haemodynamics.poiseuille import PoiseuilleModel
from ImageLynx.haemodynamics import pericyte_comparison as pericyte_comparison_haemodynamics
from ImageLynx.haemodynamics import pericyte_mask as pericyte_mask_haemodynamics
from ImageLynx.haemodynamics import probability as probability_haemodynamics

logger = logging.getLogger(__name__)


@dataclass
class HaemodynamicsApplyConfig:
    """Settings for Poiseuille conductance assignment on a vascular graph."""

    diameter_by_branch_order: dict[str, float]
    constriction_by_branch_order: dict[str, float] = field(default_factory=dict)
    custom_edges: list | dict = field(default_factory=list)
    custom_edge_diameter: float = 6.0
    constriction_length: float = 40.0
    constriction_spacing: float = 100.0
    do_pericyte_constriction: bool = False
    use_pericyte_mask_constriction: bool = False
    pericyte_mask_path: Path | str | None = None
    pericyte_mask_h5_dataset_name: str | None = None
    pericyte_max_assignment_distance_um: float | None = None
    pericyte_min_diameter_um: float | None = None
    pericyte_max_diameter_um: float | None = None
    use_probabilistic_pericyte_constriction: bool = False
    pericyte_constriction_probability: float = 0.5
    run_pericyte_resistance_comparison: bool = False
    pericyte_comparison_baseline_value: float = 1.0
    pericyte_comparison_constricted_value: float = 0.8
    reuse_comparison_pericyte_cohort_for_main_run: bool = False
    comparison_output_csv_path: Path | None = None
    resistance_node_pair: tuple[int, int] | None = None
    use_fwhm_edge_diameters: bool = False
    fwhm_raw_tiff_path: Path | str | None = None
    voxel_size_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    fwhm_sample_spacing_along_edge_um: float = 5.0
    fwhm_transverse_profile_step_um: float = 0.5
    fwhm_transverse_half_extent_um: float = 10.0
    fwhm_diameter_guess_um: float | None = None
    fwhm_min_total_extent_multiplier: float = 1.5
    fwhm_background_label: int = 0
    fwhm_junction_label: int = 2
    fwhm_allow_junction_crossing: bool = False
    fwhm_profile_baseline_mode: str = "wings"
    fwhm_profile_baseline_wing_fraction: float = 0.2
    fwhm_constrain_fitted_baseline: bool = True
    fwhm_baseline_constraint_half_width_ptp: float = 0.15
    fwhm_clip_profile_to_single_vessel: bool = True
    fwhm_clip_min_drop_fraction_of_center: float = 0.35
    fwhm_clip_re_rise_fraction_of_center: float = 0.2
    fwhm_branch_endpoint_exclusion_um: float = 2.0
    fwhm_junction_proximity_exclusion_um: float = 3.0
    fwhm_enforce_same_edge_locality: bool = True
    fwhm_same_edge_arc_window_um: float | None = None
    fwhm_same_edge_arc_window_multiplier: float = 2.0
    fwhm_same_edge_arc_window_min_um: float = 5.0
    fwhm_cap_half_extent_by_nonlocal_same_edge_distance: bool = True
    fwhm_nonlocal_same_edge_arc_separation_um: float = 15.0
    fwhm_nonlocal_same_edge_half_extent_factor: float = 0.5
    fwhm_reject_samples_with_center_offset: bool = True
    fwhm_max_fit_center_offset_um: float = 1.5
    fwhm_reject_samples_with_low_fit_r2: bool = True
    fwhm_min_fit_r2: float = 0.85


def _measure_fwhm_diameters(G: nx.MultiGraph, config: HaemodynamicsApplyConfig) -> dict[str, Any]:
    if config.fwhm_raw_tiff_path is None:
        raise ValueError("use_fwhm_edge_diameters=True requires fwhm_raw_tiff_path.")
    raw_p = io.resolve_image_path_with_optional_zip(Path(config.fwhm_raw_tiff_path))
    voxel_sz = tuple(
        float(v) for v in G.graph.get("image_voxel_size_xyz", config.voxel_size_xyz)
    )
    return automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        raw_tiff_path=raw_p,
        voxel_size_xyz=voxel_sz,
        sample_spacing_along_edge_um=float(config.fwhm_sample_spacing_along_edge_um),
        transverse_profile_step_um=float(config.fwhm_transverse_profile_step_um),
        transverse_half_extent_um=float(config.fwhm_transverse_half_extent_um),
        diameter_guess_um=(
            None if config.fwhm_diameter_guess_um is None else float(config.fwhm_diameter_guess_um)
        ),
        background_label=int(config.fwhm_background_label),
        junction_label=int(config.fwhm_junction_label),
        min_total_extent_multiplier=float(config.fwhm_min_total_extent_multiplier),
        profile_baseline_mode=config.fwhm_profile_baseline_mode,
        profile_baseline_wing_fraction=float(config.fwhm_profile_baseline_wing_fraction),
        constrain_fitted_baseline=bool(config.fwhm_constrain_fitted_baseline),
        allow_junction_crossing=bool(config.fwhm_allow_junction_crossing),
        baseline_constraint_half_width_ptp=float(config.fwhm_baseline_constraint_half_width_ptp),
        clip_profile_to_single_vessel=bool(config.fwhm_clip_profile_to_single_vessel),
        clip_min_drop_fraction_of_center=float(config.fwhm_clip_min_drop_fraction_of_center),
        clip_re_rise_fraction_of_center=float(config.fwhm_clip_re_rise_fraction_of_center),
        branch_endpoint_exclusion_um=float(config.fwhm_branch_endpoint_exclusion_um),
        junction_proximity_exclusion_um=float(config.fwhm_junction_proximity_exclusion_um),
        enforce_same_edge_locality=bool(config.fwhm_enforce_same_edge_locality),
        same_edge_arc_window_um=(
            None
            if config.fwhm_same_edge_arc_window_um is None
            else float(config.fwhm_same_edge_arc_window_um)
        ),
        same_edge_arc_window_multiplier=float(config.fwhm_same_edge_arc_window_multiplier),
        same_edge_arc_window_min_um=float(config.fwhm_same_edge_arc_window_min_um),
        cap_half_extent_by_nonlocal_same_edge_distance=bool(
            config.fwhm_cap_half_extent_by_nonlocal_same_edge_distance
        ),
        nonlocal_same_edge_arc_separation_um=float(config.fwhm_nonlocal_same_edge_arc_separation_um),
        nonlocal_same_edge_half_extent_factor=float(config.fwhm_nonlocal_same_edge_half_extent_factor),
        reject_samples_with_center_offset=bool(config.fwhm_reject_samples_with_center_offset),
        max_fit_center_offset_um=float(config.fwhm_max_fit_center_offset_um),
        reject_samples_with_low_fit_r2=bool(config.fwhm_reject_samples_with_low_fit_r2),
        min_fit_r2=float(config.fwhm_min_fit_r2),
    )


def _run_pericyte_comparison(
    G: nx.MultiGraph,
    config: HaemodynamicsApplyConfig,
) -> tuple[list[int] | None, dict[str, list[int]] | None, dict[str, Any]]:
    if not config.run_pericyte_resistance_comparison:
        return None, None, {}
    if config.comparison_output_csv_path is None:
        raise ValueError("comparison_output_csv_path required for pericyte comparison.")
    if config.resistance_node_pair is None:
        raise ValueError("resistance_node_pair required for pericyte comparison.")

    comparison_results = pericyte_comparison_haemodynamics.compare_baseline_vs_pericyte_constriction(
        G,
        diameter_by_branch_order=config.diameter_by_branch_order,
        constriction_factor_by_branch_order=config.constriction_by_branch_order,
        resistance_node_pair=config.resistance_node_pair,
        output_csv_path=config.comparison_output_csv_path,
        baseline_factor_value=float(config.pericyte_comparison_baseline_value),
        constricted_factor_value=float(config.pericyte_comparison_constricted_value),
        use_pericyte_mask_constriction=bool(config.use_pericyte_mask_constriction),
        pericyte_mask_path=config.pericyte_mask_path,
        pericyte_mask_h5_dataset_name=config.pericyte_mask_h5_dataset_name,
        max_assignment_distance_um=config.pericyte_max_assignment_distance_um,
        min_pericyte_diameter_um=config.pericyte_min_diameter_um,
        max_pericyte_diameter_um=config.pericyte_max_diameter_um,
        prefer_edge_fwhm_baseline=bool(config.use_fwhm_edge_diameters),
        constriction_length=config.constriction_length,
        constriction_spacing=config.constriction_spacing,
        use_probabilistic_pericyte_constriction=bool(config.use_probabilistic_pericyte_constriction),
        pericyte_constriction_probability=float(config.pericyte_constriction_probability),
    )

    active_pericyte_indices: list[int] | None = None
    active_center_indices_by_edge: dict[str, list[int]] | None = None
    if (
        config.reuse_comparison_pericyte_cohort_for_main_run
        and config.use_probabilistic_pericyte_constriction
    ):
        if config.use_pericyte_mask_constriction:
            selected = comparison_results.get("active_pericyte_indices")
            active_pericyte_indices = [int(idx) for idx in selected] if selected else []
        else:
            selected_map = comparison_results.get("active_center_indices_by_edge")
            if isinstance(selected_map, dict):
                active_center_indices_by_edge = {
                    str(edge_id): [int(idx) for idx in idx_list]
                    for edge_id, idx_list in selected_map.items()
                }
    return active_pericyte_indices, active_center_indices_by_edge, comparison_results


def _assign_poiseuille_weights(
    G: nx.MultiGraph,
    config: HaemodynamicsApplyConfig,
    *,
    active_pericyte_indices: list[int] | None,
    active_center_indices_by_edge: dict[str, list[int]] | None,
) -> dict[str, Any]:
    poiseuille_model = PoiseuilleModel(
        constriction_length=config.constriction_length,
        constriction_spacing=config.constriction_spacing,
    )
    results: dict[str, Any] = {}

    if config.do_pericyte_constriction:
        if config.use_pericyte_mask_constriction:
            if config.pericyte_mask_path is None:
                raise ValueError(
                    "pericyte_mask_path must be set when use_pericyte_mask_constriction=True."
                )
            G, results["pericyte_mask"] = (
                pericyte_mask_haemodynamics.set_poiseuille_weights_with_pericyte_mask(
                    G,
                    diameter_by_branch_order=config.diameter_by_branch_order,
                    constriction_factor_by_branch_order=config.constriction_by_branch_order,
                    pericyte_mask_path=config.pericyte_mask_path,
                    pericyte_mask_h5_dataset_name=config.pericyte_mask_h5_dataset_name,
                    max_assignment_distance_um=config.pericyte_max_assignment_distance_um,
                    min_pericyte_diameter_um=config.pericyte_min_diameter_um,
                    max_pericyte_diameter_um=config.pericyte_max_diameter_um,
                    prefer_edge_fwhm_baseline=bool(config.use_fwhm_edge_diameters),
                    constriction_length=config.constriction_length,
                    use_probabilistic_constriction=bool(config.use_probabilistic_pericyte_constriction),
                    constriction_probability=float(config.pericyte_constriction_probability),
                    active_pericyte_indices=(
                        active_pericyte_indices
                        if (
                            config.reuse_comparison_pericyte_cohort_for_main_run
                            and config.use_probabilistic_pericyte_constriction
                        )
                        else None
                    ),
                )
            )
        elif config.use_probabilistic_pericyte_constriction:
            G, results["probabilistic"] = (
                probability_haemodynamics.set_poiseuille_weights_with_probabilistic_periodic_constrictions(
                    G,
                    diameter_by_branch_order=config.diameter_by_branch_order,
                    constriction_factor_by_branch_order=config.constriction_by_branch_order,
                    prefer_edge_fwhm_baseline=bool(config.use_fwhm_edge_diameters),
                    constriction_length=config.constriction_length,
                    constriction_spacing=config.constriction_spacing,
                    constriction_probability=float(config.pericyte_constriction_probability),
                    active_center_indices_by_edge=(
                        active_center_indices_by_edge
                        if (
                            config.reuse_comparison_pericyte_cohort_for_main_run
                            and config.use_probabilistic_pericyte_constriction
                        )
                        else None
                    ),
                )
            )
        elif config.use_fwhm_edge_diameters:
            G, results["constrictions"] = poiseuille_model.set_poiseuille_weights_with_constrictions(
                G,
                config.diameter_by_branch_order,
                prefer_edge_fwhm_baseline=True,
                constriction_factor_by_branch_order=config.constriction_by_branch_order,
            )
        else:
            diameter_enhanced = {
                branch_order: {
                    "d1": diameter,
                    "d2": diameter * config.constriction_by_branch_order.get(branch_order, 1.0),
                }
                for branch_order, diameter in config.diameter_by_branch_order.items()
            }
            G, results["constrictions"] = poiseuille_model.set_poiseuille_weights_with_constrictions(
                G,
                diameter_enhanced,
            )
    else:
        G, results["poiseuille"] = poiseuille_model.set_poiseuille_weights(
            G,
            config.diameter_by_branch_order,
            prefer_edge_fwhm_diameter=bool(config.use_fwhm_edge_diameters),
        )

    G, results["custom_edges"] = poiseuille_model.set_poiseuille_edge_weights(
        G,
        config.custom_edges,
        edge_diameter=config.custom_edge_diameter,
    )
    return results


def apply_poiseuille_haemodynamics(
    G: nx.MultiGraph,
    *,
    diameter_by_branch_order: dict[str, float] | None = None,
    constriction_by_branch_order: dict[str, float] | None = None,
    custom_edges: list | dict | None = None,
    config: HaemodynamicsApplyConfig | None = None,
) -> tuple[nx.MultiGraph, dict[str, Any]]:
    """
    Assign Poiseuille edge conductances on ``G``.

    For the simple tutorial path, pass ``diameter_by_branch_order`` and optional
    ``custom_edges``. For the full example-pipeline path, pass a
    :class:`HaemodynamicsApplyConfig` via ``config`` (other kwargs are ignored).

    Returns
    -------
    tuple
        ``(G, results)`` where ``results`` summarizes FWHM, pericyte, and weight steps.
    """
    if config is None:
        if diameter_by_branch_order is None:
            raise ValueError(
                "diameter_by_branch_order is required when config is not provided."
            )
        config = HaemodynamicsApplyConfig(
            diameter_by_branch_order=diameter_by_branch_order,
            constriction_by_branch_order=constriction_by_branch_order or {},
            custom_edges=custom_edges or [],
        )
    summary: dict[str, Any] = {}

    if config.use_fwhm_edge_diameters:
        summary["fwhm"] = _measure_fwhm_diameters(G, config)

    active_pericyte_indices, active_center_indices_by_edge, comparison_results = (
        _run_pericyte_comparison(G, config)
    )
    if comparison_results:
        summary["pericyte_comparison"] = comparison_results

    summary["weights"] = _assign_poiseuille_weights(
        G,
        config,
        active_pericyte_indices=active_pericyte_indices,
        active_center_indices_by_edge=active_center_indices_by_edge,
    )

    return G, summary
