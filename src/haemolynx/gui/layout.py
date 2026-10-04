"""Where each row on a tab sits: its box, and the "Advanced" button it hides behind.

Most runs change a handful of settings, and a tab that shows every one of its
rows buries them. A setting the schema marks ``advanced`` therefore starts out
hidden behind an "Advanced" button, and the button sits *under the setting it
belongs to*: the FWHM gates appear under "Use FWHM edge diameters", not in one
pile at the foot of the tab. The rules, all decided here and none in Qt:

* **Anchor.** An advanced row hangs off its *anchor*: the innermost ordinary
  (not advanced) row on the same box that it needs, through its ``requires``
  chain. An advanced row whose own parent is advanced goes with that parent,
  so a child is never behind a different button from the toggle it needs. A
  row that needs nothing on its box hangs off the box itself.
* **One button per anchor**, placed after the anchor's last ordinary
  descendant in the box -- the end of the anchor's own subtree -- and indented
  one level deeper than the anchor. Two buttons after the same row are
  ordered innermost first.
* **Boxes.** A tab is one or more boxes: by default the schema's sections, as
  the panel always drew them (the first untitled, every later one a titled
  group box), or a hand-made grouping from :data:`TAB_BOXES` for the tabs
  whose sections do not read as groups.

Everything here is pure: settings names and the schema in, a
:class:`TabLayout` out. :mod:`haemolynx.gui._widget` draws it, and decides
nothing about placement itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from haemolynx.gui.tabs import section_box_title
from haemolynx.parsers.schema import ConfigError, Schema, is_active, prerequisite_name

__all__ = [
    "ADVANCED_GROUPS",
    "ADVANCED_TITLES",
    "ANCHOR_OVERRIDES",
    "BoxSpec",
    "Box",
    "DEFAULT_ADVANCED_TITLE",
    "Disclosure",
    "MOVE_BEFORE",
    "TAB_BOXES",
    "TabLayout",
    "changed_settings",
    "entry_disclosure",
    "layout_for",
    "role_nodes_disclosure",
    "section_slug",
    "shared_ilastik_disclosure",
]


@dataclass(frozen=True)
class BoxSpec:
    """One hand-made box: which rows it holds, and what it is called.

    A row is claimed by the first rule that names it, in this order over every
    box of the tab: ``names`` (listed, in that order), then ``subtree_of``
    (that setting and every row needing it, in tab order), then ``rest``
    (whatever no box has claimed yet).
    """

    key: str
    #: None draws the box as a plain run of rows rather than a titled group.
    title: str | None
    names: tuple[str, ...] = ()
    subtree_of: str | None = None
    rest: bool = False


@dataclass(frozen=True)
class Disclosure:
    """One "Advanced" button and the rows it reveals."""

    #: Unique on the panel; also what the button's Qt object name is made from.
    key: str
    #: The ordinary row this button sits under, or None for a whole box's.
    anchor: str | None
    title: str
    #: How many steps in from the box's edge the button is drawn: 0 for a
    #: box's own button, one more than its anchor's depth otherwise.
    level: int
    #: Every row behind the button, in display order.
    names: tuple[str, ...]
    #: The same rows as ``(heading, rows)`` runs: a None heading is the
    #: untitled run first, every other one a titled sub-group.
    groups: tuple[tuple[str | None, tuple[str, ...]], ...]


@dataclass(frozen=True)
class Box:
    """A run of rows, with their disclosures placed among them."""

    key: str
    title: str | None
    #: Row names and :class:`Disclosure` objects, in display order.
    items: tuple[Any, ...]

    @property
    def rows(self) -> tuple[str, ...]:
        """The ordinary rows: always shown when their prerequisites allow."""
        return tuple(item for item in self.items if isinstance(item, str))

    @property
    def disclosures(self) -> tuple[Disclosure, ...]:
        return tuple(item for item in self.items if isinstance(item, Disclosure))

    @property
    def names(self) -> tuple[str, ...]:
        """Every setting in the box, advanced ones included."""
        names: list[str] = []
        for item in self.items:
            names.extend((item,) if isinstance(item, str) else item.names)
        return tuple(names)


@dataclass(frozen=True)
class TabLayout:
    boxes: tuple[Box, ...]

    @property
    def disclosures(self) -> tuple[Disclosure, ...]:
        return tuple(d for box in self.boxes for d in box.disclosures)

    def box(self, key: str) -> Box:
        for box in self.boxes:
            if box.key == key:
                return box
        raise KeyError(f"No box {key!r}. Boxes are: {[b.key for b in self.boxes]}.")

    def disclosure_of(self, name: str) -> Disclosure | None:
        """The disclosure *name* is behind, or None for an ordinary row."""
        for disclosure in self.disclosures:
            if name in disclosure.names:
                return disclosure
        return None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for box in self.boxes for name in box.names)


#: Tabs grouped by hand, keyed by stage call. Every other tab gets one box per
#: schema section, the way the panel has always drawn them.
TAB_BOXES: Mapping[str, tuple[BoxSpec, ...]] = {
    "segment": (
        BoxSpec(
            "input",
            None,
            names=(
                "input_path",
                "use_ilastik_segmentation",
                "ilastik_unsegmented_image_path",
                "ilastik_classifier_path",
                "mid_run_postprocessing",
            ),
            rest=True,
        ),
        # The policy first: it decides whether the override is read at all.
        BoxSpec(
            "voxel_size",
            "Voxel size and axes",
            names=("image_axis_order", "voxel_size_policy", "voxel_size_override_xyz"),
        ),
        BoxSpec("memory", "Memory", subtree_of="use_memmap_loading"),
        BoxSpec("segmentation_cleanup", "Segmentation cleanup", subtree_of="segmentation_cleanup"),
        # What tunes "Check segmented image"'s score; the panel adds the check
        # and optimise buttons themselves to this box.
        BoxSpec(
            "check_and_optimise",
            "Check and optimise",
            names=(
                "expected_boundary_vessel_count",
                "min_voxels_across_vessel_radius",
                "min_acceptable_segmentation_quality",
            ),
        ),
    ),
    "assign_boundaries": (
        BoxSpec("vessel_masks", None, names=("automated_vessel_assignment",)),
        BoxSpec(
            "large_vessel_masks",
            "Large-vessel masks: inlets and outlets",
            subtree_of="use_large_vessel_masks",
        ),
        BoxSpec(
            "small_vessel_masks",
            "Small-vessel masks: arteriole and venule boundaries",
            subtree_of="use_small_vessel_masks_for_boundary_assignment",
        ),
        # Anything else about choosing boundaries that no role page places:
        # its own button above the role tabs, rather than falling in with
        # the node IDs below them.
        BoxSpec("boundary_other", None, rest=True),
        # The node-ID rows no role page owns; the page draws this box last,
        # under the role tabs.
        BoxSpec(
            "boundary_nodes",
            None,
            names=("large_arteriole_boundary_nodes", "large_venule_boundary_nodes"),
        ),
    ),
    "assign_diameters": (BoxSpec("diameters", None, rest=True),),
    "build_haemodynamic_model": (
        # The master toggle first, not below the pressures it switches off.
        BoxSpec(
            "haemodynamics",
            None,
            names=(
                "run_haemodynamics",
                "haemodynamics_solver",
                "do_equiv_resistance_calculation",
                "equivalent_resistance_solver",
            ),
        ),
        BoxSpec("boundary_pressures", "Boundary pressures", names=("inlet_p_bc", "outlet_p_bc")),
        BoxSpec(
            "blood_model",
            # The viscosity and haematocrit rows' own section is "Diameters
            # and pericytes", which names another tab.
            section_box_title("build_haemodynamic_model", "Diameters and pericytes"),
            names=(
                "viscosity_law",
                "haematocrit",
                "haematocrit_model",
                "haematocrit_junction_rule",
                "haematocrit_split_connector_length_um",
                "haematocrit_distribution_max_iterations",
                "haematocrit_distribution_tolerance",
            ),
        ),
        BoxSpec("network_handling", "Network handling", rest=True),
    ),
    "export_results": (
        # Where every output goes -- stage caches and snapshots included --
        # so it leads the tab.
        BoxSpec(
            "export",
            None,
            names=("vtk_output_prefix", "vtk_export", "visualize_vtk", "export_citations"),
            rest=True,
        ),
    ),
}

#: A row drawn just before another rather than in schema order. Only
#: ``do_fwhm_measurement``: it gates the decoy check and the raw-section
#: fallback, so it sits directly above them and its own Advanced button (the
#: decoy check) follows them, instead of heading the FWHM rows.
MOVE_BEFORE: Mapping[str, str] = {"do_fwhm_measurement": "use_raw_section_fallback"}

#: An advanced row whose anchor is not what its ``requires`` would give.
#: Tiling needs the Low RAM option on the Input tab, not anything on
#: Skeletonise, but it is part of skeletonising.
ANCHOR_OVERRIDES: Mapping[str, str] = {"skeletonize_tile_large_components": "do_skeletonize"}

#: What each button says, keyed by its anchor, or ``box:<key>`` for a box's
#: own button. Anything unlisted says :data:`DEFAULT_ADVANCED_TITLE`.
DEFAULT_ADVANCED_TITLE = "Advanced settings"
ADVANCED_TITLES: Mapping[str, str] = {
    "use_memmap_loading": "Advanced memory settings",
    "segmentation_cleanup_split_narrow_necks": "Advanced neck splitting",
    "segmentation_cleanup_reconnect_gaps": "Advanced gap reconnection",
    "box:check_and_optimise": "Advanced check and optimise settings",
    "do_skeletonize": "Advanced skeletonisation",
    "use_thick_vessel_skeletonisation": "Advanced thick-vessel settings",
    "box:pipeline_stages": "Checks",
    "do_graph_building": "Advanced graph building",
    "smooth_centrelines": "Advanced centreline smoothing",
    "use_large_vessel_masks": "Advanced large-vessel assignment",
    "cut_network_at_large_vessel_volumes": "Advanced cutting",
    "assign_large_vessel_branch_orders": "Advanced large-vessel network",
    "use_small_vessel_masks_for_boundary_assignment": "Advanced small-vessel assignment",
    "box:boundary_nodes": "Large-vessel boundary node IDs filled in by the run",
    "box:boundary_other": "Advanced boundary settings",
    "all_diams_const": "Advanced diameter table",
    "use_fwhm_edge_diameters": "Advanced FWHM settings",
    "do_fwhm_measurement": "Advanced FWHM measurement",
    "use_raw_section_fallback": "Advanced raw-section fit",
    "use_endothelial_diameters": "Advanced endothelial settings",
    "use_edt_diameter_crosscheck": "Advanced mask-diameter settings",
    "box:diameters": "Advanced diameter settings",
    "run_haemodynamics": "Advanced haemodynamics",
    "haematocrit_model": "Advanced haematocrit iteration",
    "haematocrit_junction_rule": "Advanced junction splitting",
    "box:network_handling": "Advanced network handling",
    "run_perturbations": "Advanced perturbation settings",
    "measurement_3d_to_cell_mask": "Advanced 3D distance settings",
    "statistics": "Choose statistics",
    "statistics_network_analysis": "Choose network measures",
    "vtk_export": "Advanced VTK settings",
    "box:export": "Advanced export settings",
}

#: Titled sub-groups inside the longer panels, keyed like
#: :data:`ADVANCED_TITLES`. Rows a panel holds that no group names come
#: first, untitled; a group naming rows the panel does not hold is skipped.
ADVANCED_GROUPS: Mapping[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "do_skeletonize": (
        (
            "Bridging",
            (
                "skeleton_bridge_weight_by_segmentation",
                "skeleton_bridge_z_distance_weight",
                "skeleton_bridge_min_facing_cosine",
                "bridge_require_mask_support",
                "bridge_max_background_gap_um",
                "bridge_min_mask_fraction",
                "skeleton_component_connectivity",
            ),
        ),
        (
            "Dense-bundle hubs",
            (
                "skeleton_bundle_scan_size",
                "skeleton_bundle_density_fraction",
                "skeleton_bundle_max_connections_per_hub",
                "skeleton_bundle_hub_min_spacing",
            ),
        ),
        (
            "Large volumes (needs the Low RAM option on the Input tab)",
            (
                "skeletonize_tile_large_components",
                "skeletonize_tile_max_voxels",
                "skeletonize_tile_halo_um",
            ),
        ),
    ),
    "use_thick_vessel_skeletonisation": (
        (
            "Thickness gate",
            (
                "skeleton_fill_mask_holes_before_thickness",
                "skeleton_thick_vessel_restrict_to_mask",
                "skeleton_thick_vessel_restrict_to_mask_warn_below",
            ),
        ),
        (
            "Joining thin vessels to thick ones",
            (
                "skeleton_thick_vessel_wall_absorption_um",
                "skeleton_thick_vessel_flake_filter_um",
                "skeleton_thick_vessel_max_bridge_radius_multiple",
                "skeleton_thick_vessel_max_bridge_distance_um",
                "skeleton_thick_vessel_bridge_radius_smoothing_um",
            ),
        ),
        (
            "Braiding check",
            (
                "detect_thick_vessel_braiding",
                "thick_vessel_braid_factor_limit",
                "thick_vessel_braid_min_occupied_slices",
            ),
        ),
    ),
    "box:pipeline_stages": (
        (
            "Consistency warnings",
            (
                "skeleton_mask_consistency_warn_below",
                "missing_vessel_min_voxels",
                "skeleton_missing_vessel_warn_below",
                "skeleton_graph_consistency_warn_below",
                "graph_mask_consistency_warn_below",
                "graph_missing_vessel_warn_below",
            ),
        ),
        (
            "Cartwheel hub guard",
            (
                "detect_cartwheel_hub_artifacts",
                "cartwheel_hub_min_degree",
                "cartwheel_hub_max_radial_dispersion",
                "cartwheel_hub_tangent_length_um",
            ),
        ),
    ),
    "do_graph_building": (
        (
            "Cluster collapse",
            (
                "cluster_collapse_method",
                "cluster_collapse_max_radial_dispersion",
                "cluster_collapse_persistence_search_multiple",
            ),
        ),
        (
            "Mask recovery",
            (
                "recover_uncovered_mask_vessels",
                "recovery_min_region_volume_um3",
                "recovery_min_length_um",
            ),
        ),
        ("Debugging", ("save_step_artifacts",)),
    ),
    "use_large_vessel_masks": (
        (
            "Mask cleanup",
            (
                "large_vessel_mask_dilation_microns",
                "large_vessel_min_component_volume_um3",
                "large_vessel_swap_minority_components",
                "large_vessel_swap_max_size_ratio",
                "large_vessel_swap_min_contact_fraction",
                "large_vessel_remove_small_opposite_attached_components",
                "large_vessel_opposite_attached_max_component_volume_um3",
                "large_vessel_opposite_attached_max_distance_microns",
                "exclude_smaller_overlapping_volumes",
            ),
        ),
        (
            "Overlap cleanup",
            (
                "automated_vessel_assignment_enable_overlap_cleanup",
                "automated_vessel_assignment_fast_mode",
                "automated_vessel_assignment_apply_overlap_cleanup_in_normal_mode",
                "automated_vessel_overlap_parallel_workers",
            ),
        ),
        (
            "Assignment model",
            (
                "automated_vessel_assignment_use_legacy_mode",
                "automated_vessel_confidence_margin",
                "automated_vessel_min_confidence",
                "automated_vessel_topology_penalty",
                "automated_vessel_quality_max_overlap_fraction",
                "automated_vessel_quality_min_terminal_coverage",
                "automated_vessel_quality_max_component_count",
                "automated_vessel_conservative_max_dilation_microns",
            ),
        ),
        (
            "Diagnostics",
            (
                "write_fast_mode_preassignment_large_vessel_debug_3d_html",
                "large_vessel_3d_volume_downsample_stride",
            ),
        ),
    ),
    "use_small_vessel_masks_for_boundary_assignment": (
        (
            "Mask cleanup",
            (
                "small_vessel_min_component_volume_um3",
                "small_vessel_mask_min_overlap_fraction",
                "small_vessel_swap_minority_components",
                "small_vessel_swap_max_size_ratio",
                "small_vessel_swap_min_contact_fraction",
            ),
        ),
        (
            "Overlap cleanup",
            (
                "small_vessel_boundary_assignment_enable_overlap_cleanup",
                "small_vessel_boundary_assignment_fast_mode",
                "small_vessel_boundary_assignment_apply_overlap_cleanup_in_normal_mode",
                "small_vessel_overlap_parallel_workers",
            ),
        ),
        (
            "Mask continuity",
            (
                "small_vessel_mask_continuity_enable",
                "small_vessel_mask_continuity_allow_small_to_large",
                "small_vessel_mask_continuity_allow_small_to_small",
                "small_vessel_mask_continuity_enforce_cylinder_only",
                "small_vessel_mask_continuity_min_cylindricality",
                "small_vessel_mask_continuity_max_axis_angle_degrees",
                "small_vessel_mask_continuity_min_facing_cosine",
                "small_vessel_mask_continuity_max_radius_ratio",
                "small_vessel_mask_continuity_max_bridge_distance_microns",
                "small_vessel_mask_continuity_corridor_max_distance_microns",
                "small_vessel_mask_continuity_opposite_exclusion_distance_microns",
            ),
        ),
        (
            "Tangential redefinition",
            (
                "small_vessel_tangential_redefinition_enable",
                "small_vessel_tangential_redefinition_max_contact_distance_microns",
                "small_vessel_tangential_redefinition_touch_distance_microns",
                "small_vessel_tangential_redefinition_tangency_cosine_max",
                "small_vessel_tangential_redefinition_margin",
                "small_vessel_tangential_redefinition_min_contact_fraction",
                "small_vessel_tangential_redefinition_parallel_workers",
                "small_vessel_sandwiched_reassignment_enable",
                "small_vessel_sandwiched_max_gap_microns",
                "small_vessel_sandwiched_min_facing_cosine",
                "small_vessel_sandwiched_max_axis_angle_degrees",
            ),
        ),
        (
            "Fallback",
            (
                "small_vessel_boundary_fallback_to_hop_distance",
                "small_vessel_boundary_fallback_hop_distance",
            ),
        ),
        (
            "Performance and diagnostics",
            (
                "use_gpu_mask_continuity_acceleration",
                "write_small_vessel_boundary_labelling_3d_html",
            ),
        ),
    ),
    "use_fwhm_edge_diameters": (
        (
            "Profile sampling",
            (
                "fwhm_transverse_profile_step_um",
                "fwhm_transverse_half_extent_um",
                "fwhm_min_total_extent_multiplier",
                "fwhm_max_transverse_widen_passes",
                "fwhm_transverse_sampling_mode",
                "fwhm_warn_out_of_plane_tangent_fraction",
                "fwhm_diameter_guess_um",
                "fwhm_diameter_guess_edge_attribute",
            ),
        ),
        (
            "Profile fit",
            (
                "fwhm_profile_model",
                "fwhm_profile_baseline_mode",
                "fwhm_profile_baseline_wing_fraction",
                "fwhm_constrain_fitted_baseline",
                "fwhm_baseline_constraint_half_width_ptp",
                "fwhm_min_diameter_pixels",
            ),
        ),
        (
            "Sample rejection",
            (
                "fwhm_clip_profile_to_single_vessel",
                "fwhm_clip_min_drop_fraction_of_center",
                "fwhm_clip_re_rise_fraction_of_center",
                "fwhm_reject_samples_with_center_offset",
                "fwhm_max_fit_center_offset_um",
                "fwhm_max_fit_center_offset_fraction_of_diameter",
                "fwhm_reject_samples_with_low_fit_r2",
                "fwhm_min_fit_r2",
                "fwhm_reject_samples_with_plateau_shape",
                "fwhm_max_plateau_shape_ratio",
                "fwhm_stop_at_other_vessels_in_mask",
            ),
        ),
        (
            "Junctions and neighbouring vessels",
            (
                "fwhm_branch_endpoint_exclusion_um",
                "fwhm_junction_proximity_exclusion_um",
                "fwhm_allow_junction_crossing",
                "fwhm_allow_crossing_other_edges",
                "fwhm_enforce_same_edge_locality",
                "fwhm_same_edge_arc_window_um",
                "fwhm_same_edge_arc_window_multiplier",
                "fwhm_same_edge_arc_window_min_um",
                "fwhm_cap_half_extent_by_nonlocal_same_edge_distance",
                "fwhm_nonlocal_same_edge_arc_separation_um",
                "fwhm_nonlocal_same_edge_half_extent_factor",
            ),
        ),
        ("Per-vessel width", ("fwhm_edge_diameter_aggregation",)),
    ),
    "do_fwhm_measurement": (
        (
            "Checks on the widths",
            (
                "fwhm_decoy_check",
                "fwhm_decoy_check_sample_size",
                "fwhm_demote_flagged_edges",
            ),
        ),
        ("Blur", ("fwhm_fix_blur_to_image_psf",)),
    ),
    "statistics_network_analysis": (
        (
            "Routes between inlets and outlets",
            (
                "statistics_network_robustness",
                "statistics_bottlenecks",
                "statistics_shunts",
                "statistics_shunt_max_route_fraction",
                "statistics_current_flow",
                "statistics_perfusion_territories",
                "statistics_transit_time",
            ),
        ),
        (
            "Occlusion (slow; off by default)",
            (
                "statistics_occlusion_impact",
                "statistics_occlusion_hypoperfusion_fraction",
                "statistics_occlusion_curves",
                "statistics_occlusion_curve_max_fraction",
            ),
        ),
        (
            "Network structure",
            (
                "statistics_loop_hierarchy",
                "statistics_strahler",
                "statistics_algebraic_connectivity",
                "statistics_cyclomatic_number",
                "statistics_degree_assortativity",
                "statistics_rich_club",
                "statistics_k_core",
                "statistics_flow_hierarchy",
                "statistics_path_efficiency",
                "statistics_community",
                "statistics_betweenness",
            ),
        ),
    ),
}


def section_slug(section: str) -> str:
    """`Connectivity/Network Analysis` -> `connectivity_network_analysis`."""
    return section.lower().replace(" ", "_").replace("/", "_")


def _ordered(names: Sequence[str]) -> list[str]:
    """*names* with each :data:`MOVE_BEFORE` row moved in front of its target."""
    ordered = list(names)
    for name, before in MOVE_BEFORE.items():
        if name in ordered and before in ordered:
            ordered.remove(name)
            ordered.insert(ordered.index(before), name)
    return ordered


def _prerequisites(schema: Schema, name: str) -> tuple[str, ...]:
    return tuple(prerequisite_name(rule) for rule in schema[name].requires)


def _closure(schema: Schema, name: str) -> set[str]:
    """Every setting *name* needs, directly or through another."""
    seen: set[str] = set()
    stack = list(_prerequisites(schema, name))
    while stack:
        current = stack.pop()
        if current in seen or current not in schema:
            continue
        seen.add(current)
        stack.extend(_prerequisites(schema, current))
    return seen


def _depth(schema: Schema, name: str, within: set[str], _seen=None) -> int:
    """How many rows of *within* stand above *name* in its ``requires`` chain."""
    seen = set() if _seen is None else _seen
    if name in seen:
        return 0
    seen = seen | {name}
    parents = [p for p in _prerequisites(schema, name) if p in within]
    if not parents:
        return 0
    return 1 + max(_depth(schema, p, within, seen) for p in parents)


def _anchor(schema: Schema, name: str, ordinary: set[str], advanced: set[str], order: dict) -> str | None:
    """The ordinary row *name*'s button sits under, or None for its box's own.

    An advanced parent takes precedence over any ordinary one, so a child is
    always behind the same button as the toggle it needs; among ordinary
    parents the deepest wins, and the later row breaks a tie.
    """
    override = ANCHOR_OVERRIDES.get(name)
    if override is not None and override in ordinary:
        return override
    direct = [p for p in _prerequisites(schema, name) if p in ordinary or p in advanced]
    advanced_parents = [p for p in direct if p in advanced and p != name]
    if advanced_parents:
        anchors = [
            _anchor(schema, p, ordinary, advanced, order) for p in advanced_parents
        ]
        anchors = [a for a in anchors if a is not None]
        if not anchors:
            return None
        return max(anchors, key=lambda a: (_depth(schema, a, ordinary), order[a]))
    candidates = [p for p in _closure(schema, name) if p in ordinary]
    if not candidates:
        return None
    return max(candidates, key=lambda a: (_depth(schema, a, ordinary), order[a]))


def _claim_boxes(
    stage_call: str | None, names: Sequence[str], schema: Schema
) -> list[tuple[str, str | None, list[str]]]:
    """``(key, title, claimed rows)`` per box, before advanced rows move out."""
    specs = TAB_BOXES.get(stage_call or "")
    if specs is None:
        ordinary = [n for n in names if not schema[n].advanced]
        runs: list[tuple[str, list[str]]] = []
        for name in ordinary:
            section = schema[name].section
            if runs and runs[-1][0] == section:
                runs[-1][1].append(name)
            else:
                runs.append((section, [name]))
        boxes: list[tuple[str, str | None, list[str]]] = []
        seen_keys: set[str] = set()
        for index, (section, rows) in enumerate(runs):
            key = section_slug(section)
            while key in seen_keys:
                key += "_"
            seen_keys.add(key)
            title = None if index == 0 else section_box_title(stage_call or "", section)
            boxes.append((key, title, list(rows)))
        # An advanced row joins the box of its own section when there is one,
        # and the first box when its section has no ordinary rows here.
        box_of_section: dict[str, int] = {}
        for index, (section, _rows) in enumerate(runs):
            box_of_section.setdefault(section, index)
        for name in names:
            if schema[name].advanced:
                index = box_of_section.get(schema[name].section, 0)
                if boxes:
                    boxes[index][2].append(name)
                else:
                    boxes.append((section_slug(schema[name].section), None, [name]))
        return boxes

    claimed: dict[str, int] = {}
    present = set(names)
    for index, spec in enumerate(specs):
        for name in spec.names:
            if name in present:
                claimed.setdefault(name, index)
    for index, spec in enumerate(specs):
        if spec.subtree_of is None:
            continue
        for name in names:
            if name not in claimed and (
                name == spec.subtree_of or spec.subtree_of in _closure(schema, name)
            ):
                claimed[name] = index
    rest = next((i for i, spec in enumerate(specs) if spec.rest), 0)
    for name in names:
        claimed.setdefault(name, rest)
    boxes = []
    for index, spec in enumerate(specs):
        listed = [n for n in spec.names if n in present and claimed[n] == index]
        others = [n for n in names if claimed[n] == index and n not in listed]
        boxes.append((spec.key, spec.title, listed + others))
    return boxes


def _groups(key: str, names: Sequence[str]) -> tuple[tuple[str | None, tuple[str, ...]], ...]:
    grouped: list[tuple[str, tuple[str, ...]]] = []
    placed: set[str] = set()
    for heading, members in ADVANCED_GROUPS.get(key, ()):
        chosen = tuple(n for n in names if n in members and n not in placed)
        if chosen:
            grouped.append((heading, chosen))
            placed.update(chosen)
    loose = tuple(n for n in names if n not in placed)
    runs: list[tuple[str | None, tuple[str, ...]]] = []
    if loose:
        runs.append((None, loose))
    runs.extend(grouped)
    return tuple(runs)


def _disclosure(
    tab_key: str, title_key: str, anchor: str | None, level: int, names: Sequence[str]
) -> Disclosure:
    groups = _groups(title_key, names)
    return Disclosure(
        key=f"{tab_key}:{title_key}",
        anchor=anchor,
        title=ADVANCED_TITLES.get(title_key, DEFAULT_ADVANCED_TITLE),
        level=level,
        names=tuple(name for _heading, members in groups for name in members),
        groups=groups,
    )


def layout_for(stage_call: str | None, names: Sequence[str], schema: Schema) -> TabLayout:
    """How the rows *names* are laid out on the tab of *stage_call*.

    *names* is what the tab draws as ordinary form rows, in tab order: rows
    the panel places some other way (the shared ilastik block, a role page,
    a perturbation entry's own editors) are left out by the caller.
    """
    names = _ordered([n for n in names if n in schema])
    order = {name: index for index, name in enumerate(names)}
    tab_key = stage_call or "tab"
    boxes: list[Box] = []
    for key, title, members in _claim_boxes(stage_call, names, schema):
        if stage_call not in TAB_BOXES:
            members = sorted(members, key=lambda n: order[n])
        advanced = {n for n in members if schema[n].advanced}
        ordinary_rows = [n for n in members if n not in advanced]
        ordinary = set(ordinary_rows)
        # anchor -> its advanced rows, in tab order.
        behind: dict[str | None, list[str]] = {}
        for name in sorted(advanced, key=lambda n: order[n]):
            anchor = _anchor(schema, name, ordinary, advanced, order)
            behind.setdefault(anchor, []).append(name)

        # Where each anchored button goes: after the anchor's last ordinary
        # descendant in this box.
        after: dict[int, list[str]] = {}
        for anchor in (a for a in behind if a is not None):
            position = ordinary_rows.index(anchor)
            for index, row in enumerate(ordinary_rows):
                if index > position and anchor in _closure(schema, row):
                    position = index
            after.setdefault(position, []).append(anchor)

        items: list[Any] = []
        for index, row in enumerate(ordinary_rows):
            items.append(row)
            anchors = sorted(
                after.get(index, ()),
                key=lambda a: (-_depth(schema, a, ordinary), -order[a]),
            )
            for anchor in anchors:
                items.append(
                    _disclosure(
                        tab_key,
                        anchor,
                        anchor,
                        _depth(schema, anchor, ordinary) + 1,
                        behind[anchor],
                    )
                )
        if None in behind:
            items.append(_disclosure(tab_key, f"box:{key}", None, 0, behind[None]))
        boxes.append(Box(key=key, title=title, items=tuple(items)))
    return TabLayout(boxes=tuple(box for box in boxes if box.items))


def role_nodes_disclosure(role: str, names: Sequence[str]) -> Disclosure:
    """The button at the foot of a boundary role's page: the node IDs a run fills in."""
    return Disclosure(
        key=f"assign_boundaries:role:{role}",
        anchor=None,
        title="Node IDs filled in by the run",
        level=0,
        names=tuple(names),
        groups=((None, tuple(names)),) if names else (),
    )


def shared_ilastik_disclosure(names: Sequence[str]) -> Disclosure:
    """The button under the shared ilastik rows, which move between Input and Boundaries."""
    return Disclosure(
        key="shared:ilastik",
        anchor=None,
        title="Advanced ilastik settings",
        level=1,
        names=tuple(names),
        groups=((None, tuple(names)),) if names else (),
    )


def entry_disclosure(index: int, names: Sequence[str]) -> Disclosure:
    """The button inside one perturbation entry: that type's advanced options."""
    return Disclosure(
        key=f"run_perturbations:entry:{index}",
        anchor=None,
        title="Advanced options",
        level=0,
        names=tuple(names),
        groups=((None, tuple(names)),) if names else (),
    )


def _same(setting, value: Any, reference: Any) -> bool:
    """Whether *value* and *reference* are the same setting value."""
    try:
        left = setting.coerce(value)
        right = setting.coerce(reference)
    except ConfigError:
        return value == reference
    if isinstance(left, Path) and isinstance(right, Path):
        if left == right:
            return True
        try:
            return left.resolve() == right.resolve()
        except OSError:
            return False
    return left == right


def changed_settings(
    names: Iterable[str],
    schema: Schema,
    values: Mapping[str, Any],
    baseline: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Which of *names* hold something other than where the panel started.

    A row counts only while its prerequisites hold: a changed value nothing
    reads is not worth pointing at. *baseline* overrides the schema default
    for rows the panel starts somewhere else (Produce IDE plots is off in
    napari, on in a config file).
    """
    baseline = baseline or {}
    changed = []
    for name in names:
        if name not in schema or name not in values:
            continue
        setting = schema[name]
        if not is_active(setting, values):
            continue
        reference = baseline.get(name, setting.default)
        if not _same(setting, values[name], reference):
            changed.append(name)
    return tuple(changed)
