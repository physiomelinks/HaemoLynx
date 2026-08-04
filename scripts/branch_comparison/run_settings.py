"""The settings both sides of a comparison run with.

Split out from the runner so the values a reviewer needs to trust -- the ones
that define the experiment -- sit in one readable place, and so the runner can
state exactly which of them a given branch could not accept.
"""
from __future__ import annotations

from typing import Any

#: The dataset the tool was built around, relative to a checkout root.
DEFAULT_IMAGE_RELATIVE = "examples/images/Nerve_capillaries.tif"

#: Physical voxel size of that dataset, in (x, y, z) microns.
DEFAULT_VOXEL_SIZE_XYZ = (1.167, 1.167, 2.029)

#: Settings that define the comparison. A side that cannot apply one of these
#: would be running a different experiment, so the tool refuses to report.
REQUIRED_SETTINGS: dict[str, Any] = {
    "do_skeletonize": True,
    "do_graph_building": True,
    "run_haemodynamics": True,
    "do_equiv_resistance_calculation": True,
    "skeleton_closing_radius": 1,
    "skeleton_bridge_gap_size": 1,
    "skeleton_min_branch_length": 3,
    "skeleton_max_bridge_distance": 2,
    "skeleton_component_connectivity": 3,
    "skeleton_min_component_percent": 1.0,
    "starting_node_selection_method": "volume",
    "output_node_selection_method": "volume",
    "starting_nodes": [],
    "output_nodes": [],
    "input_p_bc": 1000.0,
    "output_p_bc": 500.0,
    "min_stub_length": 3.0,
    "statistics": True,
    "statistics_mode": "fast",
    "visualize_results": True,
    "show_plots_in_ide": False,
    "visualize_vtk": False,
    "final_render_mode": "2d",
    "verbose_logging": False,
}

#: Inlet and outlet selection boxes for the nerve dataset, as
#: ``((min corner), (max corner))`` in physical (z, y, x) MICRONS -- not voxel
#: indices. Getting that wrong selects no boundary nodes at all.
NERVE_BOUNDARY_SETTINGS: dict[str, Any] = {
    "starting_node_volumes": [[[0, 0, 0], [600, 340, 700]]],
    "output_node_volumes": [[[0, 1308, 0], [600, 1730, 700]]],
}

#: Boxes that select something on the 48-voxel test fixture, for `--smoke`.
#: These prove the plumbing, not the science.
SMOKE_BOUNDARY_SETTINGS: dict[str, Any] = {
    "starting_node_volumes": [[[0, 0, 0], [47, 47, 20]]],
    "output_node_volumes": [[[0, 0, 28], [47, 47, 47]]],
}

#: Settings applied where a branch has them, and reported (not fatal) where it
#: does not: either they keep the run non-interactive, or they switch off an
#: optional stage that only newer branches know about.
BEST_EFFORT_SETTINGS: dict[str, Any] = {
    "use_volume_boxes": True,
    "interactive_plots": False,
    "hold_ide_plots_open": False,
    "ide_plot_mode": "none",
    "vtk_export": True,
    "automated_vessel_assignment": False,
    "use_large_vessel_masks": False,
    "use_small_vessel_masks_for_boundary_assignment": False,
    "use_ilastik_segmentation": False,
    "measurement_3d_to_cell_mask": False,
    "use_fwhm_edge_diameters": False,
    "do_pericyte_construction": False,
    "use_pericyte_mask_constriction": False,
    "use_probabilistic_pericyte_constriction": False,
    "run_pericyte_resistance_comparison": False,
    "strict_branch_order_assignment": False,
}

#: Names the same setting has had. The first alias a branch accepts wins.
SETTING_ALIASES: dict[str, tuple[str, ...]] = {
    "input_path": ("input_path", "image_path"),
    "image_axis_order": ("image_axis_order", "axis_order"),
    "do_pericyte_construction": (
        "do_pericyte_construction",
        "do_pericyte_constriction",
    ),
}

#: Accepted by the settings-dict entry point even though it is derived rather
#: than declared in the schema.
DERIVED_SETTINGS = ("plot_dir",)

#: Never pinned from one side onto the other: they name this run's own paths,
#: which the tool sets per side.
PER_SIDE_SETTINGS = ("input_path", "plot_dir", "base_plot_dir", "vtk_output_prefix")


def aliases_for(name: str) -> tuple[str, ...]:
    """Every name a branch might know this setting by, best first."""
    return SETTING_ALIASES.get(name, (name,))


def build_settings(
    *,
    image_path: str,
    plot_dir: str,
    vtk_output_prefix: str,
    axis_order: str = "zyx",
    voxel_size_xyz: tuple[float, float, float] | None = DEFAULT_VOXEL_SIZE_XYZ,
    boundary_settings: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], set[str]]:
    """The full settings dict for one comparison, and which names are required.

    ``voxel_size_xyz`` of ``None`` leaves the voxel size to the image metadata;
    otherwise it is pinned on both sides, because every length in the report
    scales with it.
    """
    required = dict(REQUIRED_SETTINGS)
    required.update(boundary_settings or NERVE_BOUNDARY_SETTINGS)
    required["input_path"] = image_path
    required["image_axis_order"] = axis_order
    # Both sides must write where the tool looks, or there is nothing to read
    # back and the report would be empty rather than wrong-but-visible.
    required["plot_dir"] = plot_dir
    required["vtk_output_prefix"] = vtk_output_prefix
    if voxel_size_xyz is not None:
        required["voxel_size_override_xyz"] = list(voxel_size_xyz)
        required["voxel_size_policy"] = "override"
    # A setting the caller asked for by hand defines their experiment too.
    required.update(extra or {})

    settings: dict[str, Any] = dict(BEST_EFFORT_SETTINGS)
    settings["base_plot_dir"] = plot_dir
    settings.update(required)
    return settings, set(required)
