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

#: What the tool insists on -- deliberately not the experiment.
#:
#: Which stages run, how the skeleton is cleaned, where the boundaries are and
#: what pressures drive it all come from the config file the checkout ships, so
#: comparing two branches compares what those branches actually do. Pinning
#: that here meant the tool ran something neither branch would: while
#: `examples/resistance_pipeline_config.yaml` could not run at all -- it named
#: an image that is not in the repository and gave no outlet nodes -- this file
#: replaced every setting that would have failed, so nothing said so.
#:
#: What is left is what the comparison needs mechanically, not scientifically.
#: Statistics, because the report is largely a diff of the statistics CSVs and
#: there is nothing to read without them. The display settings, because a
#: comparison runs unattended and the shipped config asks for a browser tab
#: (`show_plots_in_ide`) and a window that waits to be closed
#: (`hold_ide_plots_open`).
REQUIRED_SETTINGS: dict[str, Any] = {
    "statistics": True,
    "show_plots_in_ide": False,
    "hold_ide_plots_open": False,
    "ide_plot_mode": "none",
    "interactive_plots": False,
}

#: Boxes that select something on the 48-voxel test fixture, for `--smoke`.
#: These prove the plumbing, not the science.
SMOKE_BOUNDARY_SETTINGS: dict[str, Any] = {
    "inlet_node_volumes": [[[0, 0, 0], [47, 47, 20]]],
    "outlet_node_volumes": [[[0, 0, 28], [47, 47, 47]]],
}

#: Settings applied where a branch has them, and reported (not fatal) where it
#: does not.
#:
#: Empty, and kept as a place to put one. What was here switched optional
#: stages off -- ilastik, the vessel masks, the pericyte models, the FWHM
#: diameters -- which is the config's business now; the shipped one already
#: leaves every one of them off, and a config that turns one on is asking for
#: it to run. The display settings that were also here moved to
#: REQUIRED_SETTINGS, because an unattended comparison depends on them.
BEST_EFFORT_SETTINGS: dict[str, Any] = {}

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

    Everything not named here comes from the checkout's own config, resolved
    once from the current side and pinned onto both, so the two branches run
    the same experiment and it is the experiment the config describes.

    ``boundary_settings`` is for `--smoke`, which runs a 48-voxel fixture the
    config's boxes would select nothing in. Left out, the config's boundaries
    stand.

    ``voxel_size_xyz`` of ``None`` leaves the voxel size to the image metadata;
    otherwise it is pinned on both sides, because every length in the report
    scales with it.
    """
    required = dict(REQUIRED_SETTINGS)
    required.update(boundary_settings or {})
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
