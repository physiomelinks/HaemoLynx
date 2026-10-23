"""One image's settings in a batch run, and what a batch cannot carry over.

A batch run (``PLAN_batch_run.md``) runs the panel's settings once per
segmented image, each matched to its raw image and masks by name
(:mod:`haemolynx.io.batch_matching`). :func:`item_settings` is those settings
for one image: its own input files, and every output folder moved into
``{output root}/{image stem}``. :func:`check_batch` says which settings name
something in one image that would mean something else, or nothing, in the
next -- node IDs, a single pericyte or cell mask -- so the batch refuses them
before anything runs. :func:`batch_preflight` is the run's own preflight with
those checks added, per image.

Pure: settings in, settings and reports out.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from haemolynx.graph.boundaries import (
    BOUNDARY_ROLE_SETTINGS,
    boundary_role_configured,
    inlets_outlets_from_vessel_masks,
    network_is_synthetic,
    vessel_masks_in_use,
)
from haemolynx.io.vessel_masks import VESSEL_MASK_SETTINGS
from haemolynx.io.batch_matching import BatchMatch
from haemolynx.parsers import Schema
from haemolynx.parsers.checks import CheckReport

#: Where a run's VTK output goes when the settings name none.
DEFAULT_PREFIX_NAME = "resistance_network"

#: Output folders moved into the image's own folder: setting -> subfolder.
OUTPUT_FOLDERS = {
    "base_plot_dir": "plots",
    "perturbation_output_dir": "perturbations",
    "sweep_output_dir": "sweeps",
    "ilastik_output_dir": "segmentations",
}

#: Single files of one image that a batch cannot match per image (v1).
PER_IMAGE_FILES = (
    "pericyte_mask_path",
    "cell_mask_path",
    "measurement_3d_vessel_mask_path",
    "tissue_raw_tiff_path",
)

#: ilastik switches whose input is one raw file, not the batch's.
ILASTIK_SWITCHES = (
    "use_ilastik_segmentation",
    "use_ilastik_large_vessel_segmentation",
    "use_ilastik_small_vessel_segmentation",
)


def batch_mask_sizes(settings: Mapping[str, Any]) -> tuple[str, ...]:
    """The mask sizes a batch matches: each role in use and read from files."""
    return tuple(
        size
        for size, names in VESSEL_MASK_SETTINGS.items()
        if vessel_masks_in_use(settings, size) and not settings.get(names["use_ilastik"])
    )


def default_output_root(settings: Mapping[str, Any]) -> Path | None:
    """The folder ``vtk_output_prefix`` names the parent of, or None."""
    prefix = settings.get("vtk_output_prefix")
    if prefix in (None, ""):
        return None
    parent = Path(prefix).parent
    return None if str(parent) in ("", ".") else parent


def image_output_folder(output_root: Path | str, match: BatchMatch) -> Path:
    """``{output root}/{segmented image stem}``: the image's own folder."""
    return Path(output_root) / match.stem


def item_settings(
    base: Mapping[str, Any], match: BatchMatch, output_root: Path | str
) -> dict[str, Any]:
    """*base* for one image: its input files from *match*, every output folder
    in the image's own folder, and no pause for hand edits."""
    settings = dict(base)
    settings.update(match.settings())
    folder = image_output_folder(output_root, match)
    prefix = base.get("vtk_output_prefix")
    name = Path(prefix).name if prefix not in (None, "") else DEFAULT_PREFIX_NAME
    settings["vtk_output_prefix"] = folder / (name or DEFAULT_PREFIX_NAME)
    # An unset folder (perturbation_output_dir) already follows the VTK output.
    for setting, subfolder in OUTPUT_FOLDERS.items():
        if base.get(setting) not in (None, ""):
            settings[setting] = folder / subfolder
    settings["mid_run_postprocessing"] = False
    return settings


def _in_effect(schema: Schema, name: str, settings: Mapping[str, Any]) -> bool:
    from haemolynx.gui.form import unmet_chain

    return name in schema and not unmet_chain(schema, name, settings)


def _role_read(schema: Schema, role: str, settings: Mapping[str, Any]) -> bool:
    """Whether a boundary role's own selection method is read in this run."""
    if not _in_effect(schema, BOUNDARY_ROLE_SETTINGS[role]["method"], settings):
        return False
    if role in ("inlet", "outlet"):
        return not inlets_outlets_from_vessel_masks(settings)
    if role in ("arteriole_boundary", "venule_boundary"):
        return not vessel_masks_in_use(settings, "small") and boundary_role_configured(
            settings, role
        )
    return True


def check_batch(settings: Mapping[str, Any], schema: Schema) -> CheckReport:
    """The settings a batch refuses (errors) or doubts (warnings), whatever
    the image: each names something in one image."""
    report = CheckReport()
    if network_is_synthetic(settings):
        report.add_error("A batch runs segmented images: set Input back to a segmented image.")
        return report
    for name in ILASTIK_SWITCHES:
        if settings.get(name) and name in schema:
            report.add_error(
                f"{name} is on: a batch runs images already segmented, and ilastik "
                "would segment the one file the Input tab names. Turn it off."
            )
    for name in PER_IMAGE_FILES:
        if settings.get(name) not in (None, "") and _in_effect(schema, name, settings):
            report.add_error(
                f"{name} names one file, {Path(settings[name]).name}, which every image "
                "in the batch would read. Clear it or turn off what reads it."
            )
    for role, names in BOUNDARY_ROLE_SETTINGS.items():
        if not _role_read(schema, role, settings):
            continue
        method = str(settings.get(names["method"]) or "").strip().lower()
        if method == "node_ids":
            report.add_error(
                f"{names['method']} is 'node_ids': node IDs name one build of one "
                "image's graph. Choose another method for a batch."
            )
        elif method in ("coordinates", "volume") and boundary_role_configured(settings, role):
            report.add_warning(
                f"{names['method']} is '{method}': the same positions are used in every "
                "image, which is right only if the images share one frame."
            )
    if settings.get("crop_on_import") and settings.get("crop_method") == "coordinates":
        report.add_warning(
            "crop_method is 'coordinates': the same box is cut from every image."
        )
    for name in ("do_skeletonize", "do_graph_building"):
        if settings.get(name, True) is False:
            report.add_warning(
                f"{name} is off: each image's saved result is looked for in its own "
                "output folder, and an image with none fails."
            )
    if settings.get("run_perturbations"):
        for entry in settings.get("perturbations") or ():
            if isinstance(entry, Mapping) and str(entry.get("type")) == "capillary_block_mask":
                report.add_error(
                    f"perturbation '{entry.get('name')}' blocks from one mask of blocks, "
                    "which every image in the batch would read. Remove it from the batch's "
                    "perturbations."
                )
    if settings.get("mid_run_postprocessing"):
        report.add_warning(
            "mid_run_postprocessing is on, but a batch cannot pause for hand edits: "
            "every image runs to the end."
        )
    return report


def check_batch_item(match: BatchMatch, settings: Mapping[str, Any]) -> CheckReport:
    """What the match itself says about one image (its problems and notes)."""
    report = CheckReport()
    for problem in match.problems:
        report.add_error(problem)
    for note in match.notes:
        report.add_warning(note)
    if match.raw is None and settings.get("measure_diameters_from_raw_image"):
        report.add_warning(
            "No raw image: this image's diameters are measured from the mask."
        )
    return report


def batch_preflight(
    base: Mapping[str, Any], match: BatchMatch, output_root: Path | str, schema: Schema
) -> CheckReport:
    """One image's preflight: the run's own, the batch's and the match's."""
    from haemolynx.pipeline.checks import preflight

    settings = item_settings(base, match, output_root)
    report = check_batch_item(match, base)
    report.extend(check_batch(base, schema))
    report.extend(preflight(settings, schema))
    return report


__all__ = [
    "DEFAULT_PREFIX_NAME",
    "OUTPUT_FOLDERS",
    "PER_IMAGE_FILES",
    "batch_mask_sizes",
    "batch_preflight",
    "check_batch",
    "check_batch_item",
    "default_output_root",
    "image_output_folder",
    "item_settings",
]
