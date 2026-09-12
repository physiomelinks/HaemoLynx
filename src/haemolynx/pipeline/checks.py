"""Pre-run checks that only make sense for this pipeline.

The schema states most of what a run needs — see
:mod:`haemolynx.parsers.checks`, which reads it. What is left here is knowledge
of how *this* pipeline behaves: the files it names for itself when a stage is
skipped, and the external tool it shells out to.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from haemolynx.haemodynamics.perturbations import (
    perturbation_problems,
    perturbations_from_settings,
)
from haemolynx.parsers import Schema
from haemolynx.parsers.checks import CheckReport, check_settings, resolve_existing_path

#: Stage toggle -> the artefact a previous run must have left behind, as a
#: format string over the run's image stem.
CACHED_STAGE_ARTEFACTS = {
    "do_skeletonize": "{stem}_skeleton.npy",
    "do_graph_building": "{stem}_graph.pkl",
}


def check_cached_artefacts(settings: Mapping[str, Any]) -> CheckReport:
    """When a stage is switched off, the file it would have written must exist."""
    report = CheckReport()
    vtk_output_prefix = settings.get("vtk_output_prefix")
    if vtk_output_prefix is None:
        return report
    output_dir = Path(vtk_output_prefix).parent

    input_path = settings.get("input_path")
    if settings.get("use_ilastik_segmentation"):
        unsegmented = settings.get("ilastik_unsegmented_image_path")
        stem = f"{Path(unsegmented).stem}_segmented" if unsegmented else "input"
    else:
        stem = Path(input_path).stem if input_path else "input"

    for toggle, artefact in CACHED_STAGE_ARTEFACTS.items():
        if settings.get(toggle, True):
            continue
        expected = output_dir / artefact.format(stem=stem)
        if expected.exists():
            report.add_pass(f"cached artefact for {toggle}", str(expected))
        else:
            report.add_error(
                f"{toggle} is off but {expected} is not there. Fix: turn "
                f"{toggle} back on, or run once with it on to produce the file."
            )
    return report


def check_ilastik_executable(settings: Mapping[str, Any]) -> CheckReport:
    """ilastik is a separate program; if a run needs it, it must be findable."""
    report = CheckReport()
    needed = any(
        settings.get(name)
        for name in (
            "use_ilastik_segmentation",
            "use_ilastik_large_vessel_segmentation",
            "use_ilastik_small_vessel_segmentation",
        )
    )
    if not needed:
        return report

    executable = settings.get("ilastik_executable")
    if not executable:
        report.add_error(
            "ilastik segmentation is on but 'ilastik_executable' is not set. "
            "Fix: set it to the ilastik program name or its full path."
        )
        return report

    found = shutil.which(str(executable))
    if found:
        report.add_pass("ilastik executable", found)
        return report
    exists, detail = resolve_existing_path(executable)
    if exists:
        report.add_pass("ilastik executable", detail)
    else:
        report.add_warning(
            f"ilastik executable '{executable}' was not found on PATH. It must "
            "resolve when the segmentation stage runs."
        )
    return report


#: Stem tokens that strongly suggest a large-vessel mask was pointed at
#: ``input_path``. Skeletonising that file yields a network that only exists
#: inside that mask — cut-at-large-vessel then has almost nothing exterior to keep.
_LARGE_VESSEL_INPUT_STEM_MARKERS = (
    "large_venule_mask",
    "large_arteriole_mask",
    "large_vein_mask",
    "large_artery_mask",
)


def _paths_refer_to_same_file(left: Path, right: Path) -> bool:
    """True when both paths exist and are the same file, else name equality."""
    try:
        if left.exists() and right.exists():
            return left.resolve(strict=False).samefile(right.resolve(strict=False))
    except OSError:
        pass
    return left.resolve(strict=False) == right.resolve(strict=False)


def check_input_is_not_a_large_vessel_mask(
    settings: Mapping[str, Any],
) -> CheckReport:
    """Warn when the main input is (or looks like) a large arteriole/venule mask.

    Cut-at-large-vessel removes geometry *inside* those masks. If ``input_path``
    is itself the venule (or arteriole) mask TIFF, the skeleton and graph are
    venule-shaped from the start: there is no capillary exterior to keep, so the
    viewer looks like "network only within the venule" even when cut polarity
    is correct and both masks are loaded.
    """
    report = CheckReport()
    if settings.get("use_ilastik_segmentation"):
        return report
    input_path = settings.get("input_path")
    if not input_path:
        return report

    input_p = Path(str(input_path))
    stem_lower = input_p.stem.lower()
    looks_like_mask_name = any(
        marker in stem_lower for marker in _LARGE_VESSEL_INPUT_STEM_MARKERS
    )

    matched_roles: list[str] = []
    if settings.get("use_large_vessel_masks"):
        for role, key in (
            ("arteriole", "large_arteriole_mask_path"),
            ("venule", "large_venule_mask_path"),
        ):
            mask_path = settings.get(key)
            if not mask_path:
                continue
            if _paths_refer_to_same_file(input_p, Path(str(mask_path))):
                matched_roles.append(role)

    if not matched_roles and not looks_like_mask_name:
        return report

    cut_on = bool(settings.get("cut_network_at_large_vessel_volumes"))
    if matched_roles:
        roles = " and ".join(matched_roles)
        detail = (
            f"input_path resolves to the same file as the large {roles} mask. "
            "The pipeline skeletonises that volume as the *main* network, so "
            "the skeleton/graph only exist inside that mask before any cut runs."
        )
    else:
        detail = (
            f"input_path stem '{input_p.stem}' looks like a large-vessel mask "
            "filename. If that file is only the large arteriole/venule volume, "
            "the skeleton will be confined to that volume from the start."
        )
    detail += (
        " Use the full vessel (or capillary) segmentation as input_path; keep "
        "large arteriole/venule masks only on large_*_mask_path."
    )
    if cut_on:
        detail += (
            " With cut_network_at_large_vessel_volumes on, almost every edge is "
            "interior and is removed — remaining stubs still sit at that mask."
        )
    report.add_warning(detail)
    return report


def check_large_vessel_cut_when_masks_enabled(
    settings: Mapping[str, Any],
) -> CheckReport:
    """Warn when large-mask assignment is on but the volume cut is off."""
    report = CheckReport()
    if not bool(settings.get("use_large_vessel_masks")):
        return report
    if not bool(settings.get("automated_vessel_assignment")):
        return report
    if bool(settings.get("cut_network_at_large_vessel_volumes")):
        return report
    if bool(settings.get("assign_large_vessel_branch_orders")):
        # Cut is deliberately off: large-vessel material stays in the
        # network as Large_Art/Large_Ven branch orders instead of being
        # cut away -- not the oversight this warning otherwise flags.
        return report
    report.add_warning(
        "use_large_vessel_masks and automated_vessel_assignment are on, but "
        "cut_network_at_large_vessel_volumes is off. Interior branches inside "
        "large arteriole/venule volumes will stay in the network and napari "
        "viewer unless you enable the cut."
    )
    return report


def check_large_vessel_branch_order_mode_prerequisites(
    settings: Mapping[str, Any],
) -> CheckReport:
    """assign_large_vessel_branch_orders needs its whole prerequisite chain.

    The schema's own ``requires`` tuple only drives GUI nesting and an
    ``IneffectiveSettingWarning`` — it does not block a raw settings dict
    (e.g. loaded from a hand-edited YAML) from having the flag on without
    its prerequisites. This is the run-time check that catches that.
    """
    report = CheckReport()
    if not bool(settings.get("assign_large_vessel_branch_orders")):
        return report

    missing = [
        name
        for name in (
            "use_large_vessel_masks",
            "automated_vessel_assignment",
        )
        if not bool(settings.get(name))
    ]
    if missing:
        report.add_error(
            "assign_large_vessel_branch_orders requires "
            f"{', '.join(missing)} to also be on."
        )
    if bool(settings.get("cut_network_at_large_vessel_volumes")):
        report.add_error(
            "assign_large_vessel_branch_orders requires "
            "cut_network_at_large_vessel_volumes=False -- large-vessel "
            "material must stay in the network to be tagged Large_Art/"
            "Large_Ven, not be cut away."
        )
    hierarchical_sources = (
        "use_small_vessel_masks_for_boundary_assignment",
        "arteriole_boundary_node_coordinates",
        "arteriole_boundary_node_volumes",
        "venule_boundary_node_coordinates",
        "venule_boundary_node_volumes",
    )
    if not any(bool(settings.get(name)) for name in hierarchical_sources):
        report.add_warning(
            "assign_large_vessel_branch_orders is on, but none of "
            f"{', '.join(hierarchical_sources)} is set, so hierarchical "
            "Art*/Ven* labelling -- and the Large_Art/Large_Ven tier with "
            "it -- will never actually run."
        )
    return report


def check_thick_vessel_restriction_prerequisites(
    settings: Mapping[str, Any],
) -> CheckReport:
    """skeleton_thick_vessel_restrict_to_mask needs the mask(s) it names.

    The schema's own ``requires`` only gates ``use_thick_vessel_skeletonisation``
    -- it cannot express "large needs use_large_vessel_masks, small needs
    use_small_vessel_masks_for_boundary_assignment, both needs both" since
    that depends on *which value* the choice holds, not just whether it is
    non-default. This is the run-time check that catches that.
    """
    report = CheckReport()
    choice = str(
        settings.get("skeleton_thick_vessel_restrict_to_mask", "off")
    ).strip().lower()
    if choice == "off":
        return report

    missing = []
    if choice in ("large", "both") and not bool(settings.get("use_large_vessel_masks")):
        missing.append("use_large_vessel_masks")
    if choice in ("small", "both") and not bool(
        settings.get("use_small_vessel_masks_for_boundary_assignment")
    ):
        missing.append("use_small_vessel_masks_for_boundary_assignment")
    if missing:
        report.add_error(
            f"skeleton_thick_vessel_restrict_to_mask={choice!r} requires "
            f"{', '.join(missing)} to also be on."
        )
    return report


def check_perturbations(settings: Mapping[str, Any], schema: Schema) -> CheckReport:
    """Every configured perturbation must name a type and settings that exist.

    A perturbation is a partial config the run applies to a finished network,
    so nothing validates it when the config file loads: an entry naming a
    setting that does not exist, or a path that is not there, would not be
    found until the re-solve, after the whole pipeline had run. Hence here.
    """
    report = CheckReport()
    specs = perturbations_from_settings(settings)
    if not specs:
        return report

    for message in perturbation_problems(settings, schema):
        report.add_error(message)

    for spec in specs:
        unused = spec.unused_overrides()
        if unused:
            report.add_warning(
                f"perturbation '{spec.name}' sets {list(unused)}, which a "
                f"{spec.type} perturbation does not read, so they will have no "
                "effect."
            )
        # A path named by a perturbation is read once that perturbation runs,
        # which is after the pipeline. Check it against the settings as they
        # will be then -- the overrides are what switch its feature on.
        overrides = spec.coerced_overrides(schema)
        paths = [
            name
            for name in overrides
            if schema[name].kind == "path" and schema[name].must_exist
        ]
        if not paths:
            continue
        # Not `schema.subset(paths)`: a path's prerequisite is a setting of its
        # own, and a schema missing it will not build at all.
        checked = check_settings(
            schema,
            {**settings, **overrides},
            skip=[name for name in schema.names if name not in set(paths)],
        )
        for message in checked.errors:
            report.add_error(f"perturbation '{spec.name}': {message}")
        for label, detail in checked.passed:
            report.add_pass(f"perturbation '{spec.name}' {label}", detail)

    if report.ok:
        report.add_pass(
            "perturbations",
            f"{len(specs)} configured: " + ", ".join(spec.name for spec in specs),
        )
    return report


def check_segmentation_quality(settings: Mapping[str, Any]) -> CheckReport:
    """Warn (never block) when the segmented input's own 0-10 quality score
    -- the same score "Check segmented image" reports -- is below a
    user-configured minimum.

    Opt-in via ``min_acceptable_segmentation_quality`` (default unset, which
    skips this entirely): unlike every other check here, this one has to
    load and score the actual image, real work this module otherwise never
    does, so it only runs for someone who has deliberately asked for the
    nudge in their normal automated workflow.
    """
    report = CheckReport()
    threshold = settings.get("min_acceptable_segmentation_quality")
    if threshold is None:
        return report

    from haemolynx.io import resolve_voxel_size_xyz, voxel_size_zyx_from_xyz
    from haemolynx.io.load import _to_binary_volume_for_skeletonization
    from haemolynx.pipeline.stages import load_volume_for_skeletonise, segment
    from haemolynx.preprocessing import score_segmented_mask
    from haemolynx.preprocessing.segmentation_quality import RESOLUTION_LIMITING_THRESHOLD

    local_settings = dict(settings)
    try:
        inputs = segment(local_settings)
        image, metadata_voxel_size, voxel_meta_status = load_volume_for_skeletonise(
            local_settings, inputs.input_format
        )
    except Exception as error:  # noqa: BLE001 - a missing/unreadable image is
        # already reported by check_settings/check_cached_artefacts; this
        # optional check must not itself crash preflight over it.
        report.add_warning(
            "min_acceptable_segmentation_quality is set, but the segmented "
            f"input could not be loaded to check it: {error}"
        )
        return report

    voxel_size_xyz, _source = resolve_voxel_size_xyz(
        metadata_voxel_size_xyz=metadata_voxel_size,
        metadata_status=voxel_meta_status,
        voxel_size_override_xyz=local_settings.get("voxel_size_override_xyz"),
        voxel_size_policy=local_settings.get("voxel_size_policy", "auto"),
    )
    mask = _to_binary_volume_for_skeletonization(image)
    voxel_size_zyx = voxel_size_zyx_from_xyz(tuple(float(v) for v in voxel_size_xyz))
    score = score_segmented_mask(
        mask,
        voxel_size_zyx=voxel_size_zyx,
        target_voxels_across_radius=local_settings.get("min_voxels_across_vessel_radius"),
        boundary_patch_allowance=local_settings.get("expected_boundary_vessel_count"),
    )
    threshold = float(threshold)
    if score.total >= threshold:
        report.add_pass("segmentation quality", f"{score.total:.1f}/10 >= {threshold:.1f}")
        return report

    # Same condition segmentation_quality's own report verdict uses -- one
    # source of truth for "is this a source-data problem or a
    # pipeline-fixable one", so preflight and "Check segmented image" never
    # disagree about which tier is responsible.
    if score.resolution < RESOLUTION_LIMITING_THRESHOLD:
        tier = "source data -- resegment or re-image; cleanup will not fix this"
    else:
        tier = "pipeline-fixable -- try the segmentation_cleanup_* settings"
    report.add_warning(
        f"Segmented input quality is {score.total:.1f}/10, below the configured "
        f"minimum of {threshold:.1f}. Likely limiting tier: {tier}."
    )
    return report


def preflight(settings: Mapping[str, Any], schema: Schema) -> CheckReport:
    """Every pre-run check, printed as a checklist.

    Called for you by the examples before a run starts; call it yourself to
    check a configuration without running anything.
    """
    report = check_settings(schema, settings)
    report.extend(check_cached_artefacts(settings))
    report.extend(check_ilastik_executable(settings))
    report.extend(check_input_is_not_a_large_vessel_mask(settings))
    report.extend(check_large_vessel_cut_when_masks_enabled(settings))
    report.extend(check_large_vessel_branch_order_mode_prerequisites(settings))
    report.extend(check_thick_vessel_restriction_prerequisites(settings))
    report.extend(check_perturbations(settings, schema))
    report.extend(check_segmentation_quality(settings))
    report.print("Preflight")
    return report
