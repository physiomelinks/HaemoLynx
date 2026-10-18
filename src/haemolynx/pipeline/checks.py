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

from haemolynx.graph.boundaries import (
    VESSEL_MASK_SWITCHES,
    assigns_large_vessel_branch_orders,
    boundary_role_configured,
    vessel_masks_in_use,
)
from haemolynx.graph.boundaries import network_is_synthetic
from haemolynx.haemodynamics.perturbations import (
    perturbation_problems,
    perturbations_from_settings,
)
from haemolynx.io.automated_vessel_assignment import (
    global_vessel_mask_roles,
    vessel_masks_are_global,
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
    # A synthetic network is generated every run: nothing is read back.
    if vtk_output_prefix is None or network_is_synthetic(settings):
        return report
    output_dir = Path(vtk_output_prefix).parent

    input_path = settings.get("input_path")
    if settings.get("use_ilastik_segmentation"):
        unsegmented = settings.get("raw_image_path")
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
    if network_is_synthetic(settings):
        return CheckReport()  # no image: nothing here applies
    report = CheckReport()
    needed = bool(settings.get("use_ilastik_segmentation")) or ilastik_segments_vessel_masks(
        settings
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
    if network_is_synthetic(settings):
        return CheckReport()  # no image: nothing here applies
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
    if vessel_masks_in_use(settings, "large"):
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


def check_manual_inlets_outlets_while_masks_choose(
    settings: Mapping[str, Any],
) -> CheckReport:
    """Warn when inlets/outlets are set by hand but the masks will choose them.

    The manual Inlet/Outlet methods have no schema ``requires`` (they apply
    unless the masks choose, which ``requires`` cannot express), so this is
    what says a hand-picked node would be ignored -- and which switch to turn.
    """
    from haemolynx.graph.boundaries import inlets_outlets_from_vessel_masks

    report = CheckReport()
    if not inlets_outlets_from_vessel_masks(settings):
        return report
    set_by_hand = [
        name
        for role in ("inlet", "outlet")
        for name in (f"{role}_node_ids", f"{role}_node_volumes")
        if settings.get(name)
    ]
    if set_by_hand:
        report.add_warning(
            f"{', '.join(set_by_hand)} set, but the large-vessel masks choose the "
            "inlets and outlets (automated_vessel_assignment and "
            "inlets_outlets_from_vessel_masks are on), so they will not be used. "
            "Fix: turn inlets_outlets_from_vessel_masks off to use them; the masks "
            "still drive the large-vessel treatment."
        )
    return report


def check_global_vessel_masks(settings: Mapping[str, Any]) -> CheckReport:
    """Say what a run with global vessel masks does instead, and warn of the
    settings it then leaves unread.

    A role given its arteriole/global mask with no venule mask
    (``io.vessel_masks_are_global``) has one mask of every vessel of its
    size: the vessels are classed Unknown_Large, Unknown_Small and
    Unknown_Cap rather than given branch orders, the masks cut nothing and
    choose no inlets or outlets (the Inlet and Outlet tabs do), and a role
    still given a pair is read as the one mask the two make.
    """
    report = CheckReport()
    roles = global_vessel_mask_roles(settings)
    if not roles:
        return report
    report.add_pass(
        "global vessel masks",
        f"the {' and '.join(roles)}-vessel mask{'s are' if len(roles) > 1 else ' is'} "
        "global (no venule mask entered): vessels are classed Unknown_Large, "
        "Unknown_Small and Unknown_Cap instead of branch orders, nothing is cut "
        "at the masks, and the inlets and outlets come from the Inlet and Outlet tabs",
    )
    for role in VESSEL_MASK_SWITCHES:
        if role not in roles and vessel_masks_in_use(settings, role):
            report.add_warning(
                f"The {role}-vessel arteriole and venule masks will be read as one "
                f"global {role}-vessel mask, because the "
                f"{' and '.join(roles)}-vessel mask is global: no vessel can be "
                "classed arteriole or venule. Fix: enter the venule mask in both "
                "roles to class them, or clear it in both."
            )
    unread = [
        name
        for name in ("assign_large_vessel_branch_orders", "strict_branch_order_assignment")
        if settings.get(name)
    ] + [
        f"{vessel}_boundary_selection_method"
        for vessel in ("arteriole", "venule")
        if boundary_role_configured(settings, f"{vessel}_boundary")
    ]
    if unread:
        report.add_warning(
            f"{', '.join(unread)} set, but with global vessel masks no branch "
            "orders are assigned, so nothing reads them."
        )
    return report


def check_large_vessel_cut_when_masks_enabled(
    settings: Mapping[str, Any],
) -> CheckReport:
    """Warn when large-mask assignment is on but the volume cut is off."""
    report = CheckReport()
    if not vessel_masks_in_use(settings, "large") or vessel_masks_are_global(settings):
        # Global masks cut nothing (check_global_vessel_masks says so).
        return report
    if bool(settings.get("cut_network_at_large_vessel_volumes")):
        return report
    if assigns_large_vessel_branch_orders(settings):
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
    """assign_large_vessel_branch_orders needs arteriole/venule boundaries.

    Only while it is in effect (``graph.assigns_large_vessel_branch_orders``:
    the large masks in use, the cut off). Left on under a gate that is off,
    the panel hides it and the run reads it as off, so it is no error: the
    schema's ``IneffectiveSettingWarning`` says it is set and unread.
    """
    report = CheckReport()
    if not assigns_large_vessel_branch_orders(settings):
        return report

    # The same rule as assign_boundaries: a list the current method does not
    # read (node IDs left behind after switching back to coordinates) is not
    # a source, and edge_percent does not name arteriole or venule boundaries.
    vessel_boundaries = boundary_role_configured(
        settings, "arteriole_boundary"
    ) or boundary_role_configured(settings, "venule_boundary")
    if not (vessel_masks_in_use(settings, "small") or vessel_boundaries):
        report.add_warning(
            "assign_large_vessel_branch_orders is on, but no small-vessel "
            "masks are in use and neither the arteriole nor the venule "
            "boundary method has the coordinates, volume boxes or node IDs "
            "it reads, so hierarchical Art*/Ven* labelling -- and the "
            "Large_Art/Large_Ven tier with it -- will never actually run."
        )
    return report


#: Each vessel-mask role's ilastik switch, which sits under that role's masks.
_ILASTIK_VESSEL_MASK_FLAGS = {
    "large": "use_ilastik_large_vessel_segmentation",
    "small": "use_ilastik_small_vessel_segmentation",
}


def ilastik_segments_vessel_masks(settings: Mapping[str, Any]) -> bool:
    """Whether ilastik segments a vessel mask this run: a role's ilastik switch
    on with that role's masks in use (``graph.vessel_masks_in_use``). A switch
    left on under an unticked mask switch is hidden in the panel and runs
    nothing, so neither preflight nor the citations count it."""
    return any(
        bool(settings.get(flag)) and vessel_masks_in_use(settings, role)
        for role, flag in _ILASTIK_VESSEL_MASK_FLAGS.items()
    )


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
    if network_is_synthetic(settings):
        return CheckReport()  # no image: nothing here applies
    report = CheckReport()
    choice = str(
        settings.get("skeleton_thick_vessel_restrict_to_mask", "off")
    ).strip().lower()
    if choice == "off":
        return report

    # The masks load only while in use (graph.vessel_masks_in_use), which
    # automated_vessel_assignment gates as well as the role's own switch.
    roles = {"large": ("large",), "small": ("small",), "both": ("large", "small")}.get(
        choice, ()
    )
    missing = [
        name
        for role in roles
        for name in ("automated_vessel_assignment", VESSEL_MASK_SWITCHES[role])
        if not bool(settings.get(name))
    ]
    missing = list(dict.fromkeys(missing))
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
    if network_is_synthetic(settings):
        return CheckReport()  # no image: nothing here applies
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


#: Each image a run reads one channel of: (its path setting, its channel
#: setting). One channel of the raw image is read by the network build (the
#: raw evidence round the mask, whenever the image is set), the diameters,
#: the check, the optimiser and the tissue volume, so it is checked whenever
#: it is set; ilastik alone reads the whole file, but the build follows it.
_CHANNEL_IMAGES = (("raw_image_path", "raw_image_channel"),)


def check_image_channels(settings: Mapping[str, Any]) -> CheckReport:
    """A multi-channel image needs a channel chosen, and one it has; a
    single-channel one needs none. Caught here rather than as the load fails
    partway through a run -- or, for the raw evidence, reads the first
    channel of a composite without saying so."""
    if network_is_synthetic(settings):
        return CheckReport()  # no image: nothing here applies
    report = CheckReport()
    for path_name, channel_name in _CHANNEL_IMAGES:
        path = settings.get(path_name)
        if not path or not Path(path).is_file():
            continue
        _check_image_channel(report, path_name, path, channel_name, settings.get(channel_name))
    return report


def check_raw_image(settings: Mapping[str, Any]) -> CheckReport:
    """The raw image is optional, so it is not ``must_exist`` for every run:
    ilastik segments it and the diameters are measured from it, and without
    one each is off whatever its switch says (``stages.ilastik_segments``,
    ``stages.measures_diameters_from_raw_image``) -- a warning, so a config
    that asks for them knows it is not getting them; an error only when
    ilastik was to make the segmented image and there is none to read
    instead. One that is set must be there: the network build reads it
    whenever it is (``preprocessing.raw_evidence``), so a missing one is an
    error for every run that names it."""
    if network_is_synthetic(settings):
        return CheckReport()  # no image: nothing here applies
    report = CheckReport()
    users = [
        name for name, on in (
            ("use_ilastik_segmentation", settings.get("use_ilastik_segmentation")),
            ("measure_diameters_from_raw_image",
             settings.get("run_haemodynamics", True) and settings.get("measure_diameters_from_raw_image")),
        ) if on
    ]
    path = settings.get("raw_image_path")
    if not path:
        if users:
            skipped = "they are" if len(users) > 1 else "it is"
            report.add_warning(
                f"{' and '.join(users)} read the raw image, but raw_image_path is not set, so "
                f"{skipped} skipped: the run goes on from the segmented image alone. To use "
                "the raw image, set raw_image_path (and raw_image_channel for a multi-channel "
                "file) on the Input tab."
            )
        if settings.get("use_ilastik_segmentation"):
            # input_path's own must_exist check stands down while ilastik is
            # on (it is what ilastik would produce); here it is what is read.
            exists, detail = resolve_existing_path(settings.get("input_path"))
            if exists:
                report.add_pass("input_path", detail)
            else:
                report.add_error(
                    "use_ilastik_segmentation is on without a raw image to segment, and the "
                    f"segmented image it would read instead is not there (input_path: {detail}). "
                    "Fix: set raw_image_path for ilastik, or input_path for a segmented image."
                )
        return report
    exists, detail = resolve_existing_path(path)
    if exists:
        report.add_pass("raw_image_path", detail)
        return report
    readers = "the network build reads it" + (
        f", as {' and '.join(users)} would" if users else ""
    )
    report.add_error(
        f"raw_image_path: {detail}; {readers}. Fix: point 'raw_image_path' at the raw "
        "image, or clear it."
    )
    return report


def _check_image_channel(
    report: CheckReport, path_name: str, path: Any, channel_name: str, channel: Any
) -> None:
    from haemolynx.io import tiff_channels

    channels = tiff_channels(path)
    listing = ", ".join(c.label for c in channels)
    if channels and channel is None:
        report.add_error(
            f"{path_name} has {len(channels)} channels ({listing}); choose one with "
            f"{channel_name}."
        )
    elif channel is not None and not channels:
        report.add_error(
            f"{path_name} is a single-channel image, but {channel_name} is {channel}; "
            "leave it unset."
        )
    elif channel is not None and int(channel) >= len(channels):
        report.add_error(
            f"{channel_name} is {channel} (C{int(channel) + 1}), but {path_name} has "
            f"{len(channels)} channels ({listing})."
        )


def tissue_raw_image(settings: Mapping[str, Any]) -> tuple[Any, Any, str, str]:
    """``(path, channel, path setting, channel setting)``: the raw image the
    tissue volume is measured in. ``tissue_raw_tiff_path``, else the raw image
    -- and its channel with it, unless ``tissue_raw_channel`` picks another
    channel of that same file (one where the tissue's own background shows)."""
    if settings.get("tissue_raw_tiff_path"):
        return (
            settings["tissue_raw_tiff_path"], settings.get("tissue_raw_channel"),
            "tissue_raw_tiff_path", "tissue_raw_channel",
        )
    if settings.get("tissue_raw_channel") is not None:
        return (
            settings.get("raw_image_path"), settings["tissue_raw_channel"],
            "raw_image_path", "tissue_raw_channel",
        )
    return (
        settings.get("raw_image_path"), settings.get("raw_image_channel"),
        "raw_image_path", "raw_image_channel",
    )


def check_tissue_volume_image(settings: Mapping[str, Any]) -> CheckReport:
    """``measure_tissue_volume`` needs a raw image -- its own, or the run's
    -- that is there, with a channel chosen if it has several."""
    if network_is_synthetic(settings):
        return CheckReport()  # no image: nothing here applies
    report = CheckReport()
    if not settings.get("measure_tissue_volume"):
        return report
    path, channel, path_name, channel_name = tissue_raw_image(settings)
    if not path:
        report.add_error(
            "measure_tissue_volume needs the raw image to find the tissue in: set "
            "raw_image_path, or tissue_raw_tiff_path for another image."
        )
        return report
    if not Path(path).is_file():
        # tissue_raw_tiff_path's own must_exist reports it, and check_raw_image
        # the raw image.
        return report
    if (path_name, channel_name) != ("raw_image_path", "raw_image_channel"):
        # The raw image's own pair is checked by check_image_channels.
        _check_image_channel(report, path_name, path, channel_name, channel)
    return report


def _tiff_header(path: Any, axis_order: str) -> tuple[tuple[int, ...], tuple, dict] | None:
    """``(shape, voxel size (x, y, z), its metadata status)`` from a TIFF's
    header, as the mask loaders read them; None for anything else."""
    import tifffile

    from haemolynx.io import read_voxel_size_xyz

    if not path or Path(path).suffix.lower() not in {".tif", ".tiff"} or not Path(path).is_file():
        return None
    try:
        with tifffile.TiffFile(path) as tif:
            shape = tuple(int(n) for n in tif.series[0].shape)
    except Exception:  # noqa: BLE001 - the loader reports an unreadable file itself
        return None
    if len(shape) == 4:
        # As io.load.simplify_to_3d: the first channel of the last axis.
        shape = shape[:3]
    found = read_voxel_size_xyz(path, axis_order)
    if found is None:
        return shape, (1.0, 1.0, 1.0), {"status": "missing"}
    return shape, found[0], found[1]


def check_cell_mask_measurement(settings: Mapping[str, Any]) -> CheckReport:
    """``measurement_3d_to_cell_mask`` runs last, in the export stage, so what
    would stop it there is checked here: from the TIFF headers, that the cell
    mask and a vessel mask of its own have the input image's shape and the
    run's voxel size -- a mask with no voxel size of its own taking the run's,
    as the measurement reads it. Whether each is there is check_settings'
    (both must_exist; the vessel mask may be unset)."""
    if network_is_synthetic(settings):
        return CheckReport()  # no image: nothing here applies
    import numpy as np

    from haemolynx.io import CANONICAL_AXIS_ORDER
    from haemolynx.io.automated_vessel_assignment import (
        _VOXEL_SIZE_RELATIVE_TOLERANCE,
        _resolve_mask_voxel_size_xyz,
    )
    from haemolynx.io.voxel_validation import resolve_voxel_size_xyz

    report = CheckReport()
    if not settings.get("measurement_3d_to_cell_mask") or not settings.get("cell_mask_path"):
        # An unset cell mask is reported by check_settings: it must_exist.
        return report
    axis_order = settings.get("image_axis_order") or CANONICAL_AXIS_ORDER
    image = _tiff_header(settings.get("input_path"), axis_order)
    if image is None:
        # Checked where it is read; ilastik may not have written it yet.
        return report
    shape, image_xyz, image_status = image
    override = settings.get("voxel_size_override_xyz")
    policy = settings.get("voxel_size_policy", "auto")
    try:
        run_xyz, _source = resolve_voxel_size_xyz(image_xyz, image_status, override, policy)
    except ValueError:
        # The run stops on this itself, before any measurement.
        run_xyz = None
    for name in ("cell_mask_path", "measurement_3d_vessel_mask_path"):
        mask = _tiff_header(settings.get(name), axis_order)
        if mask is None:
            continue
        mask_shape, mask_xyz, mask_status = mask
        problems = []
        if mask_shape != shape:
            problems.append(f"is {mask_shape} voxels where the input image is {shape}")
        if run_xyz is not None:
            try:
                resolved = _resolve_mask_voxel_size_xyz(
                    mask_xyz, mask_status, voxel_size_override_xyz=override,
                    voxel_size_policy=policy, fallback_voxel_size_xyz=run_xyz,
                )
            except ValueError as error:
                problems.append(f"has no voxel size the run can use ({error})")
            else:
                if not np.allclose(resolved, run_xyz, rtol=_VOXEL_SIZE_RELATIVE_TOLERANCE, atol=0.0):
                    problems.append(
                        f"has a voxel size (x, y, z) of {tuple(resolved)} um where the run's is "
                        f"{tuple(run_xyz)} um"
                    )
        if problems:
            report.add_error(
                f"{name} {' and '.join(problems)}, so the 3D distance measurement would stop "
                "at the end of the run. Fix: save the mask from the input image, at its shape "
                "and voxel size."
            )
        else:
            report.add_pass(name, "the input image's shape and voxel size")
    return report


def check_crop_on_import(settings: Mapping[str, Any]) -> CheckReport:
    """The crop box must be one that some image could hold: three whole
    numbers for its size or each corner, and its start not past its end.
    Whether it fits *this* image is only known once it is loaded."""
    if network_is_synthetic(settings):
        return CheckReport()  # no image: nothing here applies
    report = CheckReport()
    if not settings.get("crop_on_import"):
        return report
    from haemolynx.io.crop import crop_problems

    problems = crop_problems(settings)
    for problem in problems:
        report.add_error(f"Crop on import: {problem}")
    if not problems:
        report.add_pass("crop on import", str(settings.get("crop_method")))
    return report


def check_synthetic_template(settings: Mapping[str, Any]) -> CheckReport:
    """A synthetic run's template (``synthetic_template_path``): a
    ``.haemorun`` of a microscopy-image run with its statistics saved, read
    once (``haemolynx.synthetic.load_template`` remembers it) so the error
    names what the template lacks before any work starts; and what cannot
    run without an image."""
    report = CheckReport()
    if not network_is_synthetic(settings):
        return report
    path = settings.get("synthetic_template_path")
    if path in (None, ""):
        return report  # check_settings says it is required
    file = Path(path)
    if file.suffix.lower() != ".haemorun":
        report.add_error(
            f"synthetic_template_path: {file} is not a .haemorun file. Fix: save the "
            "template run from the panel (Save run) and point synthetic_template_path at it."
        )
        return report
    if not file.exists():
        return report  # check_settings reports a missing file
    from haemolynx.gui.run_snapshot import RunSnapshotError
    from haemolynx.synthetic import TemplateError, load_template

    try:
        template = load_template(file, settings.get("synthetic_template_source", "statistics"))
    except (RunSnapshotError, TemplateError) as error:
        report.add_error(f"synthetic_template_path: {error}")
        return report
    report.add_pass(
        "synthetic template",
        f"{file}: {template.vessel_count} vessels, {template.inlet_count} inlet(s), "
        f"{template.outlet_count} outlet(s), box {tuple(round(v, 1) for v in template.extent_um)} um",
    )
    for note in template.provenance.notes:
        report.add_warning(f"synthetic template: {note}")
    for name in ("measurement_3d_to_cell_mask", "measure_tissue_volume"):
        if settings.get(name):
            report.add_error(
                f"{name} needs the run's image, and a synthetic network has none. Fix: switch it off."
            )
    unread = [
        ("use_ilastik_segmentation", "nothing is segmented"),
        ("measure_diameters_from_raw_image", "the vessels keep the widths they were drawn with"),
        ("automated_vessel_assignment", "the generator chooses the inlets, outlets and boundary nodes"),
        ("use_thick_vessel_skeletonisation", "nothing is skeletonised"),
    ]
    for name, why in unread:
        if settings.get(name):
            report.add_warning(f"{name} is on but not read for a synthetic network: {why}.")
    if settings.get("skeleton_thick_vessel_restrict_to_mask", "off") not in ("off", None):
        report.add_warning("skeleton_thick_vessel_restrict_to_mask is not read for a synthetic network: nothing is skeletonised.")
    for name in ("do_skeletonize", "do_graph_building"):
        if settings.get(name, True) is False:
            report.add_warning(f"{name} is off but a synthetic network is drawn every run; nothing is read back.")
    return report


def preflight(settings: Mapping[str, Any], schema: Schema) -> CheckReport:
    """Every pre-run check, printed as a checklist.

    Called for you by the examples before a run starts; call it yourself to
    check a configuration without running anything.
    """
    # The raw image is optional: check_raw_image checks it when it is set. A
    # synthetic network reads no file but its template: a path left set under
    # a switch the Input tab hides (an ilastik classifier, a vessel mask) is
    # not demanded, since check_settings reads a path's own gate only.
    skip: tuple[str, ...] = ("raw_image_path",)
    if network_is_synthetic(settings):
        skip += tuple(
            setting.name for setting in schema if setting.must_exist and setting.name != "synthetic_template_path"
        )
    report = check_settings(schema, settings, skip=skip)
    report.extend(check_synthetic_template(settings))
    report.extend(check_cached_artefacts(settings))
    report.extend(check_ilastik_executable(settings))
    report.extend(check_input_is_not_a_large_vessel_mask(settings))
    report.extend(check_manual_inlets_outlets_while_masks_choose(settings))
    report.extend(check_global_vessel_masks(settings))
    report.extend(check_large_vessel_cut_when_masks_enabled(settings))
    report.extend(check_large_vessel_branch_order_mode_prerequisites(settings))
    report.extend(check_thick_vessel_restriction_prerequisites(settings))
    report.extend(check_perturbations(settings, schema))
    report.extend(check_segmentation_quality(settings))
    report.extend(check_raw_image(settings))
    report.extend(check_image_channels(settings))
    report.extend(check_tissue_volume_image(settings))
    report.extend(check_crop_on_import(settings))
    report.extend(check_cell_mask_measurement(settings))
    report.print("Preflight")
    return report
