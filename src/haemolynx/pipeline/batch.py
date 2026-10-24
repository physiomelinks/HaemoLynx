"""One image's settings in a batch run, and what a batch cannot carry over.

A batch run (``PLAN_batch_run.md``) runs the panel's settings once per
segmented image -- or, with ilastik segmenting (:func:`batch_from_raw`), once
per raw image -- each matched to its raw image, masks and other files by name
(:mod:`haemolynx.io.batch_matching`). :func:`item_settings` is those settings
for one image: its own input files, and every output folder moved into
``{output root}/{image stem}``. :func:`check_batch` says which settings name
something in one image that would mean something else, or nothing, in the
next -- node IDs, a single pericyte or cell mask with no folder of them --
so the batch refuses them before anything runs.
:func:`batch_preflight` is the run's own preflight with
those checks added, per image. The run's own preflight includes
:func:`check_output_path_lengths`, for a batch's images and single runs alike:
Windows refuses a path past its limit while long paths are off, and a batch
names every file after its image twice (``{output}/{stem}/{stem}_...``).

Pure: settings in, settings and reports out.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from haemolynx.graph.boundaries import (
    BOUNDARY_ROLE_SETTINGS,
    boundary_role_configured,
    inlets_outlets_from_vessel_masks,
    network_is_synthetic,
    vessel_masks_in_use,
)
from haemolynx.io.vessel_masks import VESSEL_MASK_SETTINGS
from haemolynx.io.batch_matching import FILE_ROLES, BatchFolders, BatchMatch, FileRole, file_role
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

#: Single files of one image, read beside it: a batch matches each per image
#: from a folder of its own (:data:`~haemolynx.io.batch_matching.FILE_ROLES`),
#: and refuses one file for every image.
PER_IMAGE_FILES = tuple(role.setting for role in FILE_ROLES)

#: Settings nothing reads but a perturbation (an entry of a type reading them).
READ_BY_PERTURBATIONS_ONLY = frozenset({"capillary_block_mask_path"})


def batch_from_raw(settings: Mapping[str, Any]) -> bool:
    """Whether a batch on *settings* lists raw images for ilastik to segment
    (``use_ilastik_segmentation``) rather than segmented images. Its raw
    image is each image's own, so the switch is read whatever the panel's
    one raw image row holds."""
    return bool(settings.get("use_ilastik_segmentation")) and not network_is_synthetic(settings)


def _readers(schema: Schema, setting: str, settings: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Who reads *setting* in a run on *settings*, each with the value it
    reads: ``("", value)`` for the run itself, ``(name, value)`` for each
    perturbation whose type reads it (its own override, else the run's)."""
    readers: list[tuple[str, Any]] = []
    if setting not in READ_BY_PERTURBATIONS_ONLY and _in_effect(schema, setting, settings):
        readers.append(("", settings.get(setting)))
    if not settings.get("run_perturbations"):
        return readers
    from haemolynx.haemodynamics.perturbations import (
        perturbations_from_settings,
        settings_for_perturbation_type,
    )

    for spec in perturbations_from_settings(settings):
        if setting not in settings_for_perturbation_type(spec.type):
            continue
        merged = {**settings, **spec.applied_overrides(schema)}
        if _in_effect(schema, setting, merged):
            readers.append((spec.name, merged.get(setting)))
    return readers


def batch_file_roles(settings: Mapping[str, Any], schema: Schema) -> tuple[str, ...]:
    """The other files (:data:`~haemolynx.io.batch_matching.FILE_ROLES`, by
    key) a run on *settings* reads, so a batch matches them per image."""
    return tuple(role.key for role in FILE_ROLES if _readers(schema, role.setting, settings))


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
    """*base* for one image: its input files from *match* (a perturbation
    naming one of them reads the image's own), every output folder in the
    image's own folder, and no pause for hand edits. ilastik's segmentation
    of a raw image goes in the image's folder too."""
    settings = dict(base)
    settings.update(match.settings())
    files = {file_role(key).setting: path for key, path in match.files.items()}
    if files and isinstance(base.get("perturbations"), (list, tuple)):
        settings["perturbations"] = [
            _with_files(entry, files) for entry in base["perturbations"]
        ]
    folder = image_output_folder(output_root, match)
    prefix = base.get("vtk_output_prefix")
    name = Path(prefix).name if prefix not in (None, "") else DEFAULT_PREFIX_NAME
    settings["vtk_output_prefix"] = folder / (name or DEFAULT_PREFIX_NAME)
    # An unset folder (perturbation_output_dir) already follows the VTK output.
    for setting, subfolder in OUTPUT_FOLDERS.items():
        if base.get(setting) not in (None, "") or (
            setting == "ilastik_output_dir" and match.folders.from_raw
        ):
            settings[setting] = folder / subfolder
    settings["mid_run_postprocessing"] = False
    return settings


def _with_files(entry: Any, files: Mapping[str, Path | None]) -> Any:
    """A perturbation entry whose overrides name one of *files*' settings,
    naming the image's own file instead."""
    if not isinstance(entry, Mapping) or not isinstance(entry.get("overrides"), Mapping):
        return entry
    overrides = entry["overrides"]
    if not set(overrides) & set(files):
        return entry
    return {
        **entry,
        "overrides": {key: files[key] if key in files else value for key, value in overrides.items()},
    }


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


def _check_one_file(
    report: CheckReport, schema: Schema, role: FileRole, settings: Mapping[str, Any], folder: Path | None
) -> None:
    """*role*'s file in a batch: matched per image from *folder*, or refused
    where a run or a perturbation would read one file for every image."""
    readers = _readers(schema, role.setting, settings)
    if folder is not None:
        if not readers:
            report.add_warning(
                f"The batch has a folder of {role.label}s, {folder}, but nothing in these "
                f"settings reads {role.setting}: it is not used."
            )
        return
    for who, value in readers:
        if value in (None, ""):
            continue
        if not who:
            report.add_error(
                f"{role.setting} names one file, {Path(value).name}, which every image "
                f"in the batch would read. Clear it, turn off what reads it, or give the "
                f"batch a folder of {role.label}s."
            )
        else:
            report.add_error(
                f"perturbation '{who}' reads one {role.label}, {Path(value).name}, which "
                f"every image in the batch would read. Give the batch a folder of "
                f"{role.label}s, or remove it from the batch's perturbations."
            )


def check_batch(
    settings: Mapping[str, Any], schema: Schema, folders: BatchFolders | None = None
) -> CheckReport:
    """The settings a batch refuses (errors) or doubts (warnings), whatever
    the image: each names something in one image. *folders* are the batch's:
    a file read beside each image is refused unless they give a folder of
    them, and they must list the images the settings ask for (raw images
    while ilastik segments)."""
    report = CheckReport()
    if network_is_synthetic(settings):
        report.add_error("A batch runs segmented images: set Input back to a segmented image.")
        return report
    if folders is not None and folders.from_raw != batch_from_raw(settings):
        report.add_error(
            "use_ilastik_segmentation is on, so a batch's images are its raw folder's, "
            "for ilastik to segment, but these were matched in the segmented folder: "
            "match the folders again."
            if batch_from_raw(settings)
            else "These images were matched in the raw folder, for ilastik to segment, "
            "but use_ilastik_segmentation is off: match the folders again."
        )
    for role in FILE_ROLES:
        _check_one_file(
            report, schema, role, settings, folders.file_folder(role.key) if folders else None
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


#: Windows' longest file path and longest folder path, in characters, while
#: long paths are off (a folder keeps room for an 8.3 file name). Measured on
#: Windows 11, 2026-10-23: a 260-character file and a 248-character folder
#: are refused.
WINDOWS_MAX_FILE_PATH = 259
WINDOWS_MAX_FOLDER_PATH = 247


def path_length_limits() -> tuple[int, int] | None:
    """``(file, folder)``: the longest paths this system writes, or None when
    it has no such limit (not Windows, or Windows with long paths on)."""
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
        ) as key:
            enabled, _kind = winreg.QueryValueEx(key, "LongPathsEnabled")
    except OSError:
        enabled = 0
    return None if enabled else (WINDOWS_MAX_FILE_PATH, WINDOWS_MAX_FOLDER_PATH)


#: The run folder's files that are written only while a setting is on, by the
#: suffix after ``{stem}`` (or a perturbation's ``{name}``); every other file
#: the export dictionary lists there is written by every run.
WRITTEN_WHILE = {
    "_statistics.csv": "statistics",
    "_statistics_connected.csv": "statistics",
    "_branch_statistics.csv": "statistics",
    "_capillary_bed_joins.csv": "statistics",
    "_betweenness_communities_resistance.json": "statistics",
    "_betweenness_communities_edge_length.json": "statistics",
    "_betweenness_communities_edge_flow.json": "statistics",
    "_cell_to_vessel_3d_dual_distances.csv": "measurement_3d_to_cell_mask",
    "_cell_to_vessel_3d_dual_distances_summary.csv": "measurement_3d_to_cell_mask",
    "_tissue_volume.csv": "measure_tissue_volume",
    "_tissue_surface.vtp": "measure_tissue_volume",
}

#: Files a run writes only for a synthetic network (refused in a batch).
WRITTEN_FOR_A_SYNTHETIC_NETWORK = ("_template_comparison.csv",)

#: Files written by hand in the panel's Post processing tab, never by a run.
WRITTEN_BY_HAND = (
    "_connectivity.csv",
    "_loop_review.csv",
    "_{kind}_review.csv",
)

#: Files no batch run writes.
NOT_WRITTEN_BY_A_BATCH = WRITTEN_FOR_A_SYNTHETIC_NETWORK + WRITTEN_BY_HAND


def _documented_names(
    where: str, placeholder: str, settings: Mapping[str, Any]
) -> list[str]:
    """The file names :mod:`~haemolynx.pipeline.export_dictionary` gives for
    *where* that start with *placeholder* (``{stem}``, ``{name}``) and that a
    run on *settings* writes. A wildcard stands for each name in
    :data:`WRITTEN_WHILE` it matches."""
    from fnmatch import fnmatchcase

    from haemolynx.pipeline.export_dictionary import FILES

    synthetic = network_is_synthetic(settings)
    names: list[str] = []
    for entry in FILES:
        if entry.where != where:
            continue
        for name in entry.name.split(", "):
            if not name.startswith(f"{placeholder}_"):
                continue
            pattern = name[len(placeholder):]
            suffixes = (
                [suffix for suffix in WRITTEN_WHILE if fnmatchcase(suffix, pattern)]
                if "*" in pattern
                else [pattern]
            )
            for suffix in suffixes:
                if suffix in WRITTEN_BY_HAND:
                    continue
                if suffix in WRITTEN_FOR_A_SYNTHETIC_NETWORK and not synthetic:
                    continue
                gate = WRITTEN_WHILE.get(suffix)
                if gate is None or settings.get(gate):
                    names.append(placeholder + suffix)
    return names


def _ilastik_segmented_image(settings: Mapping[str, Any]) -> Path | None:
    """The segmented image ilastik makes of the run's raw image, which the
    run's outputs are then named after, or None when ilastik does not
    segment it."""
    from haemolynx.pipeline.stages import ilastik_segments, segmented_input_path

    if not ilastik_segments(settings) or settings.get("ilastik_output_dir") in (None, ""):
        return None
    return segmented_input_path(dict(settings))


def ilastik_output_paths(settings: Mapping[str, Any]) -> list[Path]:
    """The files ilastik's segmentations write in ``ilastik_output_dir``: the
    main image's while ilastik segments it, and each vessel-mask role's while
    ilastik makes its masks, each with the record of how it was made beside
    it (:func:`haemolynx.io.ilastik.sidecar_path`), and a vessel mask's
    chosen channels while ilastik reads them (:func:`~haemolynx.io.ilastik.channel_input_path`).
    Every vessel mask is segmented from the run's raw image, so a batch's
    image writes its own in its own folder."""
    from haemolynx.io.ilastik import channel_input_path, sidecar_path
    from haemolynx.io.vessel_masks import ilastik_vessel_mask_output_path

    folder = settings.get("ilastik_output_dir")
    if folder in (None, ""):
        return []
    suffix = str(settings.get("ilastik_output_suffix") or "")
    files: list[Path] = []
    main = _ilastik_segmented_image(settings)
    if main is not None:
        main = Path(os.path.abspath(main))
        files.extend((main, sidecar_path(main)))
    raw = settings.get("raw_image_path")
    for role, names in VESSEL_MASK_SETTINGS.items():
        if raw in (None, "") or not (
            settings.get(names["use_ilastik"]) and vessel_masks_in_use(settings, role)
        ):
            continue
        # With no venule classifier the arteriole one segments a global mask.
        classes = [("global", names["ilastik_arteriole_channels"])]
        if settings.get(names["ilastik_venule_classifier_path"]) not in (None, ""):
            classes = [
                ("arteriole", names["ilastik_arteriole_channels"]),
                ("venule", names["ilastik_venule_channels"]),
            ]
        for vessel_class, channels in classes:
            mask = Path(
                os.path.abspath(ilastik_vessel_mask_output_path(folder, raw, role, vessel_class, suffix))
            )
            files.extend((mask, sidecar_path(mask)))
            if settings.get(channels):
                files.append(channel_input_path(mask))
    return files


def output_paths(settings: Mapping[str, Any]) -> tuple[list[Path], list[Path]]:
    """``(files, folders)`` a run writes, from its settings (a batch's from
    one image's :func:`item_settings`): each stage's checkpoint, its
    ``.haemorun`` (as the ``.part`` it is written to first), the ``{stem}_``
    files the export dictionary lists for the run's folder that these
    settings write, each perturbation's in its own folder, and ilastik's
    (:func:`ilastik_output_paths`). The stem is the one the run's outputs
    will carry: ilastik's output's while ilastik segments the image. The
    checkpoints and ``.haemorun`` are the panel's, counted for every run.
    Figures are left out: their names are not fixed."""
    from haemolynx.gui.run_snapshot import SUFFIX
    from haemolynx.gui.stage_checkpoints import checkpoint_pickle_path
    from haemolynx.haemodynamics.perturbations import (
        perturbation_folder_name,
        perturbation_output_dir,
    )
    from haemolynx.pipeline.export_dictionary import PERTURBATION_FOLDER, RUN_FOLDER
    from haemolynx.pipeline.naming import run_stem
    from haemolynx.pipeline.progress import STAGES

    segmented = _ilastik_segmented_image(settings)
    stem = run_stem(settings) if segmented is None else segmented.stem
    prefix = settings.get("vtk_output_prefix")
    if stem is None or prefix in (None, ""):
        return [], []
    folder = Path(os.path.abspath(Path(prefix).parent))
    run_names = _documented_names(RUN_FOLDER, "{stem}", settings)
    files = [checkpoint_pickle_path(folder, stem, stage.call) for stage in STAGES if stage.call]
    files.append(folder / f"{stem}{SUFFIX}.part")
    files.extend(folder / name.replace("{stem}", stem) for name in run_names)
    files.extend(ilastik_output_paths(settings))
    folders = [folder]
    for setting in OUTPUT_FOLDERS:
        if settings.get(setting) not in (None, ""):
            folders.append(Path(os.path.abspath(settings[setting])))
    if settings.get("run_perturbations"):
        root = Path(os.path.abspath(perturbation_output_dir(settings)))
        names = _documented_names(PERTURBATION_FOLDER, "{name}", settings)
        for entry in settings.get("perturbations") or ():
            if not isinstance(entry, Mapping) or not entry.get("name"):
                continue
            name = str(entry["name"])
            own = root / perturbation_folder_name(name, str(entry.get("type") or ""))
            folders.append(own)
            files.extend(own / item.replace("{name}", name) for item in names)
    return files, folders


def _longest(paths: Iterable[Path]) -> Path | None:
    return max(paths, key=lambda path: len(str(path)), default=None)


def check_output_path_lengths(
    settings: Mapping[str, Any], limits: tuple[int, int] | None = None
) -> CheckReport:
    """An error when a run's longest output file or folder
    (:func:`output_paths`; a batch's from one image's :func:`item_settings`)
    is longer than *limits* (``(file, folder)``, by default
    :func:`path_length_limits`): Windows would refuse it, some writers only
    log that, and the run would go on without the file, or fail at the end.
    Part of :func:`~haemolynx.pipeline.checks.preflight`."""
    report = CheckReport()
    limits = path_length_limits() if limits is None else limits
    if limits is None:
        return report
    files, folders = output_paths(settings)
    for kind, longest, limit in (
        ("file", _longest(files), limits[0]),
        ("folder", _longest(folders), limits[1]),
    ):
        if longest is not None and len(str(longest)) > limit:
            report.add_error(
                f"This run's outputs need a {len(str(longest))}-character {kind} path, "
                f"longer than Windows allows ({limit}) while long paths are off: "
                f"{longest}. Choose a shorter output folder, or turn on Windows long paths."
            )
    return report


def batch_preflight(
    base: Mapping[str, Any], match: BatchMatch, output_root: Path | str, schema: Schema
) -> CheckReport:
    """One image's preflight: the run's own (whether its output paths fit
    among them), the batch's and the match's. Printed by its caller, not
    here."""
    from haemolynx.pipeline.checks import preflight

    settings = item_settings(base, match, output_root)
    report = check_batch_item(match, base)
    report.extend(check_batch(base, schema, match.folders))
    report.extend(preflight(settings, schema, echo=False))
    return report


__all__ = [
    "DEFAULT_PREFIX_NAME",
    "OUTPUT_FOLDERS",
    "PER_IMAGE_FILES",
    "NOT_WRITTEN_BY_A_BATCH",
    "WINDOWS_MAX_FILE_PATH",
    "WINDOWS_MAX_FOLDER_PATH",
    "WRITTEN_BY_HAND",
    "WRITTEN_FOR_A_SYNTHETIC_NETWORK",
    "WRITTEN_WHILE",
    "READ_BY_PERTURBATIONS_ONLY",
    "batch_file_roles",
    "batch_from_raw",
    "batch_mask_sizes",
    "batch_preflight",
    "check_batch",
    "check_batch_item",
    "check_output_path_lengths",
    "default_output_root",
    "ilastik_output_paths",
    "image_output_folder",
    "item_settings",
    "output_paths",
    "path_length_limits",
]
