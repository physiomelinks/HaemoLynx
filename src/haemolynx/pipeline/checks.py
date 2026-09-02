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


def preflight(settings: Mapping[str, Any], schema: Schema) -> CheckReport:
    """Every pre-run check, printed as a checklist.

    Called for you by the examples before a run starts; call it yourself to
    check a configuration without running anything.
    """
    report = check_settings(schema, settings)
    report.extend(check_cached_artefacts(settings))
    report.extend(check_ilastik_executable(settings))
    report.extend(check_perturbations(settings, schema))
    report.print("Preflight")
    return report
