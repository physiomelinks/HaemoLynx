"""Run a batch: the pipeline once per segmented image, a ``.haemorun`` each.

A batch (``PLAN_batch_run.md``) is the panel's settings run over every image a
:func:`~haemolynx.io.batch_matching.match_batch` found, each with its own
input files and its own output folder (:func:`haemolynx.pipeline.batch.item_settings`).
:func:`run_batch` runs them one after another, as the panel's Run pipeline
would, recording each stage's checkpoint as it goes so that every image's
``{output root}/{stem}/{stem}.haemorun`` opens with Load run as if it had been
run in the panel. Nothing is drawn: :class:`~haemolynx.gui.results.ResultLayers`
and :class:`~haemolynx.gui.stage_checkpoints.StageCheckpoints` need no viewer.

An image that fails is recorded and the batch goes on; a cancel stops the
image at its next stage and runs no more. ``{output root}/batch_summary.csv``
is rewritten after every image, so a batch that dies leaves it current, and
``batch_settings.yaml`` beside it holds the settings and folders the batch
ran with.

Once the images have run (or been skipped as done before), their statistics
are read back from their folders and written side by side beside the summary
(:func:`haemolynx.statistics.batch_comparison.write_batch_statistics`): every
image, each condition's summary and, given two conditions (phrases in the
images' file names), the two tested against each other.

Nothing here imports Qt: :func:`main` runs a batch from a command line
(:mod:`haemolynx.gui.batch_run_cli`).
"""
from __future__ import annotations

import csv
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from haemolynx.io.batch_matching import (
    FILE_ROLE_KEYS,
    FILE_ROLES,
    IMAGE_FOLDER_ROLES,
    BatchFolders,
    BatchMatch,
    file_role,
)
from haemolynx.parsers import Schema
from haemolynx.pipeline.batch import (
    batch_preflight,
    image_output_folder,
    item_settings,
)
from haemolynx.statistics.batch_comparison import (
    BatchConditions,
    BatchStatisticsFiles,
    write_batch_statistics,
)

logger = logging.getLogger(__name__)

SUMMARY_FILENAME = "batch_summary.csv"
SETTINGS_FILENAME = "batch_settings.yaml"

DONE = "done"
FAILED = "failed"
SKIPPED = "skipped (done before)"
CANCELLED = "cancelled"
NOT_RUN = "not run"
RUNNING = "running"

SUMMARY_COLUMNS = (
    "stem",
    "status",
    "condition",
    "segmented",
    "raw",
    "large_arteriole_mask",
    "large_venule_mask",
    "small_arteriole_mask",
    "small_venule_mask",
    *(role.setting[: -len("_path")] for role in FILE_ROLES),
    "haemorun",
    "nodes",
    "vessels",
    "wall_time_s",
    "peak_rss_bytes",
    "error",
)


def haemorun_path(output_root: Path | str, match: BatchMatch) -> Path:
    """``{output root}/{stem}/{stem}.haemorun``."""
    from haemolynx.gui.run_snapshot import SUFFIX

    return image_output_folder(output_root, match) / f"{match.stem}{SUFFIX}"


@dataclass
class BatchResult:
    """What one image of a batch came to."""

    match: BatchMatch
    status: str = NOT_RUN
    #: The condition the image's file name puts it in ("" for none).
    condition: str = ""
    haemorun: Path | None = None
    nodes: int | None = None
    vessels: int | None = None
    seconds: float | None = None
    peak_rss_bytes: int | None = None
    error: str = ""
    #: The failure's traceback (the summary CSV keeps only *error*).
    error_detail: str = ""

    def status_text(self) -> str:
        """Its status, and its error after a colon when it failed."""
        return f"{self.status}: {self.error}" if self.error else self.status

    def row(self) -> dict[str, Any]:
        settings = self.match.settings()

        def text(name: str) -> str:
            value = settings.get(name)
            return "" if value in (None, "") else str(value)

        return {
            "stem": self.match.stem,
            "status": self.status,
            "condition": self.condition,
            "segmented": text("input_path"),
            "raw": text("raw_image_path"),
            "large_arteriole_mask": text("large_arteriole_mask_path"),
            "large_venule_mask": text("large_venule_mask_path"),
            "small_arteriole_mask": text("small_arteriole_mask_path"),
            "small_venule_mask": text("small_venule_mask_path"),
            **{role.setting[: -len("_path")]: text(role.setting) for role in FILE_ROLES},
            "haemorun": "" if self.haemorun is None else str(self.haemorun),
            "nodes": "" if self.nodes is None else self.nodes,
            "vessels": "" if self.vessels is None else self.vessels,
            "wall_time_s": "" if self.seconds is None else round(self.seconds, 1),
            "peak_rss_bytes": "" if self.peak_rss_bytes is None else self.peak_rss_bytes,
            "error": self.error,
        }


@dataclass
class BatchSummary:
    """Every image of a batch, in the order run."""

    output_root: Path
    results: list[BatchResult] = field(default_factory=list)
    #: The batch's statistics, written once its images are done.
    statistics: BatchStatisticsFiles | None = None
    #: Whether it stopped after an image because it was asked to.
    stopped: bool = False

    @property
    def csv_path(self) -> Path:
        return self.output_root / SUMMARY_FILENAME

    def count(self, status: str) -> int:
        return sum(result.status == status for result in self.results)

    def message(self) -> str:
        parts = [f"{self.count(DONE)} done"]
        for status in (FAILED, SKIPPED, CANCELLED, NOT_RUN):
            if self.count(status):
                parts.append(f"{self.count(status)} {status}")
        message = f"Batch: {', '.join(parts)}. Summary: {self.csv_path}"
        if self.stopped:
            message += "\nStopped after an image, as asked."
        if self.count(FAILED) or self.count(CANCELLED) or self.count(NOT_RUN):
            message += (
                "\nOnly a finished image saves a .haemorun, so running the batch again with "
                "Skip done on runs just the ones that failed or did not run."
            )
        if self.statistics is not None and self.statistics.message:
            message += "\n" + self.statistics.message
        return message

    def write(self) -> Path | None:
        """Write the summary; never raises, so a summary file held open by
        another program (a spreadsheet, a sync client, a virus scanner) never
        stops the batch -- it is written again after the next image."""
        try:
            self.output_root.mkdir(parents=True, exist_ok=True)
            partial = self.csv_path.with_name(f"{self.csv_path.name}.part")
            with partial.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result.row())
            _replace(partial, self.csv_path)
        except OSError as error:
            logger.warning("Could not write %s: %s", self.csv_path, error)
            return None
        return self.csv_path


#: Windows refuses to replace a file another program has open for a moment
#: (a scanner or sync client looking at the one just written).
_REPLACE_ATTEMPTS = 20
_REPLACE_WAIT_S = 0.1


def _replace(source: Path, target: Path) -> None:
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_WAIT_S)


def format_duration(seconds: float | None) -> str:
    """``"under a minute"``, ``"41 min"`` or ``"1 h 50 min"``; "" for none."""
    if seconds is None:
        return ""
    minutes = int(round(seconds / 60))
    if minutes < 1:
        return "under a minute"
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def format_bytes(count: int | None) -> str:
    """``"3.2 GB"``, ``"640 MB"``; "" for none."""
    if count is None:
        return ""
    if count >= 1024**3:
        return f"{count / 1024**3:.1f} GB"
    return f"{count / 1024**2:.0f} MB"


def figure_cells(
    seconds: float | None, peak_rss_bytes: int | None, vessels: int | None
) -> tuple[str, str, str]:
    """The table's Time, Peak RAM and Vessels cells for one image."""
    return (
        format_duration(seconds),
        format_bytes(peak_rss_bytes),
        "" if vessels is None else f"{vessels:,}",
    )


def batch_progress_text(
    results: Sequence[BatchResult], *, elapsed_s: float, running_elapsed_s: float = 0.0
) -> str:
    """One line on a batch as it runs: ``"Image 3 of 12: E14_MCA -- 2 done, 0
    failed, 41 min so far, about 1 h 50 min left"``. *results* are the
    batch's images in order, as far as they have got; *elapsed_s* is the
    batch's time so far and *running_elapsed_s* the running image's. The
    estimate is the mean time of the images this batch ran to the end, so
    there is none until one has."""
    total = len(results)
    running = [index for index, result in enumerate(results) if result.status == RUNNING]
    done = [result for result in results if result.status == DONE and result.seconds is not None]
    failed = sum(result.status == FAILED for result in results)
    head = (
        f"Image {running[0] + 1} of {total}: {results[running[0]].match.stem}"
        if running
        else f"{total} image{'' if total == 1 else 's'}"
    )
    parts = [f"{len(done)} done", f"{failed} failed", f"{format_duration(elapsed_s)} so far"]
    if done:
        mean = sum(result.seconds for result in done) / len(done)
        waiting = sum(result.status == NOT_RUN for result in results)
        left = mean * waiting + (max(mean - running_elapsed_s, 0.0) if running else 0.0)
        parts.append(f"about {format_duration(left)} left" if left >= 30 else "nearly done")
    return f"{head} -- {', '.join(parts)}"


def _peak_rss(settings: Mapping[str, Any], stem: str) -> int | None:
    """The image's own peak memory: the largest of its stages' peaks in its
    run manifest. Not the manifest's ``peak_rss_bytes``: that is the
    process's, and a batch runs every image in one process, so after the
    first image it is the largest of this and every earlier one's."""
    from haemolynx.pipeline.provenance import MANIFEST_SUFFIX

    prefix = settings.get("vtk_output_prefix")
    if prefix in (None, ""):
        return None
    try:
        manifest = json.loads(
            (Path(prefix).parent / f"{stem}{MANIFEST_SUFFIX}").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    stages = manifest.get("stages") if isinstance(manifest, dict) else None
    peaks = [
        stage.get("peak_rss_bytes")
        for stage in (stages if isinstance(stages, list) else ())
        if isinstance(stage, dict) and isinstance(stage.get("peak_rss_bytes"), int)
    ]
    if peaks:
        return max(peaks)
    peak = manifest.get("peak_rss_bytes")
    return peak if isinstance(peak, int) else None


def run_image(
    values: Mapping[str, Any],
    schema: Schema,
    *,
    progress: Callable[[Any], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
    haemorun: Path | None = None,
):
    """Run the pipeline on one image's settings (form values), recording each
    stage as the panel does, and write its ``.haemorun`` to *haemorun*.
    Returns the graph. *check_cancelled* is called between stages and raises
    to stop the run."""
    from haemolynx.gui.results import ResultLayers
    from haemolynx.gui.run_snapshot import capture_run, write_run_snapshot
    from haemolynx.gui.stage_checkpoints import StageCheckpoints
    from haemolynx.pipeline import run_pipeline_stages
    from haemolynx.pipeline.settings import resolve_settings

    settings = resolve_settings(dict(values), schema=schema, config_path=None)
    results = ResultLayers(settings=settings)
    checkpoints = StageCheckpoints()

    def watched(event) -> None:
        if check_cancelled is not None:
            check_cancelled()
        if progress is not None:
            progress(event)

    def produced(stage: str, output) -> None:
        if check_cancelled is not None:
            check_cancelled()
        try:
            group = results.stage_finished(stage, output)
        except Exception:  # noqa: BLE001 - drawing must never end a run
            logger.exception("could not build layers for stage %s", stage)
            return
        try:
            checkpoints.record(stage, group, results, settings=settings, output=output)
        except Exception:  # noqa: BLE001 - a bad snapshot must not end the run
            logger.exception("could not record checkpoint for stage %s", stage)

    graph = run_pipeline_stages(settings, schema, progress=watched, on_stage_output=produced)
    if haemorun is not None:
        report = (
            "Batch run: the run produced no graph."
            if graph is None
            else f"Finished: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} vessels."
        )
        snapshot = capture_run(
            checkpoints=checkpoints,
            results=results,
            settings=dict(values),
            show_results=True,
            report=report,
        )
        write_run_snapshot(haemorun, snapshot)
    return graph


def clear_image_caches(values: Mapping[str, Any], schema: Schema) -> tuple[Path, ...]:
    """Delete what one image's run (its form *values*) left on disk that no
    later image of the batch reads; returns the files removed. Never raises.

    - its decompressed input in ``memmap_directory`` (the low-RAM option's
      :func:`~haemolynx.io.discard_cached_raw_volume`);
    - scratch volumes Windows kept while they were mapped
      (:func:`~haemolynx.preprocessing.memmap_support.delete_dropped_memmap_files`);
    - its stage checkpoint pickles: its ``.haemorun`` holds every stage, and
      Load run writes what a later Run from this stage needs;
    - its thick-vessel region cache: only a run of this same image reads it.

    Its outputs -- the skeleton, graph and voxel size a skipped stage loads
    among them -- stay.
    """
    from haemolynx.gui.stage_checkpoints import (
        discard_cached_artefacts,
        output_dir_from_prefix,
        stems_for_cached_artefacts,
    )
    from haemolynx.io import discard_cached_raw_volume
    from haemolynx.pipeline.settings import resolve_settings
    from haemolynx.pipeline.stages import thick_vessel_region_cache
    from haemolynx.preprocessing.memmap_support import delete_dropped_memmap_files

    removed: list[Path] = []
    try:
        settings = resolve_settings(dict(values), schema=schema, config_path=None)
        left = delete_dropped_memmap_files()
        if left:
            logger.info("%d scratch volume file(s) are still in use and stay for now.", left)
        if settings.get("use_memmap_loading") and settings.get("memmap_directory"):
            removed += discard_cached_raw_volume(
                settings["input_path"],
                axis_order=settings["image_axis_order"],
                memmap_directory=settings["memmap_directory"],
            )
        output_dir = output_dir_from_prefix(settings.get("vtk_output_prefix"))
        if output_dir is not None:
            for stem in stems_for_cached_artefacts(settings):
                removed += discard_cached_artefacts(output_dir, stem)
            removed += thick_vessel_region_cache(settings, output_dir).discard()
    except Exception:  # noqa: BLE001 - a cache left on disk never stops a batch
        logger.exception("could not clear this image's cached files")
    if removed:
        logger.info("Cleared %d cached file(s) this image left on disk.", len(removed))
    return tuple(removed)


def write_batch_settings(
    values: Mapping[str, Any],
    schema: Schema,
    folders: BatchFolders,
    output_root: Path,
    conditions: BatchConditions | None = None,
    exclude_sections: Sequence[str] = (),
) -> Path:
    """``batch_settings.yaml``: the settings the batch ran with, then its
    folders, conditions and the statistics sections its comparison leaves out."""
    from haemolynx.pipeline.provenance import write_settings_yaml

    path = output_root / SETTINGS_FILENAME
    write_settings_yaml(path, {k: v for k, v in values.items() if k in schema}, schema)
    lines = ["", "# Batch folders (blank: the images' folder)"]
    for name in IMAGE_FOLDER_ROLES:
        value = getattr(folders, name)
        lines.append(f"# batch_{name}: {'' if value is None else Path(value).as_posix()}")
    for name in folders.file_keys:
        lines.append(f"# batch_{name}: {Path(getattr(folders, name)).as_posix()}")
    if folders.from_raw:
        lines.append("# batch_from_raw: yes (ilastik segments each raw image)")
    lines.append(f"# batch_output: {Path(output_root).as_posix()}")
    for name in ("a", "b"):
        condition = getattr(conditions, name) if conditions is not None else None
        lines.append(f"# batch_condition_{name}: {'' if condition is None else condition.label}")
    if conditions is not None:
        lines.append(f"# batch_comparison: {conditions.design}")
        if conditions.subject_pattern:
            lines.append(f"# batch_subject_pattern: {conditions.subject_pattern}")
        if conditions.paired:
            lines.append(f"# batch_paired_aggregate: {conditions.aggregate}")
    if exclude_sections:
        lines.append(f"# batch_exclude_sections: {', '.join(exclude_sections)}")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def run_batch(
    values: Mapping[str, Any],
    schema: Schema,
    matches: Sequence[BatchMatch],
    output_root: Path | str,
    *,
    folders: BatchFolders | None = None,
    progress: Callable[[Any], None] | None = None,
    on_item: Callable[[int, int, BatchResult], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
    skip_done: bool = True,
    conditions: BatchConditions | None = None,
    exclude_sections: Sequence[str] = (),
    clear_caches: bool = True,
    stop_requested: Callable[[], bool] | None = None,
) -> BatchSummary:
    """Run *values* (the panel's form values) on each of *matches*, in order.

    *on_item* is called with ``(index, total, result)`` as each image starts
    (status :data:`RUNNING`) and once it is done. *check_cancelled* raises
    :class:`~haemolynx.gui.run_state.RunCancelled` (or anything) to stop the
    batch; the image running stops at its next stage and is recorded as
    cancelled, the rest as not run. With *skip_done*, an image whose
    ``.haemorun`` is already there is not run again. With *clear_caches*,
    each image's cached files are deleted once it ends (:func:`clear_image_caches`).
    *stop_requested* is asked before each image: once it says True, the batch
    stops there, the image that was running having finished and been saved,
    and the rest are recorded as not run.

    Once the images are done (or the batch is cancelled), the statistics of
    every image that ran or was skipped as done are written side by side,
    by *conditions* when given and without the *exclude_sections*
    (:func:`write_statistics`).
    """
    from haemolynx.gui.run_state import RunCancelled
    from haemolynx.io import clear_vessel_mask_cache

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = conditions or BatchConditions()
    summary = BatchSummary(
        output_root,
        [BatchResult(match, condition=conditions.condition_of(match.stem)) for match in matches],
    )
    if folders is not None:
        try:
            write_batch_settings(
                values, schema, folders, output_root, conditions, exclude_sections
            )
        except Exception:  # noqa: BLE001 - a record of the batch must not stop it
            logger.exception("could not write %s", SETTINGS_FILENAME)
    summary.write()
    total = len(summary.results)
    cancelled = False
    for index, result in enumerate(summary.results):
        if cancelled:
            break
        if stop_requested is not None and stop_requested():
            summary.stopped = True
            logger.info("Batch stopped before image %d of %d, as asked.", index + 1, total)
            break
        match = result.match
        target = haemorun_path(output_root, match)
        if skip_done and target.is_file():
            result.status, result.haemorun = SKIPPED, target
            logger.info("Batch image %d of %d, %s: already done, skipped.", index + 1, total, match.stem)
            summary.write()
            if on_item is not None:
                on_item(index, total, result)
            continue
        settings = item_settings(values, match, output_root)
        result.status = RUNNING
        if on_item is not None:
            on_item(index, total, result)
        logger.info("=== Batch image %d of %d: %s ===", index + 1, total, match.stem)
        start = time.perf_counter()
        try:
            graph = run_image(
                settings,
                schema,
                progress=progress,
                check_cancelled=check_cancelled,
                haemorun=target,
            )
        except RunCancelled:
            result.status = CANCELLED
            cancelled = True
        except Exception as error:  # noqa: BLE001 - one image never ends the batch
            result.status = FAILED
            result.error = f"{type(error).__name__}: {error}"
            result.error_detail = traceback.format_exc()
            logger.error("Batch image %s failed:\n%s", match.stem, result.error_detail)
        else:
            result.status = DONE
            result.haemorun = target
            if graph is not None:
                result.nodes = graph.number_of_nodes()
                result.vessels = graph.number_of_edges()
        finally:
            result.seconds = time.perf_counter() - start
            result.peak_rss_bytes = _peak_rss(settings, match.stem)
            clear_vessel_mask_cache()
            if clear_caches:
                clear_image_caches(settings, schema)
        summary.write()
        if on_item is not None:
            on_item(index, total, result)
    summary.statistics = write_statistics(
        output_root,
        [result.match.stem for result in summary.results if result.status in (DONE, SKIPPED)],
        conditions,
        exclude_sections=exclude_sections,
    )
    logger.info(summary.message())
    return summary


def _summary_rows(output_root: Path | str) -> list[dict[str, str]]:
    try:
        with (Path(output_root) / SUMMARY_FILENAME).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def read_summary_figures(output_root: Path | str) -> dict[str, tuple[str, str, str]]:
    """Each image's Time, Peak RAM and Vessels cells from *output_root*'s
    ``batch_summary.csv``, by stem (images that ran there)."""

    def number(text: str | None, kind):
        try:
            return kind(float(text)) if text not in (None, "") else None
        except ValueError:
            return None

    figures = {}
    for row in _summary_rows(output_root):
        cells = figure_cells(
            number(row.get("wall_time_s"), float),
            number(row.get("peak_rss_bytes"), int),
            number(row.get("vessels"), int),
        )
        if row.get("stem") and any(cells):
            figures[row["stem"]] = cells
    return figures


def read_summary_statuses(output_root: Path | str) -> dict[str, str]:
    """Each image's status in *output_root*'s ``batch_summary.csv`` (with its
    error), by stem: what the last batch there came to. Empty when there is
    no summary or it cannot be read."""
    statuses: dict[str, str] = {}
    for row in _summary_rows(output_root):
        stem, status = row.get("stem"), row.get("status")
        if stem and status:
            error = row.get("error") or ""
            statuses[stem] = f"{status}: {error}" if error else status
    return statuses


def stems_done_before(output_root: Path | str, matches: Sequence[BatchMatch]) -> frozenset[str]:
    """The stems of *matches* whose ``.haemorun`` is already in *output_root*:
    the images a batch run with Skip done would skip."""
    return frozenset(match.stem for match in matches if haemorun_path(output_root, match).is_file())


def statistics_warning(values: Mapping[str, Any], conditions: BatchConditions | None) -> str:
    """Why the conditions will compare nothing, or "": with Statistics off no
    image writes the statistics the batch's comparison reads."""
    if conditions is None or not conditions.conditions or values.get("statistics"):
        return ""
    return (
        "Statistics is off, so no image writes the statistics the conditions are "
        "compared on: turn on Statistics, or the batch writes no comparison."
    )


def write_statistics(
    output_root: Path | str,
    stems: Sequence[str],
    conditions: BatchConditions | None = None,
    *,
    exclude_sections: Sequence[str] = (),
) -> BatchStatisticsFiles | None:
    """The batch's statistics for *stems*, without the *exclude_sections*;
    never raises, so a batch that ran is never reported as failed for its
    statistics."""
    if not stems:
        return None
    try:
        return write_batch_statistics(
            output_root, stems, conditions, exclude_sections=exclude_sections
        )
    except Exception as error:  # noqa: BLE001 - the images' own outputs stand
        logger.exception("could not write the batch statistics")
        return BatchStatisticsFiles(message=f"Could not write the batch statistics: {error}")


#: The rows a batch fills for each image, shown as :data:`BATCH_ROW_TEXT`
#: while one is set (the other files' only while given a folder): setting ->
#: the batch folder it comes from.
BATCH_ROW_FOLDERS = {
    "input_path": "segmented",
    "raw_image_path": "raw",
    "large_arteriole_mask_path": "large",
    "large_venule_mask_path": "large",
    "small_arteriole_mask_path": "small",
    "small_venule_mask_path": "small",
    **{role.setting: role.key for role in FILE_ROLES},
}
BATCH_ROW_TEXT = "Batch run"
BATCH_BUTTON_TEXT = "Batch run"

#: Why Check segmented image, the optimisers and Run from this stage are off
#: while a batch is set: each acts on the one image the Input tab names.
SINGLE_IMAGE_IN_BATCH = (
    "This acts on one image, and a batch is set: open Batch run and press "
    "Leave batch run to use it"
)

def batch_button_greyed(
    values: Mapping[str, Any], *, batch_set: bool, running: bool, batch_running: bool
) -> str:
    """Why the Input tab's Batch run button is greyed, or "" while it is not:
    under a synthetic network, which has no images to run, and while a run of
    one image (or an optimiser) is going. Never while a batch is set or
    running: its window is where a batch is followed, stopped and left."""
    from haemolynx.graph.boundaries import network_is_synthetic
    from haemolynx.gui.chrome_tooltips import BATCH_RUN_TOOLTIPS

    if batch_set or batch_running:
        return ""
    if network_is_synthetic(values):
        return BATCH_RUN_TOOLTIPS["button_synthetic"]
    if running:
        return BATCH_RUN_TOOLTIPS["button_running"]
    return ""


#: Load run while a batch runs would stop the batch.
BATCH_RUNNING_NO_LOAD = (
    "A batch is running: open its images once it is done, or Cancel it first."
)
#: What Load run adds when it leaves the batch run to show one image.
LEFT_BATCH_FOR_RUN = (
    "The batch run was left to show this one image: Batch run, then Use this "
    "batch, goes back to it."
)

#: The table's columns.
TABLE_COLUMNS = (
    "Run", "Segmented image", "Raw image", "Large masks", "Small masks", "Other files",
    "Condition", "Time", "Peak RAM", "Vessels", "Status",
)
READY = "ready"


@dataclass(frozen=True)
class BatchSetup:
    """A batch set up in the panel: its folders and what was matched in them.
    Panel state, not settings: each image's run records the files it used."""

    folders: BatchFolders
    output_root: Path
    matches: tuple[BatchMatch, ...] = ()
    #: Stems the user unticked.
    unticked: frozenset[str] = frozenset()
    skip_done: bool = True
    #: Each image's checks (Check, or Run pipeline), by stem: its first error,
    #: or "" when it passed.
    checked: Mapping[str, str] = field(default_factory=dict)
    #: The conditions the images' file names put them in.
    conditions: BatchConditions = field(default_factory=BatchConditions)
    #: Statistics sections the comparison leaves out ("Dead Ends").
    exclude_sections: tuple[str, ...] = ()
    #: Stems whose ``.haemorun`` was in the output folder when matched.
    done_before: frozenset[str] = frozenset()
    #: Each image's status in the output folder's last ``batch_summary.csv``.
    last_statuses: Mapping[str, str] = field(default_factory=dict)
    #: Each image's Time, Peak RAM and Vessels cells (:func:`figure_cells`):
    #: this batch's, else the last summary's.
    figures: Mapping[str, tuple[str, str, str]] = field(default_factory=dict)
    #: Each failed image's traceback from this batch, for Show error.
    error_details: Mapping[str, str] = field(default_factory=dict)

    def selected(self) -> list[BatchMatch]:
        return [match for match in self.matches if match.stem not in self.unticked]

    def button_text(self) -> str:
        count = len(self.selected())
        return f"{BATCH_BUTTON_TEXT}: {count} image{'' if count == 1 else 's'}"

    def fills(self, setting: str) -> bool:
        """Whether the batch sets the row *setting* for each image: the
        images, raw images and masks always, another file while it has a
        folder of them."""
        role = BATCH_ROW_FOLDERS.get(setting)
        if role in IMAGE_FOLDER_ROLES:
            return True
        return role is not None and self.folders.file_folder(role) is not None

    def row_tooltip(self, setting: str) -> str:
        role = BATCH_ROW_FOLDERS[setting]
        if role == "segmented" and self.folders.from_raw:
            return (
                f"Made for each image by ilastik, from its raw image in {self.folders.raw}; "
                "open Batch run to change it"
            )
        if role in IMAGE_FOLDER_ROLES:
            folder = self.folders.folder(role)
        else:
            folder = self.folders.file_folder(role)
        return f"Set for each image by the batch, from {folder}; open Batch run to change it"

    def gates(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """*values* as the panel's greying reads them while this batch is set:
        a row it fills reads as set when a ticked image has that file, so
        what needs it (the raw image's switches, ilastik's among them) is not
        greyed while the panel's own row is empty."""
        gates = dict(values)
        selected = self.selected()
        for setting in BATCH_ROW_FOLDERS:
            if gates.get(setting) not in (None, "") or not self.fills(setting):
                continue
            found = next(
                (path for match in selected if (path := match.settings().get(setting)) is not None),
                None,
            )
            if found is not None:
                gates[setting] = found
        return gates

    def condition_counts(self) -> str:
        """``"3 in WT, 3 in KO, 1 in neither"`` over the ticked images."""
        return self.conditions.describe(match.stem for match in self.selected())

    def only(self, stem: str) -> "BatchSetup":
        """The batch of the one image *stem*, run whether or not it was done
        before: what Run again runs."""
        return replace(
            self,
            unticked=frozenset(match.stem for match in self.matches if match.stem != stem),
            skip_done=False,
            checked={},
        )

    def choose(self, role: str, path: Path, match: BatchMatch) -> "BatchSetup":
        """Every image's tie in *role* chosen as *path* was for *match*."""
        from haemolynx.io.batch_matching import choose_alike

        matches = choose_alike(self.matches, role, path, match.ties(role))
        return replace(self, matches=matches, checked={})


def _mask_cell(found) -> str:
    if found is None:
        return "not used"
    if found.ambiguous:
        tied = sum(len(paths) for paths in found.ambiguous.values())
        return f"{tied} files tie: choose"
    if found.is_pair:
        return f"{found.arteriole.name} + {found.venule.name}"
    if found.global_mask is not None:
        return f"{found.global_mask.name} (global)"
    if found.arteriole is not None:
        return f"{found.arteriole.name} (global)"
    if found.venule is not None:
        return f"{found.venule.name} (no arteriole)"
    return "none found"


def _files_cell(match: BatchMatch) -> str:
    """The table's Other files cell: each other file matched, by what it is."""
    parts = []
    for key, path in match.files.items():
        tied = match.file_ties.get(key)
        if tied:
            text = f"{len(tied)} files tie: choose"
        else:
            text = path.name if path is not None else "none"
        parts.append(f"{file_role(key).label}: {text}")
    return "; ".join(parts)


DONE_BEFORE_SKIPPED = "done before: will be skipped"
DONE_BEFORE_RERUN = "done before: runs again"


def match_status(
    match: BatchMatch,
    checked: Mapping[str, str] | None = None,
    *,
    done_before: bool = False,
    skip_done: bool = True,
    last: str = "",
) -> str:
    """What the table says of *match*: its first problem; its checks' first
    error or what it came to in this batch (*checked*); that its
    ``.haemorun`` is there already (*done_before*), so Skip done skips it;
    what the output folder's last batch said of it (*last*); or ready."""
    if match.problems:
        return match.problems[0]
    result = (checked or {}).get(match.stem)
    if result:
        return result
    if done_before:
        return DONE_BEFORE_SKIPPED if skip_done else DONE_BEFORE_RERUN
    if result == "":
        return f"{READY} (checked)"
    if last == RUNNING:
        return "last batch: stopped while running"
    if last:
        return f"last batch: {last}"
    return READY


def condition_cell(setup: BatchSetup, match: BatchMatch) -> str:
    """The table's Condition cell: the condition's label, "both conditions",
    or "neither" while a condition is set ("" while none is). Paired, it also
    names the image's subject, or why it has no partner."""
    conditions = setup.conditions
    if not conditions.conditions:
        return ""
    label = conditions.condition_of(match.stem) or "neither"
    if not (conditions.paired and conditions.compares) or conditions.problem():
        return label
    pairing = conditions.pairing(selected.stem for selected in setup.selected())
    pair = pairing.pair_of(match.stem)
    if pair is not None:
        own = pair.a if match.stem in pair.a else pair.b
        shared = f" ({conditions.aggregate} of {len(own)})" if len(own) > 1 else ""
        return f"{label}, subject {pair.subject}{shared}"
    reason = pairing.reason_left_out(match.stem)
    if not reason:  # unticked: not in the batch's pairing
        return label
    return label if reason.startswith("in ") else f"{label}: unpaired, {reason}"


def table_rows(setup: BatchSetup) -> list[tuple[str, ...]]:
    """The table's text, one tuple per image in :data:`TABLE_COLUMNS` order
    (the first column is the tick, "yes"/"no")."""
    rows = []
    for match in setup.matches:
        if match.raw_ties:
            raw = f"{len(match.raw_ties)} files tie: choose"
        else:
            raw = match.raw.name if match.raw is not None else "none"
        segmented = (
            match.segmented.name if match.segmented is not None else f"by ilastik: {match.stem}"
        )
        rows.append(
            (
                "no" if match.stem in setup.unticked else "yes",
                segmented,
                raw,
                _mask_cell(match.masks.get("large")),
                _mask_cell(match.masks.get("small")),
                _files_cell(match),
                condition_cell(setup, match),
                *setup.figures.get(match.stem, ("", "", "")),
                match_status(
                    match,
                    setup.checked,
                    done_before=match.stem in setup.done_before,
                    skip_done=setup.skip_done,
                    last=setup.last_statuses.get(match.stem, ""),
                ),
            )
        )
    return rows


def tied_cells(match: BatchMatch) -> dict[int, tuple[str, tuple[Path, ...]]]:
    """The table columns holding a tie for *match*: column -> (role, files).
    One role per column, the first; the next shows once it is chosen."""
    columns = {
        "raw": TABLE_COLUMNS.index("Raw image"),
        "large": TABLE_COLUMNS.index("Large masks"),
        "small": TABLE_COLUMNS.index("Small masks"),
    }
    cells: dict[int, tuple[str, tuple[Path, ...]]] = {}
    for role in match.tied_roles:
        if role in FILE_ROLE_KEYS:
            column = TABLE_COLUMNS.index("Other files")
        else:
            column = columns[role.partition("_")[0]]
        cells.setdefault(column, (role, match.ties(role)))
    return cells


def check_setup(
    values: Mapping[str, Any], schema: Schema, setup: BatchSetup
) -> tuple[BatchSetup, list[str]]:
    """Every selected image's checks (:func:`~haemolynx.pipeline.batch.batch_preflight`)
    on the resolved *values*: the setup with each image's result, and which
    images are done now (a batch since the folders were matched made more),
    and a line per image for the report."""
    from haemolynx.pipeline.settings import resolve_settings

    base = resolve_settings(dict(values), schema=schema, config_path=None)
    checked: dict[str, str] = {}
    lines: list[str] = []
    problem = setup.conditions.problem()
    if setup.conditions.conditions:
        lines.append(f"Conditions: {setup.condition_counts()}.")
        if problem:
            lines.append(f"    FAILED: {problem}")
        warning = statistics_warning(base, setup.conditions)
        if warning:
            lines.append(f"    warning: {warning}")
    for match in setup.selected():
        report = batch_preflight(base, match, setup.output_root, schema)
        checked[match.stem] = report.errors[0] if report.errors else ""
        lines.append(f"{'ok' if report.ok else 'FAILED'}: {match.stem}")
        lines.extend(f"    FAILED: {message}" for message in report.errors)
        lines.extend(f"    warning: {message}" for message in report.warnings)
    if problem:
        checked = {stem: error or problem for stem, error in checked.items()}
    done = stems_done_before(setup.output_root, setup.matches)
    return replace(setup, checked=checked, done_before=done), lines


#: The comparison CSV's columns the window shows, by how their names start
#: (the condition labels are in the rest): the metric, each condition's N
#: and median, the difference, the test's P, the q-value and the note.
COMPARISON_VIEW_COLUMNS = (
    "Section", "Metric", "Unit", "N ", "Median ", "Difference of medians",
    "Median of paired differences", "P (", "Q (", "Note",
)


def comparison_tested(header: Sequence[str], rows: Sequence[Sequence[str]]) -> int:
    """How many of :func:`comparison_table`'s rows were tested: those with a q."""
    column = next((i for i, name in enumerate(header) if name.startswith("Q (")), None)
    return 0 if column is None else sum(bool(row[column]) for row in rows)


def comparison_table(output_root: Path | str) -> tuple[list[str], list[list[str]]]:
    """The batch's ``batch_condition_comparison.csv`` as the window shows it:
    the columns of :data:`COMPARISON_VIEW_COLUMNS` (the first P only: the
    rank test's), most significant first by q (untested metrics last), and
    the numbers rounded. ``([], [])`` when there is none."""
    from haemolynx.statistics.batch_comparison import COMPARISON_FILENAME

    try:
        with (Path(output_root) / COMPARISON_FILENAME).open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = list(reader)
    except (OSError, StopIteration, UnicodeDecodeError, csv.Error):
        return [], []
    keep: list[int] = []
    for index, name in enumerate(header):
        if not name.startswith(COMPARISON_VIEW_COLUMNS):
            continue
        if name.startswith("P (") and any(header[i].startswith("P (") for i in keep):
            continue
        keep.append(index)
    q_column = next((i for i in keep if header[i].startswith("Q (")), None)

    def q_of(row: list[str]) -> float:
        try:
            value = float(row[q_column]) if q_column is not None else float("nan")
        except (ValueError, IndexError):
            value = float("nan")
        return float("inf") if value != value else value

    def cell(text: str) -> str:
        try:
            value = float(text)
        except ValueError:
            return text
        if value != value:
            return ""
        return f"{value:.3g}" if abs(value) < 1e4 else f"{value:,.0f}"

    rows.sort(key=q_of)
    return [header[i] for i in keep], [[cell(row[i]) if i < len(row) else "" for i in keep] for row in rows]


@dataclass(frozen=True)
class BatchRecord:
    """A batch as its ``batch_settings.yaml`` records it: its folders,
    conditions and the statistics sections it left out, and the file, whose
    body is the settings it ran with (a config Load config reads)."""

    path: Path
    folders: BatchFolders
    output_root: Path
    conditions: BatchConditions
    exclude_sections: tuple[str, ...] = ()


def read_batch_settings(path: Path | str) -> BatchRecord | None:
    """The batch :func:`write_batch_settings` recorded in *path*, read back
    from its ``# batch_...:`` lines; None when it has none or cannot be read."""
    from haemolynx.statistics.batch_comparison import PAIRED, parse_sections

    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    record: dict[str, str] = {}
    for line in lines:
        if line.startswith("# batch_"):
            name, _colon, value = line[len("# batch_"):].partition(":")
            record[name.strip()] = value.strip()
    from_raw = record.get("from_raw", "").startswith("yes")
    if not record.get("raw" if from_raw else "segmented"):
        return None

    def folder(name: str) -> Path | None:
        return Path(record[name]) if record.get(name) else None

    paired = record.get("comparison") == PAIRED
    return BatchRecord(
        path=path,
        folders=BatchFolders(
            folder("segmented"),
            **{name: folder(name) for name in IMAGE_FOLDER_ROLES[1:] + FILE_ROLE_KEYS},
            from_raw=from_raw,
        ),
        output_root=folder("output") or path.parent,
        conditions=BatchConditions.from_text(
            record.get("condition_a"),
            record.get("condition_b"),
            paired=paired,
            subject_pattern=record.get("subject_pattern") if paired else None,
            aggregate=record.get("paired_aggregate") if paired else None,
        ),
        exclude_sections=parse_sections(record.get("exclude_sections", "")),
    )


def mask_switch_hint(schema: Schema, values: Mapping[str, Any], size: str) -> str:
    """Why a batch matches no *size* masks, and the switches (with their tab)
    that would make it: the greyed folder row's text."""
    from haemolynx.graph.boundaries import VESSEL_MASK_SWITCHES
    from haemolynx.gui.form import label_for
    from haemolynx.gui.tabs import assign_to_stages
    from haemolynx.io.vessel_masks import VESSEL_MASK_SETTINGS

    tabs = assign_to_stages(schema)
    switch = VESSEL_MASK_SWITCHES[size]
    if values.get(switch) and values.get("automated_vessel_assignment"):
        ilastik = VESSEL_MASK_SETTINGS[size]["use_ilastik"]
        if not values.get(ilastik):
            return ""
        return f"not used: ilastik makes these masks from each image's raw image ({label_for(ilastik)})"
    needed = [name for name in ("automated_vessel_assignment", switch) if not values.get(name)]
    named = " and ".join(f"'{label_for(name)}'" for name in needed)
    where = sorted({tabs.get(name, "") for name in needed} - {""})
    return f"not used: tick {named}" + (f" ({', '.join(where)})" if where else "")


def settings_changed(values: Mapping[str, Any], schema: Schema) -> list[str]:
    """The settings *values* sets other than as its default, by name, leaving
    out the rows a batch fills for each image: what every image runs with."""

    def plain(value: Any) -> Any:
        if value in (None, ""):
            return None
        return str(value) if isinstance(value, Path) else value

    defaults = schema.defaults()
    return [
        name
        for name in schema.names
        if BATCH_ROW_FOLDERS.get(name) not in IMAGE_FOLDER_ROLES
        and plain(values.get(name)) != plain(defaults.get(name))
    ]


def selected_matches(
    matches: Sequence[BatchMatch], only: Sequence[str] | None = None
) -> list[BatchMatch]:
    """*matches* whose stem is in *only*, or all of them."""
    if not only:
        return list(matches)
    wanted = {stem.lower() for stem in only}
    return [match for match in matches if match.stem.lower() in wanted]


def main(argv: Sequence[str] | None = None) -> int:
    """Run a batch from a command line (:func:`haemolynx.gui.batch_run_cli.main`,
    which prints, so it lives in a command-line module of its own)."""
    from haemolynx.gui.batch_run_cli import main as run_command_line

    return run_command_line(argv)


__all__ = [
    "BATCH_ROW_FOLDERS",
    "BATCH_ROW_TEXT",
    "BATCH_RUNNING_NO_LOAD",
    "BatchRecord",
    "BatchResult",
    "BatchSetup",
    "BatchSummary",
    "CANCELLED",
    "DONE",
    "DONE_BEFORE_RERUN",
    "DONE_BEFORE_SKIPPED",
    "FAILED",
    "LEFT_BATCH_FOR_RUN",
    "NOT_RUN",
    "RUNNING",
    "SINGLE_IMAGE_IN_BATCH",
    "SKIPPED",
    "SUMMARY_FILENAME",
    "TABLE_COLUMNS",
    "COMPARISON_VIEW_COLUMNS",
    "batch_button_greyed",
    "check_setup",
    "clear_image_caches",
    "comparison_table",
    "comparison_tested",
    "batch_progress_text",
    "condition_cell",
    "figure_cells",
    "format_bytes",
    "format_duration",
    "haemorun_path",
    "mask_switch_hint",
    "match_status",
    "main",
    "read_batch_settings",
    "read_summary_figures",
    "read_summary_statuses",
    "run_batch",
    "run_image",
    "selected_matches",
    "settings_changed",
    "statistics_warning",
    "stems_done_before",
    "table_rows",
    "tied_cells",
    "write_statistics",
]


if __name__ == "__main__":
    sys.exit(main())
