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

Nothing here imports Qt: :func:`main` runs a batch from a command line.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from haemolynx.io.batch_matching import BatchFolders, BatchMatch, match_batch
from haemolynx.parsers import Schema
from haemolynx.pipeline.batch import (
    batch_mask_sizes,
    batch_preflight,
    default_output_root,
    image_output_folder,
    item_settings,
)

logger = logging.getLogger(__name__)

SUMMARY_FILENAME = "batch_summary.csv"
SETTINGS_FILENAME = "batch_settings.yaml"

DONE = "done"
FAILED = "failed"
SKIPPED = "skipped (done before)"
CANCELLED = "cancelled"
NOT_RUN = "not run"

SUMMARY_COLUMNS = (
    "stem",
    "status",
    "segmented",
    "raw",
    "large_arteriole_mask",
    "large_venule_mask",
    "small_arteriole_mask",
    "small_venule_mask",
    "haemorun",
    "nodes",
    "vessels",
    "seconds",
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
    haemorun: Path | None = None
    nodes: int | None = None
    vessels: int | None = None
    seconds: float | None = None
    peak_rss_bytes: int | None = None
    error: str = ""

    def row(self) -> dict[str, Any]:
        settings = self.match.settings()

        def text(name: str) -> str:
            value = settings.get(name)
            return "" if value in (None, "") else str(value)

        return {
            "stem": self.match.stem,
            "status": self.status,
            "segmented": str(self.match.segmented),
            "raw": text("raw_image_path"),
            "large_arteriole_mask": text("large_arteriole_mask_path"),
            "large_venule_mask": text("large_venule_mask_path"),
            "small_arteriole_mask": text("small_arteriole_mask_path"),
            "small_venule_mask": text("small_venule_mask_path"),
            "haemorun": "" if self.haemorun is None else str(self.haemorun),
            "nodes": "" if self.nodes is None else self.nodes,
            "vessels": "" if self.vessels is None else self.vessels,
            "seconds": "" if self.seconds is None else round(self.seconds, 1),
            "peak_rss_bytes": "" if self.peak_rss_bytes is None else self.peak_rss_bytes,
            "error": self.error,
        }


@dataclass
class BatchSummary:
    """Every image of a batch, in the order run."""

    output_root: Path
    results: list[BatchResult] = field(default_factory=list)

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
        return f"Batch: {', '.join(parts)}. Summary: {self.csv_path}"

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


def _peak_rss(settings: Mapping[str, Any], stem: str) -> int | None:
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


def write_batch_settings(
    values: Mapping[str, Any],
    schema: Schema,
    folders: BatchFolders,
    output_root: Path,
) -> Path:
    """``batch_settings.yaml``: the settings the batch ran with, then its folders."""
    from haemolynx.pipeline.provenance import write_settings_yaml

    path = output_root / SETTINGS_FILENAME
    write_settings_yaml(path, {k: v for k, v in values.items() if k in schema}, schema)
    lines = ["", "# Batch folders (blank: the segmented folder)"]
    for name in ("segmented", "raw", "large", "small"):
        value = getattr(folders, name)
        lines.append(f"# batch_{name}: {'' if value is None else Path(value).as_posix()}")
    lines.append(f"# batch_output: {Path(output_root).as_posix()}")
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
) -> BatchSummary:
    """Run *values* (the panel's form values) on each of *matches*, in order.

    *on_item* is called with ``(index, total, result)`` as each image starts
    (status ``"running"``) and once it is done. *check_cancelled* raises
    :class:`~haemolynx.gui.run_state.RunCancelled` (or anything) to stop the
    batch; the image running stops at its next stage and is recorded as
    cancelled, the rest as not run. With *skip_done*, an image whose
    ``.haemorun`` is already there is not run again.
    """
    from haemolynx.gui.run_state import RunCancelled
    from haemolynx.io import clear_vessel_mask_cache

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = BatchSummary(output_root, [BatchResult(match) for match in matches])
    if folders is not None:
        try:
            write_batch_settings(values, schema, folders, output_root)
        except Exception:  # noqa: BLE001 - a record of the batch must not stop it
            logger.exception("could not write %s", SETTINGS_FILENAME)
    summary.write()
    total = len(summary.results)
    cancelled = False
    for index, result in enumerate(summary.results):
        if cancelled:
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
        result.status = "running"
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
            logger.error(
                "Batch image %s failed:\n%s", match.stem, traceback.format_exc()
            )
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
        summary.write()
        if on_item is not None:
            on_item(index, total, result)
    logger.info(summary.message())
    return summary


#: The rows a batch fills for each image, shown as :data:`BATCH_ROW_TEXT`
#: while one is set: setting -> the batch folder it comes from.
BATCH_ROW_FOLDERS = {
    "input_path": "segmented",
    "raw_image_path": "raw",
    "large_arteriole_mask_path": "large",
    "large_venule_mask_path": "large",
    "small_arteriole_mask_path": "small",
    "small_venule_mask_path": "small",
}
BATCH_ROW_TEXT = "Batch run"
BATCH_BUTTON_TEXT = "Batch run"

#: The table's columns.
TABLE_COLUMNS = ("Run", "Segmented image", "Raw image", "Large masks", "Small masks", "Status")
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

    def selected(self) -> list[BatchMatch]:
        return [match for match in self.matches if match.stem not in self.unticked]

    def button_text(self) -> str:
        count = len(self.selected())
        return f"{BATCH_BUTTON_TEXT}: {count} image{'' if count == 1 else 's'}"

    def row_tooltip(self, setting: str) -> str:
        folder = self.folders.folder(BATCH_ROW_FOLDERS[setting])
        return f"Set for each image by the batch, from {folder}; open Batch run to change it"

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


def match_status(match: BatchMatch, checked: Mapping[str, str] | None = None) -> str:
    """What the table says of *match*: its first problem, its checks' first
    error, or ready."""
    if match.problems:
        return match.problems[0]
    if checked and match.stem in checked:
        return checked[match.stem] or f"{READY} (checked)"
    return READY


def table_rows(setup: BatchSetup) -> list[tuple[str, ...]]:
    """The table's text, one tuple per image in :data:`TABLE_COLUMNS` order
    (the first column is the tick, "yes"/"no")."""
    rows = []
    for match in setup.matches:
        if match.raw_ties:
            raw = f"{len(match.raw_ties)} files tie: choose"
        else:
            raw = match.raw.name if match.raw is not None else "none"
        rows.append(
            (
                "no" if match.stem in setup.unticked else "yes",
                match.segmented.name,
                raw,
                _mask_cell(match.masks.get("large")),
                _mask_cell(match.masks.get("small")),
                match_status(match, setup.checked),
            )
        )
    return rows


def tied_cells(match: BatchMatch) -> dict[int, tuple[str, tuple[Path, ...]]]:
    """The table columns holding a tie for *match*: column -> (role, files).
    One role per column, the first; the next shows once it is chosen."""
    columns = {"raw": 2, "large": 3, "small": 4}
    cells: dict[int, tuple[str, tuple[Path, ...]]] = {}
    for role in match.tied_roles:
        column = columns[role.partition("_")[0]]
        cells.setdefault(column, (role, match.ties(role)))
    return cells


def check_setup(
    values: Mapping[str, Any], schema: Schema, setup: BatchSetup
) -> tuple[BatchSetup, list[str]]:
    """Every selected image's checks (:func:`~haemolynx.pipeline.batch.batch_preflight`)
    on the resolved *values*: the setup with each image's result, and a line
    per image for the report."""
    from haemolynx.pipeline.settings import resolve_settings

    base = resolve_settings(dict(values), schema=schema, config_path=None)
    checked: dict[str, str] = {}
    lines: list[str] = []
    for match in setup.selected():
        report = batch_preflight(base, match, setup.output_root, schema)
        checked[match.stem] = report.errors[0] if report.errors else ""
        lines.append(f"{'ok' if report.ok else 'FAILED'}: {match.stem}")
        lines.extend(f"    FAILED: {message}" for message in report.errors)
        lines.extend(f"    warning: {message}" for message in report.warnings)
    return replace(setup, checked=checked), lines


def selected_matches(
    matches: Sequence[BatchMatch], only: Sequence[str] | None = None
) -> list[BatchMatch]:
    """*matches* whose stem is in *only*, or all of them."""
    if not only:
        return list(matches)
    wanted = {stem.lower() for stem in only}
    return [match for match in matches if match.stem.lower() in wanted]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m haemolynx.gui.batch_run",
        description=(
            "Run the HaemoLynx pipeline once per segmented image in a folder, "
            "matching each image's raw image and vessel masks by file name, and "
            "save a .haemorun and the run's outputs in {output}/{image stem}/."
        ),
    )
    parser.add_argument("--config", required=True, help="Settings file (a config YAML or a run's _run_settings.yaml).")
    parser.add_argument("--segmented", required=True, help="Folder of segmented images.")
    parser.add_argument("--raw", help="Folder of raw images (default: the segmented folder).")
    parser.add_argument("--large", help="Folder of large vessel masks (default: the segmented folder).")
    parser.add_argument("--small", help="Folder of small vessel masks (default: the segmented folder).")
    parser.add_argument("--output", help="Output folder (default: the config's vtk_output_prefix folder).")
    parser.add_argument("--only", nargs="*", help="Run only these image stems.")
    parser.add_argument("--rerun", action="store_true", help="Run images whose .haemorun is already there again.")
    parser.add_argument("--dry-run", action="store_true", help="Match and check every image, run nothing.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a batch from a command line. Returns the exit status: 0 when every
    image ran (or, with ``--dry-run``, passed its checks), 1 otherwise."""
    from haemolynx.parsers.config import load_config
    from haemolynx.pipeline import default_schema

    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    schema = default_schema()
    values = load_config(args.config, schema)
    folders = BatchFolders(args.segmented, raw=args.raw, large=args.large, small=args.small)
    output_root = Path(args.output) if args.output else default_output_root(values)
    if output_root is None:
        print("No output folder: pass --output, or set vtk_output_prefix to a path with a folder.")
        return 1
    matches = selected_matches(match_batch(folders, batch_mask_sizes(values)), args.only)
    if not matches:
        print(f"No segmented images found in {folders.segmented}.")
        return 1
    failing = []
    for match in matches:
        report = batch_preflight(values, match, output_root, schema)
        status = "ok" if report.ok else "FAILED"
        print(f"\n[{status}] {match.stem}")
        for name, value in match.settings().items():
            if value is not None:
                print(f"    {name}: {value}")
        for message in report.errors:
            print(f"    FAILED: {message}")
        for message in report.warnings:
            print(f"    warning: {message}")
        if not report.ok:
            failing.append(match.stem)
    if failing:
        print(f"\nChecks failed for {len(failing)} image(s): {', '.join(failing)}. Nothing was run.")
        return 1
    if args.dry_run:
        print(f"\nAll {len(matches)} image(s) passed their checks.")
        return 0
    summary = run_batch(
        values,
        schema,
        matches,
        output_root,
        folders=folders,
        skip_done=not args.rerun,
    )
    print(summary.message())
    return 0 if summary.count(FAILED) == 0 and summary.count(CANCELLED) == 0 else 1


__all__ = [
    "BATCH_ROW_FOLDERS",
    "BATCH_ROW_TEXT",
    "BatchResult",
    "BatchSetup",
    "BatchSummary",
    "CANCELLED",
    "DONE",
    "FAILED",
    "NOT_RUN",
    "SKIPPED",
    "SUMMARY_FILENAME",
    "TABLE_COLUMNS",
    "check_setup",
    "haemorun_path",
    "match_status",
    "main",
    "run_batch",
    "run_image",
    "selected_matches",
    "table_rows",
    "tied_cells",
]


if __name__ == "__main__":
    sys.exit(main())
