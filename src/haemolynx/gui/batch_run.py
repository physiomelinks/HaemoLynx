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
from dataclasses import dataclass, field
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

    def write(self) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        partial = self.csv_path.with_name(f"{self.csv_path.name}.part")
        with partial.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.row())
        partial.replace(self.csv_path)
        return self.csv_path


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
    "BatchResult",
    "BatchSummary",
    "CANCELLED",
    "DONE",
    "FAILED",
    "NOT_RUN",
    "SKIPPED",
    "SUMMARY_FILENAME",
    "haemorun_path",
    "main",
    "run_batch",
    "run_image",
    "selected_matches",
]


if __name__ == "__main__":
    sys.exit(main())
