"""The batch run's command line: ``python -m haemolynx.gui.batch_run``.

Kept apart from :mod:`haemolynx.gui.batch_run` because a command line prints
(the checks it reports and the summary are the answer to the flags typed, so
they belong on stdout whether or not logging is configured), and the library
module the panel runs a batch through must not.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from haemolynx.gui import batch_run
from haemolynx.io.batch_matching import BatchFolders, match_batch
from haemolynx.parsers import configure_console_logging
from haemolynx.pipeline.batch import batch_mask_sizes, batch_preflight, default_output_root
from haemolynx.statistics.batch_comparison import (
    AGGREGATE_NONE,
    AGGREGATES,
    BatchConditions,
    parse_sections,
)


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
    parser.add_argument(
        "--condition-a",
        help="Phrase(s), comma-separated, in the file names of one condition's images (e.g. 'WT').",
    )
    parser.add_argument(
        "--condition-b",
        help="Phrase(s) in the other condition's file names: the two are tested against each other.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Run nothing: write the batch statistics from the images already run in --output.",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help=(
            "Compare the conditions paired (Wilcoxon signed-rank, paired t) rather than "
            "as independent images (Mann-Whitney U, Welch t): each image with the image "
            "of the other condition whose name is most alike once the condition phrase "
            "is taken out, within its subject (the part of the name before the phrase: "
            "28_3_d in 28_3_d_contra_FITC). Images with no partner, or a tie, are left "
            "out and listed."
        ),
    )
    parser.add_argument(
        "--subject-pattern",
        help=(
            "With --paired: a regular expression whose first group (or whole match) "
            "is the subject ID, in place of the text before the condition phrase; "
            "'^(\\d+_\\d+)' reads 28_3 from 28_3_d_contra_FITC."
        ),
    )
    parser.add_argument(
        "--paired-aggregate",
        choices=AGGREGATES,
        default=AGGREGATE_NONE,
        help=(
            "With --paired: a subject with more than one image in a condition is left "
            "out (none, the default), or its images there are combined, metric by metric, "
            "by their median or mean, and those are paired."
        ),
    )
    parser.add_argument(
        "--exclude-section",
        action="append",
        default=[],
        metavar="SECTION",
        help=(
            "Leave this statistics section (and its subsections, 'Strahler Network' "
            "takes 'Strahler Network > Orders > 1'), ignoring case, out of the batch "
            "statistics and the comparison, so the Benjamini-Hochberg q-values are "
            "over the metrics left; e.g. --exclude-section \"Dead Ends\". Repeat it, "
            "or separate names by commas, for more than one."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a batch from a command line. Returns the exit status: 0 when every
    image ran (or, with ``--dry-run``, passed its checks), 1 otherwise."""
    from haemolynx.parsers.config import load_config
    from haemolynx.pipeline import default_schema

    args = _parser().parse_args(argv)
    # Timed lines: a batch runs unattended for hours, and how long each
    # image's steps took is read off its log afterwards.
    configure_console_logging(timestamps=True)
    schema = default_schema()
    values = load_config(args.config, schema)
    folders = BatchFolders(args.segmented, raw=args.raw, large=args.large, small=args.small)
    output_root = Path(args.output) if args.output else default_output_root(values)
    if output_root is None:
        print("No output folder: pass --output, or set vtk_output_prefix to a path with a folder.")
        return 1
    matches = batch_run.selected_matches(match_batch(folders, batch_mask_sizes(values)), args.only)
    if not matches:
        print(f"No segmented images found in {folders.segmented}.")
        return 1
    conditions = BatchConditions.from_text(
        args.condition_a,
        args.condition_b,
        paired=args.paired,
        subject_pattern=args.subject_pattern,
        aggregate=args.paired_aggregate,
    )
    if conditions.problem():
        print(conditions.problem())
        return 1
    exclude_sections = parse_sections(args.exclude_section)
    if conditions.conditions:
        print(f"Conditions: {conditions.describe(match.stem for match in matches)}.")
    if args.compare_only:
        written = batch_run.write_statistics(
            output_root,
            [match.stem for match in matches],
            conditions,
            exclude_sections=exclude_sections,
        )
        print(written.message if written is not None else "No images.")
        return 0 if written is not None and written.written else 1
    warning = batch_run.statistics_warning(values, conditions)
    if warning:
        print(f"warning: {warning}")
    failing = []
    for match in matches:
        report = batch_preflight(values, match, output_root, schema)
        status = "ok" if report.ok else "FAILED"
        condition = ""
        if conditions.conditions:
            condition = f" ({conditions.condition_of(match.stem) or 'no condition'})"
        print(f"\n[{status}] {match.stem}{condition}")
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
    summary = batch_run.run_batch(
        values,
        schema,
        matches,
        output_root,
        folders=folders,
        skip_done=not args.rerun,
        conditions=conditions,
        exclude_sections=exclude_sections,
    )
    print(summary.message())
    return 0 if summary.count(batch_run.FAILED) == 0 and summary.count(batch_run.CANCELLED) == 0 else 1


__all__ = ["main"]
