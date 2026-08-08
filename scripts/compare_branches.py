#!/usr/bin/env python3
"""Run the resistance network pipeline on two branches and diff the results.

A reviewer's question about a pull request is usually "does this change the
numbers?". This runs the same dataset, with the same settings, on the current
checkout and on a reference ref (``main`` by default), then reports every way
the two runs differ: graph metrics, the first graph-building stage whose output
diverges, edge attributes, the statistics CSVs, the VTK exports, and runtime.
It writes ``COMPARISON.md`` and an ``index.html`` that shows every plot from
both runs side by side.

This is not part of the test suite and does not run in CI: a full comparison
takes roughly 15 minutes per side and needs a 328 MB image that is not in the
repository.

Usage
-----
Compare the current branch against ``main`` on the nerve dataset::

    python scripts/compare_branches.py

Against another ref, with the image somewhere else::

    python scripts/compare_branches.py --ref devel \\
        --image /data/Nerve_capillaries.tif

Prove the tool itself is sound -- compare HEAD against HEAD and require the two
runs to agree exactly, on the small committed fixture (about a minute)::

    python scripts/compare_branches.py --self-check --smoke

Change a setting for both sides at once::

    python scripts/compare_branches.py --setting min_stub_length=5.0

Notes
-----
* Boundary boxes are in physical (z, y, x) MICRONS, not voxel indices.
* The reference ref is checked out into a temporary ``git worktree`` and run in
  a subprocess whose ``PYTHONPATH`` points at it, so the editable install does
  not silently make both sides run the same library. Each side verifies this
  and refuses to run if the library resolved outside its own checkout.
* A reference ref from before the ImageLynx -> HaemoLynx rename is handled:
  each side imports whichever package name its own checkout has.
* Branches differ in how the entry point takes settings -- older ones have a
  hundred-odd keyword arguments and read some settings from module constants,
  newer ones take a settings dict validated against a schema. Each side is
  inspected and adapted. A setting that defines the comparison and cannot be
  applied stops that side rather than quietly running a different
  configuration; anything else is listed in the report.
* Output goes to ``comparison_outputs/`` (gitignored) unless ``--output-dir``
  says otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branch_comparison import report as report_module, run_settings, runner

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_USAGE = 2
EXIT_DIFFERENCES = 3

SMOKE_IMAGE_RELATIVE = "tests/data/seven_vessel_noisy_3d.tif"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="compare_branches.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ref",
        default="main",
        help="reference ref to compare against (default: main)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help=(
            "input image for both runs (default: "
            f"{run_settings.DEFAULT_IMAGE_RELATIVE} in this checkout or in the "
            "repository's main working tree)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="where to write the report (default: comparison_outputs/<stamp>)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter to run each side with (default: this one)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5400.0,
        help="seconds to allow each side (default: 5400)",
    )
    parser.add_argument(
        "--voxel-size-xyz",
        default=",".join(str(v) for v in run_settings.DEFAULT_VOXEL_SIZE_XYZ),
        help=(
            "physical voxel size in x,y,z microns, pinned on both sides, or "
            "'auto' to take it from the image metadata"
        ),
    )
    parser.add_argument(
        "--axis-order", default="zyx", help="axis order of the input file"
    )
    parser.add_argument(
        "--setting",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="override one setting on both sides; repeatable",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "run the small committed fixture with boxes that select on it -- "
            "checks the plumbing, not the science"
        ),
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help=(
            "compare HEAD against HEAD and fail if anything differs; the "
            "strongest check that the tool itself is sound"
        ),
    )
    parser.add_argument(
        "--fail-on-difference",
        action="store_true",
        help="exit non-zero when the two runs differ",
    )
    parser.add_argument(
        "--keep-worktree",
        action="store_true",
        help="leave the reference worktree behind for inspection",
    )
    parser.add_argument(
        "--no-pin-all",
        action="store_true",
        help=(
            "pin only the settings this tool names, letting every other "
            "setting take each branch's own default"
        ),
    )
    parser.add_argument(
        "--allow-unapplied",
        action="store_true",
        help=(
            "run even when a side cannot apply a setting that defines the "
            "comparison; the report is marked incomplete"
        ),
    )
    return parser.parse_args(argv)


def parse_voxel_size(text: str) -> tuple[float, float, float] | None:
    if text.strip().lower() == "auto":
        return None
    parts = [part for part in text.replace(" ", "").split(",") if part]
    if len(parts) != 3:
        raise SystemExit(
            f"--voxel-size-xyz needs three comma-separated numbers or 'auto', got {text!r}"
        )
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as error:
        raise SystemExit(f"--voxel-size-xyz is not numeric: {text!r}") from error


def parse_overrides(items: list[str]) -> dict:
    overrides = {}
    for item in items:
        name, separator, raw = item.partition("=")
        if not separator:
            raise SystemExit(f"--setting needs NAME=VALUE, got {item!r}")
        try:
            overrides[name.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[name.strip()] = raw
    return overrides


def locate_image(repo: Path, explicit: Path | None, relative: str) -> Path:
    """The input image, looked for where a developer is likely to have it."""
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"--image does not exist: {path}")
        return path

    candidates = [repo / relative]
    try:
        listing = runner.git("worktree", "list", "--porcelain", cwd=repo)
        for line in listing.splitlines():
            if line.startswith("worktree "):
                candidates.append(Path(line[len("worktree "):]) / relative)
    except runner.ComparisonError:
        pass

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    looked = "\n  ".join(str(c) for c in candidates)
    raise SystemExit(
        f"Could not find the input image. Looked in:\n  {looked}\n"
        "Pass --image with its path."
    )


def per_side_settings(paths: runner.SidePaths, image: Path) -> dict:
    """The settings that name this side's own input and output locations."""
    return {
        "input_path": str(image),
        "plot_dir": str(paths.plot_dir),
        "base_plot_dir": str(paths.plot_dir),
        "vtk_output_prefix": str(paths.vtk_output_prefix),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        repo = runner.repo_root()
    except runner.ComparisonError as error:
        print(error, file=sys.stderr)
        return EXIT_USAGE

    reference_ref = "HEAD" if args.self_check else args.ref
    fail_on_difference = args.fail_on_difference or args.self_check

    warnings: list[str] = []
    if args.self_check and runner.git("status", "--porcelain", cwd=repo):
        warnings.append(
            "--self-check compares the working tree against HEAD, but the tree "
            "has uncommitted changes, so a difference here may be your own edits."
        )

    image_relative = (
        SMOKE_IMAGE_RELATIVE if args.smoke else run_settings.DEFAULT_IMAGE_RELATIVE
    )
    try:
        image = locate_image(repo, args.image, image_relative)
    except SystemExit as error:
        print(error, file=sys.stderr)
        return EXIT_USAGE

    voxel_size = parse_voxel_size(args.voxel_size_xyz)
    if args.smoke and args.image is None and args.voxel_size_xyz == ",".join(
        str(v) for v in run_settings.DEFAULT_VOXEL_SIZE_XYZ
    ):
        # The fixture has no useful metadata; unit voxels keep the smoke boxes,
        # which are written in voxels, meaning what they say.
        voxel_size = (1.0, 1.0, 1.0)

    output_dir = args.output_dir or (
        repo
        / "comparison_outputs"
        / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-vs-{reference_ref.replace('/', '_')}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_paths = runner.SidePaths("current", output_dir / "current")
    reference_paths = runner.SidePaths("reference", output_dir / "reference")

    base_settings, required = run_settings.build_settings(
        image_path=str(image),
        plot_dir=str(current_paths.plot_dir),
        vtk_output_prefix=str(current_paths.vtk_output_prefix),
        axis_order=args.axis_order,
        voxel_size_xyz=voxel_size,
        boundary_settings=(
            run_settings.SMOKE_BOUNDARY_SETTINGS if args.smoke else None
        ),
        extra=parse_overrides(args.setting),
    )

    current_ref_name = runner.current_ref(repo)
    print(f"Comparing {current_ref_name} against {reference_ref}")
    print(f"  image:  {image}")
    print(f"  output: {output_dir}")

    pinned = dict(base_settings)
    path_settings: list[str] = []
    if not args.no_pin_all:
        print("Resolving the current checkout's full settings...")
        pinned, pin_warnings, path_settings = runner.resolve_pinned_settings(
            checkout=repo,
            paths=runner.SidePaths("resolve", output_dir / "_resolve"),
            settings=base_settings,
            required=required,
            python=args.python,
            timeout=args.timeout,
        )
        warnings.extend(pin_warnings)

    try:
        with runner.reference_worktree(
            repo, reference_ref, keep=args.keep_worktree
        ) as reference_checkout:
            print(f"Running current checkout ({current_ref_name})...")
            current_result = runner.invoke_side(
                checkout=repo,
                paths=current_paths,
                settings={**pinned, **per_side_settings(current_paths, image)},
                required=required,
                mode="run",
                python=args.python,
                timeout=args.timeout,
                allow_unapplied=args.allow_unapplied,
                path_settings=path_settings,
            )
            print(f"  {'ok' if current_result.get('ok') else 'FAILED'}")

            print(f"Running reference ({reference_ref})...")
            reference_result = runner.invoke_side(
                checkout=reference_checkout,
                paths=reference_paths,
                settings={**pinned, **per_side_settings(reference_paths, image)},
                required=required,
                mode="run",
                python=args.python,
                timeout=args.timeout,
                allow_unapplied=args.allow_unapplied,
                path_settings=path_settings,
            )
            print(f"  {'ok' if reference_result.get('ok') else 'FAILED'}")

            reference_commit = runner.commit_of(repo, reference_ref)
            # Recorded here: the worktree is gone by the time we report.
            reference_checkout_path = Path(reference_checkout)
    except runner.ComparisonError as error:
        print(error, file=sys.stderr)
        return EXIT_USAGE

    current_side = runner.build_side_report(
        "current",
        current_ref_name,
        runner.commit_of(repo, "HEAD"),
        repo,
        current_paths,
        current_result,
        required,
    )
    reference_side = runner.build_side_report(
        "reference",
        reference_ref,
        reference_commit,
        reference_checkout_path,
        reference_paths,
        reference_result,
        required,
    )

    image_stem = image.stem
    current_artefacts = (
        runner.collect_side(current_paths, image_stem, root=output_dir)
        if current_side.ok
        else runner.SideArtefacts()
    )
    reference_artefacts = (
        runner.collect_side(reference_paths, image_stem, root=output_dir)
        if reference_side.ok
        else runner.SideArtefacts()
    )

    comparison = runner.build_report(
        current=current_side,
        reference=reference_side,
        current_artefacts=current_artefacts,
        reference_artefacts=reference_artefacts,
        image_path=str(image),
        warnings=warnings,
    )

    markdown_path = output_dir / "COMPARISON.md"
    html_path = output_dir / "index.html"
    markdown_path.write_text(report_module.render_markdown(comparison), encoding="utf-8")
    html_path.write_text(report_module.render_html(comparison), encoding="utf-8")

    print()
    print(comparison.status_line())
    if comparison.first_stage_difference is not None:
        print(
            "First differing stage: "
            f"{comparison.first_stage_difference.label} "
            f"({', '.join(comparison.first_stage_difference.reasons) or comparison.first_stage_difference.status})"
        )
    print(f"Markdown: {markdown_path}")
    print(f"HTML:     {html_path}")

    if comparison.failures or not comparison.complete:
        return EXIT_RUN_FAILED
    if fail_on_difference and comparison.differs:
        return EXIT_DIFFERENCES
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
