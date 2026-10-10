"""Prove a driver refactor left its outputs unchanged: snapshot driver outputs, then compare.

    python examples/cb_driver_snapshots.py snapshot .scratch/snapshots/before
    # ... refactor ...
    python examples/cb_driver_snapshots.py snapshot .scratch/snapshots/after
    python examples/cb_driver_snapshots.py compare .scratch/snapshots/before .scratch/snapshots/after

``snapshot DIR`` runs the fast ``examples/cb_*.py`` drivers, each into ``DIR/<driver>/``
(``--slow`` adds cb_h2_hypoxic_fraction and cb_h2_vtk; ``--driver NAME`` picks drivers, and a
named slow driver runs without ``--slow``). It refuses a ``DIR`` that exists. Drivers read the
live batch runs and write nothing outside ``DIR``.

``compare OLD NEW`` compares the two by content: stdout and text with the snapshot path
removed, JSON, PNG pixels, and VTK points, cells and arrays by name, dtype and value
(``ImageLynx.driver_snapshots`` says what counts as equal). Exit status: 0 if every file
matches, 1 if any differs, 2 for a usage error, 3 if nothing could be compared (a driver
failed, the two snapshots ran different drivers, arguments, Python or packages, or a file could
not be read); for a failure the reason is printed and there is no verdict.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from ImageLynx.driver_snapshots import (
    DRIVERS, STDERR_NAME, SnapshotsNotComparable, compare_snapshots, failed_driver,
    format_report, select_drivers, take_snapshot)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    snapshot = commands.add_parser("snapshot", help="run drivers into a new folder")
    snapshot.add_argument("dir", help="new folder to hold the snapshot")
    snapshot.add_argument("--driver", action="append", choices=[d.name for d in DRIVERS],
                          help="run only this driver (repeatable); default every fast driver")
    snapshot.add_argument("--slow", action="store_true",
                          help="also run the slow drivers (about an hour in all)")

    compare = commands.add_parser("compare", help="compare two snapshots by content")
    compare.add_argument("old", help="snapshot to compare against")
    compare.add_argument("new", help="snapshot to compare")

    args = parser.parse_args(argv)
    if args.command == "snapshot":
        return _snapshot(parser, args)
    return _compare(parser, args)


def _snapshot(parser, args) -> int:
    if Path(args.dir).exists():
        parser.error(f"{args.dir} exists; a snapshot goes in a new folder.")
    drivers = select_drivers(args.driver, slow=args.slow)
    print("running: " + ", ".join(driver.name for driver in drivers))
    manifest = take_snapshot(args.dir, drivers)
    failed = failed_driver(manifest)
    if failed:
        print(f"driver {failed} exited non-zero; see {Path(args.dir) / failed / STDERR_NAME}. "
              f"The snapshot stops there and cannot be compared.", file=sys.stderr)
        return 3
    print(f"wrote {args.dir} ({len(manifest['drivers'])} drivers, "
          f"commit {manifest['commit'][:9]}{', dirty' if manifest['dirty'] else ''})")
    return 0


def _compare(parser, args) -> int:
    for folder in (args.old, args.new):
        if not Path(folder).is_dir():
            parser.error(f"{folder} is not a folder.")
    try:
        comparisons = compare_snapshots(args.old, args.new)
    except SnapshotsNotComparable as error:
        print(f"not comparable: {error}", file=sys.stderr)
        return 3
    except Exception:
        # Not a difference: a file could not be read. Exit 3 so a script can tell it from 1.
        traceback.print_exc()
        return 3
    print(format_report(comparisons))
    return 0 if all(comparison.same for comparison in comparisons) else 1


if __name__ == "__main__":
    sys.exit(main())
