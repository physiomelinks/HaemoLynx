"""Compare two batch-run roots by content, such as a re-run against its archive.

    python examples/cb_compare_batch_runs.py outputs/cb_h1_batch_2026-10-01_pre_rerun outputs/cb_h1_batch
    python examples/cb_compare_batch_runs.py OLD_SENSITIVITY_ROOT NEW_SENSITIVITY_ROOT --specimen WKY-A

Each run on both sides is opened through the batch-run reader, then compared part by part:
graph, diameters, edge table, skeleton, vessel mask and TH mask
(``ImageLynx.batch_compare`` says what counts as equal). Floats must match exactly unless
``--rtol`` is given. Exit status: 0 if every run matches, 1 if any differs, 2 for a usage
error, 3 if a run could not be compared (the reader refused it, or a file could not be read);
the error's traceback is printed, and no verdict.
"""
from __future__ import annotations

import argparse
import math
import sys
import traceback

from ImageLynx.batch_compare import RootsDoNotMatch, compare_roots, format_report
from ImageLynx.specimens import SPECIMENS


def _rtol(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"--rtol must be a finite number >= 0, not {text}")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("old_root", help="batch or sensitivity root to compare against")
    parser.add_argument("new_root", help="root of the same kind to compare")
    parser.add_argument("--specimen", action="append",
                        choices=[specimen.specimen_id for specimen in SPECIMENS],
                        help="compare only this specimen (repeatable); default all six")
    parser.add_argument("--rtol", type=_rtol, default=0.0,
                        help="relative tolerance for float values (default 0, exact)")
    args = parser.parse_args(argv)

    chosen = [specimen for specimen in SPECIMENS
              if args.specimen is None or specimen.specimen_id in args.specimen]
    try:
        comparisons = compare_roots(args.old_root, args.new_root, chosen, rtol=args.rtol)
    except RootsDoNotMatch as error:
        parser.error(str(error))
    except Exception:
        # Not a difference: nothing was compared. Exit 3 so a script can tell it from 1.
        traceback.print_exc()
        return 3
    print(format_report(comparisons))
    return 0 if all(comparison.same for comparison in comparisons) else 1


if __name__ == "__main__":
    sys.exit(main())
