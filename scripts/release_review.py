#!/usr/bin/env python3
"""Score the release review (``RELEASE_REVIEW.md``) at a commit.

TEMPORARY: deleted at the release with the review (``PLAN_release_review.md``
step R8).

Re-measures every check a script can decide, reads the recorded human marks
(``scripts/release_review/human_checks.yaml``) for the H checks, carries
every other check's mark from its row, applies the ladder and writes a dated
scorecard with what moved since the last one.

Usage
-----
The quick tier (static facts, a minute or two) at HEAD::

    python scripts/release_review.py

The fixture tier too (the committed fixtures, the edge cases, the identity
runs and the phantoms; ten minutes or so)::

    python scripts/release_review.py --fixtures

The full tier too (the reference datasets named in the main checkout's
``release_review_datasets.yaml``; hours, and up to 20 GB of memory)::

    python scripts/release_review.py --full --datasets e14p5

CI scores its own clean checkout's quick tier and adds the table to the job
summary::

    python scripts/release_review.py --tree . --summary "$GITHUB_STEP_SUMMARY"

Notes
-----
* The commit is measured from a ``git archive`` of it, run with
  ``PYTHONPATH`` at the archive's ``src``: peer sessions edit the live tree,
  and the editable install would import it. ``--tree`` measures a checkout
  as it stands instead.
* Reports go to the main checkout's ``outputs/release_review/<date>_<sha>/``
  (git-ignored): ``scorecard.md``, ``scorecard.json``, and each tier's runs.
  The full tier appends to ``outputs/release_review/benchmarks.csv``.
* The scorer never writes ``RELEASE_REVIEW.md`` or ``human_checks.yaml``:
  copy the summary table and a log line into the review by hand.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from release_review import report, runner  # noqa: E402
from release_review.rubric import FIXTURES, FULL, QUICK  # noqa: E402


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--commit", default="HEAD", help="the commit to score (default HEAD)")
    parser.add_argument("--fixtures", action="store_true", help="run the fixture tier too")
    parser.add_argument("--full", action="store_true", help="run the fixture and full tiers too")
    parser.add_argument("--tree", type=Path, help="measure this checkout as it is, not a git archive")
    parser.add_argument("--out", type=Path, help="report root (default: the main checkout's outputs/release_review)")
    parser.add_argument("--coverage-xml", type=Path, help="a coverage.xml to read tests.6a from")
    parser.add_argument("--datasets-file", type=Path, help="the reference datasets file (default: the main checkout's)")
    parser.add_argument("--datasets", help="comma-separated: run only these reference datasets")
    parser.add_argument("--summary", type=Path, help="append the score table to this file (CI's job summary)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    tiers = [QUICK]
    if arguments.fixtures or arguments.full:
        tiers.append(FIXTURES)
    if arguments.full:
        tiers.append(FULL)
    repo = Path(arguments.tree or Path.cwd()).resolve()
    folder, card = runner.score(
        repo=repo,
        commit=arguments.commit,
        tiers=tiers,
        out_root=arguments.out,
        tree=arguments.tree,
        coverage_xml=arguments.coverage_xml,
        datasets_file=arguments.datasets_file,
        only_datasets=arguments.datasets.split(",") if arguments.datasets else None,
        log=lambda message: print(f"[release review] {message}", flush=True),
    )
    table = report.scorecard_table(card)
    print(f"\nLowest {card.lowest}, mean {card.mean} at {card.commit} ({', '.join(tiers)})\n")
    print(table)
    if card.disagreements:
        print("\nDisagreements with RELEASE_REVIEW.md:")
        for check_id, said, now in card.disagreements:
            print(f"  {check_id}: the row says {said}, measured {now}")
    print(f"\nReport: {folder / 'scorecard.md'}")
    if arguments.summary:
        with Path(arguments.summary).open("a", encoding="utf-8") as handle:
            handle.write(f"## Release review, quick tier at `{card.commit}`\n\n")
            handle.write(f"Lowest **{card.lowest}**, mean **{card.mean}** (the Document column is `RELEASE_REVIEW.md`'s).\n\n")
            handle.write(table + "\n\n")
            if card.disagreements:
                handle.write("Disagreements with the review's rows:\n\n")
                handle.writelines(f"- `{check_id}`: the row says {said}, measured {now}\n" for check_id, said, now in card.disagreements)
    return 0


if __name__ == "__main__":
    sys.exit(main())
