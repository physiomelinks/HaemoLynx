"""The release review's scorer (TEMPORARY: delete at the release, step R8 of
``PLAN_release_review.md``).

``RELEASE_REVIEW.md`` scores sixteen aspects of HaemoLynx out of 10, each by a
ladder of checks. This package re-measures the checks a script can decide,
reads the recorded human marks for the rest, applies the ladder and writes a
dated scorecard with what moved since the last one. The command is
``scripts/release_review.py``.

Shaped like ``scripts/branch_comparison/``: the pure modules (the rubric, the
document parser, the ladder, each collector's parsing, the report) are tested
in milliseconds by ``tests/test_release_review_*.py``; :mod:`.runner` does the
slow work and never runs in CI except the quick tier.
"""
