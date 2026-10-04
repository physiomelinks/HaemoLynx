"""Unit tests for haemolynx.optimisation.scorecard -- pure, synthetic data only."""
from __future__ import annotations

import pytest

from haemolynx.optimisation.scorecard import ScoreRow, scorecard_lines, worse_rows


@pytest.mark.parametrize(
    "before, after, higher_is_better, worse, better",
    [
        (0.90, 0.85, True, True, False),  # fell by 0.05, past the 0.02 tolerance
        (0.90, 0.89, True, False, False),  # within it: the same
        (0.90, 0.95, True, False, True),
        (3.0, 5.0, False, True, False),  # more components is worse
        (3.0, 1.0, False, False, True),
    ],
)
def test_a_row_is_worse_or_better_only_past_its_tolerance(before, after, higher_is_better, worse, better):
    row = ScoreRow("measure", before, after, higher_is_better, tolerance=0.02)
    assert row.worse is worse
    assert row.better is better


def test_a_zero_tolerance_counts_any_step_the_wrong_way():
    assert ScoreRow("cartwheel hubs", 0.0, 1.0, higher_is_better=False).worse
    assert not ScoreRow("cartwheel hubs", 1.0, 1.0, higher_is_better=False).worse


def test_worse_rows_picks_out_the_regressions():
    rows = (
        ScoreRow("kept", 1.0, 1.0, True),
        ScoreRow("coverage", 0.9, 0.8, True, 0.02),
        ScoreRow("components", 1.0, 2.0, False),
    )
    assert [row.name for row in worse_rows(rows)] == ["coverage", "components"]


def test_the_report_lines_mark_each_measure_and_warn_on_regressions():
    rows = (
        ScoreRow("vessels reached", 0.95, 0.80, True, 0.02),
        ScoreRow("graph components", 4.0, 1.0, False),
        ScoreRow("self-loops", 0.0, 0.0, False),
    )
    lines = scorecard_lines(rows)

    assert lines[0].startswith("Your starting settings against these")
    assert lines[1] == "    vessels reached   0.95 -> 0.8  worse"
    assert lines[2] == "    graph components  4 -> 1  better"
    assert lines[3] == "    self-loops        0 -> 0"
    assert lines[4] == (
        "WARNING: worse than your starting settings on vessels reached (0.95 -> 0.8). "
        "Check these before using the settings."
    )


def test_no_rows_no_lines():
    assert scorecard_lines(()) == []
    assert "WARNING" not in "\n".join(scorecard_lines((ScoreRow("a", 1.0, 1.0, True),)))
