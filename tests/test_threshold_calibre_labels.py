"""cb_h2_threshold_calibre.py reads the runs either side of the frozen threshold, whatever it is.

It hard-coded 0.85 / 0.90 / 0.95 until the 2026-09-28 re-selection moved the freeze to 0.95.
"""
import pytest

from ImageLynx import cb_settings

calibre = pytest.importorskip("cb_h2_threshold_calibre")


def test_the_three_runs_bracket_the_frozen_threshold_on_the_grid():
    grid = list(cb_settings.THRESHOLD_GRID)
    i = grid.index(cb_settings.FROZEN_THRESHOLD)
    assert (calibre.LOWER, calibre.FROZEN, calibre.UPPER) == tuple(
        f"{t:.2f}" for t in grid[i - 1:i + 2])
    labels = [label for label, _ in calibre.RUNS]
    assert labels == [calibre.LOWER, calibre.FROZEN, calibre.UPPER]


def test_the_frozen_run_is_the_batch_and_the_neighbours_are_sensitivity_runs():
    paths = [str(path("WKY-A")) for _, path in calibre.RUNS]
    assert "cb_h1_batch" in paths[1]
    assert f"cb_h1_sensitivity/t{calibre.LOWER}" in paths[0]
    assert f"cb_h1_sensitivity/t{calibre.UPPER}" in paths[2]
