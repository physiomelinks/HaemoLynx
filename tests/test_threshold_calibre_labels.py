"""cb_h2_threshold_calibre.py reads the runs either side of the frozen threshold, whatever it is.

It hard-coded 0.85 / 0.90 / 0.95 until the 2026-09-28 re-selection moved the freeze to 0.95.
"""
from types import SimpleNamespace

import pytest

from ImageLynx import cb_settings
from ImageLynx.batch_outputs import EDGE_TABLE_NAME, BatchRun

calibre = pytest.importorskip("cb_h2_threshold_calibre")


def test_the_three_runs_bracket_the_frozen_threshold_on_the_grid():
    grid = list(cb_settings.THRESHOLD_GRID)
    i = grid.index(cb_settings.FROZEN_THRESHOLD)
    assert (calibre.LOWER, calibre.FROZEN, calibre.UPPER) == tuple(
        f"{t:.2f}" for t in grid[i - 1:i + 2])
    labels = [label for label, _ in calibre.RUNS]
    assert labels == [calibre.LOWER, calibre.FROZEN, calibre.UPPER]


def test_the_frozen_run_is_the_default_batch_run_and_the_neighbours_are_sensitivity_runs():
    folders = {label: folder("WKY-A") for label, folder in calibre.RUNS}
    assert folders[calibre.FROZEN] is None
    assert folders[calibre.LOWER] == calibre.SENSITIVITY_DIR / f"t{calibre.LOWER}" / "WKY-A"
    assert folders[calibre.UPPER] == calibre.SENSITIVITY_DIR / f"t{calibre.UPPER}" / "WKY-A"


def _opened_run(tmp_path, diameters, asked):
    """A real ``BatchRun`` over an edge table holding ``diameters``, recording what was opened."""
    rows = ["u,v,key,edt_diameter_um"] + [f"{i},0,0,{d}" for i, d in enumerate(diameters)]
    (tmp_path / EDGE_TABLE_NAME).write_text("\n".join(rows) + "\n")

    def fake_open(specimen, run_dir=None):
        asked.append((specimen.specimen_id, run_dir))
        return BatchRun(specimen, tmp_path, None, None)
    return fake_open


def test_the_median_calibre_comes_from_the_opened_run_and_drops_non_positive_edges(
        tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(calibre, "open_batch_run",
                        _opened_run(tmp_path, ["4.0", "8.0", "0", "6.0"], asked))
    assert calibre.median_calibre("SHR-B", "somewhere") == 6.0
    assert asked == [("SHR-B", "somewhere")]


def test_a_blank_calibre_raises_rather_than_being_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(calibre, "open_batch_run",
                        _opened_run(tmp_path, ["4.0", "", "8.0", "6.0"], []))
    with pytest.raises(ValueError,
                       match=r"SHR-B.*empty or non-finite edt_diameter_um.*\(1, 0, 0\)"):
        calibre.median_calibre("SHR-B", "somewhere")
