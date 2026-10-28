"""The panel fills its pericyte mask rows -- the Pericyte measurements box's
and each pericyte perturbation's -- from a mask beside the input image, and a
mask entered into one into the others (``gui/pericyte_mask_search.py``)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import settings_widget  # noqa: E402

pytestmark = pytest.mark.gui

MEASURED = "pericyte_measured_constriction"


def _files(folder: Path, *names: str) -> list[Path]:
    paths = []
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        paths.append(path)
    return paths


def _path(value) -> Path | None:
    return None if value in (None, "") else Path(value)


@pytest.fixture
def panel(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((8, 8, 8)), name="stack")
    widget = settings_widget(napari_viewer=viewer)
    image, mask = _files(tmp_path, "E14_MCA_segmented.tif", "E14_MCA_pericytes.tif")
    rows = widget._haemolynx_rows()
    rows["input_path"].value = image
    rows["run_haemodynamics"].value = True
    return widget, rows, widget._haemolynx_perturbations, mask


def _entry_mask(perturbations, index: int = 0) -> Path | None:
    return _path(perturbations.entries()[index]["overrides"].get("pericyte_mask_path"))


def test_ticking_measure_pericytes_fills_its_mask_from_the_input_folder(panel):
    widget, rows, _perturbations, mask = panel
    assert _path(widget._haemolynx_values()["pericyte_measurement_mask_path"]) is None

    rows["measure_pericytes"].value = True
    assert _path(widget._haemolynx_values()["pericyte_measurement_mask_path"]) == mask
    assert "E14_MCA_pericytes.tif" in widget._haemolynx_report()


def test_choosing_the_measured_constriction_reads_the_mask_found(panel):
    _widget, _rows, perturbations, mask = panel
    perturbations.add()
    perturbations.choose_type(0, MEASURED)

    overrides = perturbations.entries()[0]["overrides"]
    assert overrides["use_pericyte_mask_constriction"] is True
    assert _entry_mask(perturbations) == mask


def test_a_mask_entered_into_a_perturbation_fills_the_measurement_row(panel, tmp_path):
    widget, rows, perturbations, mask = panel
    rows["measure_pericytes"].value = True
    perturbations.add()
    perturbations.choose_type(0, MEASURED)
    (chosen,) = _files(tmp_path / "elsewhere", "cells.tif")

    perturbations.editors()[0].editors["pericyte_mask_path"].value = chosen
    assert _entry_mask(perturbations) == chosen
    assert _path(widget._haemolynx_values()["pericyte_measurement_mask_path"]) == chosen


def test_a_mask_entered_into_the_measurement_row_fills_the_perturbations(panel, tmp_path):
    widget, rows, perturbations, mask = panel
    perturbations.add()
    perturbations.choose_type(0, MEASURED)
    rows["measure_pericytes"].value = True
    assert _entry_mask(perturbations) == mask
    (chosen,) = _files(tmp_path / "elsewhere", "cells.tif")

    rows["pericyte_measurement_mask_path"].value = chosen
    assert _entry_mask(perturbations) == chosen


def test_a_mask_the_user_chose_in_a_perturbation_is_kept(panel, tmp_path):
    widget, rows, perturbations, mask = panel
    perturbations.add()
    perturbations.choose_type(0, MEASURED)
    mine, theirs = _files(tmp_path / "elsewhere", "mine.tif", "theirs.tif")
    perturbations.editors()[0].editors["pericyte_mask_path"].value = mine
    rows["measure_pericytes"].value = True
    # The measurement row came into use and took the user's mask, not the one found.
    assert _path(widget._haemolynx_values()["pericyte_measurement_mask_path"]) == mine

    rows["pericyte_measurement_mask_path"].value = theirs
    # The perturbation's mask was the user's own: it is not replaced.
    assert _entry_mask(perturbations) == mine
