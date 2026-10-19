"""The Batch run button, its window and Run pipeline's hand-off to a batch,
built for real with a live QApplication (no viewer: a batch draws nothing)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui import _widget as widget_mod  # noqa: E402
from haemolynx.gui._widget import settings_widget  # noqa: E402
from haemolynx.gui.batch_run import BATCH_ROW_FOLDERS, BatchSetup  # noqa: E402
from haemolynx.io import BatchFolders, match_batch  # noqa: E402

pytestmark = pytest.mark.gui

FIXTURE = Path(__file__).resolve().parent / "data" / "seven_vessel_noisy_3d.tif"


@pytest.fixture
def panel(qapp):
    return settings_widget(napari_viewer=None)


def _images(folder: Path, *stems: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        shutil.copy(FIXTURE, folder / f"{stem}.tif")
    return folder


def _setup(tmp_path: Path, *stems: str) -> BatchSetup:
    folder = _images(tmp_path / "seg", *stems)
    folders = BatchFolders(folder)
    return BatchSetup(folders=folders, output_root=tmp_path / "out", matches=match_batch(folders))


def _on_input_tab(panel, widget) -> bool:
    from haemolynx.gui.tabs import tab_titles

    tabs = panel._haemolynx_tabs
    page = tabs.widget(list(tab_titles()).index("1. Input"))
    ancestor = widget.native.parent()
    while ancestor is not None:
        if ancestor is page:
            return True
        ancestor = ancestor.parent()
    return False


def test_the_button_is_on_the_input_tab_under_the_segmented_image(panel):
    batch = panel._haemolynx_batch
    assert batch.button.text == "Batch run"
    assert batch.button.tooltip.strip()
    assert _on_input_tab(panel, batch.button)


def test_every_path_row_a_batch_fills_can_say_batch_run(panel):
    assert set(panel._haemolynx_batch.labels) == set(BATCH_ROW_FOLDERS)
    for label in panel._haemolynx_batch.labels.values():
        assert label.text() == "Batch run"
        assert label.isHidden()


def test_a_batch_shows_batch_run_in_its_rows_and_leaving_puts_them_back(panel, tmp_path):
    batch = panel._haemolynx_batch
    rows = panel._haemolynx_rows()
    before = dict(panel._haemolynx_values())
    setup = _setup(tmp_path, "A_seg", "B_seg")

    batch.use(setup)
    assert batch.button.text == "Batch run: 2 images"
    for name, label in batch.labels.items():
        assert not label.isHidden(), name
        assert rows[name].line_edit.native.isHidden(), name
        assert str(tmp_path / "seg") in label.toolTip() or name != "input_path"
    assert panel._haemolynx_values() == before, "a batch must not change the rows' values"

    batch.leave()
    assert batch.button.text == "Batch run"
    for name, label in batch.labels.items():
        assert label.isHidden(), name
        assert not rows[name].line_edit.native.isHidden(), name
    assert batch.mode.setup is None


def test_the_window_matches_the_folders_and_hands_the_batch_over(panel, tmp_path):
    batch = panel._haemolynx_batch
    _images(tmp_path / "seg", "A_seg", "B_seg")
    batch.open()
    window = batch.window[0]
    window.fields["segmented"].setText(str(tmp_path / "seg"))
    window.fields["output"].setText(str(tmp_path / "out"))
    assert window.match()
    assert window.table.rowCount() == 2
    assert window.table.item(0, 1).text() == "A_seg.tif"
    # Untick the first image: the batch is the second alone.
    from qtpy.QtCore import Qt

    window.table.item(0, 0).setCheckState(Qt.Unchecked)
    window.use()
    assert [match.stem for match in batch.mode.setup.selected()] == ["B_seg"]
    assert batch.button.text == "Batch run: 1 image"
    assert window.native.isHidden()


def test_a_tie_is_chosen_in_the_table_for_every_image(panel, tmp_path):
    batch = panel._haemolynx_batch
    _images(tmp_path / "seg", "E14_seg", "E15_seg")
    for name in ("C1-E14", "C2-E14", "C1-E15", "C2-E15"):
        shutil.copy(FIXTURE, tmp_path / "seg" / f"{name}.tif")
    batch.open()
    window = batch.window[0]
    window.fields["segmented"].setText(str(tmp_path / "seg"))
    window.fields["output"].setText(str(tmp_path / "out"))
    window.match()
    choice = window.table.cellWidget(0, 2)
    assert choice is not None
    choice.setCurrentIndex(1)  # C1-E14.tif
    assert [match.raw.name for match in window.setup.matches] == ["C1-E14.tif", "C1-E15.tif"]
    assert window.table.cellWidget(1, 2) is None


def test_the_window_says_why_it_cannot_match(panel, tmp_path):
    batch = panel._haemolynx_batch
    batch.open()
    window = batch.window[0]
    window.fields["segmented"].setText(str(tmp_path / "missing"))
    window.fields["output"].setText(str(tmp_path / "out"))
    assert not window.match()
    assert "is not a folder" in window.status.text()
    window.fields["output"].setText("")
    assert not window.match()
    assert "output folder" in window.status.text()


def test_run_pipeline_runs_the_batch_once_its_checks_pass(panel, tmp_path, monkeypatch):
    batch = panel._haemolynx_batch
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_batch_in_background",
        lambda values, schema, setup, *args, **kwargs: started.append((values, setup)),
    )
    batch.use(_setup(tmp_path, "A_seg", "B_seg"))
    panel._haemolynx_run()
    assert len(started) == 1
    values, setup = started[0]
    assert [match.stem for match in setup.selected()] == ["A_seg", "B_seg"]
    assert setup.checked == {"A_seg": "", "B_seg": ""}
    # The form's own values go to the batch; each image's files are its own.
    assert "input_path" in values


def test_run_pipeline_refuses_a_batch_an_image_of_which_fails_its_checks(panel, tmp_path, monkeypatch):
    batch = panel._haemolynx_batch
    started = []
    monkeypatch.setattr(
        widget_mod, "_run_batch_in_background", lambda *args, **kwargs: started.append(args)
    )
    rows = panel._haemolynx_rows()
    rows["inlet_node_selection_method"].value = "node_ids"
    batch.use(_setup(tmp_path, "A_seg"))
    panel._haemolynx_run()
    assert started == []
    assert "Checks failed for 1 image(s)" in panel._haemolynx_report()
