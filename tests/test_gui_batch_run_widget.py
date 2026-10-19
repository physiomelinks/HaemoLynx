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


def test_the_button_is_on_the_input_tab_above_the_input_row(panel):
    batch = panel._haemolynx_batch
    assert batch.button.text == "Batch run"
    assert batch.button.tooltip.strip()
    assert _on_input_tab(panel, batch.button)
    box = list(panel._haemolynx_input_boxes["segmented_input"])
    rows = panel._haemolynx_rows()
    assert box.index(batch.button) + 1 == box.index(rows["network_source"])


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


def test_the_window_takes_the_stylesheet_napari_gives_the_panel(panel):
    # napari sets its theme's stylesheet on its main window, the panel's
    # ancestor; a parentless window never got it and came up light.
    panel.setStyleSheet("QWidget { background-color: rgb(38, 41, 48); color: rgb(240, 241, 242); }")
    batch = panel._haemolynx_batch
    batch.open()
    window = batch.window[0]
    assert window.native.isWindow()
    for widget in (window.native, window.fields["segmented"]):
        widget.ensurePolished()
        palette = widget.palette()
        assert palette.color(widget.backgroundRole()).getRgb()[:3] == (38, 41, 48)
        assert palette.color(widget.foregroundRole()).getRgb()[:3] == (240, 241, 242)


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


def test_conditions_typed_in_the_window_class_each_image_and_compare(panel, tmp_path):
    from haemolynx.gui.batch_run import TABLE_COLUMNS
    from haemolynx.statistics import export_statistics_to_csv

    batch = panel._haemolynx_batch
    _images(tmp_path / "seg", "WT_1", "WT_2", "KO_1", "KO_2")
    batch.open()
    window = batch.window[0]
    window.fields["segmented"].setText(str(tmp_path / "seg"))
    window.fields["output"].setText(str(tmp_path / "out"))
    assert window.match()
    column = TABLE_COLUMNS.index("Condition")
    assert [window.table.item(row, column).text() for row in range(4)] == ["", "", "", ""]
    window.condition_fields["condition_a"].setText("WT")
    window.condition_fields["condition_b"].setText("KO")
    assert [window.table.item(row, column).text() for row in range(4)] == ["KO", "KO", "WT", "WT"]
    assert window.status.text() == "Conditions: 2 in WT, 2 in KO."

    # Compare: the statistics of the images already run in the output folder.
    for stem, length in (("WT_1", 1.0), ("WT_2", 2.0), ("KO_1", 5.0), ("KO_2", 6.0)):
        export_statistics_to_csv(
            {"Total Edge Length (microns)": length},
            tmp_path / "out" / stem / f"{stem}_statistics.csv",
        )
    window.compare()
    assert (tmp_path / "out" / "batch_condition_comparison.csv").is_file()
    assert "KO against WT" in window.status.text()

    window.use()
    assert batch.mode.setup.conditions.compares
    # Reopened, the window shows the batch's conditions.
    window.condition_fields["condition_b"].setText("")
    batch.open()
    assert window.condition_fields["condition_b"].text() == "KO"


def test_two_conditions_alike_are_not_used(panel, tmp_path):
    batch = panel._haemolynx_batch
    _images(tmp_path / "seg", "WT_1")
    batch.open()
    window = batch.window[0]
    window.fields["segmented"].setText(str(tmp_path / "seg"))
    window.fields["output"].setText(str(tmp_path / "out"))
    window.condition_fields["condition_a"].setText("WT")
    window.condition_fields["condition_b"].setText("wt")
    window.use()
    assert batch.mode.setup is None
    assert "Both conditions" in window.status.text()


def test_the_window_compares_conditions_paired_by_subject(panel, tmp_path):
    from haemolynx.gui.batch_run import TABLE_COLUMNS

    batch = panel._haemolynx_batch
    _images(tmp_path / "seg", "s1_contra", "s1_ipsi", "s2_contra")
    batch.open()
    window = batch.window[0]
    assert not window.subject_pattern.isEnabled()
    window.fields["segmented"].setText(str(tmp_path / "seg"))
    window.fields["output"].setText(str(tmp_path / "out"))
    assert window.match()
    window.condition_fields["condition_a"].setText("contra")
    window.condition_fields["condition_b"].setText("ipsi")
    window.design.setCurrentIndex(window.design.findData("paired"))
    assert window.subject_pattern.isEnabled()
    column = TABLE_COLUMNS.index("Condition")
    assert [window.table.item(row, column).text() for row in range(3)] == [
        "contra, subject s1",
        "ipsi, subject s1",
        "contra: unpaired, no ipsi image of subject s2",
    ]
    assert "1 pair(s) by subject (s1)" in window.status.text()
    window.use()
    assert batch.mode.setup.conditions.paired
    # Reopened, the window shows the batch's design.
    window.design.setCurrentIndex(window.design.findData("independent"))
    batch.open()
    assert window.design.currentData() == "paired"


def test_the_window_combines_a_subjects_images_by_their_median(panel, tmp_path):
    from haemolynx.gui.batch_run import TABLE_COLUMNS

    batch = panel._haemolynx_batch
    _images(tmp_path / "seg", "29_1_b_contra", "29_1_f_contra", "29_1_f_ipsi")
    batch.open()
    window = batch.window[0]
    assert not window.aggregate.isEnabled()
    window.fields["segmented"].setText(str(tmp_path / "seg"))
    window.fields["output"].setText(str(tmp_path / "out"))
    assert window.match()
    window.condition_fields["condition_a"].setText("contra")
    window.condition_fields["condition_b"].setText("ipsi")
    window.design.setCurrentIndex(window.design.findData("paired"))
    window.subject_pattern.setText(r"^(\d+_\d+)")
    assert window.aggregate.isEnabled()
    column = TABLE_COLUMNS.index("Condition")
    # One to one by name, 29_1_f with 29_1_f: 29_1_b is left out.
    assert [window.table.item(row, column).text() for row in range(3)] == [
        "contra: unpaired, no ipsi image of subject 29_1 left once the closest names were paired",
        "contra, subject 29_1",
        "ipsi, subject 29_1",
    ]
    window.aggregate.setCurrentIndex(window.aggregate.findData("median"))
    assert [window.table.item(row, column).text() for row in range(3)] == [
        "contra, subject 29_1 (median of 2)",
        "contra, subject 29_1 (median of 2)",
        "ipsi, subject 29_1",
    ]
    window.use()
    assert batch.mode.setup.conditions.aggregate == "median"
    window.aggregate.setCurrentIndex(window.aggregate.findData("none"))
    batch.open()
    assert window.aggregate.currentData() == "median"
