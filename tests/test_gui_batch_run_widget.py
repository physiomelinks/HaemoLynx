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
from haemolynx.io.batch_matching import IMAGE_FOLDER_ROLES  # noqa: E402

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
        # The other files' rows only while the batch has a folder of them.
        filled = BATCH_ROW_FOLDERS[name] in IMAGE_FOLDER_ROLES
        assert label.isHidden() is not filled, name
        assert rows[name].line_edit.native.isHidden() is filled, name
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


def test_sections_typed_in_the_window_are_left_out_of_compare_and_the_batch(panel, tmp_path):
    from haemolynx.statistics import export_statistics_to_csv

    batch = panel._haemolynx_batch
    _images(tmp_path / "seg", "WT_1", "WT_2", "KO_1", "KO_2")
    batch.open()
    window = batch.window[0]
    assert window.exclude_sections.toolTip().strip()
    window.fields["segmented"].setText(str(tmp_path / "seg"))
    window.fields["output"].setText(str(tmp_path / "out"))
    assert window.match()
    window.condition_fields["condition_a"].setText("WT")
    window.condition_fields["condition_b"].setText("KO")
    window.exclude_sections.setText("dead ends, ")
    for stem, length in (("WT_1", 1.0), ("WT_2", 2.0), ("KO_1", 5.0), ("KO_2", 6.0)):
        export_statistics_to_csv(
            {"Total Edge Length (microns)": length, "Dead Ends": {"Dead Ends": length}},
            tmp_path / "out" / stem / f"{stem}_statistics.csv",
        )
    window.compare()
    text = (tmp_path / "out" / "batch_condition_comparison.csv").read_text(encoding="utf-8")
    assert "Total Edge Length" in text and "Dead Ends" not in text
    assert "left out: dead ends (1 metric)" in window.status.text()

    window.use()
    assert batch.mode.setup.exclude_sections == ("dead ends",)
    # Reopened, the window shows the batch's sections.
    window.exclude_sections.setText("")
    batch.open()
    assert window.exclude_sections.text() == "dead ends"


# -- one image's controls while a batch is set --------------------------------


def _single_image_buttons(panel):
    return [
        panel._haemolynx_check_image_button,
        panel._haemolynx_optimise_button,
        panel._haemolynx_optimise_fwhm_button,
        panel._haemolynx_optimise_endothelial_button,
    ]


def test_a_batch_greys_what_acts_on_one_image_and_leaving_restores_it(panel, tmp_path):
    from haemolynx.gui.batch_run import SINGLE_IMAGE_IN_BATCH
    from haemolynx.gui.chrome_tooltips import REVERT_STAGE_TOOLTIP

    batch = panel._haemolynx_batch
    buttons = _single_image_buttons(panel)
    before = [(button.enabled, button.tooltip) for button in buttons]
    assert all(enabled for enabled, _tooltip in before)

    batch.use(_setup(tmp_path, "A_seg"))
    for button in buttons:
        assert not button.enabled
        assert button.tooltip == SINGLE_IMAGE_IN_BATCH
    for revert in panel._haemolynx_revert_buttons.values():
        assert not revert.enabled
        assert revert.tooltip == SINGLE_IMAGE_IN_BATCH
    assert panel._haemolynx_save_run_button.tooltip != SINGLE_IMAGE_IN_BATCH

    batch.leave()
    assert [(button.enabled, button.tooltip) for button in buttons] == before
    for revert in panel._haemolynx_revert_buttons.values():
        assert revert.tooltip == REVERT_STAGE_TOOLTIP


def test_one_images_actions_refuse_while_a_batch_is_set(panel, tmp_path, monkeypatch):
    """Greyed, but a script (or a button another run re-enabled) can still
    press them: each refuses, and nothing starts."""
    from haemolynx.gui.batch_run import SINGLE_IMAGE_IN_BATCH

    started = []
    for name in (
        "_run_segmentation_quality_check_in_background",
        "_run_optimisation_in_background",
        "_run_fwhm_optimisation_in_background",
        "_run_endothelial_optimisation_in_background",
        "_run_in_background",
        "_run_batch_in_background",
    ):
        monkeypatch.setattr(widget_mod, name, lambda *args, **kwargs: started.append(args))
    panel._haemolynx_batch.use(_setup(tmp_path, "A_seg"))
    for button in [*_single_image_buttons(panel), *panel._haemolynx_revert_buttons.values()]:
        panel._haemolynx_run_checks()  # the report says something else first
        assert panel._haemolynx_report() != SINGLE_IMAGE_IN_BATCH
        button.changed.emit(True)
        assert panel._haemolynx_report() == SINGLE_IMAGE_IN_BATCH
    assert started == []


def test_run_checks_with_a_batch_checks_every_ticked_image(panel, tmp_path):
    panel._haemolynx_batch.use(_setup(tmp_path, "A_seg", "B_seg"))
    panel._haemolynx_run_checks()
    report = panel._haemolynx_report()
    assert "ok: A_seg" in report and "ok: B_seg" in report
    assert report.endswith("All 2 ticked image(s) passed their checks.")
    assert panel._haemolynx_batch.mode.setup.checked == {"A_seg": "", "B_seg": ""}


# -- the window during and after a batch --------------------------------------


def test_the_window_follows_the_batch_as_it_runs(panel, tmp_path, monkeypatch):
    from haemolynx.gui.batch_run import DONE, RUNNING, BatchResult, BatchSummary

    batch = panel._haemolynx_batch
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_batch_in_background",
        lambda *args, **kwargs: started.append(kwargs),
    )
    setup = _setup(tmp_path, "A_seg", "B_seg")
    batch.open()
    window = batch.window[0]
    window.fields["segmented"].setText(str(setup.folders.segmented))
    window.fields["output"].setText(str(setup.output_root))
    assert window.match()
    window.use()
    panel._haemolynx_run()
    (callbacks,) = started
    assert batch.mode.running

    first, second = batch.mode.setup.matches
    column = window.table.columnCount() - 1
    callbacks["on_item"](BatchResult(first, status=RUNNING))
    assert window.table.item(0, column).text() == RUNNING
    assert batch.mode.setup.checked == {"A_seg": RUNNING}
    callbacks["on_item"](BatchResult(first, status=DONE))
    callbacks["on_item"](BatchResult(second, status=RUNNING))
    assert [window.table.item(row, column).text() for row in (0, 1)] == [DONE, RUNNING]

    summary = BatchSummary(setup.output_root, [BatchResult(first, status=DONE), BatchResult(second, status=DONE)])
    callbacks["on_done"](summary)
    callbacks["on_stopped"]()
    assert not batch.mode.running
    assert [window.table.item(row, column).text() for row in (0, 1)] == [DONE, DONE]


def test_a_batch_left_while_it_ran_is_not_set_again_when_it_ends(panel, tmp_path, monkeypatch):
    from haemolynx.gui.batch_run import BatchSummary

    batch = panel._haemolynx_batch
    started = []
    monkeypatch.setattr(
        widget_mod, "_run_batch_in_background", lambda *args, **kwargs: started.append(kwargs)
    )
    batch.use(_setup(tmp_path, "A_seg"))
    panel._haemolynx_run()
    batch.leave()
    started[0]["on_done"](BatchSummary(tmp_path / "out", []))
    assert batch.mode.setup is None
    assert batch.button.text == "Batch run"


def test_the_window_says_which_images_were_done_before(panel, tmp_path):
    from haemolynx.gui.batch_run import DONE_BEFORE_RERUN, DONE_BEFORE_SKIPPED, READY, haemorun_path

    batch = panel._haemolynx_batch
    setup = _setup(tmp_path, "A_seg", "B_seg")
    done = haemorun_path(setup.output_root, setup.matches[0])
    done.parent.mkdir(parents=True)
    done.write_bytes(b"earlier")
    batch.open()
    window = batch.window[0]
    window.fields["segmented"].setText(str(setup.folders.segmented))
    window.fields["output"].setText(str(setup.output_root))
    assert window.match()
    column = window.table.columnCount() - 1
    assert [window.table.item(row, column).text() for row in (0, 1)] == [DONE_BEFORE_SKIPPED, READY]
    window.skip_done.setChecked(False)
    assert window.table.item(0, column).text() == DONE_BEFORE_RERUN
    assert window.setup.skip_done is False


# -- Load run and a batch -----------------------------------------------------


def _saved_run(tmp_path: Path) -> Path:
    from test_gui_results import a_graph, built, network

    from haemolynx.gui.results import StageLayers
    from haemolynx.gui.run_snapshot import capture_run, write_run_snapshot
    from haemolynx.gui.stage_checkpoints import StageCheckpoints

    checkpoints = StageCheckpoints()
    results = built()
    settings = {"input_path": tmp_path / "stack.tif", "vtk_output_prefix": tmp_path / "out" / "stack"}
    (tmp_path / "out").mkdir(exist_ok=True)
    results.stage_finished("build_network", network(a_graph(resistance=4.0)))
    checkpoints.record(
        "build_network",
        StageLayers(stage="build_network", title="3. Graph", note="done"),
        results,
        settings=settings,
    )
    snapshot = capture_run(checkpoints=checkpoints, results=results, settings=settings)
    return write_run_snapshot(tmp_path / "one.haemorun", snapshot)


def test_load_run_is_refused_while_a_batch_runs(panel, tmp_path, monkeypatch):
    """Loading a run stops the live one: an overnight batch would be lost to
    a double-click on a finished image."""
    from haemolynx.gui.batch_run import BATCH_RUNNING_NO_LOAD

    batch = panel._haemolynx_batch
    monkeypatch.setattr(widget_mod, "_run_batch_in_background", lambda *args, **kwargs: None)
    batch.use(_setup(tmp_path, "A_seg"))
    panel._haemolynx_run()
    assert batch.mode.running
    cancelled = []
    monkeypatch.setattr(panel._haemolynx_run_state, "cancel", lambda: cancelled.append(True))
    assert panel._haemolynx_load_run(_saved_run(tmp_path)) is False
    assert panel._haemolynx_report() == BATCH_RUNNING_NO_LOAD
    assert cancelled == []
    assert batch.mode.setup is not None


def test_loading_a_run_leaves_the_batch_set(panel, tmp_path):
    from haemolynx.gui.batch_run import LEFT_BATCH_FOR_RUN

    batch = panel._haemolynx_batch
    batch.use(_setup(tmp_path, "A_seg"))
    assert panel._haemolynx_load_run(_saved_run(tmp_path)) is True
    assert batch.mode.setup is None
    assert batch.button.text == "Batch run"
    assert LEFT_BATCH_FOR_RUN in panel._haemolynx_report()
    assert all(button.enabled for button in _single_image_buttons(panel))
    for label in batch.labels.values():
        assert label.isHidden()


# -- Phases 2-4: running from the window, the row actions, the comparison ------


def _open_window(panel, tmp_path, *stems):
    """The window matched on a folder of *stems*, output in ``tmp_path/out``."""
    batch = panel._haemolynx_batch
    _images(tmp_path / "seg", *stems)
    batch.open()
    window = batch.window[0]
    window.fields["segmented"].setText(str(tmp_path / "seg"))
    window.fields["output"].setText(str(tmp_path / "out"))
    window.fields["output"].editingFinished.emit()  # typed, then left: matched
    return batch, window


def _status_column(window):
    return window.table.columnCount() - 1


def _statuses(window):
    return [window.table.item(row, _status_column(window)).text() for row in range(window.table.rowCount())]


def test_leaving_a_folder_field_matches_by_itself(panel, tmp_path):
    _batch, window = _open_window(panel, tmp_path, "A_seg", "B_seg")
    assert window.table.rowCount() == 2
    assert window.match_button.text() == "Match again"


def test_run_batch_in_the_window_runs_it_and_keeps_the_window_open(panel, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(widget_mod, "_run_batch_in_background", lambda *args, **kwargs: started.append(kwargs))
    batch, window = _open_window(panel, tmp_path, "A_seg", "B_seg")
    window.run()
    (callbacks,) = started
    assert batch.mode.setup is not None and batch.mode.running
    assert not window.native.isHidden()
    assert not window.run_button.isEnabled() and window.stop_button.isEnabled()
    window.run()  # pressed again while it runs: nothing more starts
    assert len(started) == 1
    callbacks["on_stopped"]()
    assert window.run_button.isEnabled() and not window.stop_button.isEnabled()


def test_stop_after_this_image_asks_the_running_batch_to_stop(panel, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(widget_mod, "_run_batch_in_background", lambda *args, **kwargs: started.append(kwargs))
    batch, window = _open_window(panel, tmp_path, "A_seg")
    window.run()
    stop_requested = started[0]["stop_requested"]
    assert stop_requested() is False
    window.stop()
    assert stop_requested() is True
    assert not window.stop_button.isEnabled()
    assert window.stop_button.text() == "Stopping after this image"
    assert "Stopping once this image is done" in panel._haemolynx_report()
    started[0]["on_stopped"]()
    assert batch.mode.stop is False
    assert window.stop_button.text() == "Stop after this image"


def test_the_progress_line_and_figures_follow_the_batch(panel, tmp_path, monkeypatch):
    from haemolynx.gui.batch_run import DONE, RUNNING, TABLE_COLUMNS, BatchResult

    started = []
    monkeypatch.setattr(widget_mod, "_run_batch_in_background", lambda *args, **kwargs: started.append(kwargs))
    batch, window = _open_window(panel, tmp_path, "A_seg", "B_seg")
    window.run()
    first, second = batch.mode.setup.matches
    started[0]["on_item"](BatchResult(first, status=RUNNING))
    assert window.status.text().startswith("Image 1 of 2: A_seg -- 0 done, 0 failed")
    assert panel._haemolynx_report() == window.status.text()
    started[0]["on_item"](BatchResult(first, status=DONE, seconds=125.0, vessels=42, peak_rss_bytes=3 * 1024**3))
    started[0]["on_item"](BatchResult(second, status=RUNNING))
    assert window.status.text().startswith("Image 2 of 2: B_seg -- 1 done")
    row = [window.table.item(0, column).text() for column in range(window.table.columnCount())]
    assert row[TABLE_COLUMNS.index("Time")] == "2 min"
    assert row[TABLE_COLUMNS.index("Peak RAM")] == "3.0 GB"
    assert row[TABLE_COLUMNS.index("Vessels")] == "42"


def test_run_again_runs_the_selected_image_alone(panel, tmp_path, monkeypatch):
    from haemolynx.gui.batch_run import DONE, haemorun_path

    started = []
    monkeypatch.setattr(widget_mod, "_run_batch_in_background", lambda values, schema, setup, *args, **kwargs: started.append(setup))
    batch, window = _open_window(panel, tmp_path, "A_seg", "B_seg")
    done = haemorun_path(window.setup.output_root, window.setup.matches[1])
    done.parent.mkdir(parents=True)
    done.write_bytes(b"earlier")
    window.match()
    window.table.selectRow(1)
    assert window.row_buttons["run_again"].isEnabled()
    window.row_action("run_again")
    (setup,) = started
    assert [match.stem for match in setup.selected()] == ["B_seg"]
    assert setup.skip_done is False
    # The rest of the batch is still the batch, with A's status kept.
    assert [match.stem for match in batch.mode.setup.selected()] == ["A_seg", "B_seg"]
    assert DONE not in _statuses(window)[0]


def test_each_row_action_is_offered_only_where_it_can_act(panel, tmp_path, monkeypatch):
    from dataclasses import replace

    from haemolynx.gui.batch_run import haemorun_path

    opened, loaded = [], []
    monkeypatch.setattr(widget_mod, "_open_in_system", lambda path: opened.append(Path(path)) or True)
    batch, window = _open_window(panel, tmp_path, "A_seg", "B_seg")
    monkeypatch.setattr(window, "_on_open", lambda path: loaded.append(path))
    window.table.selectRow(0)
    assert {key: button.isEnabled() for key, button in window.row_buttons.items()} == {
        "open_run": False, "open_folder": False, "show_error": False, "run_again": True,
    }
    done = haemorun_path(window.setup.output_root, window.setup.matches[0])
    done.parent.mkdir(parents=True)
    done.write_bytes(b"run")
    window.setup = replace(
        window.setup,
        checked={"A_seg": "failed: ValueError: no"},
        error_details={"A_seg": "Traceback (most recent call last):\n  ValueError: no"},
    )
    window._fill_table()
    window.table.selectRow(0)
    assert all(button.isEnabled() for button in window.row_buttons.values())
    window.row_action("open_folder")
    assert opened == [done.parent]
    window.row_action("open_run")
    assert loaded == [done]
    window.row_action("show_error")
    assert "Traceback (most recent call last)" in window.details.toPlainText()
    assert window.splitter.sizes()[1] > 0, "the report box opens to show it"


def test_tick_all_none_and_failed(panel, tmp_path):
    from dataclasses import replace

    _batch, window = _open_window(panel, tmp_path, "A_seg", "B_seg", "C_seg")
    window.tick("none")
    assert window.setup.selected() == []
    window.tick("all")
    assert len(window.setup.selected()) == 3
    window.setup = replace(window.setup, checked={"B_seg": "failed: ValueError: no"}, last_statuses={"C_seg": "failed: x"})
    window.tick("failed")
    assert [match.stem for match in window.setup.selected()] == ["B_seg", "C_seg"]


def test_sorting_the_table_keeps_each_rows_image(panel, tmp_path):
    from qtpy.QtCore import Qt

    _batch, window = _open_window(panel, tmp_path, "A_seg", "B_seg", "C_seg")
    window.table.sortItems(1, Qt.DescendingOrder)
    assert window.table.item(0, 1).text() == "C_seg.tif"
    window.table.item(0, 0).setCheckState(Qt.Unchecked)
    assert window.setup.unticked == {"C_seg"}
    window._fill_table()  # filled again: the order the user chose stays
    assert window.table.item(0, 1).text() == "C_seg.tif"


def test_the_comparison_tab_shows_the_csv_and_opens_it(panel, tmp_path, monkeypatch):
    from haemolynx.statistics.batch_comparison import COMPARISON_FILENAME

    opened = []
    monkeypatch.setattr(widget_mod, "_open_in_system", lambda path: opened.append(Path(path)) or True)
    _batch, window = _open_window(panel, tmp_path, "A_seg")
    assert window.comparison_table.rowCount() == 0
    assert not window.open_comparison_button.isEnabled()
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    (out / COMPARISON_FILENAME).write_text(
        "Section,Metric,Unit,N WT,N KO,Median WT,Median KO,\"P (Mann-Whitney U, two-sided)\","
        "Q (Benjamini-Hochberg),Note\n"
        "Core,Length,um,3,3,10,12,0.2,0.4,\nCore,Nodes,count,3,3,5,9,0.05,0.1,\n",
        encoding="utf-8",
    )
    window.refresh_comparison()
    assert window.comparison_table.rowCount() == 2
    assert window.comparison_table.item(0, 1).text() == "Nodes"
    assert window.comparison_note.text().startswith("2 of 2 statistics tested")
    window.open_comparison()
    assert opened == [out / COMPARISON_FILENAME]


def test_the_folders_last_batch_fills_the_window_and_the_panel(panel, tmp_path):
    from haemolynx.gui.batch_run import write_batch_settings
    from haemolynx.pipeline import default_schema
    from haemolynx.statistics.batch_comparison import BatchConditions

    schema = default_schema()
    _images(tmp_path / "seg", "WT_1_seg", "KO_1_seg")
    out = tmp_path / "out"
    out.mkdir()
    values = {**panel._haemolynx_values(), "statistics": True}
    write_batch_settings(
        values, schema, BatchFolders(tmp_path / "seg"), out, BatchConditions.from_text("WT", "KO"), ("Dead Ends",)
    )
    rows = panel._haemolynx_rows()
    assert rows["statistics"].value is False
    batch = panel._haemolynx_batch
    batch.open()
    window = batch.window[0]
    window.fields["output"].setText(str(out))
    window.fields["output"].editingFinished.emit()
    assert not window.last_batch_button.isHidden()
    assert window.use_last_batch()
    assert window.fields["segmented"].text() == str(tmp_path / "seg")
    assert window.condition_fields["condition_a"].text() == "WT"
    assert window.exclude_sections.text() == "Dead Ends"
    assert rows["statistics"].value is True, "the settings it ran with are the panel's again"
    assert window.table.rowCount() == 2


def test_a_greyed_mask_row_names_its_switch(panel, tmp_path):
    _batch, window = _open_window(panel, tmp_path, "A_seg")
    assert not window.fields["large"].isEnabled()
    assert "'Use large vessel masks'" in window.fields["large"].placeholderText()


def test_the_settings_line_counts_what_differs_from_the_defaults(panel, tmp_path):
    rows = panel._haemolynx_rows()
    rows["statistics"].value = True
    _batch, window = _open_window(panel, tmp_path, "A_seg")
    assert "differ from the defaults" in window.settings_line.text()
    window.show_settings()
    assert "statistics: True" in window.details.toPlainText()


# -- the button: greyed where no batch can start ------------------------------


def test_the_batch_run_button_greys_under_a_synthetic_network(panel, tmp_path):
    from haemolynx.gui.chrome_tooltips import BATCH_RUN_TOOLTIPS

    batch = panel._haemolynx_batch
    rows = panel._haemolynx_rows()
    assert batch.button.enabled
    rows["network_source"].value = "synthetic"
    assert not batch.button.enabled
    assert batch.button.tooltip == BATCH_RUN_TOOLTIPS["button_synthetic"]
    rows["network_source"].value = "segmented_image"
    assert batch.button.enabled
    assert batch.button.tooltip == BATCH_RUN_TOOLTIPS["button"]
    # A batch already set stays in reach, to be left, whatever the input.
    batch.use(_setup(tmp_path, "A_seg"))
    rows["network_source"].value = "synthetic"
    assert batch.button.enabled


def test_the_batch_run_button_greys_while_one_image_runs_but_not_a_batch(panel, tmp_path, monkeypatch):
    from haemolynx.gui.chrome_tooltips import BATCH_RUN_TOOLTIPS

    batch = panel._haemolynx_batch
    state = panel._haemolynx_run_state
    state.start(worker=None)
    assert not batch.button.enabled
    assert batch.button.tooltip == BATCH_RUN_TOOLTIPS["button_running"]
    state.stopped()
    assert batch.button.enabled
    state.start(worker=None)
    state.supersede()  # Clear, mid-run: the panel is free at once
    assert batch.button.enabled
    # A batch running: the button opens its window, to follow and stop it.
    monkeypatch.setattr(
        widget_mod, "_run_batch_in_background", lambda *args, **kwargs: state.start(worker=None)
    )
    batch.use(_setup(tmp_path, "A_seg"))
    panel._haemolynx_run()
    assert batch.mode.running and state.running
    assert batch.button.enabled

# -- the other files beside each image, and a batch ilastik segments ----------


def test_the_window_shows_a_row_for_each_other_file_the_settings_read(panel, tmp_path):
    from haemolynx.gui.batch_run import TABLE_COLUMNS

    rows = panel._haemolynx_rows()
    batch, window = _open_window(panel, tmp_path, "A_seg", "B_seg")
    column = TABLE_COLUMNS.index("Other files")
    assert window.fields["cell"].isHidden()
    assert window.table.isColumnHidden(column)
    rows["measurement_3d_to_cell_mask"].value = True
    cells = _images(tmp_path / "cells", "A_cells", "B_cells")
    batch.open()
    assert not window.fields["cell"].isHidden() and not window.fields["vessel_3d"].isHidden()
    assert window.fields["pericyte"].isHidden() and window.fields["tissue"].isHidden()
    window.fields["cell"].setText(str(cells))
    window.fields["cell"].editingFinished.emit()
    assert not window.table.isColumnHidden(column)
    assert [window.table.item(row, column).text() for row in (0, 1)] == [
        "cell mask: A_cells.tif",
        "cell mask: B_cells.tif",
    ]
    window.use()
    assert batch.mode.setup.folders.cell == cells
    # Its row says so; a file with no folder keeps the panel's own row.
    assert not batch.labels["cell_mask_path"].isHidden()
    assert batch.labels["measurement_3d_vessel_mask_path"].isHidden()


def test_with_ilastik_on_the_window_lists_the_raw_folders_images(panel, tmp_path):
    rows = panel._haemolynx_rows()
    raw = _images(tmp_path / "raw", "E14", "E15")
    # The panel's one raw image row is empty: what reads it is greyed.
    assert not rows["use_ilastik_segmentation"].enabled
    rows["use_ilastik_segmentation"].value = True
    batch = panel._haemolynx_batch
    batch.open()
    window = batch.window[0]
    assert not window.fields["segmented"].isEnabled()
    assert "ilastik" in window.fields["segmented"].placeholderText()
    window.fields["raw"].setText(str(raw))
    window.fields["output"].setText(str(tmp_path / "out"))
    window.fields["output"].editingFinished.emit()
    assert window.table.rowCount() == 2
    assert window.table.item(0, 1).text() == "by ilastik: E14_segmented"
    assert "2 raw image(s), each segmented by ilastik," in window.status.text()
    window.use()
    assert batch.mode.setup.folders.from_raw
    # Each image has a raw image: what reads one is live again.
    assert rows["use_ilastik_segmentation"].enabled
    assert rows["measure_diameters_from_raw_image"].enabled
    assert "Made for each image by ilastik" in batch.labels["input_path"].toolTip()
    batch.leave()
    assert not rows["use_ilastik_segmentation"].enabled