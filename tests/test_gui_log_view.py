"""The log window, built for real, with a real Qt event loop behind it.

What a log window *shows* is `haemolynx.gui.run_log`, and
`tests/test_gui_run_log.py` pins all of that without a GUI. What is left is
everything that only exists once Qt is holding it: the block-count cap that
keeps a two-hour run in a fixed amount of memory, autoscroll that does not
fight a reader who has scrolled up, a per-tick cap so one burst of DEBUG cannot
stall the timer, and the three buttons.

These need a Qt binding and a display, so they are marked `gui` and skipped
where those are missing, like `test_gui_widget.py`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("qtpy")

from haemolynx.gui.log_view import (  # noqa: E402
    DEFAULT_FILENAME,
    DROPPED_FORMAT,
    LogView,
)
from haemolynx.gui.run_log import (  # noqa: E402
    DRAIN_INTERVAL_MS,
    LEVELS,
    MAX_LINES,
    MAX_PENDING,
    MAX_PER_DRAIN,
    RunLog,
    attach,
)

pytestmark = pytest.mark.gui


@pytest.fixture
def view(qtbot):
    """A log window, on screen for the length of one test."""
    view = LogView()
    qtbot.addWidget(view.native)
    view.native.resize(600, 120)
    return view


def a_record(
    message: str,
    *args,
    level: int = logging.INFO,
    name: str = "haemolynx.graph.prune",
) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=args,
        exc_info=None,
    )


def push(view: LogView, count: int, first: int = 0) -> int:
    """Put *count* lines through the buffer and onto the widget, in drainable
    batches; returns the number of the next line."""
    written = first
    while written < first + count:
        for _ in range(min(MAX_PER_DRAIN, first + count - written)):
            view.run_log.add_line(f"line {written}")
            written += 1
        view.drain()
    return written


# --- it builds at all --------------------------------------------------------


def test_the_window_builds_with_its_controls(view) -> None:
    assert view.native is not None
    assert view.view.isReadOnly()
    assert [view.level_combo.itemText(i) for i in range(view.level_combo.count())] == list(
        LEVELS
    )
    assert view.follow is True, "a log nobody has touched should follow the run"


def test_the_text_box_does_not_wrap(view) -> None:
    """A wrapped 300-character statistics line is four rows of nothing."""
    from qtpy.QtWidgets import QPlainTextEdit

    assert view.view.lineWrapMode() == QPlainTextEdit.NoWrap


def test_the_timer_runs_at_the_interval_the_buffer_names(view) -> None:
    assert view._timer.interval() == DRAIN_INTERVAL_MS
    assert view.running is False

    view.start()
    assert view.running is True

    view.stop()
    assert view.running is False


# --- the bound on the widget -------------------------------------------------


def test_three_thousand_lines_stay_inside_the_block_cap(view) -> None:
    push(view, 3000)

    assert view.view.document().blockCount() <= MAX_LINES
    assert view.view.document().blockCount() == 3000


def test_past_the_cap_qt_drops_the_oldest_and_keeps_the_newest(view) -> None:
    """The cap is the point: a run of any length costs the same memory."""
    written = push(view, MAX_LINES + 1000)

    document = view.view.document()
    assert document.blockCount() == MAX_LINES
    assert document.toPlainText().splitlines()[-1] == f"line {written - 1}"
    assert document.toPlainText().splitlines()[0] == f"line {written - MAX_LINES}"


def test_the_document_cap_is_qts_own_and_not_python_side(view) -> None:
    assert view.view.document().maximumBlockCount() == MAX_LINES


# --- following, and not fighting the reader ----------------------------------


def test_with_follow_on_the_newest_line_is_in_view(view) -> None:
    push(view, 400)

    bar = view.view.verticalScrollBar()
    assert bar.maximum() > 0, "the fixture must be big enough to scroll"
    assert bar.value() == bar.maximum()


def test_a_reader_who_scrolled_up_is_left_where_they_were(view) -> None:
    """The bug this avoids: being yanked to the bottom every 100 ms."""
    push(view, 400)
    bar = view.view.verticalScrollBar()
    bar.setValue(0)

    push(view, 400, first=400)

    assert bar.value() == 0


def test_follow_switched_off_leaves_the_view_alone(view) -> None:
    push(view, 400)
    bar = view.view.verticalScrollBar()
    view.follow = False
    bar.setValue(0)

    push(view, 400, first=400)

    assert view.follow is False
    assert bar.value() == 0


def test_scrolling_back_to_the_bottom_starts_following_again(view) -> None:
    """Position, not a mode: the checkbox is for holding still on purpose."""
    push(view, 400)
    bar = view.view.verticalScrollBar()
    bar.setValue(0)
    push(view, 400, first=400)
    assert bar.value() == 0

    bar.setValue(bar.maximum())
    written = push(view, 400, first=800)

    assert bar.value() == bar.maximum()
    assert view.view.document().toPlainText().splitlines()[-1] == f"line {written - 1}"


# --- one tick's worth --------------------------------------------------------


def test_a_huge_backlog_appends_one_batch_and_says_what_it_lost(view) -> None:
    """A DEBUG burst must cost one tick, and must not go quietly."""
    for index in range(10_000):
        view.run_log.add_line(f"line {index}")

    appended = view.drain()

    lines = view.view.document().toPlainText().splitlines()
    assert len(lines) == MAX_PER_DRAIN + 1, "the batch, plus one line owning up to it"
    assert appended == MAX_PER_DRAIN + 1
    notes = [line for line in lines if "dropped" in line]
    assert len(notes) == 1
    # 10,000 in, MAX_PENDING held, MAX_PER_DRAIN shown: the rest is the count.
    assert notes[0] == DROPPED_FORMAT.format(
        count=10_000 - MAX_PER_DRAIN, level="Info"
    )
    assert lines[-1] == "line 9999", "and what survived is the newest"


def test_the_dropped_note_names_the_level_that_caused_it(view) -> None:
    view.level_combo.setCurrentText("Debug")
    for index in range(MAX_PENDING + 600):
        view.run_log.add_line(f"line {index}")

    view.drain()

    (note,) = [
        line
        for line in view.view.document().toPlainText().splitlines()
        if "dropped" in line
    ]
    assert "log level Debug" in note


def test_a_tick_with_nothing_waiting_appends_nothing(view) -> None:
    push(view, 10)
    before = view.view.document().toPlainText()

    assert view.drain() == 0
    assert view.view.document().toPlainText() == before


def test_the_timer_really_drains_on_its_own(view, qtbot) -> None:
    view.start()
    view.run_log.add_record(a_record("Pruning complete: removed %d terminal stubs", 2))

    qtbot.waitUntil(
        lambda: "removed 2 terminal stubs" in view.view.document().toPlainText(),
        timeout=5000,
    )
    view.stop()


# --- the level combo ---------------------------------------------------------


def test_changing_the_level_changes_what_is_kept_next(view) -> None:
    view.run_log.add_record(a_record("a pass finished", level=logging.INFO))
    assert view.drain() == 1

    view.level_combo.setCurrentText("Warnings only")
    view.run_log.add_record(a_record("a pass finished", level=logging.INFO))
    assert view.drain() == 0

    view.run_log.add_record(a_record("careful", level=logging.WARNING))
    assert view.drain() == 1


def test_turning_the_level_up_mid_run_starts_showing_debug(view) -> None:
    view.level_combo.setCurrentText("Debug")

    view.run_log.add_record(a_record("per node", level=logging.DEBUG))

    assert view.level == logging.DEBUG
    assert view.drain() == 1
    assert "per node" in view.view.document().toPlainText()


def test_the_level_is_what_a_capture_should_be_attached_at(view) -> None:
    """`attach` and the combo have to agree, or the window shows nothing."""
    view.level_combo.setCurrentText("Info")
    with attach(view.run_log, level=view.level):
        logging.getLogger("haemolynx.graph.prune").info("Pruning complete: 2")
        logging.getLogger("haemolynx.graph.prune").debug("per node")

    assert view.drain() == 1
    assert "Pruning complete: 2" in view.view.document().toPlainText()


def test_lines_already_shown_are_not_taken_away_by_the_level(view) -> None:
    view.run_log.add_record(a_record("per node", level=logging.DEBUG))
    view.level_combo.setCurrentText("Debug")
    view.run_log.add_record(a_record("and another", level=logging.DEBUG))
    view.drain()

    view.level_combo.setCurrentText("Warnings only")

    assert "and another" in view.view.document().toPlainText()


# --- the buttons -------------------------------------------------------------


def test_copy_puts_the_whole_buffer_on_the_clipboard(view) -> None:
    from qtpy.QtWidgets import QApplication

    clipboard = QApplication.clipboard()
    was = clipboard.text()
    try:
        push(view, 5)

        view.copy_button.click()

        assert clipboard.text() == view.run_log.text()
        assert clipboard.text().splitlines()[-1] == "line 4"
    finally:
        clipboard.setText(was)


def test_copy_takes_the_buffer_and_not_what_the_widget_shows(view) -> None:
    """The widget has been trimmed to the last MAX_LINES blocks; the log has not."""
    from qtpy.QtWidgets import QApplication

    clipboard = QApplication.clipboard()
    was = clipboard.text()
    try:
        push(view, 20)
        view.view.clear()

        view.copy()

        assert clipboard.text() == view.run_log.text()
        assert "line 0" in clipboard.text()
    finally:
        clipboard.setText(was)


def test_save_writes_the_log_where_it_is_told(view, tmp_path, monkeypatch) -> None:
    from qtpy.QtWidgets import QFileDialog

    target = tmp_path / "somewhere-else.txt"
    asked: list[tuple] = []

    def fake_dialog(parent, caption, name, filter_):
        asked.append((caption, name, filter_))
        return str(target), filter_

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake_dialog)
    push(view, 5)

    written = view.save()

    assert written == str(target)
    assert target.read_text(encoding="utf-8") == view.run_log.text()
    assert asked[0][1] == DEFAULT_FILENAME


def test_save_cancelled_writes_nothing(view, tmp_path, monkeypatch) -> None:
    from qtpy.QtWidgets import QFileDialog

    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args: ("", "")
    )
    push(view, 5)

    assert view.save() is None
    assert list(tmp_path.iterdir()) == []


def test_clear_empties_the_window_and_the_buffer(view) -> None:
    push(view, 20)
    assert view.run_log.text() != ""

    view.clear_button.click()

    assert view.view.document().toPlainText() == ""
    assert view.run_log.text() == ""
    assert view.run_log.pending == 0


# --- the run's banners -------------------------------------------------------


def test_a_run_opens_with_a_banner_and_keeps_the_one_before_it(view) -> None:
    view.start()
    view.run_log.add_record(a_record("Pruning complete: removed %d", 2))
    view.stop()

    view.start()
    view.run_log.add_record(a_record("Pruning complete: removed %d", 5))
    view.stop()

    shown = view.view.document().toPlainText()
    assert shown.count("Run started") == 2
    assert "removed 2" in shown and "removed 5" in shown


def test_a_cancelled_run_says_so_in_the_log(view) -> None:
    view.start()
    view.cancelled()
    view.stop()

    assert "cancelled" in view.view.document().toPlainText()


def test_a_note_reaches_the_window_as_it_stands(view) -> None:
    view.note("Failed: no such file")
    view.drain()

    assert "Failed: no such file" in view.view.document().toPlainText()


def test_stopping_shows_the_last_lines_a_run_wrote(view) -> None:
    """They are written in the milliseconds before the run returns."""
    view.start()
    view.run_log.add_line("the very last thing")

    view.stop()

    assert "the very last thing" in view.view.document().toPlainText()


# --- sharing a buffer with a capture -----------------------------------------


def test_a_view_can_be_handed_the_buffer_a_run_is_writing_to(qtbot) -> None:
    log = RunLog(level=logging.INFO)
    view = LogView(log)
    qtbot.addWidget(view.native)

    assert view.run_log is log
    with attach(log, level=view.level):
        logging.getLogger("haemolynx.graph.prune").info("Step 7/11: 12 nodes")
    view.drain()

    assert "Step 7/11: 12 nodes" in view.view.document().toPlainText()


def test_the_module_needs_no_qt_to_import() -> None:
    """The panel's own promise: importing the package costs nothing without a GUI."""
    import ast

    source = Path(__file__).resolve().parents[1] / "src" / "haemolynx" / "gui" / "log_view.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    top_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    named = set()
    for node in top_level:
        if isinstance(node, ast.Import):
            named.update(alias.name.split(".")[0] for alias in node.names)
        elif node.module:
            named.add(node.module.split(".")[0])
    assert not named & {"qtpy", "napari", "magicgui", "PyQt6", "PyQt5", "PySide6"}
