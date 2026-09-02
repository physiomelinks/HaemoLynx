"""The lines a live log window shows, and what bounds them.

A run narrates itself at a rate no text box can keep up with, so
`haemolynx.gui.run_log` is a bounded buffer between the run's thread and the
GUI's, and every bound it claims is pinned here: how many lines may wait, how
many one drain hands over, how long the scrollback is, and what it does when it
has to throw a line away.

Like `haemolynx.gui.progress` and `haemolynx.gui.run_state`, none of it needs
napari, a Qt binding or a display -- `logging` is in the standard library -- so
these tests run on every Python the library supports rather than only where the
GUI is installed.
"""
from __future__ import annotations

import ast
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from haemolynx.gui.run_log import (
    DEFAULT_LEVEL,
    DRAIN_INTERVAL_MS,
    LEVELS,
    LOGGER_NAME,
    MAX_LINES,
    MAX_PENDING,
    MAX_PER_DRAIN,
    RUN_CANCELLED,
    RUN_STARTED,
    Drained,
    RunLog,
    banner,
    dropped_note,
    format_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def a_record(
    message: str,
    *args,
    level: int = logging.INFO,
    name: str = "haemolynx.graph.prune",
) -> logging.LogRecord:
    """A record as the library emits them: lazy `%` args, module-named logger."""
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=args,
        exc_info=None,
    )


# --- what a record looks like ------------------------------------------------


def test_a_record_reads_as_its_level_its_source_and_its_message() -> None:
    (line,) = format_record(
        a_record("Pruning complete: removed %d terminal stubs", 2)
    )

    assert "INFO" in line
    assert "graph.prune" in line
    assert "Pruning complete: removed 2 terminal stubs" in line


def test_the_package_prefix_is_not_repeated_on_every_line() -> None:
    """It is on every record, so it is width spent saying nothing."""
    (line,) = format_record(a_record("done", name="haemolynx.graph.prune"))

    assert "graph.prune" in line
    assert "haemolynx" not in line


def test_a_logger_outside_the_package_keeps_its_whole_name() -> None:
    (line,) = format_record(a_record("hello", name="skan.csr"))

    assert "skan.csr" in line


def test_a_warning_and_a_debug_record_say_which_they_are() -> None:
    (warning,) = format_record(a_record("careful", level=logging.WARNING))
    (debug,) = format_record(a_record("per node", level=logging.DEBUG))

    assert "WARNING" in warning
    assert "DEBUG" in debug
    assert "careful" in warning
    assert "per node" in debug


def test_the_message_arguments_are_resolved_not_left_as_a_template() -> None:
    """`%`-style args are how the library logs, so the buffer must render them."""
    (line,) = format_record(a_record("%d nodes / %d edges", 12, 15))

    assert "12 nodes / 15 edges" in line
    assert "%d" not in line


def test_a_multi_line_message_becomes_that_many_lines() -> None:
    """A diagnostics report is one record and many lines; a bound counts lines."""
    lines = format_record(a_record("first\nsecond\nthird"))

    assert len(lines) == 3
    assert "first" in lines[0]
    assert lines[1].strip() == "second"
    assert lines[2].strip() == "third"


def test_a_record_with_nothing_to_say_is_still_one_line() -> None:
    assert len(format_record(a_record(""))) == 1


# --- the banners -------------------------------------------------------------


def test_a_run_opens_with_a_banner_carrying_the_time() -> None:
    line = banner(RUN_STARTED, when=datetime(2026, 9, 2, 14, 5, 9))

    assert "14:05:09" in line
    assert "Run started" in line


def test_a_cancelled_run_says_so_rather_than_just_stopping() -> None:
    line = banner(RUN_CANCELLED, when=datetime(2026, 9, 2, 14, 5, 9))

    assert "cancelled" in line
    assert "14:05:09" in line


def test_the_dropped_note_says_how_many_are_missing() -> None:
    note = dropped_note(37)

    assert "37" in note


# --- the level a window is set to --------------------------------------------


def test_the_three_choices_are_the_levels_they_name() -> None:
    assert LEVELS == {
        "Warnings only": logging.WARNING,
        "Info": logging.INFO,
        "Debug": logging.DEBUG,
    }
    assert DEFAULT_LEVEL in LEVELS
    assert LEVELS[DEFAULT_LEVEL] == logging.INFO


def test_a_record_below_the_level_is_not_stored() -> None:
    log = RunLog(level=logging.INFO)

    log.add_record(a_record("per node", level=logging.DEBUG))
    log.add_record(a_record("a pass finished", level=logging.INFO))

    assert log.drain().lines == format_record(a_record("a pass finished"))


def test_raising_the_level_shows_what_was_being_dropped() -> None:
    log = RunLog(level=logging.WARNING)
    log.add_record(a_record("a pass finished", level=logging.INFO))
    assert log.drain().lines == ()

    log.level = logging.INFO
    log.add_record(a_record("a pass finished", level=logging.INFO))

    assert len(log.drain().lines) == 1


def test_a_warning_is_kept_at_every_choice() -> None:
    for level in LEVELS.values():
        log = RunLog(level=level)
        log.add_record(a_record("careful", level=logging.WARNING))
        assert len(log.drain().lines) == 1


# --- the bound on what is waiting --------------------------------------------


def test_the_pending_buffer_never_grows_past_its_bound() -> None:
    log = RunLog()

    for index in range(MAX_PENDING + 250):
        log.add_line(f"line {index}")

    assert log.pending == MAX_PENDING


def test_overflow_drops_the_oldest_and_counts_exactly_what_it_dropped() -> None:
    """The newest lines are the ones a live window wants; the count is the honesty."""
    log = RunLog(max_pending=10, max_per_drain=10)

    for index in range(13):
        log.add_line(f"line {index}")

    assert log.dropped == 3
    drained = log.drain()
    assert drained.lines[0] == "line 3"
    assert drained.lines[-1] == "line 12"
    assert drained.dropped == 3


def test_a_multi_line_record_does_not_slip_past_the_bound() -> None:
    log = RunLog(max_pending=10, max_per_drain=10)

    for _ in range(5):
        log.add_record(a_record("one\ntwo\nthree"))

    assert log.pending == 10
    assert log.dropped == 5


# --- draining ----------------------------------------------------------------


def test_a_drain_hands_over_at_most_its_share_and_takes_the_newest() -> None:
    log = RunLog()

    for index in range(MAX_PER_DRAIN + 120):
        log.add_line(f"line {index}")
    drained = log.drain()

    assert len(drained.lines) == MAX_PER_DRAIN
    assert drained.lines[-1] == f"line {MAX_PER_DRAIN + 119}"
    assert drained.lines[0] == "line 120"
    assert drained.dropped == 120


def test_a_drain_empties_what_it_returned() -> None:
    log = RunLog()
    log.add_line("only line")

    assert log.drain().lines == ("only line",)
    assert log.drain().lines == ()
    assert log.pending == 0


def test_the_dropped_count_is_reported_once_not_on_every_drain() -> None:
    log = RunLog(max_pending=5, max_per_drain=5)
    for index in range(8):
        log.add_line(f"line {index}")

    assert log.drain().dropped == 3
    assert log.drain().dropped == 0
    assert log.dropped == 3, "the total is still there to read"


# --- the scrollback ----------------------------------------------------------


def test_the_scrollback_holds_what_has_been_drained_in_order() -> None:
    log = RunLog()
    for index in range(3):
        log.add_line(f"line {index}")
    log.drain()

    assert log.text() == "line 0\nline 1\nline 2"


def test_the_scrollback_is_bounded_and_keeps_the_most_recent() -> None:
    log = RunLog()
    written = 0
    while written < MAX_LINES + MAX_PER_DRAIN:
        for _ in range(MAX_PER_DRAIN):
            log.add_line(f"line {written}")
            written += 1
        log.drain()

    lines = log.text().splitlines()
    assert len(lines) == MAX_LINES
    assert lines[-1] == f"line {written - 1}"


def test_the_log_is_not_cleared_between_runs() -> None:
    """Comparing one run's counts with the next is the reason for the window."""
    log = RunLog()
    log.add_line(banner(RUN_STARTED))
    log.add_record(a_record("Pruning complete: removed %d terminal stubs", 2))
    log.drain()

    log.add_line(banner(RUN_STARTED))
    log.add_record(a_record("Pruning complete: removed %d terminal stubs", 5))
    log.drain()

    text = log.text()
    assert "removed 2 terminal stubs" in text
    assert "removed 5 terminal stubs" in text
    assert text.count("Run started") == 2


def test_reset_forgets_everything_a_clear_button_should() -> None:
    log = RunLog(max_pending=5, max_per_drain=5)
    for index in range(8):
        log.add_line(f"line {index}")
    log.drain()
    log.add_line("still waiting")

    log.reset()

    assert log.pending == 0
    assert log.text() == ""
    assert log.dropped == 0
    assert log.drain() == Drained(lines=(), dropped=0)


# --- the timer the window runs on --------------------------------------------


def test_the_drain_interval_is_live_but_not_a_busy_loop() -> None:
    assert 20 <= DRAIN_INTERVAL_MS <= 250


def test_the_logger_captured_is_the_one_every_module_reports_through() -> None:
    assert LOGGER_NAME == "haemolynx"
    assert logging.getLogger("haemolynx.graph.prune").name.startswith(LOGGER_NAME)


# --- it must not need a GUI --------------------------------------------------


def test_the_module_imports_no_gui() -> None:
    """The library must import on a machine with no napari and no Qt."""
    probe = (
        "import sys; import haemolynx.gui.run_log; "
        "print([m for m in sys.modules if m.split('.')[0] in "
        "{'napari', 'magicgui', 'qtpy', 'PyQt6', 'PyQt5', 'PySide6'}])"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout


def test_no_napari_import_appears_in_the_source() -> None:
    source = REPO_ROOT / "src" / "haemolynx" / "gui" / "run_log.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"napari", "magicgui", "qtpy"}
