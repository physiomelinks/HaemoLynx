"""The lines a live log window shows, what bounds them, and how they arrive.

Two facts made this module necessary. A run narrates itself at a rate no text
box can keep up with, so `haemolynx.gui.run_log` is a bounded buffer between
the run's thread and the GUI's, and every bound it claims is pinned here: how
many lines may wait, how many one drain hands over, how long the scrollback is,
and what it does when it has to throw a line away.

And in a napari session no HaemoLynx log record reaches anywhere at all: the
root logger sits at WARNING and every `logger.info` in the library is dropped by
the level check before a handler is consulted. Adding a handler captures
nothing; the level has to come down too. That is
`test_attaching_lowers_the_level_so_info_is_captured_at_all`, and it is the one
test here that stands for the whole feature.

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
import threading
from datetime import datetime
from pathlib import Path

import pytest

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
    Attachment,
    Drained,
    RunLog,
    RunLogHandler,
    attach,
    banner,
    dropped_note,
    format_record,
    installed_handlers,
)
from haemolynx.pipeline.progress import (
    STAGE_STARTED,
    STEP,
    ProgressEvent,
    log_progress,
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


# --- capturing the library's log for the length of a run ---------------------


@pytest.fixture
def library_logger():
    """The `haemolynx` logger, put back exactly however a test leaves it."""
    logger = logging.getLogger(LOGGER_NAME)
    level = logger.level
    handlers = list(logger.handlers)
    propagate = logger.propagate
    try:
        yield logger
    finally:
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate


@pytest.fixture
def root_at_warning():
    """Root as a napari session leaves it: WARNING, and nobody's handler on it."""
    root = logging.getLogger()
    level = root.level
    handlers = list(root.handlers)
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    try:
        yield root
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def test_a_record_from_the_library_lands_in_the_buffer(library_logger) -> None:
    log = RunLog()
    with attach(log):
        logging.getLogger("haemolynx.graph.prune").info(
            "Pruning complete: removed %d terminal stubs", 2
        )

    (line,) = log.drain().lines
    assert "graph.prune" in line
    assert "removed 2 terminal stubs" in line


def test_a_record_from_another_package_does_not(library_logger) -> None:
    """Which is why this attaches to `haemolynx` and not to root."""
    log = RunLog()
    with attach(log):
        logging.getLogger("some.other.package").warning("not ours")

    assert log.drain().lines == ()


def test_attaching_lowers_the_level_so_info_is_captured_at_all(
    library_logger, root_at_warning
) -> None:
    """The fact the whole feature rests on.

    In a napari session nothing has configured logging: the library's logger
    has no level of its own, so its effective level is root's WARNING and every
    `logger.info` in the pipeline is dropped before a handler is consulted.
    Adding a handler and stopping there captures nothing whatsoever.
    """
    library_logger.setLevel(logging.NOTSET)
    assert library_logger.getEffectiveLevel() == logging.WARNING
    assert not library_logger.isEnabledFor(logging.INFO)

    log = RunLog(level=logging.INFO)
    with attach(log):
        assert library_logger.level == logging.INFO
        assert library_logger.isEnabledFor(logging.INFO)
        logging.getLogger("haemolynx.graph.prune").info("Pruning complete: 2")

    assert len(log.drain().lines) == 1


def test_attaching_leaves_the_root_logger_alone(library_logger, root_at_warning) -> None:
    """Lowering root's level would pull in matplotlib, PIL and napari as well."""
    log = RunLog()
    # Whatever is on root is the test runner's, and has to still be there.
    before = list(root_at_warning.handlers)
    with attach(log):
        assert root_at_warning.level == logging.WARNING
        assert root_at_warning.handlers == before
        logging.getLogger("matplotlib.font_manager").info("findfont: score 0.05")

    assert log.drain().lines == ()


def test_detaching_restores_the_exact_level_and_handler_list(library_logger) -> None:
    someone_elses = logging.NullHandler()
    library_logger.addHandler(someone_elses)
    library_logger.setLevel(logging.CRITICAL)
    before_level = library_logger.level
    before_handlers = list(library_logger.handlers)

    attachment = attach(RunLog(), level=logging.DEBUG)
    assert library_logger.level == logging.DEBUG
    attachment.detach()

    assert library_logger.level == before_level
    assert library_logger.handlers == before_handlers
    assert installed_handlers(library_logger) == ()


def test_a_detached_run_no_longer_fills_its_buffer(library_logger) -> None:
    log = RunLog()
    attach(log).detach()

    logging.getLogger("haemolynx.graph.prune").info("after the run")

    assert log.drain().lines == ()


def test_attaching_twice_installs_one_handler(library_logger) -> None:
    """A missed detach must cost one leaked handler, never a growing pile."""
    log = RunLog()
    attach(log)
    attachment = attach(log)

    assert len(installed_handlers(library_logger)) == 1
    assert installed_handlers(library_logger) == (attachment.handler,)

    logging.getLogger("haemolynx.graph.prune").info("said once")
    assert len(log.drain().lines) == 1
    attachment.detach()


def test_a_leaked_handler_is_not_put_back_by_a_later_detach(library_logger) -> None:
    """The snapshot is taken after the sweep, or a detach would undo it."""
    attach(RunLog())  # leaked: nothing detaches it
    second = attach(RunLog())

    second.detach()

    assert installed_handlers(library_logger) == ()


def test_detaching_twice_is_a_no_op(library_logger) -> None:
    library_logger.setLevel(logging.CRITICAL)
    attachment = attach(RunLog(), level=logging.DEBUG)

    attachment.detach()
    library_logger.setLevel(logging.INFO)  # as a later run would
    attachment.detach()

    assert attachment.detached is True
    assert library_logger.level == logging.INFO, "the second detach restored a stale level"


def test_the_console_still_gets_everything_the_buffer_gets(library_logger) -> None:
    """This is an additional handler: propagation is left alone on purpose."""
    seen: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record)

    root = logging.getLogger()
    console = Collector()
    root.addHandler(console)
    try:
        log = RunLog()
        with attach(log):
            assert library_logger.propagate is True
            logging.getLogger("haemolynx.graph.prune").info("said to both")
    finally:
        root.removeHandler(console)

    assert [r.getMessage() for r in seen] == ["said to both"]
    assert len(log.drain().lines) == 1


def test_records_from_four_threads_all_arrive(library_logger) -> None:
    """Records come from the worker thread and from graph building's pool."""
    per_thread = 200
    log = RunLog(max_pending=4000, max_per_drain=4000)

    def talk(index: int) -> None:
        logger = logging.getLogger(f"haemolynx.graph.worker{index}")
        for line in range(per_thread):
            logger.info("thread %d line %d", index, line)

    with attach(log):
        threads = [threading.Thread(target=talk, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    drained = log.drain()
    assert len(drained.lines) == 4 * per_thread
    assert drained.dropped == 0
    for index in range(4):
        said = [line for line in drained.lines if f"thread {index} " in line]
        assert len(said) == per_thread


def test_a_record_that_cannot_be_formatted_does_not_break_the_run(
    library_logger,
) -> None:
    """A log line is the least important thing happening on that thread."""
    library_logger.propagate = False  # keep the bad record off pytest's handlers
    log = RunLog()
    attachment = attach(log)
    failures: list[logging.LogRecord] = []
    attachment.handler.handleError = failures.append  # type: ignore[method-assign]
    try:
        logging.getLogger("haemolynx.graph.prune").info("%d nodes", "not a number")
    finally:
        attachment.detach()

    assert len(failures) == 1
    assert log.drain().lines == ()


def test_the_handler_holds_no_record_and_so_no_graph(library_logger) -> None:
    """The buffer holds strings; a LogRecord would keep its args -- a graph -- alive."""
    log = RunLog()
    with attach(log):
        logging.getLogger("haemolynx.graph.prune").info("%s", {"a": "graph"})

    lines = log.drain().lines
    assert all(isinstance(line, str) for line in lines)


# --- what the pipeline's own progress consumer puts in the window ------------


def a_stage_event(kind: str, **extra) -> ProgressEvent:
    return ProgressEvent(
        kind=kind, stage="build_network", title="3. Graph", index=2, total=8, **extra
    )


def test_a_stage_banner_reaches_the_buffer(library_logger) -> None:
    log = RunLog(level=logging.INFO)
    with attach(log):
        log_progress(a_stage_event(STAGE_STARTED))

    (line,) = log.drain().lines
    assert "3. Graph" in line
    assert "3/8" in line


def test_a_topology_step_is_not_shown_at_info(library_logger) -> None:
    """Eleven per graph build, and DEBUG: `Info` is a bounded amount of output."""
    log = RunLog(level=logging.INFO)
    with attach(log):
        log_progress(
            a_stage_event(STEP, step="prune_vascular_stubs", step_index=6, step_total=11)
        )

    assert log.drain().lines == ()


def test_a_topology_step_is_shown_at_debug(library_logger) -> None:
    log = RunLog()
    with attach(log, level=logging.DEBUG):
        log_progress(
            a_stage_event(STEP, step="prune_vascular_stubs", step_index=6, step_total=11)
        )

    (line,) = log.drain().lines
    assert "prune_vascular_stubs" in line


def test_an_attachment_is_usable_as_a_context_manager(library_logger) -> None:
    with attach(RunLog()) as attachment:
        assert isinstance(attachment, Attachment)
        assert isinstance(attachment.handler, RunLogHandler)
    assert attachment.detached is True


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
