"""The log window, hooked to a real run in a real viewer.

`test_gui_run_log.py` pins the buffer and the capture without a GUI, and
`test_gui_log_view.py` pins the widget. What is left is the wiring, and it is
the part with the sharp edges -- all of them about *when* the handler comes off
the logger.

A run can end three ways, and only one callback sees all three. superqt
suppresses `returned` once `quit()` has been called, so a cancelled-but-
completed run reaches neither the success handler nor the failure one; the
worker's `finished` signal is the only one that fires every time. A detach put
anywhere else survives every test that runs a pipeline to completion and leaks
on the one path a user reaches by pressing a button -- and the leaked handler
then writes every line of the *next* run into the window twice.

So: an end-to-end proof that a record logged on the worker thread reaches the
widget, and then four tests about the logger being left exactly as it was found
-- after a success, after two runs, after a cancellation, and with nothing
arriving after the cancel banner.
"""
from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import (  # noqa: E402
    LOG_DOCK_NAME,
    _run_in_background,
    settings_widget,
)
from haemolynx.gui.log_view import LogView  # noqa: E402
from haemolynx.gui.results import ResultLayers  # noqa: E402
from haemolynx.gui.run_log import LEVELS, LOGGER_NAME, installed_handlers  # noqa: E402
from haemolynx.gui.run_state import RunState  # noqa: E402
from haemolynx.pipeline.progress import STAGE_STARTED, ProgressEvent  # noqa: E402
from test_gui_results import a_graph  # noqa: E402
from test_gui_results_widget import paused_run, viewer  # noqa: E402,F401

pytestmark = pytest.mark.gui

PRUNE_LOGGER = "haemolynx.graph.prune"


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
def log(qtbot):
    view = LogView()
    qtbot.addWidget(view.native)
    return view


def shown(view: LogView) -> str:
    return view.view.document().toPlainText()


def a_run_that_logs(*messages: str, events: tuple = ()):
    """A fake pipeline that logs *messages* from whichever thread runs it."""
    seen = SimpleNamespace(thread=None, graph=a_graph())

    def fake_run(settings, schema, progress=None, on_stage_output=None):
        seen.thread = threading.current_thread()
        for event in events:
            if progress is not None:
                progress(event)
        for message in messages:
            logging.getLogger(PRUNE_LOGGER).info(message)
        return seen.graph

    return fake_run, seen


def start(log_view, state=None, viewer=None, results=None):
    """`_run_in_background` with the arguments a panel would give it."""
    report = SimpleNamespace(value="")
    button = SimpleNamespace(enabled=True)
    _run_in_background(
        {}, None, report, button, None,
        viewer=viewer,
        results=results,
        state=state if state is not None else RunState(),
        log=log_view,
    )
    return report, button


# --- the end-to-end proof ----------------------------------------------------


def test_a_record_logged_by_the_run_reaches_the_window(
    qtbot, monkeypatch, log, library_logger
) -> None:
    """The whole feature, in one test: the pipeline talks, the window shows it."""
    from haemolynx.gui import _widget

    fake_run, seen = a_run_that_logs("Pruning complete: removed 2 terminal stubs")
    monkeypatch.setattr(_widget, "run_pipeline_stages", fake_run)

    _report, button = start(log)

    qtbot.waitUntil(
        lambda: "removed 2 terminal stubs" in shown(log), timeout=5000
    )
    qtbot.waitUntil(lambda: button.enabled, timeout=5000)
    assert seen.thread is not threading.main_thread(), (
        "the point of the exercise: those records are emitted off the GUI thread"
    )


def test_the_stage_banner_arrives_before_the_counts_it_belongs_to(
    qtbot, monkeypatch, log, library_logger
) -> None:
    """One producer, one order.

    The banner becomes a record on the run's own thread, in `watched`, rather
    than beside the progress bars on the GUI thread -- so a stage's heading and
    the counts logged inside that stage cannot be interleaved by whenever the
    GUI thread next got a slice.
    """
    from haemolynx.gui import _widget

    fake_run, _seen = a_run_that_logs(
        "Pruning complete: removed 2 terminal stubs",
        events=(
            ProgressEvent(
                kind=STAGE_STARTED, stage="build_network", title="3. Graph",
                index=2, total=8,
            ),
        ),
    )
    monkeypatch.setattr(_widget, "run_pipeline_stages", fake_run)

    _report, button = start(log)
    qtbot.waitUntil(lambda: "removed 2 terminal stubs" in shown(log), timeout=5000)
    qtbot.waitUntil(lambda: button.enabled, timeout=5000)

    text = shown(log)
    assert "3. Graph" in text
    assert text.index("3. Graph") < text.index("removed 2 terminal stubs")


# --- the logger is left as it was found --------------------------------------


def test_a_finished_run_leaves_the_logger_as_it_found_it(
    qtbot, monkeypatch, log, library_logger
) -> None:
    from haemolynx.gui import _widget

    fake_run, _seen = a_run_that_logs("Pruning complete: removed 2")
    monkeypatch.setattr(_widget, "run_pipeline_stages", fake_run)
    before_level = library_logger.level
    before_handlers = list(library_logger.handlers)

    _report, button = start(log)
    qtbot.waitUntil(lambda: button.enabled, timeout=5000)
    qtbot.waitUntil(lambda: installed_handlers(library_logger) == (), timeout=5000)

    assert library_logger.level == before_level
    assert library_logger.handlers == before_handlers


def test_two_runs_say_each_thing_once(
    qtbot, monkeypatch, log, library_logger
) -> None:
    """What a leaked handler looks like from the user's chair.

    A handler left on the logger by the first run is still there for the
    second, and the second run's `attach` adds another: every line arrives
    twice, in one window, with no clue why.
    """
    from haemolynx.gui import _widget

    for message in ("line from the first run", "line from the second run"):
        fake_run, _seen = a_run_that_logs(message)
        monkeypatch.setattr(_widget, "run_pipeline_stages", fake_run)
        _report, button = start(log)
        qtbot.waitUntil(lambda: message in shown(log), timeout=5000)
        qtbot.waitUntil(lambda: button.enabled, timeout=5000)
        qtbot.waitUntil(lambda: installed_handlers(library_logger) == (), timeout=5000)

    text = shown(log)
    assert text.count("line from the first run") == 1
    assert text.count("line from the second run") == 1
    # And the first run is still there to compare against: the log accumulates.
    assert text.count("Run started") == 2


# --- the path the success and failure handlers never see ---------------------


def test_a_cancelled_run_still_takes_the_handler_off(
    qtbot, log, library_logger, viewer, paused_run  # noqa: F811
) -> None:
    """The reason the detach sits above `stopped`'s early return.

    superqt suppresses `returned` after `quit()`, and a cancellation is not
    reported as a failure, so neither of the two handlers that announce the end
    of a run runs at all here. `finished` is the only one left.
    """
    before_level = library_logger.level
    before_handlers = list(library_logger.handlers)
    state = RunState()

    _report, _button = start(log, state=state, viewer=viewer, results=ResultLayers())
    qtbot.waitUntil(paused_run.drawn.is_set, timeout=5000)
    assert installed_handlers(library_logger) != (), "it should be capturing by now"

    assert state.cancel() is True
    log.cancelled()  # what `on_clear` does
    paused_run.resume.set()

    qtbot.waitUntil(lambda: not state.running, timeout=5000)
    qtbot.waitUntil(lambda: installed_handlers(library_logger) == (), timeout=5000)
    assert library_logger.level == before_level
    assert library_logger.handlers == before_handlers


def test_nothing_arrives_after_the_cancel_banner(
    qtbot, log, library_logger, viewer, paused_run  # noqa: F811
) -> None:
    """The banner is the last word on a stopped run, and the log keeps the rest.

    What the run said before it was stopped is still true -- often it is why
    the user stopped it -- so a cancel marks the log rather than clearing it.
    """
    state = RunState()
    _report, _button = start(log, state=state, viewer=viewer, results=ResultLayers())
    qtbot.waitUntil(paused_run.drawn.is_set, timeout=5000)
    logging.getLogger(PRUNE_LOGGER).info("said while the run was going")

    state.cancel()
    log.cancelled()
    paused_run.resume.set()
    qtbot.waitUntil(lambda: installed_handlers(library_logger) == (), timeout=5000)

    logging.getLogger(PRUNE_LOGGER).info("said after the cancel")
    log.drain()

    text = shown(log)
    assert "said while the run was going" in text
    assert "said after the cancel" not in text
    assert "cancelled" in text.splitlines()[-1]


# --- what the panel does with it ---------------------------------------------


def test_the_panel_docks_the_log_in_the_viewer(make_napari_viewer) -> None:
    viewer = make_napari_viewer()

    panel = settings_widget(napari_viewer=viewer)

    assert isinstance(panel._haemolynx_log, LogView)
    dock = panel._haemolynx_log_dock
    assert dock is not None
    assert LOG_DOCK_NAME in dock.windowTitle()
    assert panel._haemolynx_log.native.parent() is not None


def test_the_panel_still_builds_with_no_viewer_to_dock_it_in() -> None:
    """There is no window to dock a log into, and that must not be an error."""
    panel = settings_widget(napari_viewer=None)

    assert panel is not None
    assert isinstance(panel._haemolynx_log, LogView)
    assert panel._haemolynx_log_dock is None


def test_starting_a_run_brings_the_log_window_back(
    make_napari_viewer, qtbot, monkeypatch, library_logger
) -> None:
    """The user's choice: a run opens its log, even one they had closed."""
    from haemolynx.gui import _widget

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    dock = panel._haemolynx_log_dock
    dock.setVisible(False)
    # `isHidden`, not `isVisible`: nothing is visible in a test, because the
    # main window this is docked in was never shown. What is being asked is
    # whether the dock is still closed, which is its own flag.
    assert dock.isHidden() is True

    monkeypatch.setattr(_widget, "preflight", lambda *a, **k: SimpleNamespace(ok=True))
    fake_run, _seen = a_run_that_logs("Pruning complete: removed 2")
    monkeypatch.setattr(_widget, "run_pipeline_stages", fake_run)

    panel._haemolynx_run()

    assert dock.isHidden() is False
    qtbot.waitUntil(lambda: panel._haemolynx_run_button.enabled, timeout=10000)


def test_the_window_starts_at_the_level_the_settings_asked_for(
    make_napari_viewer, qtbot, monkeypatch, library_logger
) -> None:
    """`verbose_logging` already means "tell me everything"; no new setting."""
    from haemolynx.gui import _widget

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    monkeypatch.setattr(_widget, "preflight", lambda *a, **k: SimpleNamespace(ok=True))
    fake_run, _seen = a_run_that_logs("Pruning complete: removed 2")
    monkeypatch.setattr(_widget, "run_pipeline_stages", fake_run)

    panel._haemolynx_run()
    qtbot.waitUntil(lambda: panel._haemolynx_run_button.enabled, timeout=10000)
    assert panel._haemolynx_log.level == LEVELS["Info"]

    panel._haemolynx_rows()["verbose_logging"].value = True
    panel._haemolynx_run()
    qtbot.waitUntil(lambda: panel._haemolynx_run_button.enabled, timeout=10000)

    assert panel._haemolynx_log.level == LEVELS["Debug"]
