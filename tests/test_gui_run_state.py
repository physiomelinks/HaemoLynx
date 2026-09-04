"""Stopping a run on purpose, and getting the panel back, without a GUI.

The bug behind this module: pressing "Clear layers" while a run was going
pulled the layers out from under the run, and left the panel unusable. The Run
button was greyed out by the run that started it and re-enabled only by that
run's own success and failure handlers, so nothing ever gave it back; the bars
sat part-way; and the worker was a local variable nobody held, so there was no
handle to cancel with. Closing the plugin and opening it again was the only way
to start another run.

`haemolynx.gui.run_state` is the whole decision -- whether a run is going, what
stops it, and what is put back when one does -- so it is tested here, on every
Python the library supports, rather than only where napari, a Qt binding and a
display are installed. The widget tests that match this one are in
`test_gui_results_widget.py`, marked `gui`.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from haemolynx.gui.progress import BarState, ProgressDisplay
from haemolynx.gui.run_state import (
    ALREADY_RUNNING,
    CANCELLED,
    FINISHED_FIRST,
    RunCancelled,
    RunState,
    clear_message,
)
from test_gui_results import built

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeWorker:
    """Stands in for napari's thread worker: all the panel asks of it is `quit`."""

    def __init__(self) -> None:
        self.quits = 0

    def quit(self) -> None:
        self.quits += 1


class DeadWorker:
    """A worker whose Qt object has already gone, which is what that looks like."""

    def quit(self) -> None:
        raise RuntimeError("wrapped C/C++ object has been deleted")


# --- the guard ---------------------------------------------------------------


def test_nothing_is_running_before_a_run_starts():
    state = RunState()
    assert state.running is False
    assert state.cancelled is False
    assert state.worker is None


def test_a_run_that_has_started_is_running():
    state = RunState()
    worker = FakeWorker()

    state.start(worker=worker, results=built())

    assert state.running is True
    assert state.worker is worker


def test_a_run_that_ended_leaves_the_panel_free():
    state = RunState()
    state.start(worker=FakeWorker())

    state.stopped()

    assert state.running is False
    assert state.worker is None


# --- the regression: a cancelled run must let the next one start -------------


def test_a_run_cancelled_part_way_lets_a_new_one_start():
    """The user-visible bug, in the object that now decides it.

    Clear the layers mid-run and the panel has to be ready for another run
    immediately -- with no restart.
    """
    state = RunState(bars=ProgressDisplay())
    state.bars.start()
    first_flag = state.start(worker=FakeWorker(), results=built())

    assert state.cancel() is True
    state.supersede()
    assert state.running is False
    with pytest.raises(RunCancelled):
        state.check(first_flag)

    second_flag = state.start(worker=FakeWorker(), results=built())

    assert state.running is True
    assert state.cancelled is False
    assert second_flag is not first_flag
    assert first_flag["cancelled"] is True
    state.check(second_flag)  # new run is not cancelled


def test_a_cancelled_run_stops_where_it_next_reports():
    """It really stops: the stages after the cancel do not happen at all."""
    state = RunState()
    state.start(worker=FakeWorker(), results=built())
    reached = []

    def a_run() -> None:
        for stage in ("skeletonise", "build_network", "assign_boundaries", "solve"):
            state.check()
            reached.append(stage)
            if stage == "build_network":
                state.cancel()  # the user presses "Clear layers"

    with pytest.raises(RunCancelled):
        a_run()

    assert reached == ["skeletonise", "build_network"]


def test_a_run_nobody_stopped_runs_past_every_checkpoint():
    state = RunState()
    state.start(worker=FakeWorker(), results=built())

    for _ in range(20):
        state.check()

    assert state.running is True


def test_cancelling_asks_the_worker_to_quit():
    """napari's own abort flag, so anything watching the worker agrees."""
    state = RunState()
    worker = FakeWorker()
    state.start(worker=worker)

    state.cancel()

    assert worker.quits == 1


def test_a_worker_that_cannot_be_quit_does_not_stop_the_cancel():
    """The flag is what actually stops the run; `quit` is the courtesy."""
    state = RunState()
    state.start(worker=DeadWorker())

    assert state.cancel() is True
    with pytest.raises(RunCancelled):
        state.check()


# --- what a cancel puts back -------------------------------------------------


def test_a_cancelled_run_forgets_the_graph_it_had_drawn():
    """`ResultLayers` remembers the graph across stages, so it must be reset.

    Left behind, it is what the next run's first stages would be drawn
    against: a boundary stage is node ids, looked up in the wrong network.
    """
    results = built()
    assert results.colour_options()  # it is holding a graph
    state = RunState()
    state.start(worker=FakeWorker(), results=results)

    state.cancel()

    assert results.colour_options() == []
    assert results.emitted == ()


def test_cancelling_puts_the_progress_bars_back_to_nothing():
    bars = ProgressDisplay()
    bars.start()
    state = RunState(bars=bars)
    state.start(worker=FakeWorker())

    state.cancel()

    assert bars.stages == BarState()
    assert bars.steps == BarState()
    assert not bars.stages.visible


def test_a_stopped_run_still_has_its_straggling_events_ignored():
    """Events emitted just before the cancel are still crossing threads."""
    state = RunState()
    state.start(worker=FakeWorker())
    state.cancel()

    state.stopped()

    assert state.cancelled is True


# --- clearing with nothing going is unchanged --------------------------------


def test_cancelling_with_no_run_in_progress_does_nothing():
    state = RunState(bars=ProgressDisplay())
    state.bars.start()
    before = state.bars.stages

    assert state.cancel() is False
    assert state.cancelled is False
    assert state.bars.stages == before
    state.check()  # nothing to raise about


def test_clearing_with_nothing_going_reports_only_what_it_removed():
    assert clear_message(4, stopping=False) == "Removed 4 HaemoLynx layer(s)."


def test_clearing_with_a_run_going_says_the_run_is_stopping():
    message = clear_message(4, stopping=True)
    assert message.startswith("Removed 4 HaemoLynx layer(s).")
    assert "Stopping the run" in message


def test_clearing_can_note_discarded_artefacts_and_restored_skips():
    message = clear_message(
        2,
        stopping=False,
        discarded_artefacts=True,
        restored_skips=True,
    )
    assert "Removed 2 HaemoLynx layer(s)." in message
    assert "Discarded cached checkpoint and resume pickles." in message
    assert "Restored skeletonize and graph-building toggles." in message


# --- a cancellation is not a failure -----------------------------------------


def test_a_cancellation_reads_as_one_and_not_as_an_error():
    """What the user sees has to tell an intention from a fault."""
    assert "ancel" in CANCELLED
    for word in ("Failed", "failed", "Error", "error", "Traceback"):
        assert word not in CANCELLED


def test_the_messages_a_run_can_end_with_are_all_different():
    assert len({CANCELLED, FINISHED_FIRST, ALREADY_RUNNING}) == 3


def test_a_run_that_beat_the_cancel_is_not_called_cancelled():
    """It finished; saying otherwise about a completed run would be a lie."""
    assert "finished" in FINISHED_FIRST


# --- it must not need a GUI --------------------------------------------------


def test_the_module_imports_no_gui():
    """The library must import on a machine with no napari and no Qt."""
    probe = (
        "import sys; import haemolynx.gui.run_state; "
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


def test_no_napari_import_appears_in_the_source():
    source = REPO_ROOT / "src" / "haemolynx" / "gui" / "run_state.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"napari", "magicgui", "qtpy"}
