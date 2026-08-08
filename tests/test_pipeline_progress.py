"""How a run says where it has got to.

Progress is the one thing a caller watches that must not change what the run
does: a missing event is a bar that stalls, a duplicated one is a bar that
jumps, and an exception raised while reporting would fail a run that worked.
So the tests here drive `run_pipeline_stages` with every stage stubbed out --
the reporting is what is under test, not the pipeline -- and pin the order, the
counts and what happens when a stage raises.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pytest

from haemolynx.graph import STEP_LABELS
from haemolynx.pipeline import stages
from haemolynx.pipeline.progress import (
    KINDS,
    STAGE_FAILED,
    STAGE_FINISHED,
    STAGE_STARTED,
    STAGES,
    STEP,
    ProgressEvent,
    RunProgress,
    log_progress,
)
from haemolynx.pipeline.stages import HaemodynamicModel, run_pipeline_stages

#: The stages a run reports, in order. Anything else is a mistake in STAGES.
STAGE_NAMES = [stage.call for stage in STAGES if stage.call]

#: Every module `haemolynx.pipeline.progress` may import. All standard library:
#: `sys.stdlib_module_names` would say so directly, but it needs Python 3.10 and
#: this library supports 3.9.
STDLIB_IMPORTS_ALLOWED_IN_PROGRESS = {
    "__future__",
    "contextlib",
    "dataclasses",
    "logging",
    "typing",
}


@pytest.fixture
def stubbed(monkeypatch):
    """Every stage replaced by a stub that only records that it ran.

    Returns the list the stubs append to, so a test can compare what ran with
    what was reported.
    """

    def install(fail_at=None, steps=()):
        called: list[str] = []
        model = HaemodynamicModel(graph=nx.MultiGraph())

        def stub(name):
            def call(*_args, **kwargs):
                called.append(name)
                if name == fail_at:
                    raise RuntimeError(f"{name} broke")
                if name == "build_network":
                    reporter = kwargs["progress"]
                    for label in steps:
                        reporter.step(label, total=len(steps))
                if name == "build_haemodynamic_model":
                    return model
                return f"{name}-result"

            return call

        for name in STAGE_NAMES:
            monkeypatch.setattr(stages, name, stub(name))
        return called, model

    return install


def _run(recorder=None, **kwargs):
    return run_pipeline_stages({}, schema=None, progress=recorder, **kwargs)


# --- the stage list ----------------------------------------------------------


def test_the_stages_are_the_pipeline_functions_in_order():
    """STAGES is what both the panel and a progress bar count through."""
    assert STAGE_NAMES == [
        "segment",
        "skeletonise",
        "build_network",
        "assign_boundaries",
        "assign_diameters",
        "build_haemodynamic_model",
        "solve",
        "export_results",
    ]
    for name in STAGE_NAMES:
        assert callable(getattr(stages, name))


# --- a whole run -------------------------------------------------------------


def test_every_stage_reports_starting_and_finishing_exactly_once(stubbed):
    called, _model = stubbed()
    events: list[ProgressEvent] = []

    _run(events.append)

    assert called == STAGE_NAMES
    assert [(event.kind, event.stage) for event in events] == [
        (kind, name)
        for name in STAGE_NAMES
        for kind in (STAGE_STARTED, STAGE_FINISHED)
    ]


def test_every_event_counts_the_same_eight_stages(stubbed):
    stubbed()
    events: list[ProgressEvent] = []

    _run(events.append)

    assert {event.total for event in events} == {len(STAGE_NAMES)}
    assert [event.index for event in events if event.kind == STAGE_STARTED] == list(
        range(len(STAGE_NAMES))
    )
    assert [event.completed for event in events if event.kind == STAGE_FINISHED] == list(
        range(1, len(STAGE_NAMES) + 1)
    )


def test_every_event_names_its_stage_the_way_the_panel_does(stubbed):
    stubbed()
    events: list[ProgressEvent] = []

    _run(events.append)

    titles = {stage.call: stage.title for stage in STAGES}
    for event in events:
        assert event.title == titles[event.stage]
        assert event.kind in KINDS


def test_a_run_returns_its_graph_whether_or_not_anyone_is_watching(stubbed):
    called, model = stubbed()
    watched = _run(lambda event: None)
    watched_calls = list(called)
    called.clear()

    unwatched = _run(None)

    assert watched is model.graph
    assert unwatched is model.graph
    assert called == watched_calls == STAGE_NAMES


def test_a_run_with_no_callback_reports_nothing_and_still_runs(stubbed):
    """The default has to be free: no callback, no events, same stages."""
    called, _model = stubbed()

    _run()

    assert called == STAGE_NAMES


# --- a stage that raises -----------------------------------------------------


def test_a_failing_stage_reports_that_it_failed_and_the_error_propagates(stubbed):
    called, _model = stubbed(fail_at="assign_boundaries")
    events: list[ProgressEvent] = []

    with pytest.raises(RuntimeError, match="assign_boundaries broke"):
        _run(events.append)

    assert called == ["segment", "skeletonise", "build_network", "assign_boundaries"]
    failed = [event for event in events if event.kind == STAGE_FAILED]
    assert [event.stage for event in failed] == ["assign_boundaries"]
    assert isinstance(failed[0].error, RuntimeError)
    assert failed[0].index == STAGE_NAMES.index("assign_boundaries")


def test_a_failing_stage_never_reports_finishing(stubbed):
    stubbed(fail_at="solve")
    events: list[ProgressEvent] = []

    with pytest.raises(RuntimeError):
        _run(events.append)

    finished = [event.stage for event in events if event.kind == STAGE_FINISHED]
    assert "solve" not in finished
    assert finished == STAGE_NAMES[: STAGE_NAMES.index("solve")]


def test_the_stages_after_a_failure_report_nothing(stubbed):
    stubbed(fail_at="skeletonise")
    events: list[ProgressEvent] = []

    with pytest.raises(RuntimeError):
        _run(events.append)

    assert {event.stage for event in events} == {"segment", "skeletonise"}


# --- the steps inside graph building ----------------------------------------


def test_graph_building_reports_its_steps_through_the_same_callback(stubbed):
    stubbed(steps=STEP_LABELS)
    events: list[ProgressEvent] = []

    _run(events.append)

    steps = [event for event in events if event.kind == STEP]
    assert [event.step for event in steps] == list(STEP_LABELS)
    assert [event.step_index for event in steps] == list(range(len(STEP_LABELS)))
    assert {event.step_total for event in steps} == {len(STEP_LABELS)}


def test_a_step_belongs_to_the_stage_it_happened_in(stubbed):
    stubbed(steps=STEP_LABELS[:3])
    events: list[ProgressEvent] = []

    _run(events.append)

    steps = [event for event in events if event.kind == STEP]
    assert {(event.stage, event.title, event.index) for event in steps} == {
        ("build_network", "3. Graph", STAGE_NAMES.index("build_network"))
    }


def test_steps_are_reported_between_their_stage_starting_and_finishing(stubbed):
    stubbed(steps=STEP_LABELS)
    events: list[ProgressEvent] = []

    _run(events.append)

    kinds = [event.kind for event in events]
    first_step, last_step = kinds.index(STEP), len(kinds) - 1 - kinds[::-1].index(STEP)
    build_started = next(
        i
        for i, event in enumerate(events)
        if event.kind == STAGE_STARTED and event.stage == "build_network"
    )
    build_finished = next(
        i
        for i, event in enumerate(events)
        if event.kind == STAGE_FINISHED and event.stage == "build_network"
    )
    assert build_started < first_step <= last_step < build_finished


def test_only_graph_building_reports_steps(stubbed):
    """Nothing else has any yet; a stray step would mis-scale the inner bar."""
    stubbed(steps=STEP_LABELS)
    events: list[ProgressEvent] = []

    _run(events.append)

    assert {event.stage for event in events if event.kind == STEP} == {"build_network"}


# --- the reporter on its own -------------------------------------------------


def test_a_reporter_with_no_callback_does_nothing_at_all():
    run = RunProgress(None)
    with run.stage("solve") as reporter:
        reporter.step("anything", total=3)
    assert run.total == len(STAGE_NAMES)


def test_a_reporter_refuses_a_stage_the_pipeline_does_not_have():
    """A typo would otherwise report a stage nothing ever finishes."""
    run = RunProgress(lambda event: None)
    with pytest.raises(KeyError, match="not a pipeline stage"):
        with run.stage("skeletonize"):  # the American spelling; not a stage
            pass


def test_a_step_with_no_total_says_so_rather_than_guessing():
    """An unknown count is an indeterminate bar, not a bar that reads 1 of 1."""
    events: list[ProgressEvent] = []
    run = RunProgress(events.append)
    with run.stage("build_network") as reporter:
        reporter.step("only_step")

    (step,) = [event for event in events if event.kind == STEP]
    assert step.step_total is None
    assert step.step_index == 0


def test_a_step_total_given_once_holds_for_the_steps_after_it():
    events: list[ProgressEvent] = []
    run = RunProgress(events.append)
    with run.stage("build_network") as reporter:
        reporter.step("first", total=2)
        reporter.step("second")

    assert [event.step_total for event in events if event.kind == STEP] == [2, 2]


def test_completed_counts_the_stage_that_just_finished():
    event = ProgressEvent(STAGE_FINISHED, "solve", "7. Solve", index=6, total=8)
    assert event.completed == 7
    assert ProgressEvent(STAGE_STARTED, "solve", "7. Solve", index=6, total=8).completed == 6


# --- the ready-made console consumer ----------------------------------------


def test_log_progress_writes_one_line_per_stage_boundary(caplog):
    import logging

    events: list[ProgressEvent] = []
    run = RunProgress(events.append)
    with run.stage("segment"):
        pass

    with caplog.at_level(logging.INFO, logger="haemolynx.pipeline.progress"):
        for event in events:
            log_progress(event)

    assert [record.message for record in caplog.records] == [
        "Stage 1/8: 1. Input",
        "Stage 1/8 done: 1. Input",
    ]


def test_log_progress_reports_a_failure_as_an_error(caplog):
    import logging

    events: list[ProgressEvent] = []
    run = RunProgress(events.append)
    with pytest.raises(ValueError):
        with run.stage("solve"):
            raise ValueError("singular matrix")

    with caplog.at_level(logging.INFO, logger="haemolynx.pipeline.progress"):
        for event in events:
            log_progress(event)

    failure = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(failure) == 1
    assert "7. Solve" in failure[0].message
    assert "singular matrix" in failure[0].message


# --- what progress must not drag in ------------------------------------------


def test_the_progress_module_imports_nothing_but_the_standard_library():
    """A callback is the whole mechanism -- it needs no progress-bar library.

    The package as a whole pulls in plenty (tqdm arrives with skan), so this
    reads the module's own imports rather than watching `sys.modules`: what
    matters is that reporting progress adds no dependency of its own, and that
    a consumer is free to be a console, a notebook or a Qt panel.
    """
    source = Path(__file__).resolve().parents[1] / "src" / "haemolynx" / "pipeline" / "progress.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= STDLIB_IMPORTS_ALLOWED_IN_PROGRESS, (
        "haemolynx.pipeline.progress imports "
        f"{sorted(imported - STDLIB_IMPORTS_ALLOWED_IN_PROGRESS)}. Progress "
        "reporting must stay a plain callback: add the name here only if it is "
        "also in the standard library."
    )


def test_reporting_progress_needs_no_gui():
    """Importing it must not drag in napari, magicgui or a Qt binding.

    Checked in a fresh interpreter, because this session may already have
    imported any of them for another test.
    """
    probe = (
        "import sys; import haemolynx.pipeline.progress; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'napari', 'magicgui', 'qtpy', 'PyQt5', 'PyQt6', 'PySide6'}))"
    )
    source = Path(__file__).resolve().parents[1] / "src"
    environment = {**os.environ, "PYTHONPATH": str(source)}
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
