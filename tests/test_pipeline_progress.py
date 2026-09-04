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
        "run_perturbations",
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
    event = ProgressEvent(STAGE_FINISHED, "solve", "Solve", index=6, total=8)
    assert event.completed == 7
    assert ProgressEvent(STAGE_STARTED, "solve", "Solve", index=6, total=8).completed == 6


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

    total = len(STAGE_NAMES)
    assert [record.message for record in caplog.records] == [
        f"Stage 1/{total}: 1. Input",
        f"Stage 1/{total} done: 1. Input",
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
    assert "Solve" in failure[0].message
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


# --- what each stage produced ------------------------------------------------


def test_every_stage_hands_over_what_it_returned(stubbed):
    """`on_stage_output` is how the panel sees a run's work as it happens."""
    _called, model = stubbed()
    produced: list[tuple[str, object]] = []

    _run(on_stage_output=lambda name, output: produced.append((name, output)))

    assert [name for name, _output in produced] == STAGE_NAMES
    by_name = dict(produced)
    assert by_name["skeletonise"] == "skeletonise-result"
    # The stage's own object, not a copy: the panel builds layers from it.
    assert by_name["build_haemodynamic_model"] is model


def test_a_stage_that_raised_hands_over_nothing(stubbed):
    """There is no output to show for a stage that did not finish."""
    stubbed(fail_at="assign_boundaries")
    produced: list[str] = []

    with pytest.raises(RuntimeError, match="assign_boundaries broke"):
        _run(on_stage_output=lambda name, _output: produced.append(name))

    assert produced == ["segment", "skeletonise", "build_network"]


def test_an_output_lands_before_the_next_stage_starts(stubbed):
    """This is what lets a consumer snapshot a stage before it is overwritten.

    Every stage after `build_network` writes attributes onto the same graph, so
    an output handed over late would describe a later stage's work under an
    earlier stage's name. Pinning the interleaving is what makes "convert it
    now" a rule rather than a hope.
    """
    stubbed()
    timeline: list[tuple[str, str]] = []

    _run(
        recorder=lambda event: timeline.append((event.kind, event.stage)),
        on_stage_output=lambda name, _output: timeline.append(("outlet", name)),
    )

    for index, stage in enumerate(STAGE_NAMES):
        finished = timeline.index((STAGE_FINISHED, stage))
        output = timeline.index(("outlet", stage))
        assert output > finished, f"{stage} handed over its output before finishing"
        if index + 1 < len(STAGE_NAMES):
            next_started = timeline.index((STAGE_STARTED, STAGE_NAMES[index + 1]))
            assert output < next_started, (
                f"{stage}'s output arrived after {STAGE_NAMES[index + 1]} had started"
            )


def test_a_consumer_that_raises_stops_the_run_at_the_right_place(stubbed):
    """It must not be reported as the *stage* failing.

    Calling the callback inside the stage's `with` block would have
    `RunProgress.stage()` catch this, emit STAGE_FAILED and re-raise -- so a
    fault in whoever is watching would blame the pipeline.
    """
    stubbed()
    events: list[ProgressEvent] = []

    def explode(name, _output):
        if name == "skeletonise":
            raise RuntimeError("the watcher broke")

    with pytest.raises(RuntimeError, match="the watcher broke"):
        _run(recorder=events.append, on_stage_output=explode)

    assert [event.kind for event in events if event.kind == STAGE_FAILED] == []
    assert (STAGE_FINISHED, "skeletonise") in [(e.kind, e.stage) for e in events]


def test_watching_the_outputs_does_not_change_the_progress_events(stubbed):
    """Adding a second watcher must not disturb the first."""
    stubbed()
    without: list[tuple[str, str, int]] = []
    _run(recorder=lambda e: without.append((e.kind, e.stage, e.index)))

    stubbed()
    with_outputs: list[tuple[str, str, int]] = []
    _run(
        recorder=lambda e: with_outputs.append((e.kind, e.stage, e.index)),
        on_stage_output=lambda _name, _output: None,
    )

    assert with_outputs == without


def test_a_run_nobody_is_watching_still_returns_its_graph(stubbed):
    _called, model = stubbed()
    assert _run() is model.graph


def test_the_sixth_stage_is_named_haemodynamics():
    """#125. One place defines it; the napari tab and the bars both read it."""
    titles = [stage.title for stage in STAGES]

    assert titles == [
        "1. Input",
        "2. Skeletonise",
        "3. Graph",
        "4. Boundaries",
        "5. Diameters",
        "6. Haemodynamics",
        "Solve",
        "7. Perturbations",
        "8. Export",
    ]
    haemodynamics = next(s for s in STAGES if s.title == "6. Haemodynamics")
    assert haemodynamics.call == "build_haemodynamic_model"


def test_the_perturbations_tab_is_the_stage_that_runs_them():
    """One entry owns the tab and does the work.

    It was panel-only while the re-solve was not written, which meant the tab
    a user configured and the stage a run performed could have disagreed. They
    cannot now: the same entry is both.
    """
    perturbations = next(s for s in STAGES if s.title == "7. Perturbations")

    assert perturbations.call == "run_perturbations"
    assert perturbations.sections == ("Perturbation runs",)
    assert perturbations.tab is None, "it opens its own tab"
    assert callable(getattr(stages, "run_perturbations"))
    assert RunProgress(None).total == 9
    assert "7. Perturbations" in [stage.title for stage in RunProgress(None).stages]


def test_a_panel_only_stage_would_not_enter_the_count():
    """The mechanism survives its first user: a `call`-less stage is not a stage.

    Nothing in `STAGES` needs this today, but the panel can still show a tab
    for something the pipeline does not run, and such an entry must stay out
    of the count a progress bar reads or every run would stop short of full.
    """
    from haemolynx.pipeline.progress import Stage

    shown_only = Stage(call=None, title="Notes", summary="Not a stage.")
    progress = RunProgress(None, stages=(*STAGES, shown_only))

    assert progress.total == 9
    assert "Notes" not in [stage.title for stage in progress.stages]


def test_the_solve_stage_shows_its_settings_on_the_haemodynamics_tab():
    """It is still a stage a run reports; only its form rows have moved.

    The boundary pressures configure the solve, but a user reads them as part
    of the haemodynamics, so the two stages share one tab. `Stage.call` and
    the progress events are untouched by that -- which is what this pins.
    """
    solve = next(stage for stage in STAGES if stage.call == "solve")

    assert solve.tab == "6. Haemodynamics"
    assert solve.settings == ("inlet_p_bc", "outlet_p_bc", "do_equiv_resistance_calculation")
    others = [stage.tab for stage in STAGES if stage.call != "solve"]
    assert others == [None] * len(others)


def test_start_from_skips_earlier_stage_bodies_but_still_reports_them(stubbed, monkeypatch):
    """A mid-pipeline rerun still counts every stage on the progress bar."""
    from types import SimpleNamespace

    from haemolynx.pipeline.stages import PipelineResume

    called, model = stubbed()
    graph = nx.MultiGraph()
    graph.add_node(0)

    def build_network(*_args, **kwargs):
        called.append("build_network")
        reporter = kwargs.get("progress")
        if reporter is not None:
            pass
        return SimpleNamespace(
            graph=graph, large_arteriole_mask=None, large_venule_mask=None
        )

    monkeypatch.setattr(stages, "build_network", build_network)
    events: list[ProgressEvent] = []
    resume = PipelineResume(
        start_from="assign_diameters",
        graph=graph,
        inlet_nodes=(0,),
        outlet_nodes=(0,),
    )

    result = run_pipeline_stages(
        {},
        schema=None,
        progress=events.append,
        start_from="assign_diameters",
        resume=resume,
    )

    assert "assign_boundaries" not in called
    assert "assign_diameters" in called
    assert "build_haemodynamic_model" in called
    assert [event.stage for event in events if event.kind == STAGE_STARTED] == STAGE_NAMES
    assert result is model.graph
