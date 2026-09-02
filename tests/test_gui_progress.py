"""What the panel's progress bars read, checked without building any.

`haemolynx.gui.progress` is the whole decision -- which bar moves, how far, and
what it says -- so it can be tested here, on every Python the library supports,
rather than only where napari, a Qt binding and a display are installed. The
widget test that matches this one is in `test_gui_widget.py`, marked `gui`.
"""
from __future__ import annotations

from haemolynx.gui.progress import TOTAL_STAGES, BarState, ProgressDisplay
from haemolynx.pipeline.progress import (
    STAGE_FAILED,
    STAGE_FINISHED,
    STAGE_STARTED,
    STAGES,
    STEP,
    ProgressEvent,
)


def _stage_event(kind, name, **extra):
    index = [stage.call for stage in STAGES if stage.call].index(name)
    stage = next(stage for stage in STAGES if stage.call == name)
    return ProgressEvent(
        kind=kind,
        stage=name,
        title=stage.title,
        index=index,
        total=TOTAL_STAGES,
        **extra,
    )


# --- the bar across the run --------------------------------------------------


def test_the_run_has_as_many_stages_as_the_pipeline_does():
    """Nine since the perturbations became a stage rather than only a tab."""
    assert TOTAL_STAGES == 9


def test_nothing_is_shown_before_a_run_starts():
    display = ProgressDisplay()
    assert display.stages == BarState()
    assert not display.stages.visible
    assert not display.steps.visible


def test_starting_a_run_shows_an_empty_bar():
    """A bar that appears only once the first stage finishes looks stuck."""
    display = ProgressDisplay()
    display.start()

    assert display.stages.visible
    assert display.stages.value == 0
    assert display.stages.total == TOTAL_STAGES


def test_a_stage_starting_leaves_the_bar_on_the_stages_already_done():
    display = ProgressDisplay()
    display.update(_stage_event(STAGE_STARTED, "assign_diameters"))

    assert display.stages.value == 4
    assert display.stages.total == TOTAL_STAGES
    assert "5. Diameters" in display.stages.text
    assert f"5/{TOTAL_STAGES}" in display.stages.text


def test_a_stage_finishing_counts_it():
    display = ProgressDisplay()
    display.update(_stage_event(STAGE_FINISHED, "assign_diameters"))

    assert display.stages.value == 5
    assert f"5/{TOTAL_STAGES}" in display.stages.text


def test_the_last_stage_finishing_fills_the_bar():
    display = ProgressDisplay()
    display.update(_stage_event(STAGE_FINISHED, "export_results"))

    assert display.stages.value == display.stages.total == TOTAL_STAGES


def test_the_whole_run_moves_the_bar_one_stage_at_a_time():
    display = ProgressDisplay()
    display.start()
    seen = []
    # Only the stages a run performs: a panel-only tab reports nothing, so it
    # would not be one of the stages the bar counts through.
    for stage in [stage for stage in STAGES if stage.call]:
        display.update(_stage_event(STAGE_STARTED, stage.call))
        seen.append(display.stages.value)
        display.update(_stage_event(STAGE_FINISHED, stage.call))
        seen.append(display.stages.value)

    assert seen == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9]


# --- the bar within a stage --------------------------------------------------


def test_a_step_shows_the_second_bar():
    display = ProgressDisplay()
    display.update(_stage_event(STAGE_STARTED, "build_network"))
    assert not display.steps.visible

    display.update(
        _stage_event(
            STEP, "build_network", step="collapse_node_clusters", step_index=4, step_total=11
        )
    )

    assert display.steps.visible
    assert display.steps.value == 5
    assert display.steps.total == 11
    assert "collapse_node_clusters" in display.steps.text


def test_a_step_with_no_known_total_asks_for_a_busy_bar():
    """Qt animates a bar whose maximum is 0, which is the honest reading."""
    display = ProgressDisplay()
    display.update(_stage_event(STEP, "build_network", step="one", step_index=0))

    assert display.steps.total == 0
    assert display.steps.visible


def test_the_step_bar_goes_away_when_the_stage_it_belonged_to_ends():
    display = ProgressDisplay()
    display.update(
        _stage_event(STEP, "build_network", step="one", step_index=0, step_total=11)
    )
    display.update(_stage_event(STAGE_FINISHED, "build_network"))

    assert not display.steps.visible


def test_the_step_bar_does_not_carry_over_into_the_next_stage():
    display = ProgressDisplay()
    display.update(
        _stage_event(STEP, "build_network", step="one", step_index=0, step_total=11)
    )
    display.update(_stage_event(STAGE_STARTED, "assign_boundaries"))

    assert not display.steps.visible
    assert display.steps.value == 0


# --- the end of a run --------------------------------------------------------


def test_a_finished_run_fills_the_bar_and_says_so():
    display = ProgressDisplay()
    display.start()
    display.update(_stage_event(STAGE_FINISHED, "solve"))
    display.finish()

    assert display.stages.value == display.stages.total == TOTAL_STAGES
    assert display.stages.text == "Finished"
    assert not display.steps.visible


def test_a_failed_stage_leaves_the_bar_where_it_stopped():
    """Filling the bar in on failure would say the run finished."""
    display = ProgressDisplay()
    display.start()
    display.update(_stage_event(STAGE_FINISHED, "skeletonise"))
    display.update(_stage_event(STAGE_FAILED, "build_network"))

    assert display.stages.value == 2
    assert "3. Graph" in display.stages.text
    assert not display.steps.visible


def test_a_failed_run_keeps_its_position_and_names_the_failure():
    display = ProgressDisplay()
    display.start()
    display.update(_stage_event(STAGE_FINISHED, "skeletonise"))
    display.fail("Failed: ValueError")

    assert display.stages.value == 2
    assert display.stages.text == "Failed: ValueError"


def test_an_event_read_out_of_order_still_gives_the_right_reading():
    """Events cross a thread boundary; a dropped one must not skew the bar."""
    display = ProgressDisplay()
    display.start()
    display.update(_stage_event(STAGE_FINISHED, "export_results"))

    assert display.stages.value == TOTAL_STAGES
