"""What the panel's "Optimise settings" progress bars read, checked without
building any -- mirrors test_gui_progress.py for the pipeline-run bars.

`haemolynx.gui.optimise_progress` is the whole decision, so it is tested here
on plain Python; the widget test that drives this for real is
`test_gui_optimise_settings_widget.py`, marked `gui`.
"""
from __future__ import annotations

from haemolynx.gui.optimise_progress import OptimisationBarState, OptimisationProgressDisplay
from haemolynx.optimisation.progress import OptimisationEvent


def _event(kind, group_name="closing_radius", **extra):
    return OptimisationEvent(kind=kind, group_index=1, group_total=16, group_name=group_name, **extra)


def test_nothing_is_shown_before_a_run_starts():
    display = OptimisationProgressDisplay()
    assert display.groups == OptimisationBarState()
    assert not display.groups.visible
    assert not display.candidates.visible


def test_starting_a_run_shows_an_empty_bar():
    display = OptimisationProgressDisplay()
    display.start()
    assert display.groups.visible
    assert display.groups.total == 0
    assert not display.candidates.visible


def test_group_started_shows_progress_and_resets_candidates():
    display = OptimisationProgressDisplay()
    display.candidates = OptimisationBarState(value=3, total=5, visible=True)
    display.update(_event("group_started"))
    assert display.groups.value == 1
    assert display.groups.total == 16
    assert "closing_radius" in display.groups.text
    assert display.groups.visible
    assert display.candidates == OptimisationBarState()


def test_group_finished_advances_and_shows_the_winner():
    display = OptimisationProgressDisplay()
    display.update(_event("group_finished", winner={"skeleton_closing_radius": 2}))
    assert display.groups.value == 2  # group_index + 1
    assert "skeleton_closing_radius=2" in display.groups.text


def test_group_finished_with_no_winner_falls_back_to_group_name():
    display = OptimisationProgressDisplay()
    display.update(_event("group_finished", winner=None))
    assert display.groups.text == "closing_radius"


def test_candidate_evaluated_updates_the_inner_bar():
    display = OptimisationProgressDisplay()
    display.update(_event("candidate_evaluated", candidate_index=2, candidate_total=5))
    assert display.candidates.value == 3
    assert display.candidates.total == 5
    assert "3/5" in display.candidates.text


def test_finish_fills_the_bar_and_drops_the_candidate_bar():
    display = OptimisationProgressDisplay()
    display.groups = OptimisationBarState(value=3, total=16, visible=True)
    display.candidates = OptimisationBarState(value=2, total=4, visible=True)
    display.finish("Optimised")
    assert display.groups.value == 16
    assert display.groups.text == "Optimised"
    assert display.candidates == OptimisationBarState()


def test_fail_leaves_the_bar_where_it_got_to():
    display = OptimisationProgressDisplay()
    display.groups = OptimisationBarState(value=5, total=16, visible=True)
    display.fail("Failed: RuntimeError")
    assert display.groups.value == 5
    assert display.groups.text == "Failed: RuntimeError"


def test_reset_clears_both_bars():
    display = OptimisationProgressDisplay()
    display.groups = OptimisationBarState(value=5, total=16, visible=True)
    display.candidates = OptimisationBarState(value=2, total=4, visible=True)
    display.reset()
    assert display.groups == OptimisationBarState()
    assert display.candidates == OptimisationBarState()
