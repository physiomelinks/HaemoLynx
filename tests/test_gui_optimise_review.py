"""Unit tests for haemolynx.gui.optimise_review -- pure, no Qt."""
from __future__ import annotations

from haemolynx.gui.optimise_review import ProposedChange, proposed_changes, review_text
from haemolynx.optimisation.scorecard import ScoreRow


def test_only_settings_that_would_change_are_listed():
    current = {
        "skeleton_closing_radius": 2,
        "skeleton_min_branch_length": 3,
        "segmentation_cleanup": False,
        "centreline_max_deviation": 1.0,
    }
    proposed = {
        "skeleton_closing_radius": 2.0,  # the same number, as a float
        "skeleton_min_branch_length": 4,
        "segmentation_cleanup": True,
        "centreline_max_deviation": 1.0 + 1e-12,  # rounding
    }

    assert proposed_changes(proposed, current) == (
        ProposedChange("skeleton_min_branch_length", 3, 4),
        ProposedChange("segmentation_cleanup", False, True),
    )


def test_a_boolean_is_never_taken_for_a_number():
    assert proposed_changes({"flag": True}, {"flag": 1}) == (ProposedChange("flag", 1, True),)


def test_a_setting_the_panel_does_not_hold_is_a_change_from_unset():
    assert proposed_changes({"fwhm_diameter_guess_um": 3.5}, {}) == (
        ProposedChange("fwhm_diameter_guess_um", None, 3.5),
    )


def test_the_review_lists_each_change_and_says_what_apply_does():
    text = review_text(
        (
            ProposedChange("skeleton_min_branch_length", 3, 4),
            ProposedChange("segmentation_cleanup", False, True),
            ProposedChange("fwhm_diameter_guess_um", None, 3.25),
        )
    )
    assert text.splitlines() == [
        "3 settings would change:",
        "  skeleton_min_branch_length: 3 -> 4",
        "  segmentation_cleanup: off -> on",
        "  fwhm_diameter_guess_um: unset -> 3.25",
        "Apply writes these into the panel; Discard leaves it as it is.",
    ]


def test_the_review_warns_of_every_measure_the_run_made_worse():
    scorecard = (
        ScoreRow("input vessels the skeleton reaches", 0.95, 0.80, True, 0.02),
        ScoreRow("graph components", 4.0, 1.0, False),
    )
    text = review_text((ProposedChange("skeleton_closing_radius", 2, 4),), scorecard)

    assert (
        "WARNING: worse than your current settings on input vessels the skeleton reaches "
        "(0.95 -> 0.8)."
    ) in text.splitlines()
    assert "graph components" not in text


def test_a_run_that_kept_everything_says_so():
    assert review_text(()).splitlines()[0] == (
        "No setting would change: the search kept every value you have."
    )
