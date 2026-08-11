"""Checking whether a quantity that should be tissue-only separates by cohort.

The segmentation threshold, the foreground fraction at a frozen threshold and the
classifier's mean output are properties of the instrument. If one of them separates cleanly
by group, part of the measured group difference is the measuring device.

Eyeballing six numbers does not settle it: at n = 3 per group, complete separation arises by
chance with probability 2/C(6,3) = 0.10, which is also the floor of any exact two-sided test
at this n. Reported alone, 0.10 reads as "nearly significant" when it is in fact the most
extreme result three against three can produce.
"""
import pytest

from ImageLynx.statistics.cohort_split import assess_cohort_split

GROUPS = {"WKY-A": "WKY", "WKY-B": "WKY", "WKY-C": "WKY",
          "SHR-A": "SHR", "SHR-B": "SHR", "SHR-C": "SHR"}


def test_overlapping_groups_are_the_reassuring_outcome():
    split = assess_cohort_split(
        {"WKY-A": 0.90, "WKY-B": 0.85, "WKY-C": 0.95,
         "SHR-A": 0.88, "SHR-B": 0.92, "SHR-C": 0.87},
        quantity="threshold", groups_by_specimen=GROUPS)

    assert not split.separated
    assert not split.concerning
    assert "does not separate" in split.verdict


def test_wide_clean_separation_is_flagged_as_concerning():
    split = assess_cohort_split(
        {"WKY-A": 0.95, "WKY-B": 0.96, "WKY-C": 0.94,
         "SHR-A": 0.70, "SHR-B": 0.71, "SHR-C": 0.69},
        quantity="threshold", groups_by_specimen=GROUPS)

    assert split.separated and split.concerning
    assert split.direction == "SHR < WKY"
    assert split.gap == pytest.approx(0.23, abs=1e-9)
    assert "part of the group difference is the instrument" in split.verdict


def test_narrow_separation_is_reported_but_not_escalated():
    """Separation inside the within-group spread is too weak to act on at this n."""
    split = assess_cohort_split(
        {"WKY-A": 0.90, "WKY-B": 0.80, "WKY-C": 0.85,
         "SHR-A": 0.91, "SHR-B": 0.99, "SHR-C": 0.95},
        quantity="threshold", groups_by_specimen=GROUPS)

    assert split.separated and not split.concerning
    assert "weak on its own" in split.verdict


def test_the_floor_is_reported_alongside_the_p_value():
    """0.10 is the most extreme p three against three can produce, and must not read as 0.05."""
    split = assess_cohort_split(
        {"WKY-A": 1.0, "WKY-B": 2.0, "WKY-C": 3.0,
         "SHR-A": 10.0, "SHR-B": 11.0, "SHR-C": 12.0},
        quantity="foreground fraction", groups_by_specimen=GROUPS)

    assert split.floor_p == pytest.approx(0.10)
    assert split.permutation_p == pytest.approx(0.10)
    assert split.permutation_p >= split.floor_p


def test_too_few_specimens_is_reported_rather_than_guessed():
    split = assess_cohort_split({"WKY-A": 1.0, "SHR-A": 2.0}, quantity="threshold",
                                groups_by_specimen=GROUPS)
    assert not split.separated
    assert "Not enough specimens" in split.verdict


def test_groups_are_looked_up_from_the_registry_by_default():
    split = assess_cohort_split(
        {"WKY-A": 0.9, "WKY-B": 0.9, "WKY-C": 0.9,
         "SHR-A": 0.5, "SHR-B": 0.5, "SHR-C": 0.5}, quantity="threshold")
    assert set(split.values_by_group) == {"WKY", "SHR"}
    assert split.separated
