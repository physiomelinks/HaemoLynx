"""Which flat rows on a tab get boxed off as a nested sub-section.

Pure function: no Qt, no display. `settings_widget`'s tab-building loop uses
this to give a tab combining more than one schema section (only "8.
Additional measurements" does, for "Statistics and measurements" plus the
nested "Connectivity/Network Analysis") a boxed group for every section after
the first, so a nested checkbox's own children read as a distinct block
instead of a continuation of the leading section's flat list.
"""
from __future__ import annotations

from haemolynx.gui._widget import _grouped_by_section
from haemolynx.gui.form import Field


def _field(name: str, section: str) -> Field:
    return Field(
        name=name,
        label=name,
        widget_type="CheckBox",
        value=True,
        options={},
        help="",
        section=section,
        advanced=False,
        enabled_by=(),
    )


def test_a_single_section_comes_back_as_one_run():
    fields = {"a": _field("a", "Statistics"), "b": _field("b", "Statistics")}
    assert _grouped_by_section(["a", "b"], fields) == [("Statistics", ["a", "b"])]


def test_a_later_section_becomes_its_own_run():
    fields = {
        "a": _field("a", "Statistics"),
        "b": _field("b", "Statistics"),
        "c": _field("c", "Connectivity"),
        "d": _field("d", "Connectivity"),
    }
    assert _grouped_by_section(["a", "b", "c", "d"], fields) == [
        ("Statistics", ["a", "b"]),
        ("Connectivity", ["c", "d"]),
    ]


def test_the_same_section_returning_later_is_kept_as_its_own_run():
    """Schema declaration keeps a section's settings contiguous, so this
    never happens for a real tab today -- but two non-adjacent runs of the
    same section must not silently merge into one if it ever does."""
    fields = {
        "a": _field("a", "X"),
        "b": _field("b", "Y"),
        "c": _field("c", "X"),
    }
    assert _grouped_by_section(["a", "b", "c"], fields) == [
        ("X", ["a"]),
        ("Y", ["b"]),
        ("X", ["c"]),
    ]


def test_an_empty_name_list_returns_no_runs():
    assert _grouped_by_section([], {}) == []
