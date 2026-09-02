"""The panel's tabs: one per pipeline stage, covering every setting exactly once.

The tab split is declared rather than derived, so the thing to guard is that it
stays complete and unambiguous as settings are added. A setting on no tab is
unreachable from the GUI; a setting on two is a value the user can set twice
and disagree with themselves about.
"""
from __future__ import annotations

import pytest

from haemolynx.gui.tabs import (
    STAGES,
    Stage,
    assign_to_stages,
    tab_titles,
    tabs_for,
    unassigned,
)
from haemolynx.parsers import Schema, Setting
from haemolynx.pipeline import default_schema

SCHEMA = default_schema()


# --- coverage ----------------------------------------------------------------


def test_every_setting_is_on_a_tab():
    """A setting no tab claims cannot be reached from the panel at all."""
    missing = unassigned(SCHEMA)
    assert missing == [], (
        f"settings on no tab: {missing}. Add each to a stage in "
        "haemolynx/gui/tabs.py -- by name, or by claiming its section."
    )


def test_no_setting_is_on_two_tabs():
    counts: dict[str, int] = {}
    for tab in tabs_for(SCHEMA):
        for field in tab.fields:
            counts[field.name] = counts.get(field.name, 0) + 1
    duplicated = sorted(name for name, count in counts.items() if count > 1)
    assert duplicated == [], f"settings on more than one tab: {duplicated}"


def test_the_tabs_hold_every_setting_between_them():
    total = sum(len(tab.fields) for tab in tabs_for(SCHEMA))
    assert total == len(SCHEMA.names)


def test_no_tab_is_empty():
    """An empty tab is a stage the panel implies has nothing to configure."""
    empty = [tab.stage.title for tab in tabs_for(SCHEMA) if not tab.fields]
    assert empty == [], f"tabs with no settings: {empty}"


# --- the stages themselves ---------------------------------------------------


def test_the_tabs_are_the_pipeline_stages_in_order():
    """The panel must read like the example: same calls, same order."""
    import haemolynx.pipeline as pipeline

    calls = [stage.call for stage in STAGES if stage.call]
    assert calls == [
        "segment",
        "skeletonise",
        "build_network",
        "assign_boundaries",
        "assign_diameters",
        "build_haemodynamic_model",
        "solve",
        "export_results",
    ]
    for name in calls:
        assert callable(getattr(pipeline, name)), f"{name} is not a pipeline stage"


def test_every_tab_says_what_its_stage_does():
    for stage in STAGES:
        assert stage.summary.strip(), f"{stage.title} has no summary"
        assert stage.summary.endswith("."), f"{stage.title}: summary reads as a sentence"


def test_tab_titles_are_unique():
    titles = [stage.title for stage in STAGES]
    assert len(set(titles)) == len(titles)


def test_the_tabs_read_in_pipeline_order():
    """What the panel actually shows, pinned: the numbering is the running order."""
    assert list(tab_titles()) == [
        "1. Input",
        "2. Skeletonise",
        "3. Graph",
        "4. Boundaries",
        "5. Diameters",
        "6. Haemodynamics",
        "7. Solve",
        "8. Export",
    ]


# --- how claims are resolved -------------------------------------------------


def _two_stage_schema() -> Schema:
    return Schema(
        [
            Setting("shared_output_dir", "path", "out", "Where output goes", "Boundaries"),
            Setting("boundary_method", "str", "volume", "How to pick nodes", "Boundaries"),
            Setting("solver_pressure", "float", 1.0, "Inlet pressure", "Solver"),
        ]
    )


def test_a_named_claim_beats_a_section_claim(monkeypatch):
    """`base_plot_dir` sits in the boundary section but belongs to output."""
    stages = (
        Stage(call="a", title="Boundaries", summary="First.", sections=("Boundaries",)),
        Stage(call="b", title="Export", summary="Second.", settings=("shared_output_dir",)),
    )
    monkeypatch.setattr("haemolynx.gui.tabs.STAGES", stages)

    owner = assign_to_stages(_two_stage_schema())
    assert owner["shared_output_dir"] == "Export"
    assert owner["boundary_method"] == "Boundaries"


def test_the_first_stage_to_name_a_setting_owns_it(monkeypatch):
    stages = (
        Stage(call="a", title="First", summary="One.", settings=("solver_pressure",)),
        Stage(call="b", title="Second", summary="Two.", settings=("solver_pressure",)),
    )
    monkeypatch.setattr("haemolynx.gui.tabs.STAGES", stages)

    assert assign_to_stages(_two_stage_schema())["solver_pressure"] == "First"


def test_the_panel_opens_one_tab_per_stage_that_names_no_other():
    """Tabs and stages are close but not one for one; `tab_titles` is the list."""
    assert [tab.stage.title for tab in tabs_for(SCHEMA)] == list(tab_titles())


def test_a_stage_can_put_its_rows_on_another_stages_tab(monkeypatch):
    """`Stage.tab` is how one tab shows two stages' settings.

    The stage still runs and still reports its own progress -- only its form
    rows move, which is what lets the boundary pressures sit next to the
    haemodynamics they belong with.
    """
    stages = (
        Stage(call="a", title="1. Boundaries", summary="First.", sections=("Boundaries",)),
        Stage(
            call="b",
            title="Solve",
            summary="Second.",
            settings=("solver_pressure",),
            tab="1. Boundaries",
        ),
    )
    monkeypatch.setattr("haemolynx.gui.tabs.STAGES", stages)

    tabs = tabs_for(_two_stage_schema())
    assert [tab.stage.title for tab in tabs] == ["1. Boundaries"]
    assert {field.name for field in tabs[0].fields} == {
        "shared_output_dir",
        "boundary_method",
        "solver_pressure",
    }
    assert unassigned(_two_stage_schema()) == []


def test_a_tab_naming_no_stage_raises_rather_than_dropping_its_rows(monkeypatch):
    """A mistyped title would take that stage's settings off the panel silently.

    `unassigned` would still read empty -- the settings *are* claimed, by a
    stage whose tab nothing draws -- so nothing else can catch this.
    """
    stages = (
        Stage(call="a", title="1. Boundaries", summary="First.", sections=("Boundaries",)),
        Stage(
            call="b",
            title="Solve",
            summary="Second.",
            settings=("solver_pressure",),
            tab="1. Boundries",  # the typo
        ),
    )
    monkeypatch.setattr("haemolynx.gui.tabs.STAGES", stages)

    with pytest.raises(ValueError, match="name no tab"):
        tabs_for(_two_stage_schema())
    with pytest.raises(ValueError, match="name no tab"):
        assign_to_stages(_two_stage_schema())


def test_a_claim_for_a_setting_that_does_not_exist_is_ignored(monkeypatch):
    """A renamed setting must not put a phantom row on a tab."""
    stages = (
        Stage(call="a", title="Only", summary="One.", settings=("gone_away", "solver_pressure")),
    )
    monkeypatch.setattr("haemolynx.gui.tabs.STAGES", stages)

    owner = assign_to_stages(_two_stage_schema())
    assert "gone_away" not in owner
    assert owner["solver_pressure"] == "Only"


# --- the rows on a tab -------------------------------------------------------


def test_a_tab_carries_the_rows_for_its_settings():
    tabs = {tab.stage.title: tab for tab in tabs_for(SCHEMA)}
    solve = tabs["7. Solve"]
    assert {field.name for field in solve.fields} == {
        "inlet_p_bc",
        "outlet_p_bc",
        "do_equiv_resistance_calculation",
    }


def test_supplied_values_reach_the_right_tab():
    tabs = {tab.stage.title: tab for tab in tabs_for(SCHEMA, {"inlet_p_bc": 1234.0})}
    row = next(f for f in tabs["7. Solve"].fields if f.name == "inlet_p_bc")
    assert row.value == 1234.0


def test_rows_keep_the_schema_order_within_a_tab():
    """Reading down a tab should match reading down the config file."""
    tabs = {tab.stage.title: tab for tab in tabs_for(SCHEMA)}
    shown = [field.name for field in tabs["1. Input"].fields]
    declared = [s.name for s in SCHEMA if s.name in set(shown)]
    assert shown == declared


@pytest.mark.parametrize("tab_title", [stage.title for stage in STAGES])
def test_every_tab_starts_with_a_number_so_the_order_is_visible(tab_title):
    assert tab_title[0].isdigit()
