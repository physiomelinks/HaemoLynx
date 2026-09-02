"""The panel's tabs: one per pipeline stage, covering every setting exactly once.

The tab split is declared rather than derived, so the thing to guard is that it
stays complete and unambiguous as settings are added. A setting on no tab is
unreachable from the GUI; a setting on two is a value the user can set twice
and disagree with themselves about.

There is one trap in here worth reading before moving a row, and it has its own
test below: which tab a setting appears on is declared by `Stage.settings` and
`Stage.sections`, never by the schema *section* the setting is declared in.
Moving it between schema sections looks like the same thing and is not -- the
pipeline hands whole schema sections to the code that consumes them.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from haemolynx.gui.tabs import (
    STAGES,
    Stage,
    assign_to_stages,
    tab_titles,
    tabs_for,
    unassigned,
)
from haemolynx.haemodynamics.apply import DIAMETER_DEFAULTS
from haemolynx.parsers import Schema, Setting
from haemolynx.pipeline import default_schema

SCHEMA = default_schema()

#: The one schema section a stage hands over whole, and the module that reads
#: it back by name. See `test_the_haemodynamics_still_gets_its_whole_section`.
DIAMETERS_AND_PERICYTES = "Diameters and pericytes"
APPLY_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src" / "haemolynx" / "haemodynamics" / "apply.py"
)


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


# --- the schema section a stage hands over whole -----------------------------


def _diameter_group_names_read_by(source_path: Path) -> set[str]:
    """Every setting `apply.py` looks up in the diameters group, by name.

    Read out of the source rather than listed here, so a name added to the
    consumer is covered without anyone remembering to add it.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    def _literal(nodes) -> str | None:
        if not nodes:
            return None
        first = nodes[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
        return None

    def _is_group(node) -> bool:
        return isinstance(node, ast.Attribute) and node.attr == "diameters"

    names: set[str] = set()
    for node in ast.walk(tree):
        # config.diameter("name")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "diameter":
                name = _literal(node.args)
                if name is not None:
                    names.add(name)
            # self.diameters.get("name", default)
            elif node.func.attr == "get" and _is_group(node.func.value):
                name = _literal(node.args)
                if name is not None:
                    names.add(name)
        # self.diameters["name"]
        elif isinstance(node, ast.Subscript) and _is_group(node.value):
            name = _literal([node.slice])
            if name is not None:
                names.add(name)
        # "name" in self.diameters
        elif isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.In, ast.NotIn)) for op in node.ops
        ):
            if any(_is_group(target) for target in node.comparators):
                name = _literal([node.left])
                if name is not None:
                    names.add(name)
    return names


def test_the_haemodynamics_still_gets_its_whole_section():
    """The guard on retabbing: move `Stage.settings`, never a schema section.

    `assign_diameters` hands `schema.section_values(settings, "Diameters and
    pericytes")` to the haemodynamics as one group, and `apply.py` reads each
    value back out of it *by name*, falling back to `DIAMETER_DEFAULTS` for
    what is not there. So a setting moved to a different schema *section* --
    which looks like the natural way to move its row to another tab -- does not
    fail. It silently reverts (e.g. ``viscosity_law`` would fall back), with
    no error and no other failing test. Which tab a row appears on is declared
    in `pipeline/progress.py`, by name or by claiming a section there.

    ``do_pericyte_construction`` must still live in this section so the group
    is complete; the pipeline then forces it False on the baseline path.
    """
    section = set(SCHEMA.section_names(DIAMETERS_AND_PERICYTES))
    read = _diameter_group_names_read_by(APPLY_SOURCE)
    assert read, "found no diameters-group lookups in apply.py; the test is broken"

    lost = sorted(read - section - set(DIAMETER_DEFAULTS))
    assert lost == [], (
        f"{lost} are read out of the '{DIAMETERS_AND_PERICYTES}' group by "
        "apply.py but are not declared in that schema section, so they will "
        "read as unset at run time. Put them back in the section and move "
        "their tab in pipeline/progress.py instead."
    )
    # Named outright as well as derived: these are the ones whose reverting to
    # a default changes the numbers rather than raising.
    for name in (
        "do_pericyte_construction",
        "use_pericyte_mask_constriction",
        "viscosity_law",
        "diameter_basis",
        "haematocrit",
        "diameter_by_branch_order",
        "pericyte_constriction_factor",
        "constriction_by_branch_order",
        "constriction_length_um",
        "constriction_spacing_um",
    ):
        assert name in section, f"{name} has left the {DIAMETERS_AND_PERICYTES} section"


def test_the_whole_section_is_on_the_stage_that_hands_it_over():
    """One tab for one group, except settings retabbed by name for the panel.

    Pericyte / constriction knobs stay in the Diameters schema section so
    apply.py still finds them, but `STAGES` claims them for Perturbations so
    they are not always-on Diameters rows. Legacy baseline / comparison flags
    are retabbed the same way without becoming typed-entry options.
    """
    from haemolynx.haemodynamics.perturbations import PERICYTE_CONSTRICTION_SETTINGS
    from haemolynx.pipeline import progress as progress_module

    owner = assign_to_stages(SCHEMA)
    retabbed = set(PERICYTE_CONSTRICTION_SETTINGS) | set(
        progress_module._LEGACY_SETTINGS_HIDDEN_FROM_DIAMETERS
    )
    tabs = {
        owner[name]
        for name in SCHEMA.section_names(DIAMETERS_AND_PERICYTES)
        if name not in retabbed
    }
    assert tabs == {"5. Diameters"}
    for name in retabbed:
        assert owner[name] == "7. Perturbations", name


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
        "run_perturbations",
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
        # `solve` renders its rows onto the haemodynamics tab rather than
        # opening one of its own.
        "7. Perturbations",
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
    """The haemodynamics tab is whether to solve, and what to solve at.

    Two stages' rows: `build_haemodynamic_model` brings the toggle,
    `solve` brings the boundary pressures it reads.
    """
    tabs = {tab.stage.title: tab for tab in tabs_for(SCHEMA)}
    haemodynamics = tabs["6. Haemodynamics"]
    assert {field.name for field in haemodynamics.fields} == {
        "run_haemodynamics",
        "inlet_p_bc",
        "outlet_p_bc",
        "do_equiv_resistance_calculation",
    }


def test_the_blood_settings_stay_on_the_diameters_tab():
    """Viscosity and measured diameters belong with the baseline model."""
    tabs = {tab.stage.title: tab for tab in tabs_for(SCHEMA)}
    shown = {field.name for field in tabs["5. Diameters"].fields}
    for name in (
        "viscosity_law",
        "diameter_basis",
        "haematocrit",
    ):
        assert name in shown, f"{name} is not on the Diameters tab"


def test_legacy_and_comparison_settings_are_not_on_the_diameters_tab():
    """Legacy flags and comparison CSV knobs leave Diameters without UI rows.

    Settings stay in the Diameters schema section for apply.py / CLI, but
    their panel rows are claimed by Perturbations and filtered from always-on
    tab chrome and typed-entry editors (``do_pericyte_construction`` is inert
    outside typed strategy paths).
    """
    from haemolynx.gui.perturbation_editing import (
        ALWAYS_VISIBLE_TAB_SETTINGS,
        EDITOR_SETTINGS,
    )
    from haemolynx.pipeline import progress as progress_module

    legacy = progress_module._LEGACY_SETTINGS_HIDDEN_FROM_DIAMETERS
    tabs = {tab.stage.title: tab for tab in tabs_for(SCHEMA)}
    diameters = {field.name for field in tabs["5. Diameters"].fields}
    perturbations = {field.name for field in tabs["7. Perturbations"].fields}
    for name in legacy:
        assert name not in diameters, f"{name} is still on Diameters"
        assert name in perturbations, f"{name} is not claimed by Perturbations"
        assert name not in ALWAYS_VISIBLE_TAB_SETTINGS, name
        assert name not in EDITOR_SETTINGS, f"{name} is still a typed-entry editor"


def test_pericyte_constriction_settings_are_not_on_the_diameters_tab():
    """They configure a typed perturbation, not the baseline diameter model."""
    from haemolynx.haemodynamics.perturbations import PERICYTE_CONSTRICTION_SETTINGS
    from haemolynx.pipeline import progress as progress_module

    assert tuple(PERICYTE_CONSTRICTION_SETTINGS) == (
        progress_module._PERICYTE_SETTINGS_ON_PERTURBATIONS_TAB
    ), "progress.py's retab list drifted from PERICYTE_CONSTRICTION_SETTINGS"

    tabs = {tab.stage.title: tab for tab in tabs_for(SCHEMA)}
    diameters = {field.name for field in tabs["5. Diameters"].fields}
    perturbations = {field.name for field in tabs["7. Perturbations"].fields}
    for name in PERICYTE_CONSTRICTION_SETTINGS:
        assert name not in diameters, f"{name} is still on Diameters"
        assert name in perturbations, f"{name} is not claimed by Perturbations"


def test_the_perturbations_tab_shows_only_the_always_on_run_settings():
    """Sweep ranges are a type's options, not permanent form rows.

    They stay declared under Perturbation runs so Field objects exist for the
    editor and defaults remain in the schema, but the tab itself only shows
    whether to run perturbations, the list, and where to write them.
    """
    tabs = {tab.stage.title: tab for tab in tabs_for(SCHEMA)}
    claimed = {field.name for field in tabs["7. Perturbations"].fields}
    # Claimed by the stage (so unassigned stays empty and the editor can clone
    # Field objects), but not shown as ordinary rows -- see
    # gui.perturbation_editing.visible_tab_settings.
    from haemolynx.gui.perturbation_editing import (
        ALWAYS_VISIBLE_TAB_SETTINGS,
        EDITOR_SETTINGS,
        visible_tab_settings,
    )

    assert set(visible_tab_settings(sorted(claimed))) == set(ALWAYS_VISIBLE_TAB_SETTINGS)
    for name in ALWAYS_VISIBLE_TAB_SETTINGS:
        assert name in claimed
    for name in EDITOR_SETTINGS:
        if name in claimed:
            assert name not in ALWAYS_VISIBLE_TAB_SETTINGS
    for name in (
        "run_pericyte_dilation_sweep",
        "pericyte_dilation_min_percent",
        "arteriole_diameter_change_percent",
        "arteriole_dilation_min_percent",
        "capillary_dilation_min_percent",
        "sweep_output_dir",
        "run_pericyte_resistance_comparison",
        "pericyte_comparison_baseline_value",
    ):
        assert name in claimed
        assert name not in ALWAYS_VISIBLE_TAB_SETTINGS


def test_supplied_values_reach_the_right_tab():
    tabs = {tab.stage.title: tab for tab in tabs_for(SCHEMA, {"inlet_p_bc": 1234.0})}
    row = next(f for f in tabs["6. Haemodynamics"].fields if f.name == "inlet_p_bc")
    assert row.value == 1234.0


def test_rows_keep_the_schema_order_within_a_tab():
    """Reading down a tab should match reading down the config file."""
    tabs = {tab.stage.title: tab for tab in tabs_for(SCHEMA)}
    shown = [field.name for field in tabs["1. Input"].fields]
    declared = [s.name for s in SCHEMA if s.name in set(shown)]
    assert shown == declared


def test_vessel_mask_settings_live_on_boundaries_not_graph():
    """Volume-assignment masks configure Boundaries, not graph topology."""
    owner = assign_to_stages(SCHEMA)
    mask_names = list(SCHEMA.section_names("Vessel masks"))
    assert mask_names[0] == "automated_vessel_assignment"
    for name in mask_names:
        assert owner[name] == "4. Boundaries", name

    tabs = {tab.stage.title: tab for tab in tabs_for(SCHEMA)}
    graph = {field.name for field in tabs["3. Graph"].fields}
    boundaries = [field.name for field in tabs["4. Boundaries"].fields]
    for name in mask_names:
        assert name not in graph, name
        assert name in boundaries, name

    assert boundaries.index("automated_vessel_assignment") < boundaries.index(
        "use_large_vessel_masks"
    )
    assert boundaries.index("use_large_vessel_masks") < boundaries.index(
        "use_small_vessel_masks_for_boundary_assignment"
    )
    assert boundaries.index(
        "use_small_vessel_masks_for_boundary_assignment"
    ) < boundaries.index("inlet_node_selection_method")


def test_automated_assignment_documents_that_it_overrides_manual_methods():
    from haemolynx.gui.boundary_picking import AUTOMATED_OVERRIDES_MANUAL_NOTE

    help_text = SCHEMA["automated_vessel_assignment"].help.lower()
    assert "override" in help_text
    assert "manual" in help_text
    assert "override" in AUTOMATED_OVERRIDES_MANUAL_NOTE.lower()
    assert "manual" in AUTOMATED_OVERRIDES_MANUAL_NOTE.lower()
    assert SCHEMA["inlet_node_selection_method"].requires == (
        "!automated_vessel_assignment",
    )
    assert SCHEMA["outlet_node_selection_method"].requires == (
        "!automated_vessel_assignment",
    )


@pytest.mark.parametrize("title", tab_titles())
def test_every_tab_starts_with_a_number_so_the_order_is_visible(title):
    """Tabs are numbered, not stages: a stage that opens none needs no number."""
    assert title[0].isdigit()
