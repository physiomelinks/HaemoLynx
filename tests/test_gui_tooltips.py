"""Every pipeline setting and panel control has hover text a new user can read.

A setting's row shows its entry in :mod:`haemolynx.gui.setting_tooltips`
(:func:`haemolynx.gui.form.field_for` appends the unit; ``_build_row`` assigns
``widget.tooltip = field.help``); its schema ``help`` is the config file's
comment. Other controls' hover text is in :mod:`haemolynx.gui.chrome_tooltips`.
This file pins that every setting has an entry, and holds every string to
one voice: a sentence or two, no setting names, code or history.
"""
from __future__ import annotations

import re

from haemolynx.gui.form import CHOICE_VALUE_LABELS, field_for, fields_for
from haemolynx.gui.setting_tooltips import SETTING_TOOLTIPS, tooltip_for
from haemolynx.gui.tabs import tabs_for, unassigned
from haemolynx.pipeline import default_schema

SCHEMA = default_schema()

#: The most words a hover text may run to: a sentence or two.
MAX_TOOLTIP_WORDS = 35
#: A setting's own name (``use_ilastik_segmentation``) or another identifier.
SNAKE_CASE = re.compile(r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b")
#: What a new user has no use for: dates, datasets, people, plans, code.
NOT_FOR_A_NEW_USER = re.compile(
    r"\bE14\.5\b|\bpatella\b|\bHarvey\b|\bMason\b|\bPLAN\b|\b\d{4}-\d{2}-\d{2}\b"
    r"|\.py\b|\bet al\b|haemolynx\.|G\.graph"
)
#: Values a dropdown shows as they are spelled, so hover text may name them.
SHOWN_CHOICE_VALUES = frozenset(
    str(value)
    for setting in SCHEMA
    for value in (setting.choices or ())
    if str(value) not in CHOICE_VALUE_LABELS.get(setting.name, {})
)


def _voice_problems(text: str) -> list[str]:
    """What keeps *text* from being hover text a new user can read."""
    problems = []
    words = len(text.split())
    if not 3 <= words <= MAX_TOOLTIP_WORDS:
        problems.append(f"{words} words")
    if text != text.strip():
        problems.append("surrounding whitespace")
    if text.rstrip().endswith("."):
        problems.append("trailing full stop")
    names = [name for name in SNAKE_CASE.findall(text) if name not in SHOWN_CHOICE_VALUES]
    if names:
        problems.append(f"names {names}")
    if NOT_FOR_A_NEW_USER.search(text):
        problems.append(f"mentions {NOT_FOR_A_NEW_USER.search(text).group()!r}")
    return problems


def _chrome_strings() -> dict[str, str]:
    """Every public string, and every string in a public dict, of the modules
    holding non-setting hover text."""
    from haemolynx.gui import boundary_picking, chrome_tooltips, perturbation_editing

    found: dict[str, str] = {}
    for module, names in (
        (chrome_tooltips, None),
        (perturbation_editing, ("NAME_TOOLTIP", "TYPE_TOOLTIP", "ADD_TOOLTIP", "REMOVE_TOOLTIP")),
        (boundary_picking, ("DISABLED_ROLE_TOOLTIP", "SYNTHETIC_CHOOSES_NOTE")),
    ):
        for name in names or [name for name in vars(module) if not name.startswith("_")]:
            value = getattr(module, name)
            if isinstance(value, str):
                found[f"{module.__name__}.{name}"] = value
            elif isinstance(value, dict):
                for key, text in value.items():
                    if isinstance(text, str):
                        found[f"{module.__name__}.{name}[{key!r}]"] = text
    return found


def test_every_default_schema_setting_has_non_empty_help():
    """Strict: every setting in ``default_schema()`` carries usable help."""
    empty = [setting.name for setting in SCHEMA if not setting.help.strip()]
    assert not empty, f"settings with empty help: {empty}"
    assert len(list(SCHEMA)) > 0


def test_every_setting_has_its_own_hover_text():
    """A setting added without an entry would show its whole config-file help."""
    missing = [setting.name for setting in SCHEMA if setting.name not in SETTING_TOOLTIPS]
    assert not missing, (
        f"settings with no hover text in haemolynx/gui/setting_tooltips.py: {missing}"
    )
    stale = sorted(set(SETTING_TOOLTIPS) - set(SCHEMA.names))
    assert not stale, f"hover text for settings the schema no longer has: {stale}"


def test_every_setting_hover_text_reads_for_a_new_user():
    problems = {
        name: found for name, text in SETTING_TOOLTIPS.items() if (found := _voice_problems(text))
    }
    assert not problems, problems


def test_every_form_row_gets_a_non_empty_tooltip_string():
    """``field.help`` is what the panel assigns to each widget's tooltip."""
    empty = [field.name for field in fields_for(SCHEMA) if not field.help.strip()]
    assert not empty, f"form rows with empty tooltip text: {empty}"
    assert {field.name for field in fields_for(SCHEMA)} == set(SCHEMA.names)


def test_every_pipeline_tab_row_has_a_non_empty_tooltip():
    """Input through Export / Perturbations: every visible form row has help."""
    assert unassigned(SCHEMA) == []
    missing: list[tuple[str, str]] = []
    for tab in tabs_for(SCHEMA):
        for field in tab.fields:
            if not field.help.strip():
                missing.append((tab.stage.title, field.name))
    assert not missing, f"tab rows with empty tooltips: {missing}"


def test_schema_help_matches_the_documented_voice():
    """Help is one imperative phrase, no trailing full stop (the YAML comment)."""
    with_period = [
        setting.name
        for setting in SCHEMA
        if setting.help.rstrip().endswith(".")
    ]
    assert not with_period, (
        f"help should not end with a full stop: {with_period}"
    )
    too_thin = [
        setting.name
        for setting in SCHEMA
        if len(setting.help.strip().split()) < 3
    ]
    assert not too_thin, f"help too short to be clear: {too_thin}"


def test_form_tooltip_is_the_hover_text_with_the_unit():
    """Unit rides on the label and is repeated in the tooltip for hover."""
    for setting in SCHEMA:
        field = field_for(setting)
        assert field.help.startswith(SETTING_TOOLTIPS[setting.name]), setting.name
        if setting.unit:
            assert field.help == f"{SETTING_TOOLTIPS[setting.name]} ({setting.unit})", setting.name
        else:
            assert field.help == SETTING_TOOLTIPS[setting.name], setting.name


def test_a_setting_with_no_hover_text_shows_its_help():
    """An example's schema adds settings the table does not know."""
    from haemolynx.parsers.schema import Setting

    setting = Setting("not_a_pipeline_setting", "int", 1, "Set how many widgets", "Example")
    assert tooltip_for(setting) == "Set how many widgets"


def test_non_schema_panel_controls_read_for_a_new_user():
    """Chrome buttons are not settings; their tooltips live in chrome_tooltips.
    That every control has one is scanned built, in
    ``test_gui_hover_text_widget.py``; this holds every string to one voice."""
    from haemolynx.gui import chrome_tooltips as chrome

    strings = _chrome_strings()
    assert len(strings) > 250
    for name in ("RUN_PIPELINE_TOOLTIP", "HELP_TOOLTIP", "OPTIMISE_SYNTHETIC_TOOLTIP"):
        assert f"haemolynx.gui.chrome_tooltips.{name}" in strings, name
    problems = {name: found for name, text in strings.items() if (found := _voice_problems(text))}
    assert not problems, problems
    assert set(chrome.ACTION_TOOLTIPS) == {
        "pick",
        "draw",
        "depth",
        "move",
        "assign",
        "clear",
        "pick_nodes",
        "clear_nodes",
        "insert_box",
        "box_choice",
        "box_size",
        "box_colour",
        "box_step",
        "box_move",
        "box_scan",
        "box_nodes",
        "use_node",
        "remove_box",
    }


def test_every_stage_tab_says_what_it_is_for():
    """A tab's hover text is the summary its page opens with, never code."""
    from haemolynx.gui.chrome_tooltips import stage_tab_tooltip

    for tab in tabs_for(SCHEMA):
        text = stage_tab_tooltip(tab.stage)
        assert text == tab.stage.summary.rstrip("."), tab.stage.title
        assert "(" not in text.split()[0] and not _voice_problems(text), (tab.stage.title, text)


def test_voice_check_catches_what_it_is_for():
    """The checker itself: each rule fires on the text it is meant for."""
    assert _voice_problems("Fill small holes inside vessels before skeletonising") == []
    assert _voice_problems("Fill small holes.") == ["trailing full stop"]
    assert _voice_problems("Too short") == ["2 words"]
    assert _voice_problems(" ".join(["word"] * 36)) == ["36 words"]
    assert _voice_problems("Needs use_ilastik_segmentation turned on") == [
        "names ['use_ilastik_segmentation']"
    ]
    # A value its dropdown shows as spelled may be named.
    assert "terminal_resistance" in SHOWN_CHOICE_VALUES
    assert _voice_problems("Drain each through terminal_resistance instead") == []
    assert _voice_problems("Fitted on E14.5 in October") == ["mentions 'E14.5'"]


def test_every_group_box_and_move_has_its_own_hover_text():
    """One string per optimiser group, box axis, box move and hover metric:
    a group, direction or metric added without one has none in the panel."""
    from haemolynx.gui import chrome_tooltips as chrome
    from haemolynx.gui.boundary_picking import MOVE_DIRECTIONS
    from haemolynx.gui.branch_hover import BRANCH_HOVER_METRICS
    from haemolynx.optimisation import FWHM_GROUP_NAMES, GROUP_NAMES

    assert set(chrome.OPTIMISE_GROUP_TOOLTIPS) == set(GROUP_NAMES)
    assert set(chrome.FWHM_OPTIMISE_GROUP_TOOLTIPS) == set(FWHM_GROUP_NAMES)
    assert set(chrome.BOX_SIZE_TOOLTIPS) == set("zyx")
    assert set(chrome.BOX_MOVE_TOOLTIPS) == set(MOVE_DIRECTIONS)
    assert set(chrome.BRANCH_HOVER_TOOLTIPS) == set(BRANCH_HOVER_METRICS)
    assert set(chrome.Z_DEPTH_TOOLTIPS) == {"min", "max"}
