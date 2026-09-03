"""Every pipeline setting must expose a non-empty GUI tooltip.

Schema ``help`` is the YAML comment and the napari hover text
(:func:`haemolynx.gui.form.field_for` appends the unit; ``_build_row`` assigns
``widget.tooltip = field.help``). An empty help string is already rejected at
``Setting`` construction; this file pins the whole ``default_schema()`` and
every tab row so a future setting cannot ship without a clear tooltip.
"""
from __future__ import annotations

from haemolynx.gui.form import field_for, fields_for
from haemolynx.gui.tabs import tabs_for, unassigned
from haemolynx.pipeline import default_schema

SCHEMA = default_schema()


def test_every_default_schema_setting_has_non_empty_help():
    """Strict: every setting in ``default_schema()`` carries usable help."""
    empty = [setting.name for setting in SCHEMA if not setting.help.strip()]
    assert not empty, f"settings with empty help: {empty}"
    assert len(list(SCHEMA)) > 0


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
    """Help is one imperative phrase, no trailing full stop (YAML + tooltip)."""
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


def test_form_tooltip_includes_unit_when_the_setting_declares_one():
    """Unit rides on the label and is repeated in the tooltip for hover."""
    for setting in SCHEMA:
        if not setting.unit:
            continue
        field = field_for(setting)
        assert field.help.strip(), setting.name
        assert field.help.endswith(f"({setting.unit})"), setting.name
        assert setting.help.strip() in field.help


def test_non_schema_panel_controls_expose_tooltip_strings():
    """Chrome buttons are not settings; their tooltips live in chrome_tooltips."""
    from haemolynx.gui import chrome_tooltips as chrome

    named = [
        chrome.SHOW_BOUNDARIES_TOOLTIP,
        chrome.SNAP_BOUNDARIES_TOOLTIP,
        chrome.LOAD_CONFIG_TOOLTIP,
        chrome.SAVE_CONFIG_TOOLTIP,
        chrome.RUN_CHECKS_TOOLTIP,
        chrome.RUN_PIPELINE_TOOLTIP,
        chrome.CLEAR_LAYERS_TOOLTIP,
        chrome.SHOW_RESULTS_TOOLTIP,
        chrome.SHOW_STEPS_TOOLTIP,
        chrome.USE_LAYER_TOOLTIP,
        chrome.REVERT_STAGE_TOOLTIP,
        *chrome.ACTION_TOOLTIPS.values(),
    ]
    empty = [text for text in named if not text.strip()]
    assert not empty
    assert set(chrome.ACTION_TOOLTIPS) == {
        "pick",
        "draw",
        "depth",
        "move",
        "assign",
        "clear",
    }
    for text in named:
        assert len(text.strip().split()) >= 3, text
        assert not text.rstrip().endswith("."), text
