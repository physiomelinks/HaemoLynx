"""Bundle-into-paths settings reach a GUI form, but are not live.

Thickness-gated skeletonisation is wired; these tests pin that the bundle knobs
stay off the live schema. Wiring them is: append the Setting tuples to
``pipeline/schema.py`` and the names to the ``skeletonise`` entry of
``pipeline/progress.py``.
"""
from __future__ import annotations

from haemolynx.gui.form import fields_for, label_for
from haemolynx.parsers import Schema, parameters_of, prefixed_arguments
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.progress import STAGES
from haemolynx.preprocessing import preprocess_skeleton_for_graph
from haemolynx.preprocessing.proposed_skeleton_settings import (
    BUNDLE_INTO_PATHS_SETTINGS,
    BUNDLE_INTO_PATHS_STAGE_SETTING_NAMES,
    PROPOSED_SKELETONISE_SETTING_NAMES,
    proposed_skeleton_schema_extension,
)


def _schema_with_proposed_rows() -> Schema:
    """What the pipeline schema would look like after the wiring step."""
    live = default_schema()
    return Schema(
        list(live) + list(proposed_skeleton_schema_extension()),
        title=live.title,
        description=live.description,
    )


def test_proposed_settings_are_not_on_the_live_schema():
    live = set(default_schema().names)
    leaked = [name for name in PROPOSED_SKELETONISE_SETTING_NAMES if name in live]
    assert not leaked, f"proposed settings were wired into default_schema: {leaked}"


def test_proposed_settings_are_not_on_the_live_skeletonise_tab():
    skeletonise = next(stage for stage in STAGES if stage.call == "skeletonise")
    leaked = [
        name
        for name in PROPOSED_SKELETONISE_SETTING_NAMES
        if name in skeletonise.settings
    ]
    assert not leaked, f"proposed settings were wired into STAGES: {leaked}"


def test_bundle_setting_names_match_preprocess_skeleton_for_graph():
    """``skeleton_bundle_*`` is what prefixed_arguments already forwards."""
    parameters = set(parameters_of(preprocess_skeleton_for_graph))
    mapped = {
        name[len("skeleton_") :]
        for name in BUNDLE_INTO_PATHS_STAGE_SETTING_NAMES
    }
    assert mapped <= parameters
    assert mapped == {
        "bundle_scan_size",
        "bundle_density_fraction",
        "bundle_max_connections_per_hub",
        "bundle_hub_min_spacing",
    }


def test_prefixed_arguments_would_forward_bundle_knobs_if_present():
    settings = {setting.name: setting.default for setting in BUNDLE_INTO_PATHS_SETTINGS}
    forwarded = prefixed_arguments(
        settings,
        "skeleton_",
        parameters_of(preprocess_skeleton_for_graph),
    )
    assert forwarded == {
        "bundle_scan_size": 9,
        "bundle_density_fraction": 0.35,
        "bundle_max_connections_per_hub": 8,
        "bundle_hub_min_spacing": 4,
    }


def test_bundle_rows_would_appear_on_the_form():
    fields = {field.name: field for field in fields_for(_schema_with_proposed_rows())}
    for setting in BUNDLE_INTO_PATHS_SETTINGS:
        row = fields[setting.name]
        assert row.section == "Pipeline stages"
        assert row.help.strip()
        assert label_for(setting.name, setting.unit)
    assert fields["skeleton_bundle_scan_size"].widget_type == "SpinBox"
    assert fields["skeleton_bundle_density_fraction"].widget_type == "FloatSpinBox"


def test_bundle_rows_grey_out_when_skeletonise_is_off():
    schema = _schema_with_proposed_rows()
    fields = {field.name: field for field in fields_for(schema)}
    values_off = {setting.name: setting.default for setting in schema}
    values_off["do_skeletonize"] = False
    values_on = dict(values_off)
    values_on["do_skeletonize"] = True
    for name in BUNDLE_INTO_PATHS_STAGE_SETTING_NAMES:
        assert fields[name].is_enabled(values_off) is False
        assert fields[name].is_enabled(values_on) is True


def test_wiring_recipe_names_match_the_declared_settings():
    assert BUNDLE_INTO_PATHS_STAGE_SETTING_NAMES == tuple(
        s.name for s in BUNDLE_INTO_PATHS_SETTINGS
    )
    assert set(PROPOSED_SKELETONISE_SETTING_NAMES) == {
        s.name for s in proposed_skeleton_schema_extension()
    }
