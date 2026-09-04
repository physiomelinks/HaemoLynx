"""The schema-to-form mapping, checked without a GUI.

`Schema.describe()` was built so a form could be generated rather than
hand-written; this is the first thing to use it. The mapping is pure, so the
part most likely to be wrong -- a setting with no widget, a range that silently
clamps a legal value, a prerequisite that greys out the wrong row -- is
testable with no napari, no Qt and no display.
"""
from __future__ import annotations

import pytest

from pathlib import Path

from haemolynx.gui.form import (
    DEFAULT_FLOAT_RANGE,
    DEFAULT_INT_RANGE,
    HIDE_WHEN_UNMET_PARENTS,
    HIDE_WHEN_UNMET_SECTIONS,
    OPTIONS_BY_WIDGET,
    SHARED_ILASTIK_SETTINGS,
    SHARED_ILASTIK_SETTING_SET,
    WIDGET_TYPES,
    Field,
    field_for,
    fields_for,
    label_for,
    sections_for,
    shared_ilastik_host,
    values_from,
    visible_diameter_settings,
    visible_graph_centreline_settings,
    visible_input_segmentation_settings,
    visible_statistics_settings,
    visible_vessel_mask_settings,
)
from haemolynx.parsers import Schema, Setting
from haemolynx.pipeline import default_schema

SCHEMA = default_schema()


# --- every setting reaches the form -----------------------------------------


def test_every_setting_becomes_a_row():
    """A setting with no row is a setting a GUI user cannot reach."""
    names = {field.name for field in fields_for(SCHEMA)}
    assert names == set(SCHEMA.names)


def test_every_schema_kind_has_a_widget():
    """A new kind must be given a widget, not silently dropped."""
    kinds = {setting.kind for setting in SCHEMA}
    missing = kinds - set(WIDGET_TYPES)
    assert not missing, f"schema kinds with no widget: {sorted(missing)}"


def test_rows_are_grouped_into_the_schema_sections():
    grouped = sections_for(SCHEMA)
    assert list(grouped) == list(dict.fromkeys(s.section for s in SCHEMA))
    assert sum(len(rows) for rows in grouped.values()) == len(SCHEMA.names)


def test_the_defaults_the_form_starts_with_are_valid_settings():
    """Opening the panel and pressing run must not fail validation."""
    values = values_from(fields_for(SCHEMA))
    validated = SCHEMA.validate(values)
    assert set(validated) == set(SCHEMA.names)


def test_supplied_values_win_over_the_defaults():
    fields = fields_for(SCHEMA, {"min_stub_length": 42.0})
    by_name = {field.name: field for field in fields}
    assert by_name["min_stub_length"].value == 42.0
    assert by_name["cluster_collapse_distance"].value == SCHEMA["cluster_collapse_distance"].default


# --- what each widget is told ------------------------------------------------


def test_a_choice_setting_offers_exactly_its_choices():
    setting = next(s for s in SCHEMA if s.kind == "choice")
    field = field_for(setting)
    assert field.widget_type == "ComboBox"
    assert field.options["choices"] == list(setting.choices)


def test_a_declared_range_reaches_the_spin_box():
    field = field_for(
        Setting("count", "int", 3, "How many", "S", minimum=1, maximum=9)
    )
    assert (field.options["min"], field.options["max"]) == (1, 9)


def test_an_undeclared_range_is_wide_rather_than_absent():
    """magicgui's own default is 0-1000, which would clamp legal values."""
    field = field_for(Setting("count", "int", 3, "How many", "S"))
    assert (field.options["min"], field.options["max"]) == DEFAULT_INT_RANGE

    field = field_for(Setting("size", "float", 3.0, "How big", "S"))
    assert (field.options["min"], field.options["max"]) == DEFAULT_FLOAT_RANGE


def test_no_spin_box_can_clamp_a_value_the_schema_allows():
    """Every numeric default must sit inside the bounds the form gives it."""
    for field in fields_for(SCHEMA):
        if field.widget_type not in {"SpinBox", "FloatSpinBox"}:
            continue
        if field.value is None:
            continue
        assert field.options["min"] <= field.value <= field.options["max"], (
            f"{field.name}: default {field.value} is outside the widget range "
            f"{field.options['min']}..{field.options['max']}"
        )


@pytest.mark.parametrize(
    "name,must_exist,expected_mode",
    [
        ("output_dir", False, "d"),   # a directory, whether or not it exists
        ("plot_dir", True, "d"),
        ("input_path", True, "r"),    # must already be there: open it
        ("report_path", False, "w"),  # the run writes it: save dialogue
    ],
)
def test_a_path_opens_the_right_kind_of_dialogue(name, must_exist, expected_mode):
    field = field_for(Setting(name, "path", None, "A path", "S", must_exist=must_exist))
    assert field.widget_type == "FileEdit"
    assert field.options["mode"] == expected_mode


def test_the_unit_is_shown_with_the_help_text():
    field = field_for(Setting("length", "float", 1.0, "How long", "S", unit="um"))
    assert field.help == "How long (um)"


def test_labels_read_as_words():
    assert label_for("skeleton_closing_radius") == "Skeleton closing radius"


def test_a_label_carries_the_unit_so_the_row_itself_says_it():
    """A tooltip is only read by someone who already suspects a problem.

    Some lengths here are voxels and some are microns, and 10 is a reasonable
    value for either, so the row has to say which without being hovered.
    """
    assert label_for("min_stub_length", "um") == "Min stub length (um)"
    assert label_for("skeleton_bridge_gap_size", "voxels") == (
        "Skeleton bridge gap size (voxels)"
    )


def test_a_label_with_no_unit_gains_no_brackets():
    assert label_for("do_skeletonize", None) == "Do skeletonize"
    assert label_for("do_skeletonize", "") == "Do skeletonize"


def test_the_row_for_a_setting_with_a_unit_is_labelled_with_it():
    field = field_for(Setting("length", "float", 1.0, "How long", "S", unit="um"))
    assert field.label == "Length (um)"


def test_every_setting_that_declares_a_unit_shows_it_on_its_row():
    """The whole point, across the real schema rather than one example."""
    from haemolynx.pipeline import default_schema

    schema = default_schema()
    for field in fields_for(schema):
        unit = schema[field.name].unit
        if unit:
            assert field.label.endswith(f" ({unit})"), (
                f"{field.name} declares {unit!r} but its row reads {field.label!r}"
            )


def test_the_label_spells_the_unit_the_way_the_config_file_does():
    """"µm" on the row and "um" in the file reads as two different units."""
    from haemolynx.pipeline import default_schema

    for field in fields_for(default_schema()):
        assert "µ" not in field.label, field.label


# --- prerequisites gate the row ---------------------------------------------


def _gated_schema() -> Schema:
    return Schema(
        [
            Setting("use_ilastik", "bool", False, "Segment with ilastik", "S"),
            Setting("classifier", "path", None, "The trained classifier", "S",
                    requires=("use_ilastik",)),
            Setting("mask", "path", None, "An existing mask", "S",
                    requires=("!use_ilastik",)),
        ]
    )


def test_a_row_is_enabled_only_when_its_prerequisite_holds():
    fields = {field.name: field for field in fields_for(_gated_schema())}

    off = {"use_ilastik": False}
    assert not fields["classifier"].is_enabled(off)
    assert fields["mask"].is_enabled(off)

    on = {"use_ilastik": True}
    assert fields["classifier"].is_enabled(on)
    assert not fields["mask"].is_enabled(on)


def test_a_disabled_row_says_which_setting_disabled_it():
    fields = {field.name: field for field in fields_for(_gated_schema())}

    assert fields["classifier"].why_disabled({"use_ilastik": False}) == (
        "Not used while 'use_ilastik' is off."
    )
    assert fields["mask"].why_disabled({"use_ilastik": True}) == (
        "Not used while 'use_ilastik' is on."
    )


def test_an_enabled_row_gives_no_reason():
    fields = {field.name: field for field in fields_for(_gated_schema())}
    assert fields["classifier"].why_disabled({"use_ilastik": True}) == ""


def test_a_row_with_no_prerequisite_is_always_enabled():
    field = field_for(Setting("always", "bool", True, "On", "S"))
    assert field.enabled_by == ()
    assert field.is_enabled({})


def test_vessel_mask_rows_hide_when_requires_unmet_rather_than_only_greying():
    """Boundaries vessel options disappear until their parent toggles hold."""
    assert "Vessel masks" in HIDE_WHEN_UNMET_SECTIONS
    fields = {f.name: f for f in fields_for(SCHEMA)}

    automated = fields["automated_vessel_assignment"]
    assert not automated.hide_when_unmet
    assert automated.is_visible({})

    large = fields["use_large_vessel_masks"]
    assert large.hide_when_unmet
    assert large.section == "Vessel masks"
    assert not large.is_visible({"automated_vessel_assignment": False})
    assert large.is_visible({"automated_vessel_assignment": True})

    # Non-vessel sections still show when unmet (greyed by the panel, not hidden)
    # unless they are themselves a hide-when-unmet section (Input).
    inlet = fields["inlet_node_selection_method"]
    assert not inlet.hide_when_unmet
    assert inlet.is_visible({"automated_vessel_assignment": True})
    assert not inlet.is_enabled({"automated_vessel_assignment": True})


def test_input_ilastik_rows_hide_when_use_ilastik_segmentation_is_off():
    """Input swaps segmented-file vs main-ilastik children; shared knobs host."""
    assert "Input and segmentation" in HIDE_WHEN_UNMET_SECTIONS
    fields = {f.name: f for f in fields_for(SCHEMA)}

    toggle = fields["use_ilastik_segmentation"]
    assert not toggle.hide_when_unmet
    assert toggle.is_visible({})

    input_path = fields["input_path"]
    assert input_path.hide_when_unmet
    assert input_path.is_visible({"use_ilastik_segmentation": False})
    assert not input_path.is_visible({"use_ilastik_segmentation": True})

    for name in ("ilastik_unsegmented_image_path", "ilastik_classifier_path"):
        child = fields[name]
        assert child.hide_when_unmet, name
        assert not child.is_visible({"use_ilastik_segmentation": False}), name
        assert child.is_visible({"use_ilastik_segmentation": True}), name

    # Shared across main / large / small ilastik — Input only while main is on.
    for name in SHARED_ILASTIK_SETTINGS:
        shared = fields[name]
        assert shared.hide_when_unmet, name
        assert name in SHARED_ILASTIK_SETTING_SET
        assert not shared.is_visible({"use_ilastik_segmentation": False}), name
        assert shared.is_visible({"use_ilastik_segmentation": True}), name


def test_visible_input_segmentation_settings_swaps_on_ilastik_toggle():
    off = {"use_ilastik_segmentation": False}
    shown_off = visible_input_segmentation_settings(SCHEMA, off)
    assert "use_ilastik_segmentation" in shown_off
    assert "input_path" in shown_off
    assert "ilastik_unsegmented_image_path" not in shown_off
    assert "ilastik_classifier_path" not in shown_off
    assert "ilastik_executable" not in shown_off
    assert "ilastik_output_dir" not in shown_off
    assert "ilastik_output_suffix" not in shown_off
    assert "voxel_size_override_xyz" in shown_off

    on = {"use_ilastik_segmentation": True}
    shown_on = visible_input_segmentation_settings(SCHEMA, on)
    assert "input_path" not in shown_on
    assert "ilastik_unsegmented_image_path" in shown_on
    assert "ilastik_classifier_path" in shown_on
    assert "ilastik_executable" in shown_on
    assert "ilastik_output_dir" in shown_on
    assert "ilastik_output_suffix" in shown_on
    assert "use_ilastik_segmentation" in shown_on


def test_shared_ilastik_host_prefers_input_then_boundaries():
    assert shared_ilastik_host({}) is None
    assert shared_ilastik_host({"use_ilastik_segmentation": False}) is None
    assert shared_ilastik_host({"use_ilastik_segmentation": True}) == "input"
    assert (
        shared_ilastik_host(
            {
                "use_ilastik_segmentation": False,
                "use_ilastik_large_vessel_segmentation": True,
            }
        )
        == "boundaries"
    )
    assert (
        shared_ilastik_host(
            {
                "use_ilastik_segmentation": False,
                "use_ilastik_small_vessel_segmentation": True,
            }
        )
        == "boundaries"
    )
    # Main wins: still Input even when vessel ilastik is also on.
    assert (
        shared_ilastik_host(
            {
                "use_ilastik_segmentation": True,
                "use_ilastik_large_vessel_segmentation": True,
            }
        )
        == "input"
    )


def test_shared_ilastik_shows_on_boundaries_when_hidden_on_input():
    """Vessel-mask ilastik hosts the same three settings on Boundaries."""
    vessel_only = {
        "use_ilastik_segmentation": False,
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": True,
        "use_ilastik_large_vessel_segmentation": True,
        "use_small_vessel_masks_for_boundary_assignment": False,
    }
    assert shared_ilastik_host(vessel_only) == "boundaries"
    assert "ilastik_executable" not in visible_input_segmentation_settings(
        SCHEMA, vessel_only
    )
    shown = visible_vessel_mask_settings(SCHEMA, vessel_only)
    for name in SHARED_ILASTIK_SETTINGS:
        assert name in shown, name

    small_only = {
        "use_ilastik_segmentation": False,
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": False,
        "use_small_vessel_masks_for_boundary_assignment": True,
        "use_ilastik_small_vessel_segmentation": True,
    }
    shown = visible_vessel_mask_settings(SCHEMA, small_only)
    for name in SHARED_ILASTIK_SETTINGS:
        assert name in shown, name

    # File-based large masks (no vessel ilastik): shared stay off Boundaries.
    file_masks = {
        **vessel_only,
        "use_ilastik_large_vessel_segmentation": False,
    }
    assert shared_ilastik_host(file_masks) is None
    shown = visible_vessel_mask_settings(SCHEMA, file_masks)
    for name in SHARED_ILASTIK_SETTINGS:
        assert name not in shown, name

    # Both main and vessel: Input hosts; Boundaries does not duplicate.
    both = {
        "use_ilastik_segmentation": True,
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": True,
        "use_ilastik_large_vessel_segmentation": True,
    }
    assert shared_ilastik_host(both) == "input"
    for name in SHARED_ILASTIK_SETTINGS:
        assert name in visible_input_segmentation_settings(SCHEMA, both), name
        assert name not in visible_vessel_mask_settings(SCHEMA, both), name


def test_centreline_smoothing_children_hide_when_smooth_centrelines_is_off():
    """Graph nests method / iterations / max deviation under smooth_centrelines."""
    assert "smooth_centrelines" in HIDE_WHEN_UNMET_PARENTS
    fields = {f.name: f for f in fields_for(SCHEMA)}

    parent = fields["smooth_centrelines"]
    # Parent still greys under do_graph_building; it does not nest-hide itself.
    assert not parent.hide_when_unmet
    assert SCHEMA["smooth_centrelines"].requires == ("do_graph_building",)
    assert parent.is_visible({"do_graph_building": False, "smooth_centrelines": False})
    assert parent.is_visible({"do_graph_building": True, "smooth_centrelines": False})

    children = (
        "centreline_smoothing_method",
        "centreline_smoothing_iterations",
        "centreline_max_deviation",
    )
    for name in children:
        child = fields[name]
        assert child.hide_when_unmet, name
        assert SCHEMA[name].requires == ("smooth_centrelines",), name
        assert not child.is_visible(
            {"do_graph_building": True, "smooth_centrelines": False}
        ), name
        assert child.is_visible(
            {"do_graph_building": True, "smooth_centrelines": True}
        ), name

    off = {"do_graph_building": True, "smooth_centrelines": False}
    assert visible_graph_centreline_settings(SCHEMA, off) == {"smooth_centrelines"}

    on = {"do_graph_building": True, "smooth_centrelines": True}
    assert visible_graph_centreline_settings(SCHEMA, on) == {
        "smooth_centrelines",
        *children,
    }


def test_thick_vessel_children_hide_when_the_toggle_is_off():
    """Skeletonise nests radius and hole-fill under the thickness-gate toggle."""
    assert "use_thick_vessel_skeletonisation" in HIDE_WHEN_UNMET_PARENTS
    fields = {f.name: f for f in fields_for(SCHEMA)}

    parent = fields["use_thick_vessel_skeletonisation"]
    assert not parent.hide_when_unmet
    assert SCHEMA["use_thick_vessel_skeletonisation"].requires == ("do_skeletonize",)
    assert parent.widget_type == "CheckBox"
    assert parent.value is False

    children = (
        "skeleton_thick_vessel_min_radius_um",
        "skeleton_fill_mask_holes_before_thickness",
    )
    off = {"do_skeletonize": True, "use_thick_vessel_skeletonisation": False}
    on = {"do_skeletonize": True, "use_thick_vessel_skeletonisation": True}
    for name in children:
        child = fields[name]
        assert child.hide_when_unmet, name
        assert SCHEMA[name].requires == ("use_thick_vessel_skeletonisation",), name
        assert not child.is_visible(off), name
        assert child.is_visible(on), name

    radius = fields["skeleton_thick_vessel_min_radius_um"]
    assert radius.widget_type == "FloatSpinBox"
    assert radius.label.endswith("(um)")
    assert SCHEMA["skeleton_thick_vessel_min_radius_um"].default == pytest.approx(6.0)


def test_diameter_rows_hide_when_parent_toggles_are_unmet():
    """Diameters nests per-order tables and FWHM knobs under parent bools."""
    assert "Diameters and pericytes" in HIDE_WHEN_UNMET_SECTIONS
    assert "FWHM diameter measurement" in HIDE_WHEN_UNMET_SECTIONS
    fields = {f.name: f for f in fields_for(SCHEMA)}

    all_const = fields["all_diams_const"]
    assert not all_const.hide_when_unmet
    assert all_const.is_visible({})

    for name in (
        "manual_capillary_diameter_by_branch_order",
        "manual_arteriole_diameter_by_branch_order",
        "manual_venule_diameter_by_branch_order",
        "diameter_by_branch_order",
    ):
        child = fields[name]
        assert child.hide_when_unmet, name
        assert SCHEMA[name].requires == ("!all_diams_const",), name
        assert not child.is_visible({"all_diams_const": True}), name
        assert child.is_visible({"all_diams_const": False}), name

    for name in ("default_diameter", "max_branch_order"):
        assert not fields[name].hide_when_unmet, name
        assert fields[name].is_visible({"all_diams_const": True}), name
        assert fields[name].is_visible({"all_diams_const": False}), name

    fwhm = fields["use_fwhm_edge_diameters"]
    assert fwhm.hide_when_unmet
    assert not fwhm.is_visible({"run_haemodynamics": False})
    assert fwhm.is_visible({"run_haemodynamics": True})

    raw = fields["fwhm_raw_tiff_path"]
    assert raw.hide_when_unmet
    assert SCHEMA["fwhm_raw_tiff_path"].requires == ("use_fwhm_edge_diameters",)
    assert not raw.is_visible(
        {"run_haemodynamics": True, "use_fwhm_edge_diameters": False}
    )
    assert raw.is_visible(
        {"run_haemodynamics": True, "use_fwhm_edge_diameters": True}
    )

    nested = fields["fwhm_baseline_constraint_half_width_ptp"]
    assert nested.hide_when_unmet
    assert not nested.is_visible(
        {
            "run_haemodynamics": True,
            "use_fwhm_edge_diameters": True,
            "fwhm_constrain_fitted_baseline": False,
        }
    )
    assert nested.is_visible(
        {
            "run_haemodynamics": True,
            "use_fwhm_edge_diameters": True,
            "fwhm_constrain_fitted_baseline": True,
        }
    )


def test_visible_diameter_settings_nests_under_all_diams_const_and_fwhm():
    const_on = {
        "all_diams_const": True,
        "run_haemodynamics": True,
        "use_fwhm_edge_diameters": False,
    }
    shown = visible_diameter_settings(SCHEMA, const_on)
    assert "all_diams_const" in shown
    assert "default_diameter" in shown
    assert "max_branch_order" in shown
    assert "manual_capillary_diameter_by_branch_order" not in shown
    assert "manual_arteriole_diameter_by_branch_order" not in shown
    assert "manual_venule_diameter_by_branch_order" not in shown
    assert "diameter_by_branch_order" not in shown
    assert "use_fwhm_edge_diameters" in shown
    assert "fwhm_raw_tiff_path" not in shown
    assert "fwhm_sample_spacing_along_edge_um" not in shown

    const_off = {**const_on, "all_diams_const": False}
    shown = visible_diameter_settings(SCHEMA, const_off)
    assert "manual_capillary_diameter_by_branch_order" in shown
    assert "manual_arteriole_diameter_by_branch_order" in shown
    assert "manual_venule_diameter_by_branch_order" in shown
    assert "diameter_by_branch_order" in shown
    assert "fwhm_raw_tiff_path" not in shown

    fwhm_on = {
        **const_on,
        "use_fwhm_edge_diameters": True,
        "fwhm_constrain_fitted_baseline": False,
        "fwhm_clip_profile_to_single_vessel": False,
        "fwhm_enforce_same_edge_locality": False,
        "fwhm_cap_half_extent_by_nonlocal_same_edge_distance": False,
        "fwhm_reject_samples_with_center_offset": False,
        "fwhm_reject_samples_with_low_fit_r2": False,
    }
    shown = visible_diameter_settings(SCHEMA, fwhm_on)
    assert "fwhm_raw_tiff_path" in shown
    assert "fwhm_sample_spacing_along_edge_um" in shown
    assert "fwhm_constrain_fitted_baseline" in shown
    assert "fwhm_baseline_constraint_half_width_ptp" not in shown
    assert "fwhm_clip_min_drop_fraction_of_center" not in shown
    assert "fwhm_same_edge_arc_window_um" not in shown
    assert "fwhm_min_fit_r2" not in shown

    fwhm_on["fwhm_constrain_fitted_baseline"] = True
    shown = visible_diameter_settings(SCHEMA, fwhm_on)
    assert "fwhm_baseline_constraint_half_width_ptp" in shown


_MEASUREMENT_3D_CHILDREN = (
    "cell_mask_path",
    "cell_mask_h5_dataset_name",
    "measurement_3d_vessel_mask_path",
    "measurement_3d_vessel_mask_h5_dataset_name",
    "measurement_3d_reference_image_path",
    "measurement_3d_reference_h5_dataset_name",
)


def test_measurement_3d_rows_hide_when_measurement_3d_to_cell_mask_is_off():
    """Export nests cell-mask paths under measurement_3d_to_cell_mask."""
    assert "Statistics and measurements" in HIDE_WHEN_UNMET_SECTIONS
    fields = {f.name: f for f in fields_for(SCHEMA)}

    parent = fields["measurement_3d_to_cell_mask"]
    assert not parent.hide_when_unmet
    assert parent.is_visible({})
    assert parent.section == "Statistics and measurements"

    for name in _MEASUREMENT_3D_CHILDREN:
        child = fields[name]
        assert child.hide_when_unmet, name
        assert child.section == "Statistics and measurements", name
        assert SCHEMA[name].requires == ("measurement_3d_to_cell_mask",), name
        assert not child.is_visible({"measurement_3d_to_cell_mask": False}), name
        assert child.is_visible({"measurement_3d_to_cell_mask": True}), name

    # statistics_mode nests under statistics the same way (same section).
    mode = fields["statistics_mode"]
    assert mode.hide_when_unmet
    assert SCHEMA["statistics_mode"].requires == ("statistics",)
    assert not mode.is_visible({"statistics": False})
    assert mode.is_visible({"statistics": True})

    # Ungated parents stay visible either way.
    assert not fields["statistics"].hide_when_unmet
    assert fields["statistics"].is_visible({"statistics": False})
    assert fields["statistics"].is_visible({"statistics": True})


def test_visible_statistics_settings_nests_under_measurement_3d_to_cell_mask():
    off = {
        "statistics": False,
        "measurement_3d_to_cell_mask": False,
    }
    shown = visible_statistics_settings(SCHEMA, off)
    assert shown == {"statistics", "measurement_3d_to_cell_mask"}
    for name in _MEASUREMENT_3D_CHILDREN:
        assert name not in shown, name
    assert "statistics_mode" not in shown

    on = {**off, "measurement_3d_to_cell_mask": True}
    shown = visible_statistics_settings(SCHEMA, on)
    assert "measurement_3d_to_cell_mask" in shown
    assert "statistics" in shown
    for name in _MEASUREMENT_3D_CHILDREN:
        assert name in shown, name
    assert "statistics_mode" not in shown

    stats_on = {**off, "statistics": True}
    shown = visible_statistics_settings(SCHEMA, stats_on)
    assert shown == {"statistics", "measurement_3d_to_cell_mask", "statistics_mode"}
    for name in _MEASUREMENT_3D_CHILDREN:
        assert name not in shown, name


def test_visible_vessel_mask_settings_nests_under_automated_and_parents():
    off = {"automated_vessel_assignment": False}
    assert visible_vessel_mask_settings(SCHEMA, off) == {"automated_vessel_assignment"}

    auto_only = {
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": False,
        "use_small_vessel_masks_for_boundary_assignment": False,
    }
    shown = visible_vessel_mask_settings(SCHEMA, auto_only)
    assert shown == {
        "automated_vessel_assignment",
        "use_large_vessel_masks",
        "use_small_vessel_masks_for_boundary_assignment",
        "remove_disconnected_io_components_after_final_assignment",
    }

    large_on = {
        **auto_only,
        "use_large_vessel_masks": True,
        "use_ilastik_large_vessel_segmentation": False,
        "large_vessel_remove_small_opposite_attached_components": False,
        "automated_vessel_assignment_enable_overlap_cleanup": True,
        "automated_vessel_assignment_fast_mode": True,
        "automated_vessel_assignment_use_legacy_mode": True,
        "cut_network_at_large_vessel_volumes": False,
    }
    shown = visible_vessel_mask_settings(SCHEMA, large_on)
    assert "large_arteriole_mask_path" in shown
    assert "large_vessel_mask_dilation_microns" in shown
    assert "automated_vessel_assignment_fast_mode" in shown
    assert "write_fast_mode_preassignment_large_vessel_debug_3d_html" in shown
    assert "ilastik_arteriole_classifier_path" not in shown
    assert "automated_vessel_confidence_margin" not in shown
    assert "orphaned_branch_max_edge_count" not in shown
    assert "use_small_vessel_masks_for_boundary_assignment" in shown
    assert "small_arteriole_mask_path" not in shown

    large_on["automated_vessel_assignment_fast_mode"] = False
    shown = visible_vessel_mask_settings(SCHEMA, large_on)
    assert "write_fast_mode_preassignment_large_vessel_debug_3d_html" not in shown
    assert "automated_vessel_assignment_apply_overlap_cleanup_in_normal_mode" in shown

    large_on["cut_network_at_large_vessel_volumes"] = True
    large_on["remove_orphaned_branches_outside_large_vessel_volumes"] = False
    shown = visible_vessel_mask_settings(SCHEMA, large_on)
    assert "cut_network_at_large_vessel_volumes" in shown
    assert "cut_large_vessel_sample_densely" not in shown
    assert "remove_orphaned_branches_outside_large_vessel_volumes" in shown
    assert "orphaned_branch_max_edge_count" not in shown

    large_on["remove_orphaned_branches_outside_large_vessel_volumes"] = True
    shown = visible_vessel_mask_settings(SCHEMA, large_on)
    assert "orphaned_branch_max_edge_count" in shown

    small_on = {
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": False,
        "use_small_vessel_masks_for_boundary_assignment": True,
        "use_ilastik_small_vessel_segmentation": False,
        "small_vessel_mask_continuity_enable": False,
        "small_vessel_tangential_redefinition_enable": False,
        "small_vessel_boundary_assignment_enable_overlap_cleanup": True,
        "small_vessel_boundary_assignment_fast_mode": True,
        "small_vessel_boundary_fallback_to_hop_distance": False,
    }
    shown = visible_vessel_mask_settings(SCHEMA, small_on)
    assert "small_arteriole_mask_path" in shown
    assert "small_vessel_mask_continuity_enable" in shown
    assert "small_vessel_mask_continuity_allow_small_to_large" not in shown
    assert "ilastik_small_arteriole_classifier_path" not in shown

    small_on["small_vessel_mask_continuity_enable"] = True
    shown = visible_vessel_mask_settings(SCHEMA, small_on)
    assert "small_vessel_mask_continuity_allow_small_to_large" in shown


def test_fields_are_immutable():
    """The form reads the schema; it must not be able to edit it by accident."""
    field = fields_for(SCHEMA)[0]
    with pytest.raises(Exception):
        field.value = "changed"  # type: ignore[misc]
    assert isinstance(field, Field)


# --- an unset path shows as empty -------------------------------------------


def test_an_unset_path_starts_empty_rather_than_at_the_working_directory():
    """`input_path` has no default: the picker must not invent one.

    magicgui's FileEdit falls back to the current directory for a null value,
    which reads as a choice somebody made. An empty box says "not set", which
    is what the pre-run checks will also say.
    """
    by_name = {field.name: field for field in fields_for(SCHEMA)}
    assert SCHEMA["input_path"].default is None
    assert by_name["input_path"].value == ""


def test_a_path_that_has_a_default_still_shows_it():
    by_name = {field.name: field for field in fields_for(SCHEMA)}
    assert by_name["ilastik_output_dir"].value == SCHEMA["ilastik_output_dir"].default


def test_a_row_is_blank_only_when_the_setting_is_unset():
    """An empty box means "not set" -- so it must not appear for a set value."""
    for field in fields_for(SCHEMA):
        if field.value != "":
            continue
        default = SCHEMA[field.name].default
        assert default is None or default == "", (
            f"{field.name} shows as empty but its default is {default!r}"
        )


# --- unset must survive the round trip through a widget ----------------------


def test_a_setting_with_no_default_gets_a_widget_that_can_be_empty():
    """A FloatSpinBox cannot hold "unset"; it reports 0.0, which is a value.

    `fwhm_diameter_guess_um` is a float with no default. Shown in a spin box,
    the panel reads back 0.0, the schema sees a setting that is neither None
    nor its default, and warns that it is set while the feature that reads it
    is off -- for a panel nobody has touched.
    """
    by_name = {field.name: field for field in fields_for(SCHEMA)}
    assert SCHEMA["fwhm_diameter_guess_um"].default is None
    assert by_name["fwhm_diameter_guess_um"].widget_type == "LineEdit"
    assert by_name["fwhm_diameter_guess_um"].value == ""


def test_a_numeric_setting_that_has_a_default_keeps_its_spin_box():
    by_name = {field.name: field for field in fields_for(SCHEMA)}
    assert by_name["min_stub_length"].widget_type == "FloatSpinBox"


@pytest.mark.parametrize(
    "kind,default,raw,expected",
    [
        ("path", None, Path("."), None),        # FileEdit's empty value
        ("path", None, "", None),
        ("path", None, Path("/data/x.tif"), Path("/data/x.tif")),
        ("path", "out", Path("out"), Path("out")),
        ("float", None, "", None),              # LineEdit left empty
        ("float", None, "4.5", 4.5),
        ("int", None, "7", 7),
        ("float", 1.0, 2.5, 2.5),               # ordinary spin box
        ("str", None, "", None),
        ("str", "auto", "", ""),                # a real empty string, not unset
    ],
)
def test_a_widget_value_reads_back_as_the_setting_value(kind, default, raw, expected):
    from haemolynx.parsers import Setting

    field = field_for(Setting("thing", kind, default, "A thing", "S"))
    assert field.to_setting_value(raw) == expected


def test_the_untouched_panel_warns_about_nothing():
    """Opening the panel and reading it back must equal the schema defaults.

    This is the whole bug: a widget that invents a value for a setting with no
    default makes the schema report "set but nothing will read it" on a panel
    the user has not touched.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SCHEMA.validate(values_from(fields_for(SCHEMA)))

    assert [str(warning.message) for warning in caught] == []


def test_the_untouched_panel_produces_the_schema_defaults():
    read_back = SCHEMA.validate(values_from(fields_for(SCHEMA)))
    defaults = SCHEMA.validate({s.name: s.default for s in SCHEMA})
    assert read_back == defaults


# --- options must suit the widget, not the kind ------------------------------


def test_no_row_passes_an_option_its_widget_would_reject():
    """magicgui raises on an unknown keyword, and the panel fails to open.

    The options used to be chosen from the setting's kind. When a float with no
    default started using a LineEdit so it could be empty, it kept the spin
    box's options and magicgui refused it:

        LineEdit got an unexpected keyword argument: min, max, step

    Options now follow the widget the setting actually gets. This checks every
    row, so the next widget swap cannot repeat it.
    """
    offenders = {}
    for field in fields_for(SCHEMA):
        allowed = OPTIONS_BY_WIDGET[field.widget_type]
        extra = sorted(set(field.options) - allowed)
        if extra:
            offenders[field.name] = (field.widget_type, extra)
    assert offenders == {}, (
        f"rows whose options their widget would reject: {offenders}"
    )


def test_every_widget_in_use_declares_which_options_it_takes():
    used = {field.widget_type for field in fields_for(SCHEMA)}
    assert used <= set(OPTIONS_BY_WIDGET), (
        f"widgets with no declared options: {sorted(used - set(OPTIONS_BY_WIDGET))}"
    )


def test_a_numeric_setting_with_no_default_gets_no_range_options():
    from haemolynx.parsers import Setting

    field = field_for(Setting("guess", "float", None, "A guess", "S", minimum=0.0))
    assert field.widget_type == "LineEdit"
    assert field.options == {}


def test_a_numeric_setting_with_a_default_keeps_its_range():
    from haemolynx.parsers import Setting

    field = field_for(Setting("guess", "float", 1.0, "A guess", "S", minimum=0.0))
    assert field.widget_type == "FloatSpinBox"
    assert field.options["min"] == 0.0


def test_every_row_starts_with_a_value_its_widget_can_read_back():
    """A LiteralEvalLineEdit parses its text; "" is not a literal.

    Blanking an unset value made three rows -- voxel_size_override_xyz and the
    two diameter tables -- start empty, and magicgui raised
    `SyntaxError: invalid syntax (<unknown>, line 0)` from `ast.literal_eval`
    as soon as the panel read them, which is at construction. None renders as
    "None" and parses straight back, so that is what they get.
    """
    import ast

    for field in fields_for(SCHEMA):
        if field.widget_type != "LiteralEvalLineEdit":
            continue
        try:
            ast.literal_eval(str(field.value))
        except (ValueError, SyntaxError) as error:  # pragma: no cover - the failure
            raise AssertionError(
                f"{field.name} starts as {field.value!r}, which its "
                f"LiteralEvalLineEdit cannot parse: {error}"
            ) from error


def test_an_unset_literal_row_is_none_rather_than_empty():
    from haemolynx.parsers import Setting

    field = field_for(Setting("table", "mapping", None, "A table", "S"))
    assert field.widget_type == "LiteralEvalLineEdit"
    assert field.value is None
    assert field.to_setting_value(None) is None


def test_an_unset_path_or_number_is_still_blank():
    """The empty string is right for those two; only literals need None."""
    from haemolynx.parsers import Setting

    assert field_for(Setting("where", "path", None, "A path", "S")).value == ""
    assert field_for(Setting("guess", "float", None, "A guess", "S")).value == ""
