"""Where each row sits on its tab, and which Advanced button hides it.

Pure: :mod:`haemolynx.gui.layout` needs no Qt, so every placement rule is
checked here against the real schema, and the widget tests only check that the
panel draws what this decides.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from haemolynx.gui._widget import FORCED_HIDDEN_SETTINGS
from haemolynx.gui.boundary_picking import ROLES, role_settings, shared_settings
from haemolynx.gui.form import SHARED_ILASTIK_SETTING_SET
from haemolynx.gui.layout import (
    ADVANCED_GROUPS,
    ADVANCED_TITLES,
    ANCHOR_OVERRIDES,
    MOVE_BEFORE,
    TAB_BOXES,
    Box,
    Disclosure,
    changed_settings,
    layout_for,
)
from haemolynx.gui.perturbation_editing import visible_tab_settings
from haemolynx.gui.tabs import section_box_title, tabs_for
from haemolynx.parsers import Schema, Setting, prerequisite_name
from haemolynx.pipeline import default_schema

SCHEMA = default_schema()

#: Rows the panel gives no row at all (see haemolynx.gui._widget.settings_widget).
FORCED_HIDDEN = set(FORCED_HIDDEN_SETTINGS)


def _tab_names(tab) -> list[str]:
    """The rows the panel hands layout_for for *tab*, as settings_widget does."""
    names = [
        field.name
        for field in tab.fields
        if field.name not in SHARED_ILASTIK_SETTING_SET and field.name not in FORCED_HIDDEN
    ]
    if tab.stage.call == "assign_boundaries":
        placed = {n for role in ROLES for n in role_settings(role)} | set(shared_settings())
        names = [n for n in names if n not in placed]
    if tab.stage.call == "run_perturbations":
        names = list(visible_tab_settings(names))
    return names


def _layouts():
    return {
        tab.stage.call or tab.stage.title: (tab, layout_for(tab.stage.call, _tab_names(tab), SCHEMA))
        for tab in tabs_for(SCHEMA)
    }


LAYOUTS = _layouts()


def _closure(name: str) -> set[str]:
    seen: set[str] = set()
    stack = [prerequisite_name(r) for r in SCHEMA[name].requires]
    while stack:
        current = stack.pop()
        if current in seen or current not in SCHEMA:
            continue
        seen.add(current)
        stack.extend(prerequisite_name(r) for r in SCHEMA[current].requires)
    return seen


# --- every row, exactly once, on the right side of its button ----------------


@pytest.mark.parametrize("call", sorted(LAYOUTS))
def test_every_row_the_tab_hands_over_is_laid_out_exactly_once(call):
    tab, layout = LAYOUTS[call]
    names = _tab_names(tab)
    assert sorted(layout.names) == sorted(names)


@pytest.mark.parametrize("call", sorted(LAYOUTS))
def test_advanced_rows_are_behind_a_button_and_ordinary_ones_are_not(call):
    _tab, layout = LAYOUTS[call]
    for box in layout.boxes:
        for row in box.rows:
            assert not SCHEMA[row].advanced, row
        for disclosure in box.disclosures:
            for name in disclosure.names:
                assert SCHEMA[name].advanced, name


@pytest.mark.parametrize("call", sorted(LAYOUTS))
def test_a_button_sits_below_its_anchor_in_the_same_box(call):
    _tab, layout = LAYOUTS[call]
    for box in layout.boxes:
        items = list(box.items)
        for index, item in enumerate(items):
            if not isinstance(item, Disclosure) or item.anchor is None:
                continue
            assert item.anchor in box.rows
            assert items.index(item.anchor) < index


@pytest.mark.parametrize("call", sorted(LAYOUTS))
def test_every_row_behind_an_anchored_button_needs_that_anchor(call):
    _tab, layout = LAYOUTS[call]
    for disclosure in layout.disclosures:
        if disclosure.anchor is None:
            continue
        for name in disclosure.names:
            lineage = {name} | _closure(name)
            if any(ANCHOR_OVERRIDES.get(each) == disclosure.anchor for each in lineage):
                continue  # placed by hand: see ANCHOR_OVERRIDES
            assert disclosure.anchor in _closure(name), name


@pytest.mark.parametrize("call", sorted(LAYOUTS))
def test_a_row_whose_toggle_is_advanced_is_behind_the_same_button(call):
    """Otherwise opening one button could show a child whose toggle is shut away."""
    _tab, layout = LAYOUTS[call]
    for disclosure in layout.disclosures:
        for name in disclosure.names:
            for parent in _closure(name):
                if parent in layout.names and SCHEMA[parent].advanced:
                    assert parent in disclosure.names, (name, parent)


@pytest.mark.parametrize("call", sorted(LAYOUTS))
def test_no_ordinary_row_needs_a_toggle_that_is_behind_a_button(call):
    """An ordinary row whose toggle is advanced would show on its own,
    detached from the toggle deciding whether it applies -- a new setting
    nested under an advanced one must be advanced too."""
    _tab, layout = LAYOUTS[call]
    on_tab = set(layout.names)
    for box in layout.boxes:
        for row in box.rows:
            hidden_parents = [p for p in _closure(row) if p in on_tab and SCHEMA[p].advanced]
            assert hidden_parents == [], (row, hidden_parents)


@pytest.mark.parametrize("call", sorted(LAYOUTS))
def test_a_button_follows_the_last_ordinary_row_of_its_anchors_subtree(call):
    _tab, layout = LAYOUTS[call]
    for box in layout.boxes:
        items = list(box.items)
        for index, item in enumerate(items):
            if not isinstance(item, Disclosure) or item.anchor is None:
                continue
            later_rows = [i for i in items[index + 1:] if isinstance(i, str)]
            for row in later_rows:
                assert item.anchor not in _closure(row), (item.title, row)


def test_disclosure_keys_are_unique_across_the_panel():
    keys = [d.key for _tab, layout in LAYOUTS.values() for d in layout.disclosures]
    assert len(keys) == len(set(keys))


def test_the_panel_hides_about_three_in_five_settings():
    """The point of the change: most settings are advanced, a minority are not."""
    advanced = sum(setting.advanced for setting in SCHEMA)
    assert 0.5 < advanced / len(SCHEMA) < 0.7


# --- the rows the user asked about -----------------------------------------


def test_viscosity_law_and_do_fwhm_measurement_stay_ordinary_rows():
    assert not SCHEMA["viscosity_law"].advanced
    assert not SCHEMA["do_fwhm_measurement"].advanced
    _tab, haemodynamics = LAYOUTS["build_haemodynamic_model"]
    assert "viscosity_law" in haemodynamics.box("blood_model").rows


def test_the_decoy_check_is_behind_a_button_under_do_fwhm_measurement():
    _tab, layout = LAYOUTS["assign_diameters"]
    decoy = layout.disclosure_of("fwhm_decoy_check")
    assert decoy is not None
    assert decoy.anchor == "do_fwhm_measurement"
    assert "fwhm_decoy_check_sample_size" in decoy.names


def test_do_fwhm_measurement_sits_directly_above_the_rows_it_gates():
    assert MOVE_BEFORE["do_fwhm_measurement"] == "use_raw_section_fallback"
    _tab, layout = LAYOUTS["assign_diameters"]
    rows = list(layout.box("diameters").rows)
    assert rows.index("use_raw_section_fallback") == rows.index("do_fwhm_measurement") + 1


def test_fwhm_buttons_nest_one_level_per_toggle():
    _tab, layout = LAYOUTS["assign_diameters"]
    fwhm = layout.disclosure_of("fwhm_min_fit_r2")
    decoy = layout.disclosure_of("fwhm_decoy_check")
    raw = layout.disclosure_of("raw_section_min_lumen_contrast")
    assert (fwhm.anchor, decoy.anchor, raw.anchor) == (
        "use_fwhm_edge_diameters",
        "do_fwhm_measurement",
        "use_raw_section_fallback",
    )
    assert fwhm.level < decoy.level < raw.level
    items = list(layout.box("diameters").items)
    # Innermost first, so each button sits closest to the row it belongs to.
    assert items.index(raw) < items.index(decoy) < items.index(fwhm)


def test_a_button_anchored_on_a_row_with_no_children_follows_it_immediately():
    _tab, layout = LAYOUTS["segment"]
    items = list(layout.box("segmentation_cleanup").items)
    necks = layout.disclosure_of("segmentation_cleanup_split_min_pinch_radius_ratio")
    assert items[items.index("segmentation_cleanup_split_narrow_necks") + 1] is necks


def test_a_box_level_button_ends_its_box():
    _tab, layout = LAYOUTS["segment"]
    box = layout.box("check_and_optimise")
    assert box.rows == ()
    (button,) = box.disclosures
    assert button.anchor is None
    assert set(button.names) == {
        "expected_boundary_vessel_count",
        "min_voxels_across_vessel_radius",
        "min_acceptable_segmentation_quality",
    }


def test_tiling_goes_behind_the_skeletonise_button_although_it_needs_low_ram():
    """ANCHOR_OVERRIDES: tiling's prerequisite is on the Input tab."""
    _tab, layout = LAYOUTS["skeletonise"]
    tiling = layout.disclosure_of("skeletonize_tile_large_components")
    assert tiling.anchor == "do_skeletonize"
    assert {"skeletonize_tile_max_voxels", "skeletonize_tile_halo_um"} <= set(tiling.names)


def test_thick_vessel_mask_restriction_moved_to_skeletonise():
    _tab, skeletonise = LAYOUTS["skeletonise"]
    _tab, boundaries = LAYOUTS["assign_boundaries"]
    for name in (
        "skeleton_thick_vessel_restrict_to_mask",
        "skeleton_thick_vessel_restrict_to_mask_warn_below",
    ):
        assert skeletonise.disclosure_of(name).anchor == "use_thick_vessel_skeletonisation"
        assert name not in boundaries.names


def test_the_cartwheel_guard_joins_the_graph_tabs_checks():
    _tab, layout = LAYOUTS["build_network"]
    checks = layout.disclosure_of("detect_cartwheel_hub_artifacts")
    assert checks.anchor is None
    assert checks.title == "Checks"
    assert "graph_mask_consistency_warn_below" in checks.names
    assert "cartwheel_hub_min_degree" in checks.names
    assert [b.key for b in layout.boxes] == ["pipeline_stages"]


def test_the_connectivity_box_keeps_its_measures_behind_its_own_button():
    _tab, layout = LAYOUTS["9. Additional measurements"]
    box = layout.box("connectivity_network_analysis")
    assert box.title == "Connectivity/Network Analysis"
    assert box.rows == ("statistics_network_analysis", "statistics_mode")
    (measures,) = box.disclosures
    assert measures.title == "Choose network measures"
    assert "statistics_cyclomatic_number" in measures.names
    statistics = layout.disclosure_of("statistics_basic")
    assert statistics.title == "Choose statistics"
    assert statistics.anchor == "statistics"


# --- hand-made boxes -------------------------------------------------------


def test_the_haemodynamics_tab_leads_with_its_master_toggle():
    _tab, layout = LAYOUTS["build_haemodynamic_model"]
    assert [(b.key, b.title) for b in layout.boxes] == [
        ("haemodynamics", None),
        ("boundary_pressures", "Boundary pressures"),
        ("blood_model", "Blood model"),
        ("network_handling", "Network handling"),
    ]
    assert layout.boxes[0].rows == ("run_haemodynamics",)
    iteration = layout.disclosure_of("haematocrit_distribution_tolerance")
    assert iteration.anchor == "haematocrit_model"
    # The junction rule is a dropdown in plain view, under the model it
    # belongs to, not behind that model's Advanced button.
    blood = layout.boxes[2].rows
    assert blood.index("haematocrit_junction_rule") == blood.index("haematocrit_model") + 1
    assert layout.disclosure_of("haematocrit_junction_rule") is None
    # The split's connector length only matters to split_junctions, so it sits
    # behind a button of its own under the rule, not with the iteration's.
    splitting = layout.disclosure_of("haematocrit_split_connector_length_um")
    assert splitting.anchor == "haematocrit_junction_rule"
    assert splitting.title == "Advanced junction splitting"


def test_the_export_tab_leads_with_where_the_outputs_go():
    _tab, layout = LAYOUTS["export_results"]
    assert layout.boxes[0].rows[0] == "vtk_output_prefix"
    assert layout.disclosure_of("visualize_vtk").anchor == "vtk_export"
    plots = layout.disclosure_of("show_plots_in_ide")
    assert plots.anchor is None
    assert {"visualize_results", "base_plot_dir", "verbose_logging"} <= set(plots.names)


def test_boundaries_box_each_mask_kind_by_its_own_toggle():
    _tab, layout = LAYOUTS["assign_boundaries"]
    keys = [b.key for b in layout.boxes]
    # "boundary_other" appears only once some setting needs it.
    assert [k for k in keys if k != "boundary_other"] == [
        "vessel_masks",
        "large_vessel_masks",
        "small_vessel_masks",
        "boundary_nodes",
    ]
    if "boundary_other" in keys:
        assert keys.index("boundary_other") < keys.index("boundary_nodes")
    large = layout.box("large_vessel_masks")
    small = layout.box("small_vessel_masks")
    for name in large.names:
        assert name == "use_large_vessel_masks" or "use_large_vessel_masks" in _closure(name)
    for name in small.names:
        assert name == "use_small_vessel_masks_for_boundary_assignment" or (
            "use_small_vessel_masks_for_boundary_assignment" in _closure(name)
        )
    assert set(layout.box("boundary_nodes").names) == {
        "large_arteriole_boundary_nodes",
        "large_venule_boundary_nodes",
    }


def test_an_unplaced_boundaries_setting_gets_its_own_button_not_the_node_ids():
    """A new Boundaries setting no role page places must not be filed under
    the node IDs a run fills in: it gets its own button, above the role tabs."""
    schema = Schema([
        Setting("automated_vessel_assignment", "bool", False, "Assign from masks for a test", "Vessel masks"),
        Setting(
            "large_arteriole_boundary_nodes", "any", [], "Node IDs for a test", "Boundary assignment",
            advanced=True,
        ),
        Setting(
            "large_venule_boundary_nodes", "any", [], "Node IDs for a test", "Boundary assignment",
            advanced=True,
        ),
        Setting(
            "open_end_distance", "float", 10.0, "Distance to a face for a test", "Boundary assignment",
            advanced=True,
        ),
    ])
    layout = layout_for("assign_boundaries", list(schema.names), schema)
    assert [b.key for b in layout.boxes] == ["vessel_masks", "boundary_other", "boundary_nodes"]
    other = layout.disclosure_of("open_end_distance")
    assert other.title == "Advanced boundary settings"
    assert "open_end_distance" not in layout.box("boundary_nodes").names


def test_the_input_tab_puts_the_voxel_size_policy_above_its_override():
    _tab, layout = LAYOUTS["segment"]
    box = layout.box("voxel_size")
    assert box.rows == ("image_axis_order", "voxel_size_policy", "voxel_size_override_xyz")


def test_every_hand_made_box_names_real_settings():
    for call, specs in TAB_BOXES.items():
        for spec in specs:
            for name in spec.names:
                assert name in SCHEMA, (call, name)
            if spec.subtree_of is not None:
                assert spec.subtree_of in SCHEMA, (call, spec.subtree_of)


def test_every_title_and_group_key_names_a_button_the_panel_draws():
    """A stale key would silently stop titling or grouping anything. A
    hand-made box's own button may be waiting for its first setting, so its
    title counts as long as the box does."""
    keys = {d.key.split(":", 1)[1] for _tab, layout in LAYOUTS.values() for d in layout.disclosures}
    hand_made = {f"box:{spec.key}" for specs in TAB_BOXES.values() for spec in specs}
    assert set(ADVANCED_TITLES) <= keys | hand_made
    assert set(ADVANCED_GROUPS) <= keys


def test_every_group_names_real_settings_behind_its_button():
    by_key = {d.key.split(":", 1)[1]: d for _tab, layout in LAYOUTS.values() for d in layout.disclosures}
    for key, groups in ADVANCED_GROUPS.items():
        for heading, names in groups:
            for name in names:
                assert name in SCHEMA, (key, heading, name)
    # Every row named by a group of the Skeletonise button really is behind it.
    skeletonise = by_key["do_skeletonize"]
    for heading, names in skeletonise.groups:
        assert set(names) <= set(skeletonise.names)


# --- the default boxes: one per schema section ------------------------------


def _synthetic(*declared: tuple[str, str, bool]) -> Schema:
    return Schema(
        [
            Setting(name, "bool", False, f"Toggle {name} for a test", section, advanced=advanced)
            for name, section, advanced in declared
        ]
    )


def test_one_section_is_one_untitled_box():
    schema = _synthetic(("a", "Statistics", False), ("b", "Statistics", False))
    layout = layout_for("no_such_stage", ["a", "b"], schema)
    assert [(b.key, b.title, b.items) for b in layout.boxes] == [("statistics", None, ("a", "b"))]


def test_a_later_section_becomes_its_own_titled_box():
    schema = _synthetic(
        ("a", "Statistics", False),
        ("b", "Statistics", False),
        ("c", "Connectivity", False),
        ("d", "Connectivity", False),
    )
    layout = layout_for("no_such_stage", ["a", "b", "c", "d"], schema)
    assert [(b.key, b.title, b.items) for b in layout.boxes] == [
        ("statistics", None, ("a", "b")),
        ("connectivity", section_box_title("no_such_stage", "Connectivity"), ("c", "d")),
    ]


def test_the_same_section_returning_later_is_kept_as_its_own_box():
    """Schema declaration keeps a section's settings contiguous, but two
    non-adjacent runs of the same section must not silently merge."""
    schema = _synthetic(("a", "X", False), ("b", "Y", False), ("c", "X", False))
    layout = layout_for("no_such_stage", ["a", "b", "c"], schema)
    assert [b.items for b in layout.boxes] == [("a",), ("b",), ("c",)]
    assert len({b.key for b in layout.boxes}) == 3


def test_no_rows_is_no_boxes():
    assert layout_for("no_such_stage", [], _synthetic()).boxes == ()


def test_an_advanced_row_with_no_ordinary_section_mates_joins_the_first_box():
    schema = _synthetic(("a", "Main", False), ("b", "Checks", True))
    layout = layout_for("no_such_stage", ["a", "b"], schema)
    (box,) = layout.boxes
    assert box.rows == ("a",)
    (button,) = box.disclosures
    assert button.anchor is None and button.names == ("b",)


def test_an_advanced_row_hangs_off_the_innermost_ordinary_toggle_it_needs():
    schema = Schema(
        [
            Setting("outer", "bool", False, "Outer toggle for a test", "S"),
            Setting("inner", "bool", False, "Inner toggle for a test", "S", requires=("outer",)),
            Setting("knob", "int", 1, "A knob for a test", "S", requires=("outer", "inner"), advanced=True),
            Setting("other", "int", 1, "Another knob for a test", "S", requires=("outer",), advanced=True),
        ]
    )
    layout = layout_for("no_such_stage", ["outer", "inner", "knob", "other"], schema)
    (box,) = layout.boxes
    assert [item if isinstance(item, str) else item.anchor for item in box.items] == [
        "outer",
        "inner",
        "inner",
        "outer",
    ]
    inner, outer = box.disclosures
    assert inner.level == 2 and outer.level == 1


# --- what an Advanced button counts as changed --------------------------------


def test_changed_settings_counts_only_values_that_differ_and_take_effect():
    values = dict(SCHEMA.defaults())
    values["skeleton_bundle_scan_size"] = 11
    values["thick_vessel_braid_factor_limit"] = 3.0  # its braid check is off
    names = ["skeleton_bundle_scan_size", "skeleton_bundle_density_fraction", "thick_vessel_braid_factor_limit"]
    assert changed_settings(names, SCHEMA, values) == ("skeleton_bundle_scan_size",)


def test_changed_settings_measures_against_the_panels_own_starting_value():
    values = dict(SCHEMA.defaults())
    values["visualize_results"] = False
    assert changed_settings(["visualize_results"], SCHEMA, values) == ("visualize_results",)
    assert changed_settings(
        ["visualize_results"], SCHEMA, values, baseline={"visualize_results": False}
    ) == ()


def test_changed_settings_reads_the_same_path_two_ways_as_unchanged():
    values = dict(SCHEMA.defaults())
    values["base_plot_dir"] = Path(SCHEMA["base_plot_dir"].default).resolve()
    assert changed_settings(["base_plot_dir"], SCHEMA, values) == ()
    values["base_plot_dir"] = Path("somewhere_else")
    assert changed_settings(["base_plot_dir"], SCHEMA, values) == ("base_plot_dir",)


def test_box_names_include_the_rows_behind_its_buttons():
    box = Box(
        key="k",
        title=None,
        items=(
            "a",
            Disclosure(key="t:a", anchor="a", title="T", level=1, names=("b", "c"), groups=((None, ("b", "c")),)),
        ),
    )
    assert box.names == ("a", "b", "c")
    assert box.rows == ("a",)
