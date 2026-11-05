"""Boundaries vessel-mask path field order and ilastik hide rules."""
from __future__ import annotations

from haemolynx.gui.form import field_for, fields_for, visible_vessel_mask_settings
from haemolynx.gui.tabs import tabs_for
from haemolynx.pipeline import default_schema

SCHEMA = default_schema()


def _ordered_visible_vessel_mask_names(values: dict) -> list[str]:
    """Vessel-mask setting names that should appear, in schema/form order."""
    ordered: list[str] = []
    for setting in SCHEMA:
        if setting.section != "Vessel masks":
            continue
        if field_for(setting).is_visible(values):
            ordered.append(setting.name)
    return ordered


def test_large_mask_paths_sit_immediately_under_use_large_vessel_masks():
    """Schema/form order: arteriole then venule path right under the toggle."""
    names = list(SCHEMA.section_names("Vessel masks"))
    i = names.index("use_large_vessel_masks")
    assert names[i : i + 3] == [
        "use_large_vessel_masks",
        "large_arteriole_mask_path",
        "large_venule_mask_path",
    ]
    assert names[i + 3] == "use_ilastik_large_vessel_segmentation"

    tabs = {tab.stage.title: tab for tab in tabs_for(SCHEMA)}
    boundaries = [field.name for field in tabs["4. Boundaries"].fields]
    j = boundaries.index("use_large_vessel_masks")
    assert boundaries[j : j + 3] == [
        "use_large_vessel_masks",
        "large_arteriole_mask_path",
        "large_venule_mask_path",
    ]


def test_small_mask_paths_sit_immediately_under_use_small_vessel_masks():
    names = list(SCHEMA.section_names("Vessel masks"))
    i = names.index("use_small_vessel_masks_for_boundary_assignment")
    assert names[i : i + 3] == [
        "use_small_vessel_masks_for_boundary_assignment",
        "small_arteriole_mask_path",
        "small_venule_mask_path",
    ]
    assert names[i + 3] == "use_ilastik_small_vessel_segmentation"

    tabs = {tab.stage.title: tab for tab in tabs_for(SCHEMA)}
    boundaries = [field.name for field in tabs["4. Boundaries"].fields]
    j = boundaries.index("use_small_vessel_masks_for_boundary_assignment")
    assert boundaries[j : j + 3] == [
        "use_small_vessel_masks_for_boundary_assignment",
        "small_arteriole_mask_path",
        "small_venule_mask_path",
    ]


def test_visible_large_mask_rows_keep_paths_immediately_under_toggle():
    """Among visible rows: paths follow Use large vessel masks until ilastik."""
    values = {
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": True,
        "use_ilastik_large_vessel_segmentation": False,
        "use_small_vessel_masks_for_boundary_assignment": False,
        "large_vessel_remove_small_opposite_attached_components": False,
        "automated_vessel_assignment_enable_overlap_cleanup": False,
        "automated_vessel_assignment_use_legacy_mode": True,
        "cut_network_at_large_vessel_volumes": False,
    }
    shown = _ordered_visible_vessel_mask_names(values)
    i = shown.index("use_large_vessel_masks")
    assert shown[i : i + 3] == [
        "use_large_vessel_masks",
        "large_arteriole_mask_path",
        "large_venule_mask_path",
    ]
    assert shown[i + 3] == "use_ilastik_large_vessel_segmentation"
    assert "large_arteriole_mask_path" in visible_vessel_mask_settings(SCHEMA, values)
    assert "large_venule_mask_path" in visible_vessel_mask_settings(SCHEMA, values)
    assert "ilastik_arteriole_classifier_path" not in visible_vessel_mask_settings(
        SCHEMA, values
    )


def test_large_mask_paths_hidden_when_ilastik_large_vessel_segmentation_on():
    values = {
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": True,
        "use_ilastik_large_vessel_segmentation": True,
        "use_small_vessel_masks_for_boundary_assignment": False,
        "large_vessel_remove_small_opposite_attached_components": False,
        "automated_vessel_assignment_enable_overlap_cleanup": False,
        "automated_vessel_assignment_use_legacy_mode": True,
        "cut_network_at_large_vessel_volumes": False,
    }
    shown = visible_vessel_mask_settings(SCHEMA, values)
    assert "large_arteriole_mask_path" not in shown
    assert "large_venule_mask_path" not in shown
    assert "ilastik_arteriole_classifier_path" in shown
    assert "ilastik_venule_classifier_path" in shown

    fields = {f.name: f for f in fields_for(SCHEMA)}
    for name in ("large_arteriole_mask_path", "large_venule_mask_path"):
        assert "!use_ilastik_large_vessel_segmentation" in fields[name].enabled_by
        assert not fields[name].is_visible(values)


def test_visible_small_mask_rows_keep_paths_immediately_under_toggle():
    values = {
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": False,
        "use_small_vessel_masks_for_boundary_assignment": True,
        "use_ilastik_small_vessel_segmentation": False,
        "small_vessel_mask_continuity_enable": False,
        "small_vessel_tangential_redefinition_enable": False,
        "small_vessel_boundary_assignment_enable_overlap_cleanup": False,
        "small_vessel_boundary_assignment_fast_mode": False,
        "small_vessel_boundary_fallback_to_hop_distance": False,
    }
    shown = _ordered_visible_vessel_mask_names(values)
    i = shown.index("use_small_vessel_masks_for_boundary_assignment")
    assert shown[i : i + 3] == [
        "use_small_vessel_masks_for_boundary_assignment",
        "small_arteriole_mask_path",
        "small_venule_mask_path",
    ]
    assert shown[i + 3] == "use_ilastik_small_vessel_segmentation"


def test_small_mask_paths_hidden_when_ilastik_small_vessel_segmentation_on():
    values = {
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": False,
        "use_small_vessel_masks_for_boundary_assignment": True,
        "use_ilastik_small_vessel_segmentation": True,
        "small_vessel_mask_continuity_enable": False,
        "small_vessel_tangential_redefinition_enable": False,
        "small_vessel_boundary_assignment_enable_overlap_cleanup": False,
        "small_vessel_boundary_assignment_fast_mode": False,
        "small_vessel_boundary_fallback_to_hop_distance": False,
    }
    shown = visible_vessel_mask_settings(SCHEMA, values)
    assert "small_arteriole_mask_path" not in shown
    assert "small_venule_mask_path" not in shown
    assert "ilastik_small_arteriole_classifier_path" in shown
    assert "ilastik_small_venule_classifier_path" in shown

    fields = {f.name: f for f in fields_for(SCHEMA)}
    for name in ("small_arteriole_mask_path", "small_venule_mask_path"):
        assert "!use_ilastik_small_vessel_segmentation" in fields[name].enabled_by
        assert not fields[name].is_visible(values)
