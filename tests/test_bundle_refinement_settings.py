"""Bundle-into-paths settings are live: on the schema, the Skeletonise tab,
and forwarded to ``preprocess_skeleton_for_graph`` by name.

These replace ``test_proposed_skeleton_gui_settings.py``, which pinned the
opposite state while the four ``skeleton_bundle_*`` settings were declared in
``preprocessing/proposed_skeleton_settings.py`` but not yet wired into
``pipeline/schema.py`` / ``pipeline/progress.py``.
"""
from __future__ import annotations

from haemolynx.gui.form import fields_for, label_for
from haemolynx.parsers import parameters_of, prefixed_arguments
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.progress import STAGES
from haemolynx.preprocessing import preprocess_skeleton_for_graph

BUNDLE_SETTING_NAMES: tuple[str, ...] = (
    "skeleton_bundle_scan_size",
    "skeleton_bundle_density_fraction",
    "skeleton_bundle_max_connections_per_hub",
    "skeleton_bundle_hub_min_spacing",
)

BUNDLE_SETTING_DEFAULTS: dict[str, object] = {
    "skeleton_bundle_scan_size": 9,
    "skeleton_bundle_density_fraction": 0.35,
    "skeleton_bundle_max_connections_per_hub": 8,
    "skeleton_bundle_hub_min_spacing": 4,
}


def test_bundle_settings_are_on_the_live_schema():
    live = default_schema()
    missing = [name for name in BUNDLE_SETTING_NAMES if name not in live.names]
    assert not missing, f"bundle settings missing from default_schema(): {missing}"
    for name in BUNDLE_SETTING_NAMES:
        assert live[name].default == BUNDLE_SETTING_DEFAULTS[name]
        assert live[name].requires == ("do_skeletonize",)


def test_bundle_settings_are_on_the_live_skeletonise_tab():
    skeletonise = next(stage for stage in STAGES if stage.call == "skeletonise")
    missing = [name for name in BUNDLE_SETTING_NAMES if name not in skeletonise.settings]
    assert not missing, f"bundle settings missing from the Skeletonise tab: {missing}"


def test_bundle_setting_names_match_preprocess_skeleton_for_graph():
    """``skeleton_bundle_*`` is what ``prefixed_arguments`` forwards."""
    parameters = set(parameters_of(preprocess_skeleton_for_graph))
    mapped = {name[len("skeleton_"):] for name in BUNDLE_SETTING_NAMES}
    assert mapped <= parameters
    assert mapped == {
        "bundle_scan_size",
        "bundle_density_fraction",
        "bundle_max_connections_per_hub",
        "bundle_hub_min_spacing",
    }


def test_prefixed_arguments_forwards_bundle_knobs():
    settings = dict(BUNDLE_SETTING_DEFAULTS)
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


def test_bundle_rows_appear_on_the_form():
    schema = default_schema()
    fields = {field.name: field for field in fields_for(schema)}
    for name in BUNDLE_SETTING_NAMES:
        row = fields[name]
        assert row.section == "Pipeline stages"
        assert row.help.strip()
        assert label_for(name, schema[name].unit)
    assert fields["skeleton_bundle_scan_size"].widget_type == "SpinBox"
    assert fields["skeleton_bundle_density_fraction"].widget_type == "FloatSpinBox"


def test_bundle_rows_grey_out_when_skeletonise_is_off():
    schema = default_schema()
    fields = {field.name: field for field in fields_for(schema)}
    values_off = dict(schema.defaults())
    values_off["do_skeletonize"] = False
    values_on = dict(values_off)
    values_on["do_skeletonize"] = True
    for name in BUNDLE_SETTING_NAMES:
        assert fields[name].is_enabled(values_off) is False
        assert fields[name].is_enabled(values_on) is True


# --- hub links: one per branch, not one per octant ------------------------------


def _two_branches_in_one_octant():
    """A hub centre and two separate branches both leaving it toward
    (+z, +y, +x): one a pair of adjacent voxels, one a single voxel."""
    import numpy as np

    result = np.zeros((21, 21, 21), dtype=bool)
    center = np.array([10, 10, 10])
    branch_a = np.array([[14, 11, 11], [15, 11, 11]])
    branch_b = np.array([[11, 14, 12]])
    result[tuple(center)] = True
    for point in np.vstack([branch_a, branch_b]):
        result[tuple(point)] = True
    return result, center, np.vstack([branch_a, branch_b])


def test_two_branches_leaving_a_hub_in_the_same_octant_both_get_a_link():
    """Regression: links were kept one per sign of the direction, so the
    nearer of two branches in one octant was left disconnected."""
    import numpy as np
    from scipy.ndimage import generate_binary_structure, label

    from haemolynx.preprocessing.skeleton import _draw_hub_links

    result, center, boundary = _two_branches_in_one_octant()
    _draw_hub_links(result, center, boundary, max_connections_per_hub=8)

    assert label(result, structure=generate_binary_structure(3, 3))[1] == 1


def test_adjacent_boundary_voxels_are_one_branch_with_one_link():
    import numpy as np

    from haemolynx.preprocessing.skeleton import _draw_hub_links

    result, center, boundary = _two_branches_in_one_octant()
    before = result.copy()
    _draw_hub_links(result, center, boundary[:2], max_connections_per_hub=8)

    # One line, centre (10, 10, 10) to the farther voxel (15, 11, 11): the
    # voxels between, one per z step, not already set.
    drawn = np.argwhere(result & ~before)
    assert sorted(drawn[:, 0].tolist()) == [11, 12, 13]
    assert set(map(tuple, drawn[:, 1:].tolist())) <= {(10, 10), (11, 11)}


def test_the_link_cap_keeps_the_farthest_branches():
    import numpy as np

    from haemolynx.preprocessing.skeleton import _draw_hub_links

    result, center, boundary = _two_branches_in_one_octant()
    _draw_hub_links(result, center, boundary, max_connections_per_hub=1)

    assert result[13, 10, 10] or result[13, 11, 11] or result[13, 10, 11] or result[13, 11, 10]
    assert not (result[10, 12, 11] or result[10, 13, 11] or result[11, 12, 11] or result[11, 13, 12])
