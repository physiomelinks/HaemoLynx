"""Bundle-into-paths settings are live: on the schema, the Skeletonise tab,
and forwarded to ``preprocess_skeleton_for_graph`` by name.

These replace ``test_proposed_skeleton_gui_settings.py``, which pinned the
opposite state while the four ``skeleton_bundle_*`` settings were declared in
``preprocessing/proposed_skeleton_settings.py`` but not yet wired into
``pipeline/schema.py`` / ``pipeline/progress.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

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


def _clump_in_three_slices() -> np.ndarray:
    mask = np.zeros((11, 28, 28), dtype=bool)
    mask[4:7, 10:19, 10:19] = True  # 6 x 4.5 x 4.5 um on a 0.5 x 0.5 x 2 um stack
    return mask


@pytest.mark.parametrize("use_memmap", [False, True])
def test_the_density_window_is_the_same_width_in_microns_on_every_axis(
    monkeypatch, tmp_path, use_memmap
):
    """Regression (audit): a window of 9 voxels on every axis spanned 18 um in
    z but 4.5 um in-plane, diluting a clump a few slices deep -- here a dense
    6 x 4.5 x 4.5 um one read as a third full, under the 0.35 a bundle needs,
    and was never collapsed to a hub."""
    import haemolynx.preprocessing.skeleton as skeleton_module

    hubs = []
    real = skeleton_module._collapse_hubs

    def recording(result, mask, dense, selected, scan, *rest):
        hubs.append((len(selected), tuple(scan)))
        return real(result, mask, dense, selected, scan, *rest)

    monkeypatch.setattr(skeleton_module, "_collapse_hubs", recording)
    kwargs = {"use_memmap": True, "memmap_directory": tmp_path} if use_memmap else {}

    skeleton_module.skeletonize_voxel_bundles_into_paths(
        _clump_in_three_slices(), 9, voxel_size_zyx=(2.0, 0.5, 0.5), **kwargs
    )

    assert hubs and hubs[0][0] >= 1
    assert hubs[0][1] == (3, 9, 9)


def test_hub_spacing_is_measured_in_microns():
    """Two peaks three 2 um slices apart are 6 um apart: both are hubs when
    hubs must be 4 finest-axis voxels (2 um) apart."""
    from haemolynx.preprocessing.skeleton import _select_hub_centres

    peaks = np.array([[2, 10, 10], [5, 10, 10]])
    density = np.array([0.9, 0.8])

    assert len(_select_hub_centres(peaks, density, 4, np.array([4.0, 1.0, 1.0]))) == 2
    assert len(_select_hub_centres(peaks, density, 4)) == 1  # voxel cubes, as before
