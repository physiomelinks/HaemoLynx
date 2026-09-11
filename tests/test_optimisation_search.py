"""End-to-end tests for haemolynx.optimisation.search on tiny, real synthetic volumes.

These run the actual preprocessing/graph-building code (not mocks) on masks
small enough to finish in well under a second per call, so the whole 10-group
search still runs in a few seconds.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from haemolynx.optimisation.progress import (
    CANDIDATE_EVALUATED,
    GROUP_FINISHED,
    GROUP_STARTED,
)
from haemolynx.optimisation.search import (
    AUTO_DOWNSAMPLE_TARGET_VOXELS,
    DOWNSAMPLE_FACTORS,
    GRAPH_SETTING_NAMES,
    GROUP_NAMES,
    OPTIMISE_SETTING_NAMES,
    SKELETON_SETTING_NAMES,
    _VOXEL_SCALED_SETTING_NAMES,
    optimise_skeleton_and_graph_settings,
    resolve_auto_downsample_factor,
)


def _draw_line(mask: np.ndarray, start, end) -> None:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    steps = int(np.ceil(np.linalg.norm(end - start))) * 2 + 1
    for t in np.linspace(0.0, 1.0, steps):
        point = np.round(start + t * (end - start)).astype(int)
        if np.all(point >= 0) and np.all(point < mask.shape):
            mask[tuple(point)] = True


def _y_shaped_vessel(shape=(30, 30, 30), radius=2) -> np.ndarray:
    """A Y-shaped tube: one trunk splitting into two arms, radius ~2 voxels."""
    skeleton = np.zeros(shape, dtype=bool)
    centre = np.array([15, 15, 15])
    _draw_line(skeleton, centre + [-10, 0, 0], centre)
    _draw_line(skeleton, centre, centre + [8, 8, 0])
    _draw_line(skeleton, centre, centre + [8, -8, 0])
    structure = np.ones((2 * radius + 1,) * 3, dtype=bool)
    return binary_dilation(skeleton, structure=structure)


#: Schema defaults for every setting this search decides, matching
#: pipeline/schema.py -- kept local so this test does not depend on
#: haemolynx.pipeline (the optimisation package's own purity boundary).
_DEFAULT_STARTING_VALUES = {
    "segmentation_cleanup_fill_cavities": False,
    "segmentation_cleanup_remove_whiskers": False,
    "segmentation_cleanup_whisker_radius_um": 1.0,
    "segmentation_cleanup_split_narrow_necks": False,
    "segmentation_cleanup_split_min_marker_separation_um": 10.0,
    "segmentation_cleanup_split_min_pinch_radius_ratio": 0.6,
    "segmentation_cleanup_split_min_body_radius_um": 1.0,
    "segmentation_cleanup_close_gaps": False,
    "segmentation_cleanup_close_gaps_radius_um": 0.5,
    "segmentation_cleanup_reconnect_gaps": False,
    "segmentation_cleanup_reconnect_max_bridge_distance_um": 30.0,
    "segmentation_cleanup_reconnect_min_cylindricality": 0.5,
    "segmentation_cleanup_reconnect_max_axis_angle_degrees": 30.0,
    "segmentation_cleanup_reconnect_min_facing_cosine": 0.85,
    "segmentation_cleanup_reconnect_max_radius_ratio": 3.0,
    "segmentation_cleanup_smooth_surfaces": False,
    "segmentation_cleanup_smooth_method": "gaussian",
    "segmentation_cleanup_smooth_sigma_um": 1.0,
    "segmentation_cleanup_smooth_morphological_radius_um": 1.0,
    "segmentation_cleanup_remove_small_volumes": False,
    "segmentation_cleanup_remove_small_min_volume_um3": 5.0,
    "use_thick_vessel_skeletonisation": False,
    "skeleton_thick_vessel_min_radius_um": 6.0,
    "skeleton_fill_mask_holes_before_thickness": True,
    "skeleton_thick_vessel_wall_absorption_um": None,
    "skeleton_thick_vessel_flake_filter_um": None,
    "skeleton_thick_vessel_max_bridge_radius_multiple": 4.0,
    "skeleton_thick_vessel_max_bridge_distance_um": None,
    "skeleton_thick_vessel_bridge_radius_smoothing_um": 10.0,
    "skeleton_min_branch_length": 3,
    "skeleton_bundle_scan_size": 9,
    "skeleton_bundle_density_fraction": 0.35,
    "skeleton_bundle_max_connections_per_hub": 8,
    "skeleton_bundle_hub_min_spacing": 4,
    "skeleton_closing_radius": 2,
    "skeleton_bridge_gap_size": 3,
    "skeleton_max_bridge_distance": 4,
    "skeleton_component_connectivity": 3,
    "skeleton_min_component_percent": 0.0,
    "graph_reconnect_threshold": 10.0,
    "final_orphan_reconnect_threshold": 3.0,
    "cluster_collapse_distance": 5.0,
    "cluster_collapse_method": "distance_only",
    "cluster_collapse_max_radial_dispersion": 0.5,
    "cluster_collapse_persistence_search_multiple": 3.0,
    "min_stub_length": 10.0,
    "smooth_centrelines": True,
    "centreline_smoothing_method": "taubin",
    "centreline_smoothing_iterations": 10,
    "centreline_max_deviation": 1.0,
}


@pytest.fixture
def y_shaped_mask() -> np.ndarray:
    return _y_shaped_vessel()


def test_optimise_settings_returns_every_setting(y_shaped_mask):
    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
    )
    assert set(result.settings) == set(OPTIMISE_SETTING_NAMES)
    assert len(result.trials) > 0


def test_optimise_settings_only_sweeps_the_chosen_cluster_collapse_methods_own_knob(y_shaped_mask):
    """Only the method the search actually picked gets its own extra sweep --
    the other method-specific setting is left untouched, since it is not read
    by the graph builder either way."""
    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
    )
    tried_settings = {trial.setting for trial in result.trials}
    method = result.settings["cluster_collapse_method"]
    assert ("cluster_collapse_max_radial_dispersion" in tried_settings) == (method == "direction_aware")
    assert ("cluster_collapse_persistence_search_multiple" in tried_settings) == (method == "persistence")


def test_optimise_settings_produces_sane_values(y_shaped_mask):
    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
    )
    settings = result.settings
    assert settings["skeleton_closing_radius"] >= 0
    assert settings["skeleton_bridge_gap_size"] >= 0
    assert settings["skeleton_max_bridge_distance"] >= 0
    assert 1 <= settings["skeleton_component_connectivity"] <= 3
    assert 0.0 <= settings["skeleton_min_component_percent"] <= 100.0
    assert settings["graph_reconnect_threshold"] >= 0.0
    assert settings["final_orphan_reconnect_threshold"] >= 0.0
    assert settings["cluster_collapse_distance"] >= 0.0
    assert settings["cluster_collapse_method"] in ("distance_only", "direction_aware", "persistence")
    assert 0.0 <= settings["cluster_collapse_max_radial_dispersion"] <= 1.0
    assert settings["cluster_collapse_persistence_search_multiple"] >= 0.0
    assert settings["min_stub_length"] >= 0.0
    assert settings["centreline_smoothing_method"] in ("taubin", "chaikin")
    assert settings["centreline_smoothing_iterations"] > 0
    assert settings["centreline_max_deviation"] > 0.0
    # A modest tube well under the guard radius: thickness gating should not
    # turn itself on for its own sake.
    assert settings["use_thick_vessel_skeletonisation"] is False


def test_optimise_settings_reports_progress_in_order(y_shaped_mask):
    events = []
    optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
        progress=events.append,
    )
    assert events, "no progress events were emitted"
    kinds = {event.kind for event in events}
    assert kinds == {GROUP_STARTED, GROUP_FINISHED, CANDIDATE_EVALUATED}
    # group_index must never go backwards.
    indices = [event.group_index for event in events]
    assert indices == sorted(indices)


def test_optimise_settings_on_empty_mask_falls_back_to_starting_values():
    empty = np.zeros((10, 10, 10), dtype=bool)
    result = optimise_skeleton_and_graph_settings(
        empty, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
    )
    assert result.settings == _DEFAULT_STARTING_VALUES


def test_optimise_settings_on_a_single_blob_does_not_raise():
    """A solid block has no gaps and one component -- every guard must degrade
    gracefully rather than divide by zero or index past an empty array.

    A solid cube's own medial axis is degenerate (both Lee thinning and
    thickness-gated skeletonisation reduce it to nothing, a 0.0-vs-0.0 tie
    the "off" epsilon then wins), so this fixture never reaches the
    thick-vessel refinement sweeps -- see
    ``test_optimise_settings_thick_vessel_refinement_sweeps_run_when_on``
    for those, forced via a fake rather than this fixture's own shape.
    """
    blob = np.zeros((20, 20, 20), dtype=bool)
    blob[5:15, 5:15, 5:15] = True
    result = optimise_skeleton_and_graph_settings(
        blob, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
    )
    assert set(result.settings) == set(OPTIMISE_SETTING_NAMES)


def test_optimise_settings_thick_vessel_refinement_sweeps_run_when_on(monkeypatch):
    """The five thick-vessel refinement settings (wall absorption, flake
    filter, the two bridge caps, bridge-radius smoothing) get their own sweep
    once thick-vessel skeletonisation is on.

    Forces that path with a fast fake for ``skeletonize_thickness_gated`` and
    calls the one group directly, rather than the whole search: the real
    function is already exercised (and is what the on/off decision itself
    tests) elsewhere, and everything downstream of the raw skeleton (graph
    building) is a different concern this test is not about -- a fake raw
    skeleton has no reason to survive real cleaning and graph-building
    unscathed. A cheap, deterministic stand-in keeps this fast and
    independent of exactly how any one real mask happens to skeletonise.
    """
    from haemolynx import preprocessing as preprocessing_module
    from haemolynx.optimisation.search import _Search

    fat_line = np.zeros((10, 10, 10), dtype=bool)
    fat_line[5, 5, :] = True  # a real, single-component "skeleton"

    monkeypatch.setattr(
        preprocessing_module, "skeletonize_thickness_gated", lambda binary, **kwargs: fat_line
    )

    blob = np.zeros((20, 20, 20), dtype=bool)
    blob[5:15, 5:15, 5:15] = True  # Lee thinning of this is empty -- see the test above
    search = _Search(
        blob, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES, progress=None,
    )
    search._group_thick_vessel_gating()

    assert search.current["use_thick_vessel_skeletonisation"] is True
    tried_settings = {trial.setting for trial in search.trials}
    for name in (
        "skeleton_thick_vessel_wall_absorption_um",
        "skeleton_thick_vessel_flake_filter_um",
        "skeleton_thick_vessel_max_bridge_radius_multiple",
        "skeleton_thick_vessel_max_bridge_distance_um",
        "skeleton_thick_vessel_bridge_radius_smoothing_um",
    ):
        assert name in tried_settings, f"{name} was never swept"
    assert search.current["skeleton_thick_vessel_wall_absorption_um"] is None or (
        search.current["skeleton_thick_vessel_wall_absorption_um"] >= 0.0
    )
    assert search.current["skeleton_thick_vessel_max_bridge_radius_multiple"] >= 0.0
    assert search.current["skeleton_thick_vessel_bridge_radius_smoothing_um"] >= 0.0
    assert search.raw_skeleton.any()


def test_optimise_settings_on_scattered_speckle_does_not_raise():
    """All-disconnected noise: many tiny components, no coherent network."""
    rng = np.random.default_rng(0)
    speckle = rng.random((15, 15, 15)) > 0.97
    result = optimise_skeleton_and_graph_settings(
        speckle, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
    )
    assert set(result.settings) == set(OPTIMISE_SETTING_NAMES)


def test_optimise_settings_respects_anisotropic_voxel_size(y_shaped_mask):
    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 2.0), starting_values=_DEFAULT_STARTING_VALUES,
    )
    assert set(result.settings) == set(OPTIMISE_SETTING_NAMES)


# ---------------------------------------------------------------------------
# Optimisation downsampling
# ---------------------------------------------------------------------------
def test_downsample_mask_is_a_no_op_at_factor_1():
    from haemolynx.optimisation.search import _downsample_mask

    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3, 3, 3] = True
    assert _downsample_mask(mask, 1) is mask


def test_downsample_mask_never_loses_a_single_foreground_voxel():
    """Block-max reduction: even a lone voxel in an otherwise-empty block must
    survive, unlike plain striding which could sample right past it."""
    from haemolynx.optimisation.search import _downsample_mask

    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[1, 1, 1] = True  # would be skipped entirely by mask[::4, ::4, ::4]
    result = _downsample_mask(mask, 4)
    assert result.shape == (2, 2, 2)
    assert result.any()
    assert result[0, 0, 0]  # the block containing (1,1,1)


def test_resolve_auto_downsample_factor_is_1_under_the_target():
    assert resolve_auto_downsample_factor((100, 100, 100)) == 1


def test_resolve_auto_downsample_factor_scales_up_for_a_huge_volume():
    # A whole-brain-scale stack: comfortably past the target at every factor
    # except the largest offered. resolve_auto_downsample_factor only does
    # arithmetic on the shape tuple, so this never allocates the array.
    huge = (5000, 5000, 5000)
    assert huge[0] * huge[1] * huge[2] > AUTO_DOWNSAMPLE_TARGET_VOXELS * (DOWNSAMPLE_FACTORS[-1] ** 3)
    assert resolve_auto_downsample_factor(huge) == DOWNSAMPLE_FACTORS[-1]


def test_resolve_auto_downsample_factor_picks_the_smallest_sufficient_factor():
    # Chosen so factor=2 clears the target but factor=1 does not.
    target = AUTO_DOWNSAMPLE_TARGET_VOXELS
    side = round((target * 1.5) ** (1 / 3))
    shape = (side, side, side)
    assert side ** 3 > target
    assert resolve_auto_downsample_factor(shape) == 2


def test_optimise_settings_downsample_factor_1_is_a_no_op(y_shaped_mask):
    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
        downsample_factor=1,
    )
    assert result.downsample_factor == 1


def test_optimise_settings_downsample_rescales_voxel_settings_but_not_micron_ones(monkeypatch):
    """The settings measured in voxels of the search's own grid must come back
    multiplied by the downsample factor; physical-micron settings must not
    change just because the grid the search ran on was coarser."""
    import haemolynx.optimisation.search as search_module

    # A deterministic, fast fake standing in for the whole search: only the
    # rescaling logic in `optimise_skeleton_and_graph_settings` itself is
    # under test here, not any group's own decision-making.
    class _FakeSearch:
        def __init__(self, mask, voxel_size_xyz, starting_values, progress, enabled_groups=None):
            self.mask = mask
            self.voxel_size_xyz_seen = voxel_size_xyz
            self.current = dict(starting_values)
            self.current["skeleton_closing_radius"] = 3
            self.current["skeleton_min_branch_length"] = 5
            self.current["graph_reconnect_threshold"] = 12.5  # a micron setting: must pass through unchanged
            self.trials = []
            self.groups_run = list(search_module.GROUP_NAMES)

        def run(self):
            pass

    monkeypatch.setattr(search_module, "_Search", _FakeSearch)

    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[10:30, 10:30, 10:30] = True
    result = optimise_skeleton_and_graph_settings(
        mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
        downsample_factor=4,
    )
    assert result.downsample_factor == 4
    assert result.settings["skeleton_closing_radius"] == 12  # 3 * 4
    assert result.settings["skeleton_min_branch_length"] == 20  # 5 * 4
    assert result.settings["graph_reconnect_threshold"] == pytest.approx(12.5)  # unchanged


def test_optimise_settings_downsample_scales_the_voxel_size_the_search_sees(monkeypatch):
    import haemolynx.optimisation.search as search_module

    seen = {}

    class _FakeSearch:
        def __init__(self, mask, voxel_size_xyz, starting_values, progress, enabled_groups=None):
            seen["mask_shape"] = mask.shape
            seen["voxel_size_xyz"] = voxel_size_xyz
            self.current = dict(starting_values)
            self.trials = []
            self.groups_run = []

        def run(self):
            pass

    monkeypatch.setattr(search_module, "_Search", _FakeSearch)

    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[10:30, 10:30, 10:30] = True
    optimise_skeleton_and_graph_settings(
        mask, voxel_size_xyz=(1.0, 1.0, 2.0), starting_values=_DEFAULT_STARTING_VALUES,
        downsample_factor=4,
    )
    assert seen["mask_shape"] == (10, 10, 10)
    assert seen["voxel_size_xyz"] == pytest.approx((4.0, 4.0, 8.0))


def test_voxel_scaled_setting_names_are_all_real_settings():
    assert set(_VOXEL_SCALED_SETTING_NAMES) <= set(OPTIMISE_SETTING_NAMES)


def test_optimise_settings_runs_for_real_at_an_explicit_downsample_factor(y_shaped_mask):
    """No mocking: the real search, actually run on a real (2x) downsampled
    copy of a real mask, still completes and returns every setting."""
    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
        downsample_factor=2,
    )
    assert result.downsample_factor == 2
    assert set(result.settings) == set(OPTIMISE_SETTING_NAMES)
    for name in _VOXEL_SCALED_SETTING_NAMES:
        assert result.settings[name] % 2 == 0, f"{name} was not scaled back up by the factor"


# ---------------------------------------------------------------------------
# Choose optimisation types (selective groups)
# ---------------------------------------------------------------------------
def test_optimise_settings_with_no_groups_selected_changes_nothing(y_shaped_mask):
    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
        groups=(),
    )
    assert result.settings == _DEFAULT_STARTING_VALUES
    assert result.groups_run == ()
    assert result.trials == ()


def test_optimise_settings_with_one_skeleton_group_only_leaves_graph_settings_untouched(y_shaped_mask):
    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
        groups=("closing_radius",),
    )
    assert result.groups_run == ("closing_radius",)
    for name in (
        "graph_reconnect_threshold", "final_orphan_reconnect_threshold",
        "cluster_collapse_distance", "min_stub_length",
    ):
        assert result.settings[name] == _DEFAULT_STARTING_VALUES[name]
    tried_settings = {trial.setting for trial in result.trials}
    assert tried_settings == {"skeleton_closing_radius"}


def test_optimise_settings_with_only_graph_groups_still_builds_a_graph(y_shaped_mask):
    """Skipping every skeleton group must not stop the graph groups from
    having something to build on."""
    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
        groups=("min_stub_length",),
    )
    assert result.groups_run == ("min_stub_length",)
    tried_settings = {trial.setting for trial in result.trials}
    assert tried_settings == {"min_stub_length"}
    assert result.settings["skeleton_closing_radius"] == _DEFAULT_STARTING_VALUES["skeleton_closing_radius"]


def test_optimise_settings_with_only_segmentation_cleanup_group_leaves_other_settings_untouched(y_shaped_mask):
    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
        groups=("segmentation_cleanup",),
    )
    assert result.groups_run == ("segmentation_cleanup",)
    for name in SKELETON_SETTING_NAMES + GRAPH_SETTING_NAMES:
        assert result.settings[name] == _DEFAULT_STARTING_VALUES[name]
    tried_groups = {trial.group for trial in result.trials}
    assert tried_groups == {"segmentation_cleanup"}


def test_segmentation_cleanup_group_removes_a_small_disconnected_speck():
    """An end-to-end demonstration that the new group actually improves the
    mask's own quality score, not just that it runs without crashing.

    Either `remove_whiskers` (an isolated single voxel has no core to survive
    an opening) or `remove_small_volumes` can clear a lone speck like this --
    which one wins is an implementation detail of where each sits in the
    fixed cleanup order, so this asserts on the outcome (the speck is gone,
    the score improved), not on which specific toggle did it.
    """
    from haemolynx.preprocessing import clean_segmented_mask_for_skeletonisation, score_segmented_mask

    mask = _y_shaped_vessel()
    mask[1, 1, 1] = True  # a single-voxel speck, far from the vessel body
    baseline_score = score_segmented_mask(mask, voxel_size_zyx=(1.0, 1.0, 1.0)).total

    result = optimise_skeleton_and_graph_settings(
        mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
        groups=("segmentation_cleanup",),
    )
    cleanup_kwargs = {
        name[len("segmentation_cleanup_"):]: value
        for name, value in result.settings.items()
        if name.startswith("segmentation_cleanup_")
    }
    cleaned, _raw = clean_segmented_mask_for_skeletonisation(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), **cleanup_kwargs
    )
    assert not cleaned[1, 1, 1]
    assert score_segmented_mask(cleaned, voxel_size_zyx=(1.0, 1.0, 1.0)).total > baseline_score


def test_group_names_cover_every_group_a_trial_could_report():
    """Every group string search.py actually uses to record a trial must be
    one of the selectable names, or a user could never disable it."""
    result = optimise_skeleton_and_graph_settings(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
    )
    used_groups = {trial.group for trial in result.trials}
    assert used_groups <= set(GROUP_NAMES)
