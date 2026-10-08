"""End-to-end tests for haemolynx.optimisation.search on tiny, real synthetic volumes.

These run the actual preprocessing/graph-building code (not mocks) on masks
small enough to finish in well under a second per call, so the whole 10-group
search still runs in a few seconds.
"""
from __future__ import annotations

import networkx as nx
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
    "segmentation_cleanup": False,
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
    "skeleton_bridge_weight_by_segmentation": False,
    "skeleton_bridge_z_distance_weight": 1.0,
    "skeleton_component_connectivity": 3,
    "skeleton_min_component_percent": 0.0,
    "graph_reconnect_threshold": 10.0,
    "final_orphan_reconnect_threshold": 3.0,
    "cluster_collapse_distance": 5.0,
    "cluster_collapse_method": "distance_only",
    "cluster_collapse_max_radial_dispersion": 0.5,
    "cluster_collapse_persistence_search_multiple": 3.0,
    "min_stub_length": 10.0,
    "min_stub_length_radius_multiple": 1.5,
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

    blob = np.zeros((20, 20, 20), dtype=bool)
    blob[5:15, 5:15, 5:15] = True  # Lee thinning of this is empty -- see the test above

    fat_line = np.zeros(blob.shape, dtype=bool)
    fat_line[10, 10, :] = True  # a real, single-component "skeleton", same shape as blob

    monkeypatch.setattr(
        preprocessing_module, "skeletonize_thickness_gated", lambda binary, **kwargs: fat_line
    )
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


# ---------------------------------------------------------------------------
# Input-data-consistency guards (reusing the existing diagnose_* functions)
# ---------------------------------------------------------------------------
def test_regression_penalty_is_zero_when_current_meets_or_beats_baseline():
    from haemolynx.optimisation.search import _GUARD_PENALTY, _Search

    assert _Search._regression_penalty(current=0.9, baseline=0.9, tolerance=0.02) == 0.0
    assert _Search._regression_penalty(current=0.95, baseline=0.9, tolerance=0.02) == 0.0
    # Within tolerance: a small dip must not trip the guard.
    assert _Search._regression_penalty(current=0.89, baseline=0.9, tolerance=0.02) == 0.0


def test_regression_penalty_fires_past_tolerance():
    from haemolynx.optimisation.search import _GUARD_PENALTY, _Search

    assert _Search._regression_penalty(current=0.5, baseline=0.9, tolerance=0.02) == _GUARD_PENALTY


def test_reconnect_thresholds_baseline_is_refreshed_between_sub_sweeps():
    """Regression: `baseline_graph_coverage` used to be computed once
    before the graph_reconnect_threshold sub-sweep and reused unchanged for
    the final_orphan_reconnect_threshold sub-sweep that runs right after it
    -- even though the first sub-sweep can move
    self.current["graph_reconnect_threshold"], which the baseline's own
    `self._build_graph({})` call reads (see project memory: "matched
    processing stage"). The same stale-baseline-across-sequential-sub-
    sweeps bug, and the same fix (refresh immediately before the next
    sub-sweep), also applies to `_group_bundle_refinement`,
    `_group_thick_vessel_gating`'s five refinement sub-sweeps,
    `_group_gap_bridging`, `_group_cluster_collapse`, and
    `_group_segmentation_cleanup`'s ~20 sequential sub-sweeps -- this test
    covers the mechanism once, on the simplest of the six groups to
    instrument precisely.
    """
    from haemolynx import preprocessing
    from haemolynx.optimisation.search import _Search

    mask = _y_shaped_vessel()
    starting_values = dict(_DEFAULT_STARTING_VALUES)
    search = _Search(
        mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=starting_values, progress=None,
    )
    search.current_skeleton = preprocessing.skeletonize_volume(mask)

    recorded_thresholds_at_baseline_calls = []
    real_build_graph = search._build_graph

    def recording_build_graph(overrides):
        graph = real_build_graph(overrides)
        if not overrides:  # every baseline call in this group uses no overrides
            recorded_thresholds_at_baseline_calls.append(
                float(search.current["graph_reconnect_threshold"])
            )
        return graph

    search._build_graph = recording_build_graph
    search._group_reconnect_thresholds()

    # Three no-override `_build_graph({})` calls happen in the fixed group:
    # the graph_reconnect_threshold sub-sweep's own baseline, the refreshed
    # baseline right before final_orphan_reconnect_threshold's sub-sweep,
    # and the group's own closing `self.current_graph = self._build_graph({})`.
    # The stale-baseline bug this guards against never made that middle,
    # refreshing call at all -- only two no-override calls would have
    # happened (the initial baseline and the closing one).
    assert len(recorded_thresholds_at_baseline_calls) == 3, (
        "expected the orphan-threshold sub-sweep's baseline to be refreshed "
        f"as its own _build_graph({{}}) call, got calls={recorded_thresholds_at_baseline_calls}"
    )
    winning_threshold = float(search.current["graph_reconnect_threshold"])
    assert recorded_thresholds_at_baseline_calls[1] == winning_threshold, (
        "the refreshed baseline must reflect the graph_reconnect_threshold "
        "sub-sweep's own already-decided winner, not a stale pre-sweep value"
    )


def test_search_never_aliases_the_callers_own_mask_array():
    """`self.original_raw_mask` must be the search's own private copy, never
    the caller's own array -- even when the caller already passed a bool
    array, where `np.asarray(..., dtype=bool)` would otherwise return that
    same object unchanged. Mutating the caller's array after construction
    must not be visible through either `raw_mask` or `original_raw_mask`."""
    from haemolynx.optimisation.search import _Search

    caller_mask = np.zeros((10, 10, 10), dtype=bool)
    caller_mask[2:5, 2:5, 2:5] = True

    search = _Search(
        caller_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES, progress=None,
    )
    caller_mask[:] = False  # mutate after construction

    assert search.raw_mask.any()
    assert search.original_raw_mask.any()
    assert search.raw_mask is not caller_mask
    assert search.original_raw_mask is not caller_mask


def test_run_default_is_a_single_pass_matching_todays_behaviour(monkeypatch):
    """max_passes defaults to 1 -- calling run() the way every existing
    caller does (no arguments) must still do exactly one pass, even
    against a fake _run_one_pass that would keep "improving" forever."""
    from haemolynx.optimisation.search import _Search

    search = _Search(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0),
        starting_values=dict(_DEFAULT_STARTING_VALUES), progress=None,
    )
    calls = {"n": 0}

    def fake_pass():
        calls["n"] += 1
        search.current["skeleton_min_branch_length"] = calls["n"] + 100

    monkeypatch.setattr(search, "_run_one_pass", fake_pass)
    search.run()

    assert calls["n"] == 1
    assert search.passes_run == 1


def test_run_converges_early_when_a_pass_changes_nothing(monkeypatch):
    """Regression for the coordinate-descent search order limitation: a
    single pass decides each setting once, in a fixed order, and never
    revisits it, so it cannot see an improvement that only appears once
    two settings have both moved from their starting values. max_passes > 1
    repeats the sequence to catch that -- but must not cost extra real work
    once nothing is left to improve: a pass that changes no setting means
    every later pass would repeat it identically.
    """
    from haemolynx.optimisation.search import _Search

    search = _Search(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0),
        starting_values=dict(_DEFAULT_STARTING_VALUES), progress=None,
    )
    calls = {"n": 0}

    def fake_pass():
        calls["n"] += 1  # never touches self.current -- nothing to converge on

    monkeypatch.setattr(search, "_run_one_pass", fake_pass)
    search.run(max_passes=5)

    assert calls["n"] == 1
    assert search.passes_run == 1


def test_run_keeps_going_while_a_pass_still_changes_something(monkeypatch):
    """A pass that moves a setting earns exactly one more pass, to confirm
    the new value is itself stable -- not the full max_passes budget."""
    from haemolynx.optimisation.search import _Search

    search = _Search(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0),
        starting_values=dict(_DEFAULT_STARTING_VALUES), progress=None,
    )
    calls = {"n": 0}

    def fake_pass():
        calls["n"] += 1
        if calls["n"] == 1:
            search.current["skeleton_min_branch_length"] = 99  # changes on pass 1 only

    monkeypatch.setattr(search, "_run_one_pass", fake_pass)
    search.run(max_passes=5)

    assert calls["n"] == 2
    assert search.passes_run == 2
    assert search.current["skeleton_min_branch_length"] == 99


def test_run_respects_max_passes_as_a_hard_cap_when_never_converging(monkeypatch):
    """A pathological cost function that keeps finding an improvement every
    single pass must not loop forever -- max_passes is a hard budget."""
    from haemolynx.optimisation.search import _Search

    search = _Search(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0),
        starting_values=dict(_DEFAULT_STARTING_VALUES), progress=None,
    )
    calls = {"n": 0}

    def fake_pass():
        calls["n"] += 1
        search.current["skeleton_min_branch_length"] = calls["n"]  # always different

    monkeypatch.setattr(search, "_run_one_pass", fake_pass)
    search.run(max_passes=3)

    assert calls["n"] == 3
    assert search.passes_run == 3


def test_run_scales_the_progress_total_by_max_passes():
    """A progress bar's own fraction must never exceed 1 just because
    convergence needed more than one pass through the group sequence."""
    from haemolynx.optimisation.search import _GROUP_TOTAL_UPPER_BOUND, _Search

    search = _Search(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0),
        starting_values=dict(_DEFAULT_STARTING_VALUES), progress=None,
    )
    assert search._group_total == _GROUP_TOTAL_UPPER_BOUND

    search.run(max_passes=3)
    assert search._group_total == _GROUP_TOTAL_UPPER_BOUND * 3


def test_optimise_settings_forwards_max_passes_and_reports_passes_run(y_shaped_mask):
    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
        max_passes=2,
    )
    assert 1 <= result.passes_run <= 2
    assert set(result.settings) == set(OPTIMISE_SETTING_NAMES)


def test_optimise_settings_default_max_passes_reports_exactly_one_pass(y_shaped_mask):
    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
    )
    assert result.passes_run == 1


def test_sweep_with_refinement_finds_a_value_between_coarse_grid_points():
    """Regression: the coarse multiplier grids this search's candidate
    generators use (0.5x/1x/2x/4x a base scale) cannot land on an
    arbitrary true optimum -- a genuinely better value can sit between two
    grid points, which a single coarse `_sweep` can never find. The
    refinement round's neighbourhood around the coarse winner can get
    closer.
    """
    from haemolynx.optimisation.search import _Search

    search = _Search(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0),
        starting_values=dict(_DEFAULT_STARTING_VALUES), progress=None,
    )
    true_optimum = 2.4  # between the coarse grid's 2.0 and 4.0, closer to 2.5 (2.0 * 1.25)

    def cost(value):
        return (value - true_optimum) ** 2

    coarse_only = search._sweep("test_group", "test_setting", [0.5, 1.0, 2.0, 4.0], cost)
    assert coarse_only == 2.0  # the closest coarse grid point alone

    search.current["test_setting"] = 0.5  # reset before the refined run
    refined = search._sweep_with_refinement("test_group", "test_setting", [0.5, 1.0, 2.0, 4.0], cost)
    assert abs(refined - true_optimum) < abs(coarse_only - true_optimum)
    assert search.current["test_setting"] == refined


def test_sweep_with_refinement_never_regresses_the_coarse_winner():
    """The coarse winner is always included in the refine round's own
    candidate set, so refinement can only match or improve on it -- never
    silently replace it with something worse."""
    from haemolynx.optimisation.search import _Search

    search = _Search(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0),
        starting_values=dict(_DEFAULT_STARTING_VALUES), progress=None,
    )
    # The coarse winner (2.0) is already the true optimum -- every refined
    # neighbour (1.5, 2.5) is strictly worse.
    def cost(value):
        return (value - 2.0) ** 2

    refined = search._sweep_with_refinement("test_group", "test_setting", [0.5, 1.0, 2.0, 4.0], cost)
    assert refined == 2.0


def test_sweep_with_refinement_respects_the_max_value_cap():
    """Refining outward must not reintroduce a candidate the coarse grid's
    own cap (e.g. a typical_radius_um safety bound) excluded for real
    safety reasons, not just because it was not on the grid."""
    from haemolynx.optimisation.search import _Search

    search = _Search(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0),
        starting_values=dict(_DEFAULT_STARTING_VALUES), progress=None,
    )
    # True optimum is past the cap -- 2.5 (2.0 * 1.25) must never be tried.
    def cost(value):
        return (value - 10.0) ** 2

    refined = search._sweep_with_refinement(
        "test_group", "test_setting", [0.5, 1.0, 2.0], cost, max_value=2.0,
    )
    assert refined == 2.0
    assert all(
        trial.value <= 2.0
        for trial in search.trials
        if trial.setting == "test_setting"
    )


def test_sweep_with_refinement_ignores_non_numeric_settings():
    """A boolean/choice setting's winner is not a radius to refine around;
    _sweep_with_refinement must pass it through unchanged rather than
    trying to multiply it by a fraction."""
    from haemolynx.optimisation.search import _Search

    search = _Search(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0),
        starting_values=dict(_DEFAULT_STARTING_VALUES), progress=None,
    )
    winner = search._sweep_with_refinement(
        "test_group", "test_setting", ["gaussian", "morphological"],
        lambda value: 0.0 if value == "morphological" else 1.0,
    )
    assert winner == "morphological"


def test_raw_mask_local_radius_map_is_cached_across_guard_calls(monkeypatch):
    """Regression: `_vessels_represented_fraction`/
    `_skeleton_mask_coverage_fraction` used to trigger a fresh full-volume
    EDT (via `inscribed_radius_map`) inside `diagnose_*` on every single
    call, even though `self.raw_mask` is fixed for most of a run -- dozens
    of avoidable EDT/labelling passes on a real, whole-brain-scale volume
    across one "Optimise settings" run. Confirms the cache means only one
    such computation happens across many guard calls against the same mask.
    """
    from haemolynx import preprocessing as preprocessing_module
    from haemolynx.optimisation.search import _Search

    mask = _y_shaped_vessel()
    starting_values = dict(_DEFAULT_STARTING_VALUES)
    search = _Search(
        mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=starting_values, progress=None,
    )
    search.raw_skeleton = preprocessing_module.skeletonize_volume(mask)

    real_inscribed_radius_map = preprocessing_module.inscribed_radius_map
    call_count = {"n": 0}

    def counting(mask_arr, voxel_size_zyx):
        call_count["n"] += 1
        return real_inscribed_radius_map(mask_arr, voxel_size_zyx)

    monkeypatch.setattr(preprocessing_module, "inscribed_radius_map", counting)

    search._vessels_represented_fraction(search.raw_skeleton)
    search._skeleton_mask_coverage_fraction(search.raw_skeleton)
    search._vessels_represented_fraction(search.raw_skeleton)
    search._skeleton_mask_coverage_fraction(search.raw_skeleton)

    assert call_count["n"] == 1, (
        "self.raw_mask's own local radius map should be computed once and "
        f"reused across guard calls, got {call_count['n']} separate "
        "full-volume EDT calls"
    )

    # The cache must not go stale once self.raw_mask is reassigned to a
    # genuinely different array (segmentation_cleanup does this exactly
    # once, mid-run) -- a new mask needs its own fresh radius map.
    search.raw_mask = np.zeros_like(search.raw_mask)
    search.raw_mask[0:3, 0:3, 0:3] = True
    search._vessels_represented_fraction(search.raw_skeleton)
    assert call_count["n"] == 2


def test_min_branch_length_guard_rejects_a_candidate_that_drops_a_real_small_vessel():
    """Without the vessels-missing guard, pruning a real-but-tiny (and far
    from the main body, so never re-bridged) vessel's own skeleton barely
    moves `largest_fraction` -- the guard is what actually catches this."""
    from haemolynx.optimisation.search import _GUARD_PENALTY, _Search

    shape = (30, 30, 30)
    main_mask = np.zeros(shape, dtype=bool)
    main_mask[10:20, 14:16, 14:16] = True  # a solid main vessel body
    small_vessel_mask = np.zeros(shape, dtype=bool)
    small_vessel_mask[2:6, 2:6, 2:6] = True  # a real, separate small vessel (64 voxels)
    raw_mask = main_mask | small_vessel_mask

    raw_skeleton = np.zeros(shape, dtype=bool)
    raw_skeleton[10:20, 15, 15] = True  # 10-voxel main skeleton line
    raw_skeleton[3, 3, 3:5] = True  # 2-voxel stub for the small vessel, far from the main line

    starting_values = dict(_DEFAULT_STARTING_VALUES)
    # The guard's own baseline is measured at the group's *starting*
    # min_branch_length value (run through the full pipeline, see
    # `_group_min_branch_length`'s own comment) -- it must start below the
    # small vessel's own 2-voxel stub, or the baseline would already prune
    # it and no candidate could "regress" any further.
    starting_values["skeleton_min_branch_length"] = 0
    search = _Search(
        raw_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=starting_values, progress=None,
    )
    search.raw_skeleton = raw_skeleton
    search._group_min_branch_length()

    trials_by_value = {
        trial.value: trial.score
        for trial in search.trials
        if trial.setting == "skeleton_min_branch_length"
    }
    removing_candidates = [value for value in trials_by_value if value >= 3]
    assert removing_candidates, "expected at least one candidate >= the small vessel's own 2-voxel stub"
    for value in removing_candidates:
        assert trials_by_value[value] >= _GUARD_PENALTY, (
            f"skeleton_min_branch_length={value} removes the small vessel's stub "
            f"entirely and should have tripped the vessels-missing guard, got "
            f"score={trials_by_value[value]}"
        )


def test_min_branch_length_guard_ignores_segmentation_noise_flecks():
    """A real-data run showed the guard's default noise floor
    (`diagnose_vessels_missing_from_skeleton`'s own `min_vessel_voxels=2`)
    treated tiny segmentation-noise flecks in the mask as "real vessels" it
    had to protect, blocking min_branch_length from pruning almost
    anything at all (3 of 4 real-data candidates rejected) -- see
    `_MIN_VESSEL_VOLUME_UM3_FOR_GUARD`'s own docstring. This proves the
    fix: pruning a skeleton stub backed only by a mask-noise fleck (well
    under the physical volume floor) must not trip the guard."""
    from haemolynx.optimisation.search import _GUARD_PENALTY, _Search

    shape = (30, 30, 30)
    main_mask = np.zeros(shape, dtype=bool)
    main_mask[10:20, 14:16, 14:16] = True
    noise_fleck_mask = np.zeros(shape, dtype=bool)
    noise_fleck_mask[2, 2, 2:4] = True  # 2 voxels -- segmentation noise, not a real vessel
    raw_mask = main_mask | noise_fleck_mask

    raw_skeleton = np.zeros(shape, dtype=bool)
    raw_skeleton[10:20, 15, 15] = True  # 10-voxel main skeleton line
    raw_skeleton[2, 2, 2:4] = True  # 2-voxel stub matching the noise fleck, far from the main line

    starting_values = dict(_DEFAULT_STARTING_VALUES)
    # See the sibling test above: the baseline is measured at the group's
    # starting value, which must start below the fleck's own 2-voxel stub.
    starting_values["skeleton_min_branch_length"] = 0
    search = _Search(
        raw_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=starting_values, progress=None,
    )
    search.raw_skeleton = raw_skeleton
    search._group_min_branch_length()

    trials_by_value = {
        trial.value: trial.score
        for trial in search.trials
        if trial.setting == "skeleton_min_branch_length"
    }
    removing_candidates = [value for value in trials_by_value if value >= 3]
    assert removing_candidates, "expected at least one candidate >= the noise fleck's 2-voxel stub"
    for value in removing_candidates:
        assert trials_by_value[value] < _GUARD_PENALTY, (
            f"skeleton_min_branch_length={value} only removes a 2-voxel "
            f"segmentation-noise fleck (well under the physical volume "
            f"floor) and should not trip the vessels-missing guard, got "
            f"score={trials_by_value[value]}"
        )


def _cartwheel_graph() -> nx.MultiGraph:
    """One hub with 6 edges radiating evenly around it in the y-z plane --
    a textbook cartwheel artifact (see graph.cartwheel_guard's own module
    docstring), not a real vessel junction."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pos = np.array([0.0, 10.0 * np.cos(angle), 10.0 * np.sin(angle)])
        G.add_node(i + 1, pos=pos)
        G.add_edge(0, i + 1, length=10.0)
    return G


def _clean_hub_graph() -> nx.MultiGraph:
    """A degree-3 bifurcation whose daughters continue in a coherent
    general direction -- a real vessel junction, never flagged."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.0, 10.0, 0.0]))
    G.add_node(2, pos=np.array([0.0, -7.0, 7.0]))
    G.add_node(3, pos=np.array([0.0, -7.0, -7.0]))
    G.add_edge(0, 1, length=10.0)
    G.add_edge(0, 2, length=10.0)
    G.add_edge(0, 3, length=10.0)
    return G


def test_cartwheel_hub_count_detects_a_wheel_and_ignores_a_real_junction():
    from haemolynx.optimisation.search import _Search

    search = _Search(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0),
        starting_values=dict(_DEFAULT_STARTING_VALUES), progress=None,
    )
    assert search._cartwheel_hub_count(_cartwheel_graph()) == 1
    assert search._cartwheel_hub_count(_clean_hub_graph()) == 0


def test_cluster_collapse_guard_rejects_a_candidate_that_creates_a_cartwheel_hub(monkeypatch):
    """Regression: graph_topology_metrics alone (nodes/edges/components/
    self-loops/degree-2 count) cannot see a "cartwheel" hub -- collapsing a
    cluster of nearby, real junctions into one representative is exactly
    the mechanism graph.cartwheel_guard's own module docstring names as
    the cause, and it moves none of those five numbers on its own. A
    candidate that introduces one must still be rejected in favour of one
    that does not, even though the cartwheel graph here has fewer degree-2
    nodes (0, vs 0 for the clean graph too -- deliberately tied on the
    score `_group_cluster_collapse` otherwise minimises, so only the new
    guard can be why the sweep prefers the non-cartwheel candidate).
    """
    from haemolynx.optimisation import candidates as cand_module
    from haemolynx.optimisation.search import _GUARD_PENALTY, _Search

    search = _Search(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0),
        starting_values=dict(_DEFAULT_STARTING_VALUES), progress=None,
    )
    search.current_skeleton = np.zeros((5, 5, 5), dtype=bool)
    search.current_graph = _clean_hub_graph()
    monkeypatch.setattr(search, "_skeleton_graph_coverage_fraction", lambda G: 1.0)

    small, large = 1.0, 50.0
    graphs_by_distance = {small: _clean_hub_graph(), large: _cartwheel_graph()}

    def fake_build_graph(overrides):
        value = overrides.get("cluster_collapse_distance", search.current["cluster_collapse_distance"])
        return graphs_by_distance[value]

    monkeypatch.setattr(search, "_build_graph", fake_build_graph)
    monkeypatch.setattr(
        cand_module, "cluster_collapse_distance_candidates", lambda *_a, **_k: [small, large]
    )
    search.current["cluster_collapse_distance"] = small

    search._group_cluster_collapse()

    trials_by_value = {
        trial.value: trial.score
        for trial in search.trials
        if trial.setting == "cluster_collapse_distance"
    }
    assert trials_by_value[large] >= _GUARD_PENALTY, (
        f"the large candidate creates a cartwheel hub and should have "
        f"tripped the guard, got trials={trials_by_value}"
    )
    assert trials_by_value[small] < _GUARD_PENALTY
    assert search.current["cluster_collapse_distance"] == small


def test_segmentation_cleanup_raw_image_guard_rejects_invented_foreground():
    """Turning close_gaps on bridges a real gap the raw signal does not
    support -- quality() alone (fragmentation/connectivity) would favour
    bridging it, but the raw-image guard must reject that once a reference
    image shows there is nothing there."""
    from haemolynx.optimisation.search import _GUARD_PENALTY, _Search

    shape = (16, 16, 16)
    mask = np.zeros(shape, dtype=bool)
    mask[4:12, 4:7, 4:7] = True
    mask[4:12, 8:11, 4:7] = True  # two blocks, a genuine 1-voxel gap at y=7

    # Raw image: bright exactly where the mask already is, dark everywhere
    # else -- including the gap -- so bridging it invents unsupported
    # foreground with no raw signal behind it at all.
    raw_image = np.where(mask, 200.0, 10.0).astype(np.float32)

    starting_values = dict(_DEFAULT_STARTING_VALUES)
    starting_values["segmentation_cleanup_close_gaps"] = False

    search = _Search(
        mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=starting_values, progress=None,
        raw_image=raw_image,
    )
    search._group_segmentation_cleanup()

    toggle_trials = {
        trial.value: trial.score
        for trial in search.trials
        if trial.setting == "segmentation_cleanup_close_gaps"
    }
    # Comfortably above any un-guarded quality() score (roughly [-10, 10]) --
    # the guard's own `_GUARD_PENALTY` (1000) minus quality()'s own
    # contribution, not the raw 1000 itself.
    assert toggle_trials.get(True, 0.0) > 500.0, (
        "turning close_gaps on bridges a gap the raw image does not "
        f"support and should have tripped the guard, got trials={toggle_trials}"
    )
    assert toggle_trials[True] > toggle_trials[False]
    assert search.current["segmentation_cleanup_close_gaps"] is False


def test_segmentation_cleanup_raw_image_guard_rejects_erasing_real_signal():
    """The symmetric case to the invented-foreground test above: without
    this guard, `compare_segmentation_to_raw_image`'s own `removed_fraction`
    was computed but never read, so nothing here caught a candidate that
    erases real, raw-supported foreground the way over-aggressive
    smoothing/closing can on real data (confirmed there to erase ~88% of a
    true vasculature, none of it flagged as "added", since nothing was
    invented -- it was all removed). Whether or not score_segmented_mask's
    own geometry-only sub-scores happen to already disfavour a given
    candidate, the guard itself must independently reject one that erases
    real signal a reference image confirms is genuinely there.
    """
    from haemolynx.optimisation.search import _GUARD_PENALTY, _Search

    shape = (20, 20, 20)
    mask = np.zeros(shape, dtype=bool)
    mask[4:16, 4:16, 4:16] = True  # a solid main body
    mask[16:19, 9:11, 9:11] = True  # a thin (radius-1) real whisker off one face

    # Raw image: bright exactly where the mask is -- body and whisker alike
    # -- so the whisker is real, raw-supported signal, not noise.
    raw_image = np.where(mask, 200.0, 10.0).astype(np.float32)

    starting_values = dict(_DEFAULT_STARTING_VALUES)
    starting_values["segmentation_cleanup_remove_whiskers"] = False
    starting_values["segmentation_cleanup_whisker_radius_um"] = 1.0

    search = _Search(
        mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=starting_values, progress=None,
        raw_image=raw_image,
    )
    search._group_segmentation_cleanup()

    toggle_trials = {
        trial.value: trial.score
        for trial in search.trials
        if trial.setting == "segmentation_cleanup_remove_whiskers"
    }
    assert toggle_trials.get(True, 0.0) > 500.0, (
        "removing the real whisker erases raw-supported signal and should "
        f"have tripped the removed-fraction guard, got trials={toggle_trials}"
    )
    assert toggle_trials[True] > toggle_trials[False]
    assert search.current["segmentation_cleanup_remove_whiskers"] is False


def test_segmentation_cleanup_whisker_radius_gets_a_refinement_round():
    """Integration check that the real segmentation-cleanup group actually
    wires `refine=True` through for its radius-style settings, not just
    that `_sweep_with_refinement` works in isolation: the trials recorded
    for whisker_radius_um must include a value that is not on
    `small_radius_candidates`'s own coarse grid, proving a second, refined
    round of candidates really ran.
    """
    from haemolynx.optimisation import candidates as cand_module
    from haemolynx.optimisation.search import _Search

    mask = _y_shaped_vessel()
    starting_values = dict(_DEFAULT_STARTING_VALUES)
    starting_values["segmentation_cleanup_remove_whiskers"] = True

    search = _Search(
        mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=starting_values, progress=None,
    )
    search._group_segmentation_cleanup()

    coarse_grid = set(
        cand_module.small_radius_candidates(
            (1.0, 1.0, 1.0),
            float(starting_values["segmentation_cleanup_whisker_radius_um"]),
            typical_radius_um=search.typical_radius_um,
        )
    )
    radius_trial_values = {
        trial.value
        for trial in search.trials
        if trial.setting == "segmentation_cleanup_whisker_radius_um"
    }
    assert radius_trial_values - coarse_grid, (
        "expected at least one refined candidate beyond the coarse grid "
        f"{coarse_grid}, got trials={radius_trial_values}"
    )


def test_segmentation_cleanup_reuses_one_precomputed_raw_image_foreground(monkeypatch):
    """Regression: `_group_segmentation_cleanup` used to redo Otsu
    thresholding and connected-components labelling of `self.raw_image`
    inside every single `compare_segmentation_to_raw_image` call -- once
    per candidate, across ~20 sequential sub-sweeps. Confirms the
    `RawImageForeground` computed once in `_Search.__init__` is what every
    one of those calls reuses instead."""
    from haemolynx.preprocessing import segmentation_raw_comparison as src_module
    from haemolynx.optimisation.search import _Search

    mask = _y_shaped_vessel()
    raw_image = np.where(mask, 200.0, 10.0).astype(np.float32)
    starting_values = dict(_DEFAULT_STARTING_VALUES)

    search = _Search(
        mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=starting_values, progress=None,
        raw_image=raw_image,
    )
    assert search._raw_image_foreground is not None

    def boom(*_args, **_kwargs):
        raise AssertionError(
            "threshold_otsu/label should not run again once the group has "
            "a precomputed RawImageForeground for its own raw_image"
        )

    monkeypatch.setattr(src_module, "threshold_otsu", boom)
    monkeypatch.setattr(src_module, "label", boom)

    search._group_segmentation_cleanup()  # must not raise


def test_segmentation_cleanup_raw_image_none_matches_todays_behaviour():
    """raw_image=None (the default) must not change anything about how
    segmentation_cleanup scores its candidates."""
    from haemolynx.optimisation.search import _Search

    mask = _y_shaped_vessel()
    starting_values = dict(_DEFAULT_STARTING_VALUES)

    without_raw = _Search(mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=starting_values, progress=None)
    without_raw._group_segmentation_cleanup()

    with_none_raw = _Search(
        mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=starting_values, progress=None, raw_image=None,
    )
    with_none_raw._group_segmentation_cleanup()

    assert without_raw.current == with_none_raw.current


def test_optimise_settings_on_scattered_speckle_does_not_raise():
    """All-disconnected noise: many tiny components, no coherent network."""
    rng = np.random.default_rng(0)
    speckle = rng.random((15, 15, 15)) > 0.97
    result = optimise_skeleton_and_graph_settings(
        speckle, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
    )
    assert set(result.settings) == set(OPTIMISE_SETTING_NAMES)


def test_a_search_whose_graph_comes_back_empty_keeps_the_graph_settings(monkeypatch):
    import haemolynx.optimisation.search as search_module
    from haemolynx.optimisation.report import build_report_text

    monkeypatch.setattr(
        search_module.graph_mod, "build_graph_from_skeleton", lambda *a, **k: nx.MultiGraph()
    )
    result = optimise_skeleton_and_graph_settings(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0),
        starting_values=_DEFAULT_STARTING_VALUES, downsample_factor=1,
    )
    later = ("cluster_collapse_distance", "min_stub_length", "centreline_smoothing")
    assert not set(later) & set(result.groups_run)
    assert result.settings["min_stub_length"] == _DEFAULT_STARTING_VALUES["min_stub_length"]
    not_optimised = next(
        line for line in build_report_text(result).splitlines() if line.startswith("Not optimised")
    )
    assert all(group in not_optimised for group in later)


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


def test_estimate_downsample_factor_for_time_budget_falls_back_on_an_empty_mask():
    """An empty mask has nothing to time a real probe against; falling back
    to the voxel-count heuristic must not raise."""
    from haemolynx.optimisation.search import estimate_downsample_factor_for_time_budget

    empty = np.zeros((10, 10, 10), dtype=bool)
    factor = estimate_downsample_factor_for_time_budget(empty, (1.0, 1.0, 1.0))
    assert factor == resolve_auto_downsample_factor(empty.shape)


def test_estimate_downsample_factor_for_time_budget_picks_full_resolution_for_a_generous_budget(
    y_shaped_mask,
):
    """A tiny synthetic mask's own real probe is fast enough that even a
    demanding budget (the default, five minutes) should never need to
    sacrifice detail."""
    from haemolynx.optimisation.search import estimate_downsample_factor_for_time_budget

    factor = estimate_downsample_factor_for_time_budget(
        y_shaped_mask, (1.0, 1.0, 1.0), target_seconds=300.0,
    )
    assert factor == 1


def _proportional_model():
    """Graph builds timed at 0.01 s on a one-voxel grid -- the 16x grid of a
    volume of 16**3 voxels."""
    from haemolynx.optimisation.search import _SearchTimeModel

    return _SearchTimeModel({"graph": 0.01}, probe_voxels=1.0)


def test_factor_from_time_model_picks_the_coarsest_factor_for_a_tiny_budget():
    """A budget no real probe could ever fit under must fall back to the
    coarsest offered factor, not raise or return something outside
    DOWNSAMPLE_FACTORS. Pure arithmetic (see this function's own docstring
    for why it is split out from the real probe) -- no timing involved, so a
    genuinely unmeetable budget can be tested directly rather than relying
    on real wall-clock timing.
    """
    from haemolynx.optimisation.search import _factor_from_time_model

    factor = _factor_from_time_model(_proportional_model(), 16.0**3, target_seconds=1e-9)
    assert factor == DOWNSAMPLE_FACTORS[-1]


def test_factor_from_time_model_picks_full_resolution_for_a_generous_budget():
    from haemolynx.optimisation.search import _factor_from_time_model

    factor = _factor_from_time_model(_proportional_model(), 16.0**3, target_seconds=1e9)
    assert factor == DOWNSAMPLE_FACTORS[0]


def test_factor_from_time_model_is_monotonic_in_target_seconds():
    """A smaller time budget must never pick a *finer* (smaller) factor than
    a larger budget, for the same measured model."""
    from haemolynx.optimisation.search import _factor_from_time_model

    generous = _factor_from_time_model(_proportional_model(), 16.0**3, target_seconds=1e9)
    stingy = _factor_from_time_model(_proportional_model(), 16.0**3, target_seconds=1e-9)
    assert stingy >= generous


def test_factor_from_time_model_free_evaluations_mean_full_resolution():
    """Probes that measured as zero give no evidence downsampling would help;
    default to full detail."""
    from haemolynx.optimisation.search import _factor_from_time_model, _SearchTimeModel

    assert _factor_from_time_model(_SearchTimeModel({}, 1.0), 1e9, 1e-9) == DOWNSAMPLE_FACTORS[0]


def test_optimise_settings_auto_downsample_uses_the_time_budget_resolver(monkeypatch):
    """Regression: 'Auto' (downsample_factor=None) used to resolve purely
    from the mask's own voxel count (resolve_auto_downsample_factor),
    which assumes a fixed voxels-per-second rate true of neither a
    specific machine nor a specific dataset's own topological complexity.
    It must now consult the time-budget estimate
    (estimate_downsample_factor_for_time_budget's own `_auto_downsample`,
    which also returns the time it estimated and the resolution cap it
    applied, for the report) instead,
    forwarding the caller's own auto_downsample_target_seconds and
    use_thick_vessel_skeletonisation starting value.
    """
    import haemolynx.optimisation.search as search_module

    recorded = {}

    def fake_estimate(raw_mask, voxel_size_zyx, *, target_seconds, starting_values):
        recorded["target_seconds"] = target_seconds
        recorded["use_thick_vessel_skeletonisation"] = starting_values["use_thick_vessel_skeletonisation"]
        return search_module._AutoDownsample(
            4, estimated_seconds=123.0, typical_radius_um=9.0, resolution_cap=4,
        )

    monkeypatch.setattr(search_module, "_auto_downsample", fake_estimate)
    monkeypatch.setattr(
        search_module, "resolve_auto_downsample_factor",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not use the voxel-count heuristic for Auto")),
    )

    starting_values = dict(_DEFAULT_STARTING_VALUES)
    starting_values["use_thick_vessel_skeletonisation"] = True
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[10:30, 10:30, 10:30] = True

    result = search_module.optimise_skeleton_and_graph_settings(
        mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=starting_values,
        auto_downsample_target_seconds=42.0,
    )

    assert recorded["target_seconds"] == 42.0
    assert recorded["use_thick_vessel_skeletonisation"] is True
    assert result.downsample_factor == 4
    assert result.estimated_seconds == pytest.approx(123.0)
    assert (result.typical_radius_um, result.resolution_cap) == (9.0, 4)


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
        def __init__(self, mask, voxel_size_xyz, starting_values, progress, enabled_groups=None, raw_image=None):
            self.mask = mask
            self.voxel_size_xyz_seen = voxel_size_xyz
            self.current = dict(starting_values)
            self.current["skeleton_closing_radius"] = 3
            self.current["skeleton_min_branch_length"] = 5
            self.current["graph_reconnect_threshold"] = 12.5  # a micron setting: must pass through unchanged
            self.trials = []
            self.groups_run = list(search_module.GROUP_NAMES)
            self.passes_run = 1
            self.group_seconds = {}

        def run(self, *, max_passes=1):
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
        def __init__(self, mask, voxel_size_xyz, starting_values, progress, enabled_groups=None, raw_image=None):
            seen["mask_shape"] = mask.shape
            seen["voxel_size_xyz"] = voxel_size_xyz
            self.current = dict(starting_values)
            self.trials = []
            self.groups_run = []
            self.passes_run = 1
            self.group_seconds = {}

        def run(self, *, max_passes=1):
            pass

    monkeypatch.setattr(search_module, "_Search", _FakeSearch)

    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[10:30, 10:30, 10:30] = True
    optimise_skeleton_and_graph_settings(
        mask, voxel_size_xyz=(1.0, 1.0, 2.0), starting_values=_DEFAULT_STARTING_VALUES,
        downsample_factor=4,
    )
    # z is twice as coarse, so it is reduced half as much: 4 um voxels on
    # every axis, not 4 x 4 x 8.
    assert seen["mask_shape"] == (20, 10, 10)
    assert seen["voxel_size_xyz"] == pytest.approx((4.0, 4.0, 4.0))


def test_voxel_scaled_setting_names_are_all_real_settings():
    assert set(_VOXEL_SCALED_SETTING_NAMES) <= set(OPTIMISE_SETTING_NAMES)


def test_optimise_settings_runs_for_real_at_an_explicit_downsample_factor(y_shaped_mask):
    """No mocking: the real search, actually run on a real (2x) downsampled
    copy of a real mask, still completes and returns every setting -- each
    voxel-counted one as the user had it when its sweep kept it, and its
    search-grid value times the factor when its sweep moved it.

    Regression: this used to require every one to be a multiple of the
    factor, kept or not -- the bug that turned a closing radius of 2 into 32
    on a 16x search."""
    from haemolynx.optimisation.search import _to_search_grid

    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
        downsample_factor=2,
    )
    assert result.downsample_factor == 2
    assert set(result.settings) == set(OPTIMISE_SETTING_NAMES)
    on_search_grid = _to_search_grid(_DEFAULT_STARTING_VALUES, 2)
    for name in _VOXEL_SCALED_SETTING_NAMES:
        if name == "skeleton_bundle_hub_min_spacing":
            continue  # derived from the scan size, below
        sweeps = [t for t in result.trials if t.setting == name and t.chosen]
        assert len(sweeps) == 1, name
        if sweeps[0].value == on_search_grid[name]:
            assert result.settings[name] == _DEFAULT_STARTING_VALUES[name], name
        else:
            assert result.settings[name] == sweeps[0].value * 2, name
    assert result.settings["skeleton_bundle_hub_min_spacing"] == max(
        1, result.settings["skeleton_bundle_scan_size"] // 2
    )


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
    # Stubs are judged by a multiple of their parent vessel's radius by
    # default, so that multiple is what the group tunes.
    assert tried_settings == {"min_stub_length_radius_multiple"}
    assert result.settings["skeleton_closing_radius"] == _DEFAULT_STARTING_VALUES["skeleton_closing_radius"]


def test_the_stub_group_tunes_the_fixed_length_when_radius_pruning_is_off(y_shaped_mask):
    starting = {**_DEFAULT_STARTING_VALUES, "min_stub_length_radius_multiple": 0.0}
    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=starting,
        groups=("min_stub_length",),
    )
    assert {trial.setting for trial in result.trials} == {"min_stub_length"}
    assert result.settings["min_stub_length_radius_multiple"] == 0.0


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
    # Regression: whichever toggle cleared the speck, the master switch that
    # gates all seven (pipeline.schema's segmentation_cleanup) must also end
    # up on, or a real run applying result.settings would silently ignore
    # the very toggle that just fixed this.
    assert any(cleanup_kwargs.values())
    assert result.settings["segmentation_cleanup"] is True


def _capillary_bed_with_one_resolved_trunk(shape=(40, 60, 60)) -> np.ndarray:
    """One thick (radius ~4 voxel), clearly-resolved trunk, with many thin
    (radius ~1 voxel) capillary-scale branches off it -- like a real, dense
    microvascular bed where a handful of larger vessels feed a much larger
    number of near-resolution-limit capillaries. Many more ridge points
    come from the thin branches than from the trunk, so a plain median
    over all of them -- even a medial-ridge one -- is dominated by the
    thin branches, not the only part of this mask any radius-based
    candidate could safely operate at the scale of.
    """
    skeleton = np.zeros(shape, dtype=bool)
    trunk_a = np.array([20, 5, 30])
    trunk_b = np.array([20, 55, 30])
    _draw_line(skeleton, trunk_a, trunk_b)
    trunk_mask = binary_dilation(skeleton, structure=np.ones((9, 9, 9), dtype=bool))

    thin_skeleton = np.zeros(shape, dtype=bool)
    rng = np.random.default_rng(0)
    for y in range(6, 55, 2):
        far = [20 + int(rng.integers(-10, 11)), y, 30 + int(rng.integers(-25, 26))]
        _draw_line(thin_skeleton, [20, y, 30], far)
    thin_mask = binary_dilation(thin_skeleton, structure=np.ones((3, 3, 3), dtype=bool))

    return trunk_mask | thin_mask


def test_typical_radius_um_is_not_dragged_down_by_many_thin_capillaries():
    """Regression: ``typical_radius_um`` used to be the median inscribed
    radius over every foreground voxel of the whole mask -- a whole-mask
    bias (most of a vessel's own volume sits near its surface, so this
    underestimates its true radius by roughly 3-4x -- see
    ``segmentation_quality.py``'s own fix for the same bias) stacked with,
    even switching to the medial ridge alone, still just its raw median.
    On a mask with many more thin (~1 voxel radius) capillary branches
    than points on one much wider, clearly-resolved trunk (~4 voxel
    radius), the raw ridge median is dominated by the thin branches even
    though they sit at or below this codebase's own discretisation-bias
    floor (``DEFAULT_TARGET_VOXELS_ACROSS_RADIUS``). Restricting to ridge
    points at or above that floor recovers the trunk's own scale instead
    -- the only scale any of this group's size-based candidates could
    safely use.
    """
    from haemolynx.optimisation.search import _Search

    mask = _capillary_bed_with_one_resolved_trunk()
    search = _Search(
        mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES, progress=None,
    )
    assert search.typical_radius_um >= 3.0


def test_smooth_sigma_candidates_capped_by_typical_radius_do_not_erase_the_mask():
    """Regression, isolating the actual mechanism (see
    ``test_typical_radius_um_is_not_dragged_down_by_many_thin_capillaries``
    for why ``typical_radius_um`` itself needed fixing first): applying
    Gaussian smooth-then-rethreshold at the *uncapped* candidate
    ``small_radius_candidates`` used to offer (up to 4x the coarsest voxel
    spacing, with no reference to vessel size) against a mask whose real
    vessels are only ~1 voxel radius must still behave exactly as before
    -- badly, confirmed on real data to erase ~88% of a comparable network
    -- while the same call at the *capped* candidate (typical_radius_um
    aware) leaves most of the mask intact. This is what
    ``_group_segmentation_cleanup``'s own sweep now always chooses from,
    for every dataset, without needing its own end-to-end test here.
    """
    from haemolynx.optimisation import candidates as cand
    from haemolynx.preprocessing import clean_segmented_mask_for_skeletonisation

    mask = _capillary_bed_with_one_resolved_trunk()
    voxel_size_zyx = (1.0, 1.0, 1.0)
    typical_radius_um = 1.0  # this mask's own thin-capillary scale

    uncapped_candidates = cand.small_radius_candidates(voxel_size_zyx, default=1.0)
    capped_candidates = cand.small_radius_candidates(
        voxel_size_zyx, default=1.0, typical_radius_um=typical_radius_um
    )
    assert max(uncapped_candidates) > max(capped_candidates)

    def _smoothed_fraction(sigma_um: float) -> float:
        cleaned, _raw = clean_segmented_mask_for_skeletonisation(
            mask,
            voxel_size_zyx=voxel_size_zyx,
            fill_cavities=False,
            remove_whiskers=False,
            whisker_radius_um=1.0,
            split_narrow_necks=False,
            split_min_marker_separation_um=10.0,
            split_min_pinch_radius_ratio=0.6,
            split_min_body_radius_um=1.0,
            close_gaps=False,
            close_gaps_radius_um=0.5,
            reconnect_gaps=False,
            reconnect_max_bridge_distance_um=30.0,
            reconnect_min_cylindricality=0.5,
            reconnect_max_axis_angle_degrees=30.0,
            reconnect_min_facing_cosine=0.85,
            reconnect_max_radius_ratio=3.0,
            smooth_surfaces=True,
            smooth_method="gaussian",
            smooth_sigma_um=sigma_um,
            smooth_morphological_radius_um=1.0,
            remove_small_volumes=False,
            remove_small_min_volume_um3=5.0,
        )
        return cleaned.sum() / mask.sum()

    assert _smoothed_fraction(max(uncapped_candidates)) < 0.5
    assert _smoothed_fraction(max(capped_candidates)) >= 0.5


def test_group_names_cover_every_group_a_trial_could_report():
    """Every group string search.py actually uses to record a trial must be
    one of the selectable names, or a user could never disable it."""
    result = optimise_skeleton_and_graph_settings(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
    )
    used_groups = {trial.group for trial in result.trials}
    assert used_groups <= set(GROUP_NAMES)


def test_downsampling_reduces_each_axis_towards_isotropic_voxels():
    """Regression: every axis was reduced by the same factor, so a 4x search
    of a 0.5 x 0.5 x 2 um stack ran on 2 x 2 x 8 um voxels -- a vessel's z
    extent at four times the resolution loss of its in-plane width."""
    from haemolynx.optimisation.search import _downsample_mask, axis_downsample_factors

    spacing_zyx = (2.0, 0.5, 0.5)
    assert axis_downsample_factors(4, spacing_zyx) == (1, 4, 4)
    assert axis_downsample_factors(8, spacing_zyx) == (2, 8, 8)
    assert axis_downsample_factors(2, spacing_zyx) == (1, 2, 2)
    assert axis_downsample_factors(4, (1.0, 1.0, 1.0)) == (4, 4, 4)
    assert axis_downsample_factors(4) == (4, 4, 4)

    mask = np.zeros((8, 32, 32), dtype=bool)
    mask[3, 10, 10] = True
    reduced = _downsample_mask(mask, 4, spacing_zyx)
    assert reduced.shape == (8, 8, 8)
    assert reduced[3, 2, 2]


def test_a_search_on_an_anisotropic_stack_runs_on_near_isotropic_voxels(y_shaped_mask):
    stretched = np.repeat(np.repeat(y_shaped_mask, 4, axis=1), 4, axis=2)
    result = optimise_skeleton_and_graph_settings(
        stretched,
        voxel_size_xyz=(0.5, 0.5, 2.0),
        starting_values=_DEFAULT_STARTING_VALUES,
        downsample_factor=4,
        groups=["closing_radius"],
    )

    assert result.downsample_factor == 4
    assert result.downsample_factors_zyx == (1, 4, 4)


def test_the_time_estimate_counts_the_voxels_each_factor_keeps():
    """A factor that leaves the coarse axis alone keeps more voxels than its
    cube: at 0.5 x 0.5 x 2 um, factor 4 over factor 2 reduces only y and x
    further, 4x fewer voxels, not 8x."""
    from haemolynx.optimisation.search import _voxel_reduction

    spacing_zyx = (2.0, 0.5, 0.5)
    assert _voxel_reduction(4, spacing_zyx) / _voxel_reduction(2, spacing_zyx) == 4
    assert _voxel_reduction(4, None) / _voxel_reduction(2, None) == 8


def test_the_typical_radius_floor_is_judged_in_plane():
    """Regression (audit): only ridge radii of at least three of the coarsest
    voxels counted -- 6 um on a 2 um z -- so four 2.5 um capillaries beside one
    8 um vessel read as a typical radius of the large vessel's."""
    from types import SimpleNamespace

    import haemolynx.optimisation.search as search_module

    spacing = (2.0, 0.5, 0.5)
    z, y, x = np.indices((20, 120, 80))

    def tube(yc, radius):
        return ((z - 10) * spacing[0]) ** 2 + ((y - yc) * spacing[1]) ** 2 <= radius**2

    mask = tube(60, 8.0)
    for yc in (15, 30, 90, 105):
        mask |= tube(yc, 2.5)
    mask &= (x >= 5) & (x < 75)

    radius = search_module._Search._measure_typical_radius_um(
        SimpleNamespace(voxel_size_zyx=spacing), mask
    )

    assert radius < 4.0


def test_bundle_refinement_scores_density_over_its_own_window(monkeypatch):
    """Regression (audit): the bundle group's leftover-density cost filtered
    with ``size=scan_size`` -- a cube of voxels, four times deeper than wide on
    a 2 um z -- not the window bundle refinement itself scans with."""
    import scipy.ndimage

    from haemolynx import preprocessing
    from haemolynx.optimisation.search import _Search
    from haemolynx.preprocessing.skeleton import bundle_scan_window

    mask = _y_shaped_vessel()
    search = _Search(
        mask, voxel_size_xyz=(0.5, 0.5, 2.0), starting_values=dict(_DEFAULT_STARTING_VALUES),
        progress=None,
    )
    search.current_skeleton = preprocessing.skeletonize_volume(mask)

    sizes = []
    real_uniform_filter = scipy.ndimage.uniform_filter

    def recording_uniform_filter(values, size=3, *args, **kwargs):
        sizes.append(size)
        return real_uniform_filter(values, size=size, *args, **kwargs)

    monkeypatch.setattr(scipy.ndimage, "uniform_filter", recording_uniform_filter)
    search._group_bundle_refinement()

    assert sizes
    for size in sizes:
        assert isinstance(size, tuple) and size[0] < size[1] == size[2]
    scan = int(search.current["skeleton_bundle_scan_size"])
    assert bundle_scan_window(scan, search.voxel_size_zyx) in sizes


def test_closing_and_bridge_gap_candidates_are_in_finest_axis_voxels():
    """Regression (audit): the closing radius and the bridge gap size are
    read in voxels of the finest axis (a physical footprint), but their
    candidates came from gaps counted in plain voxels -- a 6 um gap of three
    2 um slices offered a bridge gap of 3, 1.5 um, which never reaches across
    a slice. The same gap now offers 12 (6 um / 0.5 um)."""
    from haemolynx.optimisation.search import _Search

    mask = _y_shaped_vessel()
    search = _Search(
        mask, voxel_size_xyz=(0.5, 0.5, 2.0), starting_values=dict(_DEFAULT_STARTING_VALUES),
        progress=None,
    )
    skeleton = np.zeros((20, 9, 9), dtype=bool)
    skeleton[0:8, 4, 4] = True
    skeleton[10:20, 4, 4] = True  # three slices on from z=7: 6 um
    search.current_skeleton = skeleton

    swept = {}

    def recording_sweep(group, setting, candidate_values, cost_fn):
        swept[setting] = list(candidate_values)
        return search.current[setting]

    search._sweep = recording_sweep
    search._preprocess_trial = lambda overrides: skeleton
    search._skeleton_mask_coverage_fraction = lambda cleaned: 1.0
    search._group_gap_bridging()

    # Beyond 0 and the default (3), the measured gap's own candidates.
    assert set(swept["skeleton_bridge_gap_size"]) - {0, 3} == {12}
    assert 12 in swept["skeleton_max_bridge_distance"]


def test_the_max_bridge_distance_candidates_keep_the_z_weight():
    from haemolynx.optimisation.search import _Search

    mask = _y_shaped_vessel()
    starting_values = dict(_DEFAULT_STARTING_VALUES, skeleton_bridge_z_distance_weight=2.0)
    search = _Search(
        mask, voxel_size_xyz=(0.5, 0.5, 2.0), starting_values=starting_values, progress=None,
    )
    skeleton = np.zeros((20, 9, 9), dtype=bool)
    skeleton[0:8, 4, 4] = True
    skeleton[10:20, 4, 4] = True
    search.current_skeleton = skeleton

    assert search._gap_distances_finest_voxels().tolist() == [12.0]
    assert search._gap_distances_finest_voxels(z_distance_weight=2.0).tolist() == [24.0]


# ---------------------------------------------------------------------------
# The shared sweep: the current value is kept unless something clearly beats it
# ---------------------------------------------------------------------------
def _bare_search():
    from haemolynx.optimisation.search import _Search

    return _Search(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0),
        starting_values=dict(_DEFAULT_STARTING_VALUES), progress=None,
    )


def test_a_sweep_keeps_the_current_value_on_a_tie():
    """Regression: on a tie the first candidate won -- the smallest, since
    candidates are sorted -- so a setting moved off the user's value for
    nothing."""
    search = _bare_search()
    search.current["test_setting"] = 4

    winner = search._sweep("test_group", "test_setting", [1.0, 2.0, 4.0], lambda value: 0.5)

    assert winner == 4 and isinstance(winner, int)
    assert search.current["test_setting"] == 4


def test_a_sweep_moves_only_for_an_improvement_beyond_the_margin():
    search = _bare_search()
    search.current["test_setting"] = 2.0

    # Better by 0.0005 on a score of 0.9: under SWEEP_MIN_IMPROVEMENT.
    kept = search._sweep("g", "test_setting", [2.0, 3.0], {2.0: -0.9, 3.0: -0.9005}.get)
    moved = search._sweep("g", "test_setting", [2.0, 3.0], {2.0: -0.9, 3.0: -0.95}.get)

    assert kept == 2.0
    assert moved == 3.0


def test_chosen_index_rules():
    from haemolynx.optimisation.search import _chosen_index

    inf = float("inf")
    # The incumbent is not among the candidates: the lowest, first on a tie.
    assert _chosen_index([1, 2, 3], [0.2, 0.1, 0.1], incumbent=9) == 1
    # The incumbent failed: the best of the rest.
    assert _chosen_index([1, 2, 3], [inf, 0.5, 0.3], incumbent=1) == 2
    # Everything failed: nothing chosen, the caller keeps its value.
    assert _chosen_index([1, 2], [inf, inf], incumbent=1) is None
    # A refined candidate rounded to six places is still the incumbent.
    assert _chosen_index([0.98, 1.225], [-1.0, -1.0], incumbent=round(0.98 * 1.0, 6)) == 0
    # A boolean is never mistaken for 0 or 1.
    assert _chosen_index([1, 0], [0.0, 0.0], incumbent=False) == 0
    # A relative margin for a large score: 0.1% of 5000 is 5.
    assert _chosen_index([10, 20], [5000.0, 4996.0], incumbent=10) == 0
    assert _chosen_index([10, 20], [5000.0, 4990.0], incumbent=10) == 1


def test_trials_record_their_sweep_the_incumbent_the_choice_and_its_measurements():
    search = _bare_search()
    search.current["test_setting"] = 1.0

    def cost(value):
        search._note(width_um=value * 10.0)
        return -value

    search._sweep("g", "test_setting", [1.0, 2.0], cost)
    search._sweep("g", "test_setting", [2.0, 3.0], cost)

    one, two, two_again, three = [t for t in search.trials if t.setting == "test_setting"]
    assert (one.sweep, two.sweep, two_again.sweep, three.sweep) == (0, 0, 1, 1)
    assert one.incumbent and not one.chosen
    assert two.chosen and not two.incumbent
    assert two_again.incumbent and not two_again.chosen
    assert three.chosen
    assert two.metrics == {"width_um": 20.0}
    assert all(t.seconds >= 0.0 for t in (one, two, two_again, three))


def test_a_failed_candidate_records_no_measurements_from_the_one_before():
    search = _bare_search()

    def cost(value):
        if value == 2.0:
            raise RuntimeError("boom")
        search._note(width_um=value)
        return -value

    search._sweep("g", "test_setting", [1.0, 2.0], cost)

    failed = next(t for t in search.trials if t.value == 2.0)
    assert failed.note == "failed: boom"
    assert failed.metrics == {}


# ---------------------------------------------------------------------------
# Trials a search has just run are reused, not run again
# ---------------------------------------------------------------------------
def test_min_branch_length_runs_each_candidates_skeleton_once(monkeypatch):
    """Regression: the sweep's baseline was the current value's own trial,
    run a second time, and the skeleton it left behind was the winner's,
    run a third -- two extra full preprocessing calls on every sweep."""
    from haemolynx import preprocessing as preprocessing_module
    from haemolynx.optimisation.search import _skeleton_kwargs

    search = _bare_search()
    search.raw_skeleton = preprocessing_module.skeletonize_volume(search.raw_mask)
    real = preprocessing_module.preprocess_skeleton_for_graph
    calls = []

    def counting(*args, **kwargs):
        calls.append(kwargs["min_branch_length"])
        return real(*args, **kwargs)

    monkeypatch.setattr(preprocessing_module, "preprocess_skeleton_for_graph", counting)
    search._group_min_branch_length()

    tried = [t for t in search.trials if t.setting == "skeleton_min_branch_length"]
    assert sorted(calls) == sorted(t.value for t in tried)
    assert any(t.incumbent for t in tried)  # the baseline was one of them
    # What the group left behind is the winner's skeleton, as a fresh run makes it.
    np.testing.assert_array_equal(
        search.current_skeleton,
        real(
            search.raw_skeleton, segmentation_mask=search.raw_mask,
            voxel_size_zyx=search.voxel_size_zyx, **_skeleton_kwargs(search.current),
        ),
    )


def test_reconnect_thresholds_build_each_candidates_graph_once(monkeypatch):
    from haemolynx import graph as graph_module
    from haemolynx import preprocessing as preprocessing_module

    search = _bare_search()
    search.current_skeleton = preprocessing_module.skeletonize_volume(search.raw_mask)
    real = graph_module.build_graph_from_skeleton
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(graph_module, "build_graph_from_skeleton", counting)
    search._group_reconnect_thresholds()

    candidates = [t for t in search.trials if t.group == "reconnect_thresholds"]
    # One zero-threshold probe for the gap distances, then at most one build
    # per candidate: every baseline, and the graph the group leaves behind,
    # is a candidate's graph already built. Before, each sweep added two.
    assert calls["n"] <= 1 + len(candidates)
    assert search.current_graph.number_of_nodes() > 0


# ---------------------------------------------------------------------------
# Timing, for the report and for checking Auto's estimate against it
# ---------------------------------------------------------------------------
def test_the_result_reports_how_long_the_search_and_each_group_took(y_shaped_mask):
    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
        downsample_factor=1,
    )
    assert result.seconds > 0.0
    assert result.estimated_seconds is None  # a factor was given: nothing was estimated
    assert set(result.group_seconds) == set(result.groups_run)
    assert sum(result.group_seconds.values()) <= result.seconds + 1e-6
    assert all(trial.seconds >= 0.0 for trial in result.trials)


def test_the_estimate_auto_reports_is_the_one_it_chose_its_factor_by():
    from haemolynx.optimisation.search import (
        DOWNSAMPLE_FACTORS,
        _factor_from_time_model,
        _SearchTimeModel,
        _voxel_reduction,
    )

    # Graph builds timed at 1 s on a million-voxel grid; a 75M-voxel stack.
    model, total, target = _SearchTimeModel({"graph": 1.0}, probe_voxels=1e6), 75e6, 300.0
    factor = _factor_from_time_model(model, total, target)
    finer = DOWNSAMPLE_FACTORS[DOWNSAMPLE_FACTORS.index(factor) - 1]

    assert factor == 2
    assert model.search_seconds(total / _voxel_reduction(factor, None)) <= target
    assert model.search_seconds(total / _voxel_reduction(finer, None)) > target


def test_auto_reports_no_estimate_when_it_falls_back_to_the_voxel_count():
    from haemolynx.optimisation.search import _auto_downsample

    auto = _auto_downsample(
        np.zeros((8, 8, 8), dtype=bool), (1.0, 1.0, 1.0),
        target_seconds=300.0, starting_values=dict(_DEFAULT_STARTING_VALUES),
    )
    assert auto.factor == 1
    assert auto.estimated_seconds is None
    assert auto.typical_radius_um is None and auto.resolution_cap is None


# ---------------------------------------------------------------------------
# Voxel-counted settings in and out of a downsampled search
# ---------------------------------------------------------------------------
_E14_STARTING_VOXEL_SETTINGS = {
    # The E14.5 run's own starting values, which a 16x search returned 16x over.
    "skeleton_closing_radius": 2,
    "skeleton_bridge_gap_size": 3,
    "skeleton_min_branch_length": 3,
    "skeleton_max_bridge_distance": 10,
    "skeleton_bundle_scan_size": 9,
    "skeleton_bundle_hub_min_spacing": 4,
}


def test_voxel_settings_go_into_the_search_in_its_own_grids_voxels():
    from haemolynx.optimisation.search import _to_search_grid

    on_grid = _to_search_grid(dict(_DEFAULT_STARTING_VALUES, **_E14_STARTING_VOXEL_SETTINGS), 16)

    assert {name: on_grid[name] for name in _E14_STARTING_VOXEL_SETTINGS} == {
        "skeleton_closing_radius": 0,
        "skeleton_bridge_gap_size": 0,
        "skeleton_min_branch_length": 0,
        "skeleton_max_bridge_distance": 1,
        "skeleton_bundle_scan_size": 1,
        "skeleton_bundle_hub_min_spacing": 0,
    }
    assert on_grid["graph_reconnect_threshold"] == _DEFAULT_STARTING_VALUES["graph_reconnect_threshold"]
    assert _to_search_grid(_E14_STARTING_VOXEL_SETTINGS, 1) == _E14_STARTING_VOXEL_SETTINGS


def test_a_kept_voxel_setting_comes_back_as_the_user_had_it():
    """Regression: the E14.5 run's 16x search kept every one of these and
    returned it multiplied by 16 -- a closing radius of 32 voxels (~31 um),
    a maximum bridge distance of 160 (~157 um)."""
    from haemolynx.optimisation.search import _to_full_resolution, _to_search_grid

    on_grid = _to_search_grid(_E14_STARTING_VOXEL_SETTINGS, 16)
    decided = dict(on_grid, skeleton_max_bridge_distance=2)  # the one the search moved

    settings, finer = _to_full_resolution(
        decided, _E14_STARTING_VOXEL_SETTINGS, on_grid, 16, bundle_refinement_ran=True,
    )

    assert settings == {
        "skeleton_closing_radius": 2,
        "skeleton_bridge_gap_size": 3,
        "skeleton_min_branch_length": 3,
        "skeleton_max_bridge_distance": 32,
        "skeleton_bundle_scan_size": 9,
        "skeleton_bundle_hub_min_spacing": 4,  # half the scan window, as a full search derives it
    }
    assert finer == (
        "skeleton_closing_radius", "skeleton_bridge_gap_size", "skeleton_min_branch_length",
    )


def test_a_downsampled_search_that_changes_nothing_returns_the_starting_values(monkeypatch):
    import haemolynx.optimisation.search as search_module

    class _KeepingSearch:
        def __init__(self, mask, voxel_size_xyz, starting_values, progress, enabled_groups=None, raw_image=None):
            self.current = dict(starting_values)
            self.trials = []
            self.groups_run = list(search_module.GROUP_NAMES)
            self.passes_run = 1
            self.group_seconds = {}

        def run(self, *, max_passes=1):
            pass

    monkeypatch.setattr(search_module, "_Search", _KeepingSearch)
    starting_values = dict(_DEFAULT_STARTING_VALUES, **_E14_STARTING_VOXEL_SETTINGS)
    mask = np.zeros((64, 64, 64), dtype=bool)
    mask[20:40, 20:40, 20:40] = True

    result = optimise_skeleton_and_graph_settings(
        mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=starting_values, downsample_factor=16,
    )

    for name, value in _E14_STARTING_VOXEL_SETTINGS.items():
        assert result.settings[name] == value, name
    assert set(result.finer_than_search_grid) == {
        "skeleton_closing_radius", "skeleton_bridge_gap_size", "skeleton_min_branch_length",
    }


# ---------------------------------------------------------------------------
# Auto: a fixed overhead per evaluation, and a grid that still sees the vessels
# ---------------------------------------------------------------------------
def test_a_search_time_is_each_step_times_how_often_a_search_runs_it():
    from haemolynx.optimisation.search import _SEARCH_STEP_CALLS, _SearchTimeModel

    model = _SearchTimeModel({"graph": 1.0, "smooth": 0.5}, probe_voxels=1000.0)

    one_search = _SEARCH_STEP_CALLS["graph"] * 1.0 + _SEARCH_STEP_CALLS["smooth"] * 0.5
    assert model.search_seconds(1000.0) == pytest.approx(one_search)
    assert model.search_seconds(8000.0) == pytest.approx(8 * one_search)  # scaled with voxels
    assert _SearchTimeModel({"not a step": 9.0}, 1.0).search_seconds(1.0) == 0.0


def test_auto_times_the_search_on_the_grid_it_will_search(monkeypatch):
    """Regression: Auto timed one whole pipeline cycle at 16x and scaled it
    with voxel count, which read a 4x search on the E14.5 stack as 637 s;
    it took 57 s. Coarse grids do not predict fine ones -- vessels merge into
    blobs there -- so the steps are timed on the grid the resolution cap
    allows."""
    import haemolynx.optimisation.search as search_module

    probed = []

    def recording_probe(raw_mask, voxel_size_zyx, *, factor, starting_values=None):
        probed.append(factor)
        return search_module._SearchTimeModel({"graph": 0.001}, probe_voxels=raw_mask.size / 8)

    monkeypatch.setattr(search_module, "_probe_time_model", recording_probe)
    auto = search_module._auto_downsample(
        _y_shaped_vessel(shape=(64, 64, 64), radius=4), (1.0, 1.0, 1.0),
        target_seconds=300.0, starting_values=dict(_DEFAULT_STARTING_VALUES),
    )

    assert probed == [auto.resolution_cap] == [2]


@pytest.mark.parametrize(
    "typical_radius_um, voxel_size_zyx, expected",
    [
        (3.67, (1.0, 0.98, 0.98), 2),  # the E14.5 stack: 1.9 voxels at 2x, 0.94 at 4x
        (20.0, (1.0, 0.98, 0.98), 8),
        (1.0, (1.0, 0.98, 0.98), 1),  # unresolved even at full resolution
        (3.0, (2.0, 0.5, 0.5), 4),  # judged in-plane on an anisotropic stack
    ],
)
def test_auto_never_searches_a_grid_too_coarse_for_the_typical_vessel(
    typical_radius_um, voxel_size_zyx, expected
):
    from haemolynx.optimisation.search import _coarsest_factor_resolving

    assert _coarsest_factor_resolving(typical_radius_um, voxel_size_zyx) == expected


def test_auto_caps_a_time_budget_that_would_go_coarser_than_the_vessels_allow(monkeypatch):
    """Regression: with nothing but a time budget, Auto searched the E14.5
    stack on an 18 x 32 x 32 grid where a capillary is one voxel."""
    import haemolynx.optimisation.search as search_module

    # Evaluations so slow that only the coarsest grid fits any budget.
    mask = _y_shaped_vessel(shape=(64, 64, 64), radius=4)  # ridge radius 5 voxels
    monkeypatch.setattr(
        search_module, "_probe_time_model",
        lambda *_a, **_k: search_module._SearchTimeModel({"graph": 1e6}, probe_voxels=mask.size / 8),
    )
    auto = search_module._auto_downsample(
        mask, (1.0, 1.0, 1.0), target_seconds=300.0, starting_values=dict(_DEFAULT_STARTING_VALUES),
    )

    assert 4.0 <= auto.typical_radius_um <= 5.5
    assert auto.resolution_cap == 2
    assert auto.factor == 2
    assert auto.estimated_seconds == pytest.approx(1e6 * search_module._SEARCH_STEP_CALLS["graph"])


def _tube_lattice(n: int = 160, spacing: int = 40, radius: int = 3) -> np.ndarray:
    """Tubes crossing the whole volume, so even a 16x grid keeps a skeleton."""
    skeleton = np.zeros((n, n, n), dtype=bool)
    for a in range(spacing // 2, n, spacing):
        for b in range(spacing // 2, n, spacing):
            skeleton[a, b, :] = True
            skeleton[a, :, b] = True
    return binary_dilation(skeleton, structure=np.ones((2 * radius + 1,) * 3, dtype=bool))


def test_auto_times_every_costly_step_for_real():
    from haemolynx.optimisation.search import (
        _SEARCH_STEP_CALLS,
        _auto_downsample,
        _probe_time_model,
    )

    mask = _tube_lattice()
    starting = dict(_DEFAULT_STARTING_VALUES)
    model = _probe_time_model(mask, (1.0, 1.0, 1.0), factor=4, starting_values=starting)
    auto = _auto_downsample(mask, (1.0, 1.0, 1.0), target_seconds=300.0, starting_values=starting)

    assert model is not None
    assert set(model.step_seconds) == set(_SEARCH_STEP_CALLS)
    assert all(seconds >= 0.0 for seconds in model.step_seconds.values())
    assert model.probe_voxels == pytest.approx((160 / 4) ** 3)
    assert auto.estimated_seconds is not None and auto.estimated_seconds > 0.0
    assert auto.factor <= auto.resolution_cap


def test_the_typical_radius_is_measured_on_the_densest_crop():
    from haemolynx.optimisation.search import _densest_crop

    mask = np.zeros((90, 90, 90), dtype=bool)
    mask[70:80, 60:85, 5:12] = True
    small = np.zeros((10, 10, 10), dtype=bool)

    crop = _densest_crop(mask, 30**3)

    assert crop.shape == (30, 30, 30)
    assert np.count_nonzero(crop) == np.count_nonzero(mask)
    assert _densest_crop(small, 30**3) is small


def test_the_densest_crop_is_found_anywhere_in_a_large_volume():
    """Crops at the start, middle and end of each axis alone left gaps on a
    large volume, where the vessels could sit and never be measured."""
    from haemolynx.optimisation.search import _densest_crop

    mask = np.zeros((150, 150, 150), dtype=bool)
    mask[25:30, 105:110, 30:40] = True  # between the start, middle and end crops

    crop = _densest_crop(mask, 25**3)

    assert crop.shape == (25, 25, 25)
    assert np.count_nonzero(crop) > 0


# ---------------------------------------------------------------------------
# Cleanup without a raw image: the input's own vessels guard it
# ---------------------------------------------------------------------------
def _thin_vessels_beside_a_thick_one() -> np.ndarray:
    """Eight capillaries three voxels across and one vessel nine across, each
    well over the 20 um3 floor at 1 um voxels."""
    skeleton = np.zeros((40, 48, 48), dtype=bool)
    for i in range(8):
        skeleton[5:35, 4 + 5 * i, 6] = True
    mask = binary_dilation(skeleton, structure=np.ones((3, 3, 3), dtype=bool))
    thick = np.zeros_like(skeleton)
    thick[5:35, 24, 30] = True
    return mask | binary_dilation(thick, structure=np.ones((9, 9, 9), dtype=bool))


def test_real_vessels_kept_counts_a_vessel_while_half_of_it_is_left():
    from haemolynx.optimisation.search import _Search

    mask = _thin_vessels_beside_a_thick_one()
    search = _Search(mask, (1.0, 1.0, 1.0), dict(_DEFAULT_STARTING_VALUES), progress=None)
    without_two = mask.copy()
    without_two[:, 2:12, :20] = False  # the first two capillaries
    thinned = mask.copy()
    thinned[:20, :, :20] = False  # a third of every capillary: still kept

    assert search._real_vessels_kept_fraction(mask) == 1.0
    assert search._real_vessels_kept_fraction(without_two) == pytest.approx(7 / 9)
    assert search._real_vessels_kept_fraction(thinned) == 1.0


def test_cleanup_without_a_raw_image_cannot_erase_the_capillaries(monkeypatch):
    """Regression: with no raw image, nothing but the mask's own score judged
    a cleanup candidate, and a radius erasing thin vessels can score better
    -- confirmed on real data, where a too-large smoothing erased ~88% of a
    capillary network. Here the score prefers whatever removes most, so only
    the guard stands between the search and erasing all eight capillaries."""
    from types import SimpleNamespace

    from haemolynx import preprocessing as preprocessing_module
    from haemolynx.optimisation.search import _Search

    # On the real score's 0-10 scale: 10 for an empty mask, 0 for a full one.
    monkeypatch.setattr(
        preprocessing_module, "score_segmented_mask",
        lambda mask, **_kwargs: SimpleNamespace(total=10.0 * (1.0 - float(np.mean(mask)))),
    )
    mask = _thin_vessels_beside_a_thick_one()
    search = _Search(mask, (1.0, 1.0, 1.0), dict(_DEFAULT_STARTING_VALUES), progress=None)

    search._group_segmentation_cleanup()

    from scipy.ndimage import label

    labels, n_vessels = label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    left = np.bincount(labels[search.raw_mask], minlength=sizes.size)
    assert n_vessels == 9
    assert all(left[i] >= 0.5 * sizes[i] for i in range(1, n_vessels + 1)), left
    guarded = [t for t in search.trials if t.metrics.get("guard_penalty", 0.0) > 0.0]
    assert guarded, "no candidate was ever rejected: the test did not exercise the guard"


# ---------------------------------------------------------------------------
# The scorecard: the starting settings against the chosen ones
# ---------------------------------------------------------------------------
def test_the_result_scores_the_starting_and_the_chosen_settings_alike(y_shaped_mask):
    from haemolynx.optimisation.search import _SCORECARD_MEASURES

    result = optimise_skeleton_and_graph_settings(
        y_shaped_mask, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
        downsample_factor=1,
    )

    assert [row.name for row in result.scorecard] == [name for name, _h, _t in _SCORECARD_MEASURES]
    by_name = {row.name: row for row in result.scorecard}
    assert by_name["input vessels the skeleton reaches"].before == pytest.approx(1.0)
    assert by_name["graph components"].after >= 1.0


def test_a_search_ending_worse_than_it_started_says_so(monkeypatch):
    """Regression: every sweep guards against its own starting point, so
    small losses could add up -- or a destructive choice slip through -- and
    nothing compared the run's result with what it was given."""
    import haemolynx.optimisation.search as search_module
    from haemolynx.optimisation.report import build_report_text
    from haemolynx.optimisation.scorecard import worse_rows

    class _DestructiveSearch:
        """Ends with a whisker radius that erases every capillary."""

        def __init__(self, mask, voxel_size_xyz, starting_values, progress, enabled_groups=None, raw_image=None):
            self.current = dict(starting_values)
            self.trials = []
            self.groups_run = list(search_module.GROUP_NAMES)
            self.passes_run = 1
            self.group_seconds = {}

        def run(self, *, max_passes=1):
            self.current.update(
                segmentation_cleanup=True,
                segmentation_cleanup_remove_whiskers=True,
                segmentation_cleanup_whisker_radius_um=2.0,
            )

    monkeypatch.setattr(search_module, "_Search", _DestructiveSearch)
    result = optimise_skeleton_and_graph_settings(
        _thin_vessels_beside_a_thick_one(), voxel_size_xyz=(1.0, 1.0, 1.0),
        starting_values=_DEFAULT_STARTING_VALUES, downsample_factor=1,
    )

    worse = {row.name for row in worse_rows(result.scorecard)}
    assert "input vessels the skeleton reaches" in worse
    assert "WARNING: worse than your starting settings on" in build_report_text(result)


def test_a_scorecard_that_cannot_be_measured_does_not_fail_the_search(monkeypatch):
    import haemolynx.optimisation.search as search_module

    def boom(*_args, **_kwargs):
        raise RuntimeError("no graph")

    monkeypatch.setattr(search_module, "_scorecard_measures", boom)
    result = optimise_skeleton_and_graph_settings(
        _y_shaped_vessel(), voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
        downsample_factor=1, groups=["min_stub_length"],
    )
    assert result.scorecard == ()
    assert result.settings
