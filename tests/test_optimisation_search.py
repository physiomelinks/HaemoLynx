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


def test_factor_from_probe_seconds_picks_the_coarsest_factor_for_a_tiny_budget():
    """A budget no real probe could ever fit under must fall back to the
    coarsest offered factor, not raise or return something outside
    DOWNSAMPLE_FACTORS. Pure arithmetic (see this function's own docstring
    for why it is split out from estimate_downsample_factor_for_time_budget) --
    no real probe or timing involved, so a genuinely unmeetable budget can
    be tested directly rather than relying on real wall-clock timing.
    """
    from haemolynx.optimisation.search import _factor_from_probe_seconds

    factor = _factor_from_probe_seconds(
        probe_seconds=0.01, probe_factor=DOWNSAMPLE_FACTORS[-1], target_seconds=1e-9,
    )
    assert factor == DOWNSAMPLE_FACTORS[-1]


def test_factor_from_probe_seconds_picks_full_resolution_for_a_generous_budget():
    from haemolynx.optimisation.search import _factor_from_probe_seconds

    factor = _factor_from_probe_seconds(
        probe_seconds=0.01, probe_factor=DOWNSAMPLE_FACTORS[-1], target_seconds=1e9,
    )
    assert factor == DOWNSAMPLE_FACTORS[0]


def test_factor_from_probe_seconds_is_monotonic_in_target_seconds():
    """A smaller time budget must never pick a *finer* (smaller) factor than
    a larger budget, for the same measured probe."""
    from haemolynx.optimisation.search import _factor_from_probe_seconds

    generous = _factor_from_probe_seconds(
        probe_seconds=0.01, probe_factor=DOWNSAMPLE_FACTORS[-1], target_seconds=1e9,
    )
    stingy = _factor_from_probe_seconds(
        probe_seconds=0.01, probe_factor=DOWNSAMPLE_FACTORS[-1], target_seconds=1e-9,
    )
    assert stingy >= generous


def test_factor_from_probe_seconds_zero_probe_means_full_resolution():
    """A probe that measured as zero (or negative, from clock jitter) gives
    no evidence downsampling would help; default to full detail rather
    than dividing by/scaling a meaningless zero."""
    from haemolynx.optimisation.search import _factor_from_probe_seconds

    assert _factor_from_probe_seconds(0.0, DOWNSAMPLE_FACTORS[-1], 1e-9) == DOWNSAMPLE_FACTORS[0]


def test_optimise_settings_auto_downsample_uses_the_time_budget_resolver(monkeypatch):
    """Regression: 'Auto' (downsample_factor=None) used to resolve purely
    from the mask's own voxel count (resolve_auto_downsample_factor),
    which assumes a fixed voxels-per-second rate true of neither a
    specific machine nor a specific dataset's own topological complexity.
    It must now consult estimate_downsample_factor_for_time_budget
    instead, forwarding the caller's own auto_downsample_target_seconds
    and use_thick_vessel_skeletonisation starting value.
    """
    import haemolynx.optimisation.search as search_module

    recorded = {}

    def fake_estimate(raw_mask, voxel_size_zyx, *, target_seconds, use_thick_vessel_skeletonisation):
        recorded["target_seconds"] = target_seconds
        recorded["use_thick_vessel_skeletonisation"] = use_thick_vessel_skeletonisation
        return 4

    monkeypatch.setattr(
        search_module, "estimate_downsample_factor_for_time_budget", fake_estimate
    )
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

        def run(self, *, max_passes=1):
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
