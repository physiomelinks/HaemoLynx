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
    OPTIMISE_SETTING_NAMES,
    optimise_skeleton_and_graph_settings,
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


#: Schema defaults for the 21 settings this search decides, matching
#: pipeline/schema.py -- kept local so this test does not depend on
#: haemolynx.pipeline (the optimisation package's own purity boundary).
_DEFAULT_STARTING_VALUES = {
    "use_thick_vessel_skeletonisation": False,
    "skeleton_thick_vessel_min_radius_um": 6.0,
    "skeleton_fill_mask_holes_before_thickness": True,
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
    gracefully rather than divide by zero or index past an empty array."""
    blob = np.zeros((20, 20, 20), dtype=bool)
    blob[5:15, 5:15, 5:15] = True
    result = optimise_skeleton_and_graph_settings(
        blob, voxel_size_xyz=(1.0, 1.0, 1.0), starting_values=_DEFAULT_STARTING_VALUES,
    )
    assert set(result.settings) == set(OPTIMISE_SETTING_NAMES)


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
