"""Physical-unit vs voxel-index handling across the graph and benchmark layers.

Node ``pos`` and edge ``voxels`` are stored in physical units by
``build_graph_segment_skan_stitched_loops``, while ``image_shape``, array subscripts and
several thresholds are in voxels. At a voxel size of (1, 1, 1) the two are numerically
identical, so no fixture built at unit spacing can tell them apart - which is exactly why
this class of bug survived. Every test here therefore runs at a genuinely anisotropic
spacing, where a physical coordinate used as a subscript lands somewhere visibly wrong.
"""
import networkx as nx
import numpy as np
import pytest

from ImageLynx.graph.boundaries import select_boundary_terminal_nodes
from ImageLynx.specimens import PROCESSING_VOXEL_UM
from ImageLynx.statistics.benchmarking import (
    evaluate_centerline_centricity,
    evaluate_completeness_and_overpruning,
    redilate_skeleton_to_volume,
)

# The measured acquisition voxel size for the WKY carotid body volume, (z, y, x) in microns.
# The anisotropy is on z, not x: the raw TIFF's ImageJ metadata gives a slice spacing of
# 1.86386 um and an in-plane pixel size of 1.86600 um in both y and x. This was previously
# written (1.8660, 1.8660, 1.8639), which put the odd axis on x under a (z, y, x) reading.
SPACING = PROCESSING_VOXEL_UM
SHAPE = (20, 20, 20)

# A straight centreline along z at voxel (y, x) = (10, 10), spanning voxels z = 5..15.
CENTRELINE_VOXELS = [(z, 10, 10) for z in range(5, 16)]


def _physical(voxel_coords, spacing=SPACING):
    """Convert (z, y, x) voxel indices to the physical coordinates the graph actually stores."""
    return [
        [z * spacing[0], y * spacing[1], x * spacing[2]] for z, y, x in voxel_coords
    ]


def _centreline_graph(diameter_um=2.0, spacing=SPACING):
    G = nx.MultiGraph()
    physical = _physical(CENTRELINE_VOXELS, spacing)
    G.add_node(0, pos=np.array(physical[0]))
    G.add_node(1, pos=np.array(physical[-1]))
    G.add_edge(0, 1, voxels=physical, assigned_diameter_um=diameter_um)
    return G


@pytest.mark.parametrize("spacing", [SPACING, (1.0, 1.0, 1.0)], ids=["anisotropic", "unit"])
def test_redilate_skeleton_to_volume_divides_physical_coordinates_by_the_spacing(spacing):
    """Physical edge coordinates must be divided by the spacing before being used as indices.

    Truncating them directly places the reconstruction at voxel (z * 1.866, 18, 18) instead
    of (z, 10, 10) - an offset that silently corrupts the Dice coefficient the skeletonisation
    tuner optimises against. At unit spacing the division is a no-op, which is what every
    existing caller uses, so the same centreline must land in the same place.
    """
    reconstructed = redilate_skeleton_to_volume(_centreline_graph(spacing=spacing), SHAPE, spacing)

    assert reconstructed[10, 10, 10], \
        "the true centreline voxel is not covered by the reconstruction"
    assert not reconstructed[10, 18, 18], \
        "reconstruction landed at int(physical), i.e. the coordinates were used as indices"

    # The reconstruction should sit on the centreline in y/x, not be smeared to the corner.
    occupied = np.argwhere(reconstructed)
    assert occupied.size > 0
    assert abs(occupied[:, 1].mean() - 10) < 1.5
    assert abs(occupied[:, 2].mean() - 10) < 1.5


def test_evaluate_completeness_and_overpruning_round_trips_anisotropic_spacing():
    """Orphaned volume is measured as a distance from the drawn skeleton to mask tissue.

    If the physical coordinates are used as subscripts the skeleton is drawn in the wrong
    place, every tissue voxel looks far from a centreline, and a healthy graph is reported
    as massively over-pruned.
    """
    mask = np.zeros(SHAPE, dtype=bool)
    for z, y, x in CENTRELINE_VOXELS:
        mask[z, y - 1:y + 2, x - 1:x + 2] = True

    result = evaluate_completeness_and_overpruning(
        _centreline_graph(), mask, SPACING, max_capillary_radius_um=5.0
    )

    # Every tissue voxel is within ~1 voxel (1.87 um) of the true centreline, so nothing is
    # orphaned. Mis-indexing puts the skeleton ~8 voxels (15 um) away and orphans everything.
    assert result["orphaned_volume_fraction"] < 0.05, (
        "tissue appears orphaned because the skeleton was drawn at the wrong coordinates: "
        f"{result['orphaned_volume_fraction']:.3f}"
    )


def test_evaluate_centerline_centricity_divides_physical_coordinates_by_the_spacing():
    """Centricity reads the mask's EDT at each centreline point, so the point must be a voxel.

    Truncating the physical coordinate reads the EDT at (9, 18, 18), (11, 18, 18), ... outside
    the tube, where the distance to the wall is 0: a mean error of 4.0 um, the whole radius.
    """
    G = _centreline_graph(diameter_um=8.0)
    mask = redilate_skeleton_to_volume(G, SHAPE, SPACING)

    result = evaluate_centerline_centricity(G, mask, SPACING)

    # On the axis the EDT reads a mean error of 0.17 um against the 4 um radius.
    assert result["mean_centricity_error_um"] < 1.0, (
        "centricity read the EDT away from the centreline: "
        f"{result['mean_centricity_error_um']:.2f} um"
    )


def test_select_boundary_terminal_nodes_compares_like_with_like():
    """Node positions are physical; image_shape is in voxels. The extent must be converted.

    Comparing a physical coordinate against a voxel extent shrinks the apparent volume by the
    voxel size, so interior nodes drift into the outlet band and are wired up as venous
    boundaries.
    """
    G = nx.Graph()
    # A hub at voxel z = 10 with three degree-1 terminals at voxels z = 0, 8 and 19.
    positions = {"hub": 10, "inlet": 0, "interior": 8, "outlet": 19}
    for name, z_voxel in positions.items():
        G.add_node(name, pos=np.array([z_voxel * SPACING[0], 10 * SPACING[1], 10 * SPACING[2]]))
    for terminal in ("inlet", "interior", "outlet"):
        G.add_edge("hub", terminal)

    starting, outputs = select_boundary_terminal_nodes(
        G, SHAPE, edge_percent=25.0, end_percent=25.0, axis=0, voxel_size=SPACING,
    )

    assert "inlet" in starting
    assert "outlet" in outputs
    # z = 8 of 19 is 42% of the way along, comfortably outside the bottom 25%. Against a
    # voxel-denominated extent it lands at 79% and is misclassified as an outlet.
    assert "interior" not in outputs, \
        "an interior node was selected as an outlet; the extent was not converted to physical units"
    assert "interior" not in starting


def test_resolve_voxel_size_prefers_explicit_configuration():
    """An explicitly configured voxel size must win over file metadata.

    The probability TIFFs declare no resolution, so metadata detection yields (1, 1, 1) and
    silently reports voxel counts as microns. Configuration has to override it.
    """
    import carotid_image_to_model as C

    original = C.VOXEL_SIZE_UM
    try:
        C.VOXEL_SIZE_UM = SPACING
        # Deliberately a format with no metadata path at all: configuration must still win.
        assert C._resolve_voxel_size("no/such/file.npy", "npy") == SPACING

        C.VOXEL_SIZE_UM = None
        assert C._resolve_voxel_size("no/such/file.npy", "npy") == (1.0, 1.0, 1.0)
    finally:
        C.VOXEL_SIZE_UM = original


# --- The configured acquisition voxel size ------------------------------------------------
#
# 2705b38 added the mechanism for an explicit voxel size, but PipelineConfig.voxel_size_um was
# left at None and neither group YAML set it. The pipeline consumes the Ilastik probability
# TIFF, which carries no resolution tag, so _resolve_voxel_size fell back to (1, 1, 1) and
# every "micron" in the output was really a voxel count - the exact failure 2705b38 was meant
# to close. These tests pin the configured value to the acquisition file it came from.

def _pipeline_module():
    return pytest.importorskip("carotid_image_to_model")


@pytest.mark.realdata
def test_configured_voxel_size_tracks_the_raw_acquisition_metadata():
    """The constant must be traceable to the file, not typed in beside it.

    It is no longer required to equal the acquisition metadata exactly. Every volume was
    preprocessed with (1.8639, 1.866, 1.866) and the supplied distance transform is
    calibrated in those units, so that is what the pipeline has to compute in - otherwise
    radii and skeleton lengths end up on different scales. The rounding is 0.0024% and the
    between-group z difference it also absorbs is 0.014%; both are far below anything that
    changes a result, and both are recorded per specimen in ImageLynx.specimens.

    What still has to hold is that the configured value is the same physical quantity: a
    drift to voxels, to the 1x1x1 acquisition, or to a different specimen's calibration all
    remain failures.
    """
    mod = _pipeline_module()
    if not mod.RAW_IMAGE_PATH.exists():
        pytest.skip(f"raw acquisition volume not present at {mod.RAW_IMAGE_PATH}")

    from ImageLynx.io import get_tif_spacing

    detected = get_tif_spacing(mod.RAW_IMAGE_PATH)
    # That the config equals PROCESSING_VOXEL_UM is checked without the data, by
    # test_specimens.py::test_pipeline_voxel_size_is_the_shared_processing_value.
    configured = tuple(mod.PipelineConfig().voxel_size_um)

    assert configured == pytest.approx(detected, rel=1e-4), (
        f"configured voxel_size_um {configured} has drifted from the acquisition "
        f"metadata {detected} by more than the recorded rounding"
    )


# --- Terminal-reconnection thresholds are microns, not voxels (item 17/(e)) ----------------
#
# The converse of the usual calibration bug: 2705b38 did not change these literals, it changed
# what they mean. At the old (1, 1, 1) spacing a threshold of 3.0 was both 3 voxels and 3
# "microns"; once the voxel size was fixed it silently became 3 um where it had behaved as
# 3 * 1.866 = 5.6 um, and the conservative cap of 1.5 became 1.5 um where it had behaved as
# 2.8 um. Nothing in the diff of 2705b38 pointed at either.

def test_reconnect_thresholds_are_declared_in_microns():
    from ImageLynx.graph._helpers import (
        CONSERVATIVE_RECONNECT_CAP_UM,
        RECONNECT_THRESHOLD_UM,
    )

    # 3 voxels and 1.5 voxels at the measured 1.866 um in-plane pixel size.
    assert RECONNECT_THRESHOLD_UM == pytest.approx(3 * PROCESSING_VOXEL_UM[1], abs=0.05)
    assert CONSERVATIVE_RECONNECT_CAP_UM == pytest.approx(1.5 * PROCESSING_VOXEL_UM[1], abs=0.05)


def test_both_reconnection_entry_points_default_to_the_same_physical_distance():
    """They reconnect the same terminals; a divergence here is silent and asymmetric."""
    import inspect

    from ImageLynx.graph._helpers import RECONNECT_THRESHOLD_UM
    from ImageLynx.graph.build import build_graph_segment_skan_stitched_loops
    from ImageLynx.graph.optimise import optimise_graph_topology_fixed

    build_default = inspect.signature(
        build_graph_segment_skan_stitched_loops).parameters["reconnect_threshold"].default
    optimise_default = inspect.signature(
        optimise_graph_topology_fixed).parameters["reconnect_threshold"].default

    assert build_default == optimise_default == RECONNECT_THRESHOLD_UM


def test_reconnection_threshold_is_compared_against_physical_positions():
    """A 2 um gap must be bridged and a 4 um gap must not, at anisotropic spacing.

    Without skeleton_data the reconnection takes the conservative branch, capped at
    min(reconnect_threshold * 0.5, CONSERVATIVE_RECONNECT_CAP_UM) = 2.8 um, so these are the
    distances that actually decide anything here.

    Under the old voxel reading the cap was 1.5 and neither gap would be bridged. That is the
    only way to tell the two readings apart: a pure rescaling is invisible unless the test
    fixes an absolute physical distance, which is exactly why this class of bug survived.
    """
    from ImageLynx.graph._helpers import RECONNECT_THRESHOLD_UM
    from ImageLynx.graph.optimise import optimise_graph_topology_fixed

    G = nx.MultiGraph()
    G.graph["voxel_size"] = SPACING

    def _pair(prefix, z_start, gap_um, y):
        """Two degree-1 terminals separated along z by exactly gap_um."""
        a, b = f"{prefix}_a", f"{prefix}_b"
        anchor_a, anchor_b = f"{prefix}_anchor_a", f"{prefix}_anchor_b"
        z0 = z_start * SPACING[0]
        G.add_node(a, pos=np.array([z0, y * SPACING[1], 10 * SPACING[2]]))
        G.add_node(b, pos=np.array([z0 + gap_um, y * SPACING[1], 10 * SPACING[2]]))
        # Anchors keep a and b at degree 1 without being reconnection candidates themselves.
        G.add_node(anchor_a, pos=np.array([z0 - 40.0, y * SPACING[1], 10 * SPACING[2]]))
        G.add_node(anchor_b, pos=np.array([z0 + gap_um + 40.0, y * SPACING[1], 10 * SPACING[2]]))
        G.add_edge(a, anchor_a)
        G.add_edge(b, anchor_b)
        return a, b

    near_a, near_b = _pair("near", 5, gap_um=2.0, y=4)
    far_a, far_b = _pair("far", 5, gap_um=4.0, y=14)

    out, _ = optimise_graph_topology_fixed(
        G, voxel_loops=set(), loop_edges=set(), skeleton_data=None, debug=False,
        reconnect_threshold=RECONNECT_THRESHOLD_UM, use_spatial_index=False,
        remove_degree2_nodes=False, improve_junctions=False,
    )

    assert out.has_edge(near_a, near_b), \
        "a 2 um gap was not bridged by a 2.8 um cap; the threshold is not in microns"
    assert not out.has_edge(far_a, far_b), \
        "a 4 um gap was bridged by a 2.8 um cap"


# --- Mask calibre: the one classifier-independent diagnostic (Phase 1.5) -------------------
#
# Every other preprocessing benchmark scores the mask against the probability field, so all of
# them inherit the classifier's miscalibration. This one measures how thick the segmented
# structures physically are. With hand annotation infeasible (#98 item 23) it is the only
# evidence about the segmentation that does not come from the classifier being assessed.

def _tube(shape, radius_voxels, spacing):
    """A solid cylinder along z, centred, of the given radius in voxels."""
    zz, yy, xx = np.indices(shape)
    cy, cx = shape[1] // 2, shape[2] // 2
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius_voxels ** 2


def test_mask_calibre_reports_microns_not_voxels():
    """The whole point is comparing against a capillary calibre, which is a physical quantity."""
    from ImageLynx.statistics.benchmarking import evaluate_mask_calibre

    shape = (12, 41, 41)
    mask = _tube(shape, radius_voxels=6, spacing=SPACING)

    voxels = evaluate_mask_calibre(mask, (1.0, 1.0, 1.0))
    microns = evaluate_mask_calibre(mask, SPACING)

    # Same mask, spacing 1.866x larger, so every radius must scale by 1.866.
    assert microns["max_radius_um"] == pytest.approx(voxels["max_radius_um"] * SPACING[1], rel=0.02)
    assert microns["max_radius_um"] > voxels["max_radius_um"]


def test_mask_calibre_separates_a_capillary_from_a_flooded_blob():
    """The discrimination the diagnostic exists for, at the real voxel size.

    A 2-voxel-radius tube is a capillary at 1.866 um voxels; a mask covering most of the volume
    is a block of tissue. Measured on real data the same contrast appears as a median radius of
    2.64 um against 20.09 um.
    """
    from ImageLynx.statistics.benchmarking import evaluate_mask_calibre

    shape = (12, 61, 61)
    capillary = evaluate_mask_calibre(_tube(shape, 2, SPACING), SPACING)
    flooded = evaluate_mask_calibre(np.ones(shape, dtype=bool), SPACING)

    assert capillary["max_radius_um"] < 5.0, "a 2-voxel tube should read as a few microns"
    assert flooded["max_radius_um"] > 20.0, "a filled volume should read as tens of microns"
    assert flooded["median_radius_um"] > 5 * capillary["median_radius_um"]
    assert capillary["foreground_fraction"] < flooded["foreground_fraction"] == 1.0


def test_mask_calibre_handles_an_empty_mask():
    from ImageLynx.statistics.benchmarking import evaluate_mask_calibre

    result = evaluate_mask_calibre(np.zeros((5, 5, 5), dtype=bool), SPACING)
    assert result["foreground_fraction"] == 0.0
    assert result["max_radius_um"] == 0.0


def test_mask_calibre_is_reported_but_never_optimised_against():
    """It must not become a loss term: that would assert the capillary radius it exists to test."""
    from ImageLynx.statistics.auto_tuner import PreprocessingObjective
    from ImageLynx.statistics.benchmarking import run_all_preprocessing_benchmarks

    prob = np.zeros((10, 21, 21), dtype=np.float32)
    prob[:, 8:13, 8:13] = 0.9
    mask = prob > 0.5

    reported = run_all_preprocessing_benchmarks(prob, mask, None, voxel_size_xyz=SPACING)
    calibre_keys = {"foreground_fraction", "median_radius_um", "p90_radius_um",
                    "p99_radius_um", "max_radius_um"}
    assert calibre_keys <= set(reported), "the diagnostic is not being reported"

    objective = PreprocessingObjective(lambda kwargs: None)
    baseline = objective._calculate_loss(reported)
    for key in calibre_keys:
        perturbed = dict(reported)
        perturbed[key] = reported[key] * 1000.0 + 7.0
        assert objective._calculate_loss(perturbed) == baseline, \
            f"{key} moved the loss; the diagnostic has leaked into the objective"


# --- The default thresholds must be reachable by the tuner (Phase 1.5) --------------------

def test_default_hysteresis_thresholds_lie_inside_the_tuner_search_range():
    """A default outside the searched range cannot be reproduced or refined by tuning."""
    C = pytest.importorskip("carotid_image_to_model")
    optuna = pytest.importorskip("optuna")
    from ImageLynx.statistics.auto_tuner import PreprocessingObjective

    ranges = {}

    class _Recorder:
        def suggest_float(self, name, low, high, **k):
            ranges[name] = (low, high)
            return (low + high) / 2.0

        def suggest_int(self, name, low, high, **k):
            ranges[name] = (low, high)
            return low

        def suggest_categorical(self, name, choices):
            return choices[0]

    # The recorder's run returns no mask, so the objective prunes the trial after drawing
    # every parameter. Any other error is a real one and must not be swallowed.
    try:
        PreprocessingObjective(lambda kwargs: None)(_Recorder())
    except optuna.TrialPruned:
        pass

    cfg = C.PreprocessingConfig()
    lo_lo, lo_hi = ranges["hysteresis_threshold_low"]
    hi_lo, hi_hi = ranges["hysteresis_threshold_high"]
    assert lo_lo <= cfg.hysteresis_threshold_low <= lo_hi
    assert hi_lo <= cfg.hysteresis_threshold_high <= hi_hi


# --- The bundle-collapse operator must not eat vascular loops (#98 Tier 1 item 5) ---------
#
# skeletonize_voxel_bundles_into_paths deletes dense skeleton regions and replaces each with one
# hub node, merging everything passing through the window. Density is measured on the SKELETON,
# so one centreline crossing a 9^3 window is 9/729 = 0.0123 and two are 0.0247 - the old default
# of 0.025 meant "collapse anywhere two capillaries pass within 16.8 um", which in a capillary
# bed is the normal condition. Measured on the reference subvolume it took beta-1 from 307 to
# 99 and deleted 29% of the skeleton.

def _loop_skeleton():
    """Two centrelines 4 voxels apart, joined at both ends: one unambiguous vascular loop."""
    vol = np.zeros((11, 20, 31), dtype=bool)
    vol[5, 6, 5:26] = True       # first centreline
    vol[5, 10, 5:26] = True      # second, well inside a 9-voxel window
    vol[5, 6:11, 5] = True       # joined at the left
    vol[5, 6:11, 25] = True      # and at the right
    return vol


def _beta1(skeleton):
    """beta-1 via the Euler characteristic: chi = b0 - b1 + b2, and b2 = 0 for a curve.

    Not computed from voxel adjacency directly. Face connectivity reports a thinned skeleton as
    broken wherever it steps diagonally, and 26-connectivity closes a spurious triangle at every
    right-angle corner - the same fixture reads as 0 or as 5 depending which you pick. The Euler
    number is the quantity the pipeline's own graph_fundamental_loops is built on.
    """
    if not skeleton.any():
        return 0
    from scipy import ndimage
    from skimage.measure import euler_number

    _, b0 = ndimage.label(skeleton, structure=np.ones((3, 3, 3)))
    return b0 - int(euler_number(skeleton, connectivity=3))


def test_bundle_collapse_is_disabled_by_default():
    C = pytest.importorskip("carotid_image_to_model")
    assert C.SkeletonConfig().bundle_density_fraction >= 1.0, (
        "the bundle-collapse operator is enabled; it took beta-1 from 307 to 99 when measured"
    )


def test_bundle_collapse_short_circuits_when_disabled():
    """A window cannot exceed 100% foreground, so >= 1.0 is the documented off switch.

    The contract is that it does nothing beyond the baseline skeletonisation the function always
    performs, so the output must equal that baseline exactly.
    """
    from ImageLynx.preprocessing.skeleton import (
        skeletonize_3d,
        skeletonize_voxel_bundles_into_paths,
    )

    skeleton = _loop_skeleton()
    out = skeletonize_voxel_bundles_into_paths(skeleton, scan_size=9, density_fraction=1.0)
    assert np.array_equal(out, skeletonize_3d(skeleton).astype(bool))
    assert _beta1(out) == 1


def test_bundle_collapse_at_the_old_default_corrupts_loop_topology():
    """The behaviour being disabled, pinned so the reason cannot quietly be lost.

    On this fixture the operator does not delete the loop but replaces it with three, because
    collapsing the crossing to a hub and re-linking the boundary invents connections. Either
    direction is disqualifying for a hypothesis whose readout IS beta-1: on real data the error
    ran the other way, 307 loops down to 99.
    """
    from ImageLynx.preprocessing.skeleton import (
        skeletonize_3d,
        skeletonize_voxel_bundles_into_paths,
    )

    skeleton = skeletonize_3d(_loop_skeleton()).astype(bool)
    assert _beta1(skeleton) == 1, "fixture does not contain exactly one loop"

    collapsed = skeletonize_voxel_bundles_into_paths(
        skeleton, scan_size=9, density_fraction=0.025)
    assert _beta1(collapsed) != 1, (
        "the old default no longer alters this loop; the fixture has drifted and this test is "
        "no longer pinning the behaviour it documents"
    )


def test_default_skeleton_config_preserves_a_loop_through_the_full_cleanup():
    """End to end through preprocess_skeleton_for_graph at the shipped configuration."""
    C = pytest.importorskip("carotid_image_to_model")
    from ImageLynx.preprocessing import preprocess_skeleton_for_graph

    s = C.SkeletonConfig()
    cleaned = preprocess_skeleton_for_graph(
        _loop_skeleton(),
        min_branch_length=s.min_branch_length,
        max_bridge_distance=s.max_bridge_distance,
        component_connectivity=s.component_connectivity,
        min_component_fraction=0.0,
        bundle_scan_size=s.bundle_scan_size,
        bundle_density_fraction=s.bundle_density_fraction,
        bundle_max_connections_per_hub=s.bundle_max_connections,
        bundle_hub_min_spacing=s.bundle_hub_min_spacing,
    )

    assert _beta1(cleaned) == 1, "the shipped skeleton cleanup altered a vascular loop"


# --- Stub pruning is micron-denominated and cannot touch beta-1 (#98 Tier 1 item 6) -------

def _stub_graph(stub_lengths_um):
    """A square loop plus one terminal stub of each requested physical length."""
    G = nx.MultiGraph()
    ring = [(0.0, 0.0, 0.0), (0.0, 40.0, 0.0), (0.0, 40.0, 40.0), (0.0, 0.0, 40.0)]
    for i, pos in enumerate(ring):
        G.add_node(i, pos=np.array(pos))
    for i in range(4):
        j = (i + 1) % 4
        G.add_edge(i, j, voxels=[list(ring[i]), list(ring[j])])

    for k, length in enumerate(stub_lengths_um):
        tip = f"tip{k}"
        anchor = ring[0]
        end = (anchor[0], anchor[1] - length, anchor[2])
        G.add_node(tip, pos=np.array(end))
        G.add_edge(0, tip, voxels=[list(anchor), list(end)])
    return G


def _graph_beta1(G):
    return G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)


def test_stub_pruning_cannot_change_beta1():
    """Structural guarantee: only degree-1 nodes are removed, and those lie on no cycle.

    Confirmed on real data too - beta-1 was 307 at every threshold from 0 to 30 um. This is
    what distinguishes stub pruning from bundle collapse, which destroyed 68% of it.
    """
    from ImageLynx.graph.prune import prune_vascular_stubs

    G = _stub_graph([2.0, 4.0, 8.0, 20.0])
    assert _graph_beta1(G) == 1

    for threshold in [0.0, 3.0, 5.6, 10.0, 30.0, 1000.0]:
        pruned = prune_vascular_stubs(
            G.copy(), min_stub_length=threshold, voxel_size=(1.0, 1.0, 1.0))
        assert _graph_beta1(pruned) == 1, f"beta-1 changed at min_stub_length={threshold}"


def test_stub_pruning_threshold_is_in_microns():
    """A 4 um stub must survive a 3 um threshold and be cut by a 5.6 um one."""
    from ImageLynx.graph.prune import prune_vascular_stubs

    G = _stub_graph([4.0])
    kept = prune_vascular_stubs(G.copy(), min_stub_length=3.0, voxel_size=(1.0, 1.0, 1.0))
    cut = prune_vascular_stubs(G.copy(), min_stub_length=5.6, voxel_size=(1.0, 1.0, 1.0))

    assert "tip0" in kept
    assert "tip0" not in cut


def test_pipeline_configures_the_stub_threshold_explicitly(monkeypatch):
    """It was never passed, so the pipeline silently took the library default of 10.0.

    That default behaved as 10 voxels = 18.7 um before 2705b38 and as 10 um after, without the
    literal ever changing. Runs the graph step on a Y skeleton and records what it passes, at a
    threshold other than the default so a value typed in at the call fails too.
    """
    C = pytest.importorskip("carotid_image_to_model")

    cfg = C.GraphConfig()
    # Bounded by the measured p99 inscribed radius of the mask, 5.60 um: a skeletonisation spur
    # cannot be longer than the local vessel radius.
    assert 0.0 < cfg.min_stub_length_um <= 6.0

    passed = []
    prune = C.graph.prune_vascular_stubs

    def recording_prune(G, *args, **kwargs):
        passed.append(kwargs.get("min_stub_length", "not passed"))
        return prune(G, *args, **kwargs)

    monkeypatch.setattr(C.graph, "prune_vascular_stubs", recording_prune)

    # The Y skeleton of test_optuna_regression.py's graph-step harness.
    skeleton = np.zeros((40, 40, 40), dtype=np.uint8)
    skeleton[5:35, 20, 20] = 1
    skeleton[20, 20:35, 20] = 1
    cfg.keep_largest_component_only = False
    cfg.min_stub_length_um = 4.25
    pipeline_config = C.PipelineConfig()
    pipeline_config.enable_diagnostic_plots = False
    # input_format != "tif" keeps get_tif_spacing away from the filesystem.
    C._build_and_optimize_graph(skeleton, skeleton, None, "npy",
                                C.SkeletonConfig(), cfg, pipeline_config)

    assert passed == [4.25], \
        "the configured threshold is not reaching prune_vascular_stubs"


# --- Centreline smoothing provenance (#98 Tier 1 item 7) ----------------------------------
#
# `length` is only rewritten for edges that smooth successfully, so the section 1.4 tortuosity
# numerator mixes two operators: a B-spline arc length for some edges and a raw 26-connected
# staircase for the rest. Measured on the reference subvolume, 967 edges smoothed, 4 relaxed,
# 214 fell back and 149 were too short - 27.2% keeping the staircase - and the staircase runs
# about 8% longer (smoothed/raw ratio: p50 0.9336, mean 0.9263).

def _smoothing_graph():
    """One long wobbly edge that can be splined, and one 2-point reconnection edge."""
    G = nx.MultiGraph()
    wobble = [[0.0, float(i), 2.0 * ((i % 2) - 0.5)] for i in range(20)]
    G.add_node("a", pos=np.array(wobble[0]))
    G.add_node("b", pos=np.array(wobble[-1]))
    G.add_edge("a", "b", voxels=wobble, length=100.0)

    G.add_node("c", pos=np.array([0.0, 40.0, 0.0]))
    G.add_edge("b", "c", voxels=[wobble[-1], [0.0, 40.0, 0.0]], length=20.0)

    skeleton = np.zeros((3, 45, 9), dtype=bool)
    for pt in wobble:
        skeleton[int(round(pt[0])), int(round(pt[1])), int(round(pt[2])) + 4] = True
    return G, skeleton


def test_every_edge_is_tagged_with_its_smoothing_provenance():
    """An untagged graph silently mixes two operators in the tortuosity numerator."""
    from ImageLynx.graph._helpers import smooth_graph_edge_centerlines_continuous

    G, skeleton = _smoothing_graph()
    stats = smooth_graph_edge_centerlines_continuous(G, skeleton, voxel_size=(1.0, 1.0, 1.0))

    tags = [d.get("centreline_smoothing") for _, _, d in G.edges(data=True)]
    assert all(t is not None for t in tags), "some edges carry no smoothing provenance"
    assert set(tags) <= {"bspline", "chaikin", "bspline_relaxed", "raw_fallback", "raw_too_short",
                         "raw_unchanged"}
    assert sum(stats["centreline_smoothing_counts"].values()) == G.number_of_edges()


def test_two_point_edges_are_tagged_as_structurally_unsplinable():
    """These are terminal reconnections. Their tortuosity is 1.0 by construction, not anatomy.

    They must be distinguishable from edges where smoothing was tried and rejected, because the
    two need different treatment in a section 1.4 distribution.
    """
    from ImageLynx.graph._helpers import smooth_graph_edge_centerlines_continuous

    G, skeleton = _smoothing_graph()
    smooth_graph_edge_centerlines_continuous(G, skeleton, voxel_size=(1.0, 1.0, 1.0))

    two_point = [d for u, v, d in G.edges(data=True) if len(d["voxels"]) == 2]
    assert two_point, "fixture no longer contains a 2-point edge"
    assert all(d["centreline_smoothing"] == "raw_too_short" for d in two_point)


def test_edges_the_spline_returns_unchanged_are_not_corridor_fallbacks(caplog):
    """A 3-point edge cannot be splined, so it comes back as it went in (package T).

    It was tagged raw_fallback and warned about as if it had left the skeleton corridor. It
    did not: smoothing was a no-op, and it needs its own tag so the warning stays true.
    """
    import logging

    from ImageLynx.graph._helpers import smooth_graph_edge_centerlines_continuous

    voxels = [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 2.0, 1.0]]
    G = nx.MultiGraph()
    G.add_node("a", pos=np.array(voxels[0]))
    G.add_node("b", pos=np.array(voxels[-1]))
    G.add_edge("a", "b", voxels=[list(p) for p in voxels], length=3.0)
    skeleton = np.zeros((1, 3, 2), dtype=bool)
    for z, y, x in voxels:
        skeleton[int(z), int(y), int(x)] = True

    with caplog.at_level(logging.WARNING, logger="ImageLynx.graph._helpers"):
        stats = smooth_graph_edge_centerlines_continuous(G, skeleton, voxel_size=(1.0, 1.0, 1.0))

    data = G["a"]["b"][0]
    assert data["centreline_smoothing"] == "raw_unchanged"
    assert data["voxels"] == [list(p) for p in voxels]
    assert data["length"] == 3.0
    assert stats["unchanged_edges"] == 1
    assert stats["fallback_edges"] == 0
    assert "raw centreline" not in caplog.text


def test_unsmoothed_edges_keep_the_longer_staircase_numerator():
    """The bias being made visible: a raw voxel path is longer than its spline.

    Tortuosity is length over Euclidean distance and the denominator is unaffected, so an
    unsmoothed edge reports a proportionally higher tortuosity for purely instrumental reasons.
    """
    from ImageLynx.graph._helpers import bspline_smooth_polyline, calculate_path_length

    wobble = [[0.0, float(i), 2.0 * ((i % 2) - 0.5)] for i in range(20)]
    raw_length = calculate_path_length(wobble)
    smoothed = np.asarray(bspline_smooth_polyline(np.asarray(wobble, dtype=float), smoothness=0.75))
    smoothed_length = calculate_path_length(smoothed.tolist())

    assert smoothed_length < raw_length, (
        "the staircase bias has vanished; the provenance tag is no longer motivated"
    )
