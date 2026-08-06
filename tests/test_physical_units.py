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
from ImageLynx.statistics.benchmarking import (
    evaluate_completeness_and_overpruning,
    redilate_skeleton_to_volume,
)

# The measured acquisition voxel size for the WKY carotid body volume, (z, y, x) in microns.
# The anisotropy is on z, not x: the raw TIFF's ImageJ metadata gives a slice spacing of
# 1.86386 um and an in-plane pixel size of 1.86600 um in both y and x. This was previously
# written (1.8660, 1.8660, 1.8639), which put the odd axis on x under a (z, y, x) reading.
SPACING = (1.8639, 1.8660, 1.8660)
SHAPE = (20, 20, 20)

# A straight centreline along z at voxel (y, x) = (10, 10), spanning voxels z = 5..15.
CENTRELINE_VOXELS = [(z, 10, 10) for z in range(5, 16)]


def _physical(voxel_coords):
    """Convert (z, y, x) voxel indices to the physical coordinates the graph actually stores."""
    return [
        [z * SPACING[0], y * SPACING[1], x * SPACING[2]] for z, y, x in voxel_coords
    ]


def _centreline_graph(diameter_um=2.0):
    G = nx.MultiGraph()
    physical = _physical(CENTRELINE_VOXELS)
    G.add_node(0, pos=np.array(physical[0]))
    G.add_node(1, pos=np.array(physical[-1]))
    G.add_edge(0, 1, voxels=physical, assigned_diameter_um=diameter_um)
    return G


def test_redilate_skeleton_to_volume_round_trips_anisotropic_spacing():
    """Physical edge coordinates must be divided by the spacing before being used as indices.

    Truncating them directly places the reconstruction at voxel (z * 1.866, 18, 18) instead
    of (z, 10, 10) - an offset that silently corrupts the Dice coefficient the skeletonisation
    tuner optimises against.
    """
    reconstructed = redilate_skeleton_to_volume(_centreline_graph(), SHAPE, SPACING)

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
    import sys
    from pathlib import Path

    examples_path = Path(__file__).parent.parent / "examples"
    if str(examples_path) not in sys.path:
        sys.path.insert(0, str(examples_path))
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


def test_unit_spacing_is_unchanged():
    """The fixes must be a no-op at (1, 1, 1), which is what every existing caller uses today."""
    unit = (1.0, 1.0, 1.0)
    G = nx.MultiGraph()
    voxels = [[float(z), 10.0, 10.0] for z in range(5, 16)]
    G.add_node(0, pos=np.array(voxels[0]))
    G.add_node(1, pos=np.array(voxels[-1]))
    G.add_edge(0, 1, voxels=voxels, assigned_diameter_um=2.0)

    reconstructed = redilate_skeleton_to_volume(G, SHAPE, unit)
    assert reconstructed[10, 10, 10]
    assert not reconstructed[10, 18, 18]


# --- The configured acquisition voxel size ------------------------------------------------
#
# 2705b38 added the mechanism for an explicit voxel size, but PipelineConfig.voxel_size_um was
# left at None and neither group YAML set it. The pipeline consumes the Ilastik probability
# TIFF, which carries no resolution tag, so _resolve_voxel_size fell back to (1, 1, 1) and
# every "micron" in the output was really a voxel count - the exact failure 2705b38 was meant
# to close. These tests pin the configured value to the acquisition file it came from.

def _pipeline_module():
    return pytest.importorskip("carotid_image_to_model")


def test_configured_voxel_size_is_calibrated_not_unit():
    mod = _pipeline_module()
    configured = mod.PipelineConfig().voxel_size_um

    assert configured is not None, "voxel_size_um is unset; the pipeline will fall back to voxels"
    assert tuple(configured) != (1.0, 1.0, 1.0)
    # The anisotropy belongs on z. y and x are the in-plane pixel size and must agree.
    z, y, x = configured
    assert y == x, "y and x are the same in-plane pixel size and must be equal"
    assert z != y, "the slice spacing differs from the in-plane pixel size"


def test_configured_voxel_size_matches_the_raw_acquisition_metadata():
    """The constant must be derived from the file, not typed in beside it."""
    mod = _pipeline_module()
    if not mod.RAW_IMAGE_PATH.exists():
        pytest.skip(f"raw acquisition volume not present at {mod.RAW_IMAGE_PATH}")

    from ImageLynx.io import get_tif_spacing

    detected = get_tif_spacing(mod.RAW_IMAGE_PATH)
    configured = tuple(mod.PipelineConfig().voxel_size_um)

    assert configured == pytest.approx(detected), (
        f"configured voxel_size_um {configured} has drifted from the acquisition "
        f"metadata {detected}"
    )


def test_resolve_voxel_size_uses_the_configuration_for_the_untagged_probability_tiff():
    """The probability TIFF declares no resolution; the configured value must win anyway."""
    mod = _pipeline_module()
    configured = tuple(mod.PipelineConfig().voxel_size_um)

    saved = mod.VOXEL_SIZE_UM
    try:
        mod.VOXEL_SIZE_UM = configured
        resolved = mod._resolve_voxel_size(mod.RAW_IMAGE_PATH, "tiff")
    finally:
        mod.VOXEL_SIZE_UM = saved

    assert resolved == pytest.approx(configured)
    assert resolved != (1.0, 1.0, 1.0)


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
    assert RECONNECT_THRESHOLD_UM == pytest.approx(3 * 1.866, abs=0.05)
    assert CONSERVATIVE_RECONNECT_CAP_UM == pytest.approx(1.5 * 1.866, abs=0.05)


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


# --- The default filter chain must survive a capillary (Phase 1.5) ------------------------

def _capillary_probability_volume(radius_voxels=1.6, p_vessel=0.95, p_background=0.05):
    """A capillary-scale tube in a probability map: the thing the chain must not erase."""
    shape = (24, 41, 41)
    _, yy, xx = np.indices(shape)
    tube = ((yy - 20) ** 2 + (xx - 20) ** 2) <= radius_voxels ** 2
    prob = np.full(shape, p_background, dtype=np.float32)
    prob[tube] = p_vessel
    return prob, tube


def test_default_preprocessing_chain_preserves_a_capillary():
    """A 6 um capillary is 3.2 voxels across; median 7 and opening 1 filter it away.

    Both are applied to the float probability map before thresholding, so this is signal
    destruction rather than mask cleanup. Measured on real data, the old chain left 71
    connected structures where an unfiltered one left 199.
    """
    C = pytest.importorskip("carotid_image_to_model")
    prob, tube = _capillary_probability_volume()

    _, mask = C._apply_preprocessing_filters(
        prob, None, C.PreprocessingConfig().__dict__, boundary_permeability_mode="caged")
    assert mask.shape == prob.shape

    recovered = float((mask & tube).sum()) / float(tube.sum())
    assert recovered > 0.5, (
        f"the default filter chain destroyed the capillary: only {recovered:.0%} recovered"
    )


def test_greyscale_opening_has_no_capillary_preserving_radius():
    """The reason morphological_opening_radius is 0 rather than merely reduced."""
    from scipy import ndimage
    from skimage.morphology import ball

    _, tube = _capillary_probability_volume()
    assert ndimage.binary_opening(tube, structure=ball(1)).sum() < 0.7 * tube.sum()
    assert ndimage.binary_opening(tube, structure=ball(2)).sum() == 0


# --- The default thresholds must not flood (Phase 1.5) ------------------------------------

def test_default_hysteresis_thresholds_do_not_flood_a_soft_probability_field():
    """The failure mode the old 0.2 / 0.4 defaults had on the real data.

    Hysteresis grows from seeds above `high` into any connected region above `low`. When the
    background probability sits above `low` - which it does here, and does on the real field,
    whose median is 0.42 - the whole volume is one connected superlevel set containing seeds,
    so the mask floods regardless of `high`. Measured on real data, the old defaults gave 84.7%
    foreground as a single component with a median inscribed radius of 11.8 um.
    """
    C = pytest.importorskip("carotid_image_to_model")

    shape = (24, 41, 41)
    _, yy, xx = np.indices(shape)
    prob = np.full(shape, 0.35, dtype=np.float32)          # soft background, above the old low
    prob[((yy - 20) ** 2 + (xx - 20) ** 2) <= 1.6 ** 2] = 0.95   # a capillary

    _, mask = C._apply_preprocessing_filters(
        prob, None, C.PreprocessingConfig().__dict__, boundary_permeability_mode="caged")

    assert mask.mean() < 0.25, (
        f"the default thresholds flooded: {mask.mean():.1%} foreground on a volume whose "
        "vessel occupies about 1%"
    )


def test_default_hysteresis_thresholds_lie_inside_the_tuner_search_range():
    """A default outside the searched range cannot be reproduced or refined by tuning."""
    C = pytest.importorskip("carotid_image_to_model")
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

    try:
        PreprocessingObjective(lambda kwargs: None)(_Recorder())
    except Exception:
        pass

    cfg = C.PreprocessingConfig()
    lo_lo, lo_hi = ranges["hysteresis_threshold_low"]
    hi_lo, hi_hi = ranges["hysteresis_threshold_high"]
    assert lo_lo <= cfg.hysteresis_threshold_low <= lo_hi
    assert hi_lo <= cfg.hysteresis_threshold_high <= hi_hi
    assert cfg.hysteresis_threshold_high > cfg.hysteresis_threshold_low


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
