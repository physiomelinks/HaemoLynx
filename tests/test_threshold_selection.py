"""Choosing the segmentation threshold from calibre, constrained by fragmentation.

The segmentation handover picks the threshold from connected-component statistics: take the
value just above where component count climbs steeply and the largest component's share
starts falling. Measured on the real WKY-C probability field, that criterion does not work.

    low   fg     comps  largest share  n>=50vox  d_med um
    0.20  0.507  10621         0.9945       114     13.96
    0.30  0.456   7787         0.9945       105     12.92
    0.70  0.310   2844         0.9369       115      9.14
    0.90  0.224   2003         0.9492       135      7.46
    0.95  0.184   2347         0.9507       139      5.28
    0.99  0.105   7151         0.9580       138      3.73

The largest component's share never falls - it is *higher* at 0.99, where the network has
visibly shattered, than at 0.70 - because share is measured in voxels and this network is one
dominant mass at every threshold, with fragments too small to move a voxel fraction. The
count of components above a size floor is equally flat, wandering between 94 and 139 with no
structure. No mask-component statistic discriminates here.

Two things do. Median inscribed diameter moves monotonically from 13.96 um to 3.73 um and has
an external target - the handover's own validation table expects a capillary mode of 4-7 um.
And skeleton endpoint density is flat until the network actually beads, then doubles: on a
mid-stack subvolume it runs 2.3, 2.8, 3.2, 2.9, 2.4, 2.1 per mm from 0.30 to 0.97 and then
jumps to 4.8 at 0.99, where skeleton components go 172 -> 467 and mean component length
halves.

So calibre is the objective and fragmentation is the constraint, which is the reverse of the
handover's ordering.
"""
import numpy as np
import pytest

from ImageLynx.statistics.threshold_selection import (
    CAPILLARY_DIAMETER_RANGE_UM,
    ThresholdSample,
    evaluate_threshold,
    select_threshold,
    sweep_thresholds,
)

VOXEL = (1.8639, 1.866, 1.866)


def _tube(shape=(64, 48, 48), radius_vox=4.0, beading=False, seed=0):
    """A capillary-scale tube whose probability optionally dips periodically along z.

    With ``beading`` the axial peak probability cycles between 0.62 and 1.0, so the tube is
    continuous at a low threshold and breaks into beads at a high one - fragmentation with a
    known onset, which is what the constraint has to detect.
    """
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cy, cx = shape[1] // 2, shape[2] // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    prob = 1.0 / (1.0 + np.exp((r - radius_vox) * 2.5))
    prob = np.broadcast_to(prob, shape).astype(np.float32).copy()
    if beading:
        axial = 0.81 + 0.19 * np.cos(2 * np.pi * zz[:, 0, 0] / 8.0)
        prob *= axial[:, None, None].astype(np.float32)
    return np.clip(prob, 0.0, 1.0).astype(np.float32)


def test_a_single_evaluation_reports_calibre_and_both_topologies():
    sample = evaluate_threshold(_tube(), 0.5, VOXEL)

    assert isinstance(sample, ThresholdSample)
    assert sample.threshold == 0.5
    assert 0.0 < sample.foreground_fraction < 1.0
    assert sample.median_diameter_um > 0
    # Mask-component statistics are still recorded - they are just not decisive.
    assert sample.mask_components >= 1
    assert 0.0 < sample.largest_mask_component_share <= 1.0
    # The skeleton measures are what the constraint uses.
    assert sample.skeleton_components >= 1
    assert sample.endpoint_density_per_mm >= 0.0


def test_calibre_is_the_objective_and_picks_the_capillary_scale_threshold():
    """The handover's own validation table expects a 4-7 um capillary diameter mode."""
    prob = _tube(radius_vox=4.0)
    samples = sweep_thresholds(prob, np.arange(0.1, 1.0, 0.1), VOXEL)
    selection = select_threshold(samples)

    assert selection.threshold is not None, selection.reason
    chosen = next(s for s in samples if s.threshold == selection.threshold)
    lo, hi = CAPILLARY_DIAMETER_RANGE_UM
    assert lo <= chosen.median_diameter_um <= hi
    assert selection.require() == selection.threshold


def test_fragmentation_is_detected_from_the_skeleton_not_the_mask():
    """The mask stays one component while the centreline is already beading.

    This is the failure the handover's criterion is blind to, reproduced deliberately: the
    beaded tube's mask-component statistics barely move while endpoint density climbs.
    """
    samples = sweep_thresholds(_tube(beading=True), np.arange(0.3, 1.0, 0.05), VOXEL)
    selection = select_threshold(samples)

    assert selection.fragmentation_onset is not None, "beading must be detected"
    beyond = [s for s in samples if s.threshold >= selection.fragmentation_onset]
    below = [s for s in samples if s.threshold < selection.fragmentation_onset]
    assert beyond and below

    worst = max(beyond, key=lambda s: s.endpoint_density_per_mm)
    calmest = min(below, key=lambda s: s.endpoint_density_per_mm)
    assert worst.endpoint_density_per_mm > calmest.endpoint_density_per_mm

    # Nothing at or beyond the onset may be selected, whatever its calibre.
    assert selection.threshold is None or selection.threshold < selection.fragmentation_onset


def test_no_threshold_satisfying_both_is_reported_not_papered_over():
    """If calibre and connectivity never agree, that is a segmentation problem.

    Returning a best-effort threshold would convert an unusable segmentation into a plausible
    set of numbers, which is the failure this whole module exists to avoid.
    """
    # A tube far too fat for capillary calibre at any threshold.
    samples = sweep_thresholds(_tube(radius_vox=8.0), np.arange(0.1, 1.0, 0.1), VOXEL)
    selection = select_threshold(samples)

    assert selection.threshold is None
    assert "calibre" in selection.reason.lower()
    with pytest.raises(ValueError, match="No threshold"):
        selection.require()


def test_the_mask_component_share_criterion_is_recorded_but_never_decides():
    """Regression guard on the reason this module exists.

    largest_mask_component_share is kept because it is what the handover specifies and it
    belongs in the record, but a selection must never turn on it - it is empirically flat.
    """
    import inspect

    from ImageLynx.statistics import threshold_selection

    source = inspect.getsource(threshold_selection.select_threshold)
    assert "largest_mask_component_share" not in source
    assert "median_diameter_um" in source
    assert "endpoint_density_per_mm" in source


def test_sweep_is_ordered_and_covers_every_requested_threshold():
    thresholds = [0.9, 0.3, 0.6]
    samples = sweep_thresholds(_tube(), thresholds, VOXEL)
    assert [s.threshold for s in samples] == [0.3, 0.6, 0.9]


def test_an_empty_mask_is_skipped_rather_than_crashing():
    """Sweeps run to the top of the range, where the mask legitimately empties out."""
    samples = sweep_thresholds(_tube(), [0.5, 0.99999], VOXEL)
    assert all(s.foreground_fraction > 0 for s in samples)
    assert len(samples) < 2 or samples[-1].skeleton_length_mm >= 0


def test_selection_can_be_rendered_as_a_table():
    """The sweep is something a person reads before freezing a parameter."""
    samples = sweep_thresholds(_tube(), np.arange(0.2, 1.0, 0.2), VOXEL)
    text = select_threshold(samples).format_table()
    assert "d_med" in text and "ep/mm" in text
    assert str(samples[0].threshold) in text or f"{samples[0].threshold:.2f}" in text
