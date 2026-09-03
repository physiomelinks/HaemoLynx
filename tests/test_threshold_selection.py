"""Choosing the segmentation threshold from calibre, constrained by fragmentation.

The segmentation handover picks the threshold from connected-component statistics: take the
value just above where component count climbs steeply and the largest component's share
starts falling. Measured on the 160^3 ROI the pipeline actually selects on - WKY-C - that
criterion cannot name a value.

    thr   fg     comps  largest share  n>=50vox  d_med um  ep/mm
    0.30  0.652    753         0.9980         6      9.14   3.10
    0.50  0.530    572         0.9972        21      8.34   5.13
    0.70  0.417    405         0.9899        11      6.46   5.13
    0.90  0.261    400         0.9917        10      5.27   5.17
    0.95  0.187    640         0.9925        11      3.73   7.83
    0.99  0.063   2700         0.9239        32      3.73  32.57

Both mask statistics move - the share falls from 0.9980 to 0.9239, in this and every other
specimen, and component count is U-shaped. Neither has a knee. Component count accelerates
smoothly, so "just above where it climbs steeply" names no threshold; and the share only
moves once the network has already shattered, so reading a value off it lands at 0.95-0.97,
by which point endpoint density has already doubled. They move too late, not too little.

(An earlier version of this docstring quoted a whole-volume sweep - 10621 components at 0.20,
7151 at 0.99, above-floor counts of 94-139 - and concluded the share never falls. That was
measured before ROI placement existed, on a different sub-volume, and does not hold here.)

Two things do better. Median inscribed diameter falls monotonically and has an external
target - the handover's own validation table expects a capillary mode of 4-7 um. And skeleton
endpoint density is flat while the network is intact, then climbs sharply once it beads.

So calibre is the objective and fragmentation is the constraint, which is the reverse of the
handover's ordering. On these six specimens the constraint never actually binds: the
fragmentation onset always sits above the top of the calibre window.
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


def _sample(threshold, d_med, ep_per_mm):
    """A minimal sample carrying only the two fields selection reads."""
    return ThresholdSample(
        threshold=threshold,
        foreground_fraction=0.3,
        median_diameter_um=d_med,
        p90_diameter_um=d_med * 2,
        mask_components=100,
        mask_components_above_floor=10,
        largest_mask_component_share=0.99,
        skeleton_length_mm=1.0,
        endpoints=int(ep_per_mm),
        endpoint_density_per_mm=ep_per_mm,
        skeleton_components=10,
    )


def test_only_the_lower_calibre_bound_can_select():
    """The window's upper bound is structurally inert, and that is not obvious.

    Median calibre falls monotonically with threshold and select_threshold takes the *highest*
    threshold in the window, so the upper bound only ever prunes from the low-threshold end -
    which the maximum never reads. The one thing it can do is empty the window and force a
    refusal. This is why a sensitivity analysis that sweeps the window's width symmetrically
    is testing one live parameter and one dead one.
    """
    # calibre falling monotonically, as it does on every real specimen
    samples = [
        _sample(0.30, 9.14, 3.0),
        _sample(0.50, 8.34, 3.5),
        _sample(0.70, 6.46, 4.0),
        _sample(0.85, 5.27, 4.5),
        _sample(0.90, 5.27, 5.0),
        _sample(0.95, 3.73, 12.0),
    ]
    # The fragmentation veto is switched off throughout, so that what is being measured is
    # the calibre window alone. With it live it would mask the effect at the low end.
    def choose(lo, hi):
        return select_threshold(
            samples, diameter_range=(lo, hi), fragmentation_tolerance=1e9
        ).threshold

    baseline = choose(4.0, 7.0)
    assert baseline == 0.90

    # raising the ceiling to anything that still admits 5.27 changes nothing
    for hi in (5.5, 6.0, 7.0, 8.0, 12.0, 25.0):
        assert choose(4.0, hi) == baseline

    # the floor is what selects
    assert choose(3.70, 7.0) == 0.95
    assert choose(5.30, 7.0) == 0.70

    # and a ceiling below every observed calibre is a refusal, not a different pick
    assert choose(4.0, 3.9) is None


def test_the_fragmentation_veto_is_a_guard_that_the_real_data_never_trips():
    """On all six specimens the onset sits above the calibre window, so nothing is vetoed.

    Pinned because it is the reason skeletonisation - the expensive half of the sweep - is
    currently buying a guard rather than a decision. If a future classifier makes this bind,
    this test should fail and the section's claim be revisited.
    """
    # the real shape: calibre window tops out at 0.90, endpoint density does not spike until 0.95
    samples = [
        _sample(0.70, 6.46, 5.13),
        _sample(0.80, 5.28, 4.73),
        _sample(0.85, 5.27, 5.01),
        _sample(0.90, 5.27, 5.17),
        _sample(0.93, 3.73, 6.09),
        _sample(0.95, 3.73, 7.83),
        _sample(0.97, 3.73, 10.89),
    ]
    selection = select_threshold(samples)
    assert selection.threshold == 0.90
    assert selection.fragmentation_onset == 0.95
    # the onset is above the top of the calibre window, so it removed no candidate
    assert max(selection.calibre_window) < selection.fragmentation_onset
    # and disabling the constraint gives the identical answer
    unconstrained = select_threshold(samples, fragmentation_tolerance=1e9)
    assert unconstrained.fragmentation_onset is None
    assert unconstrained.threshold == selection.threshold


def test_the_above_floor_component_count_is_computed_but_never_shown():
    """MIN_COMPONENT_VOXELS feeds a field format_table does not print and nothing reads.

    The 'maskcmp' column is the unfiltered total. Guarding it so the discrepancy is noticed
    rather than rediscovered: if the column is ever switched to the filtered count, the
    section 2.2 note about it saying so must change too.
    """
    sample = evaluate_threshold(_tube(), 0.5, VOXEL)
    assert sample.mask_components_above_floor <= sample.mask_components

    table = select_threshold([sample]).format_table()
    assert str(sample.mask_components) in table
    if sample.mask_components_above_floor != sample.mask_components:
        assert f"{sample.mask_components_above_floor:>9d}" not in table
