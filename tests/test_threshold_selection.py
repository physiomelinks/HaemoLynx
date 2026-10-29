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
from dataclasses import replace

import numpy as np
import pytest

from ImageLynx.specimens import PROCESSING_VOXEL_UM
from ImageLynx.statistics.threshold_selection import (
    CAPILLARY_DIAMETER_RANGE_UM,
    ThresholdSample,
    evaluate_threshold,
    select_on_network_calibre,
    select_threshold,
    sweep_thresholds,
)

VOXEL = PROCESSING_VOXEL_UM


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
    """One straight tube: one mask component, one centreline with its two ends."""
    sample = evaluate_threshold(_tube(), 0.5, VOXEL)

    assert isinstance(sample, ThresholdSample)
    assert sample.threshold == 0.5
    assert 0.0 < sample.foreground_fraction < 1.0
    # Radius ~4 voxels at the 0.5 cut: the centreline median reads ~15.4 um across.
    assert sample.median_diameter_um == pytest.approx(15.39, abs=0.01)
    # Mask-component statistics are still recorded - they are just not decisive.
    assert sample.mask_components == 1
    assert sample.largest_mask_component_share == 1.0
    # The skeleton measures are what the constraint uses.
    assert sample.skeleton_components == 1
    assert sample.endpoints == 2


def test_calibre_is_the_objective_and_picks_the_capillary_scale_threshold():
    """The handover's own validation table expects a 4-7 um capillary diameter mode.

    A capillary-scale tube: radius 1.5 voxels, ~5.6 um across. It used to be radius 4 (~15 um),
    which only reached the window through the voxel-weighted median.
    """
    prob = _tube(radius_vox=1.5)
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


def test_sweep_is_ordered_and_covers_every_requested_threshold():
    thresholds = [0.9, 0.3, 0.6]
    samples = sweep_thresholds(_tube(), thresholds, VOXEL)
    assert [s.threshold for s in samples] == [0.3, 0.6, 0.9]


def test_an_empty_mask_is_skipped_rather_than_crashing():
    """Sweeps run to the top of the range, where the mask legitimately empties out."""
    # The seed must sit above every threshold (the pipeline's rule), so it goes to 1.0 here;
    # nothing reaches it, and the network-mask calibre is then NaN rather than an error.
    samples = sweep_thresholds(_tube(), [0.5, 0.99999], VOXEL, seed=1.0)
    # The empty 0.99999 cut is dropped, not returned as a zero sample.
    assert [s.threshold for s in samples] == [0.5]


def test_selection_can_be_rendered_as_a_table():
    """The sweep is something a person reads before freezing a parameter."""
    samples = sweep_thresholds(_tube(), np.arange(0.2, 1.0, 0.2), VOXEL)
    lines = select_threshold(samples).format_table().splitlines()
    header = lines[1].split()
    assert "d_med" in header and "ep/mm" in header
    # Recorded and printed, never read by the selection: the voxel-level median and the
    # network-mask median sit beside the plain-cut one so a person can compare them.
    assert "d_vox" in header and "d_net" in header
    # One row per threshold, each led by the threshold to two places.
    rows = lines[2:-1]
    assert [row.split()[0] for row in rows] == ["0.20", "0.40", "0.60", "0.80"]


def _sample(threshold, d_med, ep_per_mm):
    """A minimal sample carrying only the two fields selection reads."""
    return ThresholdSample(
        threshold=threshold,
        foreground_fraction=0.3,
        median_diameter_um=d_med,
        p90_diameter_um=d_med * 2,
        median_voxel_diameter_um=d_med * 0.8,
        mask_components=100,
        mask_components_above_floor=10,
        largest_mask_component_share=0.99,
        skeleton_length_mm=1.0,
        endpoints=int(ep_per_mm),
        endpoint_density_per_mm=ep_per_mm,
        skeleton_components=10,
    )


# The real shape: calibre window tops out at 0.90, endpoint density does not spike until 0.95.
_REAL_SHAPE_SWEEP = [
    _sample(0.70, 6.46, 5.13),
    _sample(0.80, 5.28, 4.73),
    _sample(0.85, 5.27, 5.01),
    _sample(0.90, 5.27, 5.17),
    _sample(0.93, 3.73, 6.09),
    _sample(0.95, 3.73, 7.83),
    _sample(0.97, 3.73, 10.89),
]


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
    selection = select_threshold(_REAL_SHAPE_SWEEP)
    assert selection.threshold == 0.90
    assert selection.fragmentation_onset == 0.95
    # the onset is above the top of the calibre window, so it removed no candidate
    assert max(selection.calibre_window) < selection.fragmentation_onset
    # and disabling the constraint gives the identical answer
    unconstrained = select_threshold(_REAL_SHAPE_SWEEP, fragmentation_tolerance=1e9)
    assert unconstrained.fragmentation_onset is None
    assert unconstrained.threshold == selection.threshold


def test_the_above_floor_component_count_is_computed_but_never_shown():
    """MIN_COMPONENT_VOXELS feeds a field format_table does not print and nothing reads.

    The 'maskcmp' column is the unfiltered total. Guarding it so the discrepancy is noticed
    rather than rediscovered: if the column is ever switched to the filtered count, every
    description of it as unfiltered must change too.
    """
    sample = _sample(0.90, 5.27, 5.17)  # 100 components, 10 of them above the floor

    lines = select_threshold([sample]).format_table().splitlines()
    column = lines[1].split().index("maskcmp")
    assert lines[2].split()[column] == "100"


def _quantised(**kw):
    """The tube, quantised to hundredths the way the Ilastik export is."""
    return (np.round(_tube(**kw) * 100.0).astype(np.float32)
            / np.float32(100.0))


def test_the_cut_keeps_the_level_on_the_threshold():
    """The cut is inclusive, as the pipeline's hysteresis is.

    The Ilastik export is quantised to hundredths - 101 distinct values on every real ROI -
    and every threshold in the sweep grid is a whole number of hundredths, so each lands
    exactly on an occupied level. The strict `>` used before dropped that level
    entire: (p > t) == (p >= t + 0.01), and p > 0.99 was exactly p == 1.0. On the six
    specimens the dropped mass ran 0.5% of the ROI at 0.30 to 4.2-5.9% at 0.99.
    """
    prob = _quantised()
    assert len(np.unique(prob)) <= 101

    # the tube's discrete radii do not occupy every level; test the grid values that are
    occupied = [t for t in (0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99)
                if (prob == np.float32(t)).any()]
    assert len(occupied) >= 3, occupied
    for t in occupied:
        on_level = prob == np.float32(t)
        sample = evaluate_threshold(prob, t, VOXEL)
        assert sample.foreground_fraction == pytest.approx(float((prob >= np.float32(t)).mean()))
        # and strictly more than the old strict cut, by exactly the level on the threshold
        assert sample.foreground_fraction == pytest.approx(
            float((prob > np.float32(t)).mean() + on_level.mean()))


def test_calibre_is_read_on_the_centreline_not_over_every_voxel():
    """The window is a capillary diameter, which is a centreline quantity.

    For a solid tube of radius R the centreline EDT is ~R everywhere, while the median over
    every foreground voxel sits near R(1 - 1/sqrt(2)), because half a disc's area lies in its
    outer ring. The selector used to use the latter and read 0.63-1.00x the centreline
    median on the real masks.
    """
    radius_vox = 6.0
    prob = _tube(radius_vox=radius_vox)
    sample = evaluate_threshold(prob, 0.5, VOXEL)

    centreline_diameter = 2 * radius_vox * VOXEL[1]
    assert sample.median_diameter_um == pytest.approx(centreline_diameter, rel=0.1)
    assert sample.median_voxel_diameter_um < 0.75 * sample.median_diameter_um


def _everywhere(value):
    return lambda s: value


def _at_or_above_090(value, below):
    return lambda s: value if s.threshold >= 0.90 else below


@pytest.mark.parametrize("field, moves", [
    pytest.param(field, moves, id=field) for field, moves in (
        ("median_voxel_diameter_um", (_everywhere(0.0), _everywhere(1e6))),
        ("median_network_diameter_um", (_everywhere(0.0), _everywhere(1e6))),
        ("largest_mask_component_share", (_at_or_above_090(0.5, 0.99), _everywhere(0.0))),
        ("mask_components_above_floor", (_everywhere(0), _everywhere(10**6))),
    )
])
def test_a_recorded_only_field_never_moves_the_selection(field, moves):
    """The field is in the record but never read by the selection.

    ``largest_mask_component_share`` is kept because it is what the handover specifies, but a
    selection must never turn on it: it is empirically flat until the network has already
    shattered (module docstring). ``mask_components_above_floor`` is computed and never shown.
    ``median_network_diameter_um`` (d_net) shows the calibre the network would read at each
    threshold. It is a diagnostic: the window is still tested on the plain-cut
    centreline median, because every candidate replacement moves calibre by about the window's
    width and so would choose the threshold by itself.

    Each case moves the field on the samples, one way per move: a calibre far outside the
    window on either side, the share falling to 0.5 from the chosen 0.90 up (the knee the
    handover reads) or to 0 everywhere, the above-floor count to 0 or a million. A selection
    that read the field would refuse, or pick another threshold.
    """
    baseline = select_threshold(_REAL_SHAPE_SWEEP)
    assert baseline.threshold == 0.90

    for i, move in enumerate(moves):
        moved = [replace(s, **{field: move(s)}) for s in _REAL_SHAPE_SWEEP]
        selection = select_threshold(moved)
        assert selection.threshold == baseline.threshold, (field, i)
        assert selection.calibre_window == baseline.calibre_window, (field, i)
        assert selection.fragmentation_onset == baseline.fragmentation_onset, (field, i)


def test_the_network_mask_uses_the_given_seed():
    """Nothing reaches a seed of 1.0 here, so the network mask is empty and d_net is NaN,
    while the plain-cut calibre is untouched: the seed reaches d_net and nothing else."""
    prob = _tube()
    fixed = evaluate_threshold(prob, 0.5, VOXEL)
    unseeded = evaluate_threshold(prob, 0.5, VOXEL, seed=1.0)
    assert np.isfinite(fixed.median_network_diameter_um)
    assert np.isnan(unseeded.median_network_diameter_um)
    assert unseeded.median_diameter_um == fixed.median_diameter_um


def test_a_seed_at_or_below_the_threshold_raises():
    with pytest.raises(ValueError, match="seed"):
        evaluate_threshold(_tube(), 0.95, VOXEL, seed=0.95)


def _with_network(sample, d_net):
    return replace(sample, median_network_diameter_um=d_net)


# WKY-B-like: the plain cut enters the window at 0.93, the network mask never does uncorrected.
_NETWORK_SWEEP = [
    _with_network(_sample(0.90, 7.46, 4.0), 8.35),
    _with_network(_sample(0.93, 5.27, 4.0), 7.46),
    _with_network(_sample(0.95, 5.27, 4.0), 7.46),
    _with_network(_sample(0.97, 3.73, 4.0), 7.40),
]


def test_the_default_calibre_is_the_plain_cut_median():
    """Passing the default reader explicitly changes nothing: the plain cut still selects."""
    default = select_threshold(_NETWORK_SWEEP)
    explicit = select_threshold(_NETWORK_SWEEP, calibre=lambda s: s.median_diameter_um)
    assert default.threshold == explicit.threshold == 0.95
    assert default.calibre_window == explicit.calibre_window == (0.93, 0.95)


def test_a_different_calibre_moves_the_window():
    uncorrected = select_threshold(
        _NETWORK_SWEEP, calibre=lambda s: s.median_network_diameter_um)
    assert uncorrected.threshold is None
    assert "No threshold reaches capillary calibre" in uncorrected.reason
    assert "7.40-8.35" in uncorrected.reason


def test_the_network_check_takes_one_mean_voxel_off_d_net():
    """d_net less one voxel (the EDT half-step bias) re-enters the window."""
    check = select_on_network_calibre(_NETWORK_SWEEP, VOXEL)
    pitch = float(np.mean(VOXEL))
    expected = tuple(s.threshold for s in _NETWORK_SWEEP
                     if 4.0 <= s.median_network_diameter_um - pitch <= 7.0)
    assert check.calibre_window == expected == (0.90, 0.93, 0.95, 0.97)
    assert check.threshold == 0.97
    # A scalar voxel works the same way.
    assert select_on_network_calibre(_NETWORK_SWEEP, pitch).threshold == 0.97


def test_the_network_check_leaves_the_selection_alone():
    """The check is a second call, not a change to the first."""
    select_on_network_calibre(_NETWORK_SWEEP, VOXEL)
    assert select_threshold(_NETWORK_SWEEP).threshold == 0.95

