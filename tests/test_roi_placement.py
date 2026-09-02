"""Placing the sub-volume so the six samples are comparable anatomy, not comparable indices.

A matched ROI size makes the samples the same size; it does not make them the same anatomy.
The stacks differ in depth - 435 slices for WKY, 495 for SHR - so the peak is compared as a
fraction of depth. It ranges from 0.244 (WKY-B) to 0.529 (WKY-A), so a centred box lands
mid-organ in one specimen and in the sparse margin of another, and the resulting density
difference is partly a difference in where the box was put. That scatter is the argument for
per-specimen placement.

The misplacement is also group-correlated, but weakly: WKY means 0.402 against SHR's 0.371,
a gap of 0.031 sitting inside a within-WKY spread of 0.285. The test below asserts only that
the gap exceeds 0.02, which is all three specimens per group can support.
"""
import numpy as np
import pytest

from ImageLynx.roi_placement import (
    centre_to_offsets,
    clamp_centre,
    place_roi,
    tissue_centroid_yx,
)
from ImageLynx.specimens import SPECIMENS, get_specimen


def test_the_axial_tissue_peak_is_not_the_volume_centre_and_differs_by_group():
    """The premise: centring on the array is not centring on the organ."""
    fractions = {}
    for specimen in SPECIMENS:
        record = specimen.qc_record()
        assert record is not None, specimen.specimen_id
        peak = record["z_profile"]["peak_slice"]
        fractions[specimen.specimen_id] = peak / specimen.shape_zyx[0]

    assert any(abs(f - 0.5) > 0.15 for f in fractions.values()), (
        "if every peak were near the middle, placement would not matter"
    )
    wky = [f for s, f in fractions.items() if s.startswith("WKY")]
    shr = [f for s, f in fractions.items() if s.startswith("SHR")]
    assert abs(np.mean(wky) - np.mean(shr)) > 0.02, (
        "the group-correlated part of the misplacement is what makes this load-bearing"
    )


def test_centroid_follows_the_signal_not_the_array():
    volume = np.zeros((10, 100, 100), dtype=np.float32)
    volume[:, 70:80, 20:30] = 1.0
    y, x = tissue_centroid_yx(volume)
    assert 70 <= y < 80 and 20 <= x < 30


def test_centroid_is_not_dragged_to_the_middle_by_background():
    """A plain centre of mass over a background-subtracted volume is mostly background."""
    rng = np.random.default_rng(0)
    volume = rng.random((8, 120, 120)).astype(np.float32) * 0.02
    volume[:, 90:100, 15:25] += 1.0

    y, x = tissue_centroid_yx(volume)
    assert 88 <= y <= 101 and 13 <= x <= 27

    naive_y = np.average(np.arange(120), weights=volume.max(axis=0).sum(axis=1))
    assert abs(naive_y - 60) < abs(y - 60), "the naive centroid should be the one pulled centre"


def test_the_box_is_pulled_inside_the_volume():
    """A box hanging over the edge is silently truncated, unmatching the sample size."""
    assert clamp_centre((5, 5, 5), (40, 40, 40), (100, 100, 100)) == (20, 20, 20)
    assert clamp_centre((95, 95, 95), (40, 40, 40), (100, 100, 100)) == (80, 80, 80)
    assert clamp_centre((50, 50, 50), (40, 40, 40), (100, 100, 100)) == (50, 50, 50)
    # A box larger than the volume falls back to the centre rather than erroring.
    assert clamp_centre((10, 10, 10), (200, 200, 200), (100, 100, 100)) == (50, 50, 50)


def test_offsets_round_trip_to_the_centre_crop_roi_would_use():
    shape = (435, 315, 255)
    centre = (230, 100, 200)
    offsets = centre_to_offsets(centre, shape)
    recovered = tuple(int(round(o * e + e / 2.0)) for o, e in zip(offsets, shape))
    assert recovered == centre


@pytest.mark.parametrize("specimen", SPECIMENS, ids=lambda s: s.specimen_id)
def test_every_specimen_gets_a_placement_that_fits(specimen):
    placement = place_roi(specimen, (160, 160, 160))

    assert placement.size_zyx == (160, 160, 160)
    for start_stop, extent in zip(placement.bounds, specimen.shape_zyx):
        assert start_stop.start >= 0
        assert start_stop.stop <= extent
        assert start_stop.stop - start_stop.start == 160


def test_placement_uses_the_qc_peak_for_z_and_says_where_it_came_from():
    """A silent fallback to centred placement would reintroduce the bias being removed."""
    specimen = get_specimen("WKY-B")
    placement = place_roi(specimen, (160, 160, 160))

    assert placement.peak_slice == 106
    assert "z=qc_peak_slice" in placement.source
    # 106 is closer to the top of the stack than a 160-deep box allows, so it is clamped in -
    # but it must still sit well above the volume centre of 217.
    assert placement.centre_zyx[0] < specimen.shape_zyx[0] // 2


def test_placement_differs_between_specimens():
    """If every ROI landed in the same relative spot there would be nothing to correct."""
    centres = {s.specimen_id: place_roi(s, (160, 160, 160)).centre_zyx for s in SPECIMENS}
    assert len(set(centres.values())) > 1
