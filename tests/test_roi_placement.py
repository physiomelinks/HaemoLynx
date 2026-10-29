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
    clamp_centre,
    place_roi,
    placement_fell_back,
    tissue_centroid_yx,
)
from ImageLynx.specimens import SPECIMENS


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


@pytest.mark.parametrize("shape, peak, tissue_yx, size", [
    ((120, 40, 40), 3, (2, 37), (60, 16, 16)),        # peak at the top, tissue in a corner
    ((121, 41, 45), 118, (38, 1), (61, 17, 17)),      # peak at the bottom, odd extents
    ((99, 37, 50), 49, (18, 25), (60, 17, 16)),       # peak and tissue in the middle
], ids=["top", "bottom", "middle"])
def test_every_specimen_gets_a_placement_that_fits(tmp_path, shape, peak, tissue_yx, size):
    """The box is the size asked for, wholly inside the volume, and centred on its centre.

    The QC peak and the bright tissue sit at the edges in two cases, so the box only fits if
    place_roi pulls it in on every axis.
    """
    from types import SimpleNamespace

    import h5py

    data = np.zeros(shape + (1,), dtype=np.float32)
    y, x = tissue_yx
    data[:, max(y - 2, 0):y + 3, max(x - 2, 0):x + 3, 0] = 1.0
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as handle:
        handle["data"] = data
    specimen = SimpleNamespace(
        specimen_id="TEST", shape_zyx=shape, ilastik_input_path=path,
        qc_record=lambda: {"z_profile": {"peak_slice": peak}},
    )

    placement = place_roi(specimen, size)

    # A fall back to the volume centre would fit on y and x without any pulling in.
    assert not placement_fell_back(placement), placement.source
    assert placement.size_zyx == size
    for axis, (start_stop, extent) in enumerate(zip(placement.bounds, shape)):
        assert start_stop.start >= 0, axis
        assert start_stop.stop <= extent, axis
        assert start_stop.stop - start_stop.start == size[axis], axis
        # Every CB driver slices with .bounds, so the box is exactly centred as well as sized.
        assert (start_stop.start + start_stop.stop) // 2 == placement.centre_zyx[axis], axis


def test_the_lateral_centroid_is_measured_over_the_rois_own_slices(tmp_path):
    """Tissue the box will never contain must not vote on where it goes.

    Bright tissue sits outside the ROI's z-band at one (y, x), and inside it at another. A
    whole-stack projection lands between the two; the band projection lands on the tissue
    the box actually holds.
    """
    from types import SimpleNamespace

    import h5py

    shape = (200, 96, 96)
    data = np.zeros(shape + (3,), dtype=np.float32)
    data[0:40, 8:24, 8:24, 0] = 1.0          # outside the band, and larger
    data[110:170, 56:64, 56:64, 0] = 1.0     # inside the band
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as handle:
        handle["data"] = data

    specimen = SimpleNamespace(
        specimen_id="TEST", shape_zyx=shape, ilastik_input_path=path,
        qc_record=lambda: {"z_profile": {"peak_slice": 140}},
    )
    placement = place_roi(specimen, (80, 32, 32))

    assert placement.centre_zyx[0] == 140
    # A silent fallback to centred placement would reintroduce the bias being removed.
    assert "z=qc_peak_slice" in placement.source
    assert placement.bounds[0] == slice(100, 180)
    assert placement.centre_zyx[1:] == (60, 60)
    assert "over z 100-180" in placement.source

    whole_stack = tissue_centroid_yx(data[::4, ::2, ::2, 0])
    assert abs(whole_stack[0] * 2 - 60) > 10, "the whole-stack centroid must differ, or this proves nothing"
