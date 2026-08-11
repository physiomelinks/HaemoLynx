"""Where the labels sit relative to the boundary the classifier has to place.

Tripling the labels on WKY-C - 4488 to 13504 voxels, background up 3.9x, depths 2 to 4 -
left the prediction unchanged to four decimal places, and the uncertain band slightly worse.
The reason was not how many labels there were but where:

    WKY-C                  n        p10    p25  median    p75    p90   in 2-10um band
    vessel labels       8217       0.00   0.00    0.00   0.00   2.64             6.0%
    background labels   5287       5.60  11.04   26.95  45.16  64.37            19.8%
    uncertain voxels   17.5M       1.87   3.73    8.13  16.05  29.21            41.5%

Distances are to the nearest confidently-predicted vessel. The classifier was shown vessel
cores and far-field emptiness and almost nothing between, so in the band where the decision
boundary actually lies it has no evidence and returns about 0.5 - across half the volume.
Nothing tells it where a vessel ends, which is also why the masks come out roughly twice
capillary calibre at every usable threshold.

Measured from the labels alone, with no prediction involved, the same gap is in all six:
between 1.0% and 8.6% of background labels lie within 9.33 um of a vessel label.

This check exists so that gap is visible in the two seconds after a save, rather than after a
relabel-predict-measure round trip.
"""
import numpy as np
import pytest

from ImageLynx.specimens import SPECIMENS
from ImageLynx.statistics.label_placement import (
    BOUNDARY_BAND_UM,
    MIN_BOUNDARY_FRACTION,
    analyse_label_placement,
    format_placement_table,
)


def _write_project(path, slices_by_lane):
    """Build a minimal .ilp whose label blocks carry real 2D geometry.

    ``slices_by_lane`` maps a specimen's preproc_stem to {z: (vessel_yx, background_yx)},
    each a list of (y, x) positions, so a test can place background at a chosen distance from
    vessel rather than merely counting it.
    """
    import h5py

    with h5py.File(path, "w") as project:
        project["PixelClassification/LabelNames"] = np.array([b"vessel", b"background"],
                                                             dtype="S32")
        for position, (stem, by_z) in enumerate(slices_by_lane.items()):
            lane = f"lane{position:04d}"
            project[f"Input Data/infos/{lane}/Raw Data/filePath"] = \
                f"{stem}_ilastik.h5/data".encode()
            group = project.create_group(
                f"PixelClassification/LabelSets/labels{position:03d}")
            for index, (z, (vessel, background)) in enumerate(sorted(by_z.items())):
                extent = 1 + max(max((c for c, _ in vessel + background), default=0),
                                 max((c for _, c in vessel + background), default=0))
                block = np.zeros((1, extent, extent, 1), dtype=np.uint8)
                for y, x in vessel:
                    block[0, y, x, 0] = 1
                for y, x in background:
                    block[0, y, x, 0] = 2
                dataset = group.create_dataset(f"block{index:04d}", data=block)
                dataset.attrs["blockSlice"] = \
                    f"[{z}:{z + 1},0:{extent},0:{extent},0:1]".encode()


def _lane(vessel_x=20, background_x=23):
    """One lane: a vertical vessel stroke, and a background stroke at a chosen offset."""
    return {
        z: ([(y, vessel_x) for y in range(5, 35)],
            [(y, background_x) for y in range(5, 35)])
        for z in (100, 200, 300)
    }


def test_background_labelled_beside_the_vessel_passes(tmp_path):
    """3 voxels out is inside the band where the decision boundary lives."""
    path = tmp_path / "adjacent.ilp"
    _write_project(path, {s.preproc_stem: _lane(20, 23) for s in SPECIMENS})

    rows = analyse_label_placement(path)
    assert len(rows) == len(SPECIMENS)
    for row in rows:
        assert row.background_within_band_fraction == pytest.approx(1.0)
        assert row.ok, row


def test_background_labelled_in_the_far_field_fails_and_says_how_far(tmp_path):
    """The real failure: labels on the cores and in empty space, nothing in between."""
    path = tmp_path / "far.ilp"
    _write_project(path, {s.preproc_stem: _lane(20, 200) for s in SPECIMENS})

    rows = analyse_label_placement(path)
    for row in rows:
        assert not row.ok
        assert row.background_median_um > BOUNDARY_BAND_UM[1]
        assert row.background_within_band_fraction < MIN_BOUNDARY_FRACTION


def test_the_band_edges_are_respected(tmp_path):
    """Labels on top of the vessel are not boundary evidence either."""
    path = tmp_path / "onvessel.ilp"
    # Background one voxel from vessel: below the band, since it overlaps the wall itself.
    _write_project(path, {s.preproc_stem: _lane(20, 21) for s in SPECIMENS})

    rows = analyse_label_placement(path)
    lo, hi = BOUNDARY_BAND_UM
    for row in rows:
        assert row.background_median_um < lo


def test_a_slice_missing_one_class_is_skipped_not_counted(tmp_path):
    """Distance is undefined on a slice with no vessel label to measure from."""
    lanes = {}
    for specimen in SPECIMENS:
        by_z = _lane(20, 23)
        by_z[400] = ([], [(y, 40) for y in range(5, 35)])   # background only
        lanes[specimen.preproc_stem] = by_z
    path = tmp_path / "partial.ilp"
    _write_project(path, lanes)

    rows = analyse_label_placement(path)
    for row in rows:
        assert row.slices_measured == 3
        assert row.slices_skipped == 1
        assert row.background_within_band_fraction == pytest.approx(1.0)


def test_the_table_names_the_specimens_and_the_target(tmp_path):
    path = tmp_path / "table.ilp"
    _write_project(path, {s.preproc_stem: _lane(20, 200) for s in SPECIMENS})

    text = format_placement_table(analyse_label_placement(path))
    for specimen in SPECIMENS:
        assert specimen.specimen_id in text
    assert "median" in text
    assert f"{MIN_BOUNDARY_FRACTION:.0%}" in text


def test_placement_is_measured_without_any_prediction(tmp_path):
    """The point of the check: it runs in the GUI loop, on the .ilp alone.

    A prediction-based version would be more accurate - distance to the nearest labelled
    vessel overstates distance to the nearest real one, because most vessels are unlabelled -
    but it would cost a headless run per iteration and could not be used while labelling.
    """
    import inspect

    from ImageLynx.statistics import label_placement

    source = inspect.getsource(label_placement)
    for forbidden in ("read_ilastik_probabilities", "Probabilities", "probabilities_path"):
        assert forbidden not in source


def test_runs_against_the_real_project_if_present():
    from ImageLynx.specimens import POOLED_CLASSIFIER

    if not POOLED_CLASSIFIER.exists():
        pytest.skip("no trained classifier present")
    rows = analyse_label_placement()
    assert len(rows) == len(SPECIMENS)
    assert all(0.0 <= r.background_within_band_fraction <= 1.0 for r in rows)
