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

from conftest import write_ilp_project
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
    lanes = []
    for stem, by_z in slices_by_lane.items():
        blocks = []
        for z, (vessel, background) in sorted(by_z.items()):
            extent = 1 + max(max((c for c, _ in vessel + background), default=0),
                             max((c for _, c in vessel + background), default=0))
            labels = np.zeros((extent, extent), dtype=np.uint8)
            for y, x in vessel:
                labels[y, x] = 1
            for y, x in background:
                labels[y, x] = 2
            blocks.append((z, labels))
        lanes.append((f"{stem}_ilastik.h5", blocks))
    write_ilp_project(path, lanes)


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


def test_placement_is_measured_without_any_prediction(tmp_path, monkeypatch):
    """The point of the check: it runs in the GUI loop, on the .ilp alone.

    A prediction-based version would be more accurate - distance to the nearest labelled
    vessel overstates distance to the nearest real one, because most vessels are unlabelled -
    but it would cost a headless run per iteration and could not be used while labelling.
    """
    from pathlib import Path

    import h5py

    import ImageLynx.io
    import ImageLynx.io.load

    path = tmp_path / "labels_only.ilp"
    _write_project(path, {s.preproc_stem: _lane(20, 23) for s in SPECIMENS})

    # Recorded as well as raised, so a read swallowed by a broad except still fails the test.
    reads = []
    open_h5 = h5py.File

    def only_the_project(name, *args, **kwargs):
        if Path(name).resolve() != path.resolve():
            reads.append(str(name))
            raise AssertionError(f"opened {name}, which is not the .ilp")
        return open_h5(name, *args, **kwargs)

    def no_prediction(*args, **kwargs):
        reads.append("read_ilastik_probabilities")
        raise AssertionError("read a prediction")

    monkeypatch.setattr(h5py, "File", only_the_project)
    monkeypatch.setattr(ImageLynx.io, "read_ilastik_probabilities", no_prediction)
    monkeypatch.setattr(ImageLynx.io.load, "read_ilastik_probabilities", no_prediction)

    rows = analyse_label_placement(path)

    assert reads == []
    assert len(rows) == len(SPECIMENS)


@pytest.mark.realdata
def test_runs_against_the_real_project_if_present():
    from ImageLynx.specimens import POOLED_CLASSIFIER

    if not POOLED_CLASSIFIER.exists():
        pytest.skip("no trained classifier present")
    rows = analyse_label_placement()
    assert len(rows) == len(SPECIMENS)
    assert all(0.0 <= r.background_within_band_fraction <= 1.0 for r in rows)
