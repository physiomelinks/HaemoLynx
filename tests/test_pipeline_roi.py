"""The network pipeline crops the placed ROI, and says which box it cut (open item 27).

Until item 27, ``cb_h1_batch.py --stage run`` passed ``--roi-voxels`` and no offsets, so every
network was built on the array-centre box while the threshold stage, the TH metrics and the
H2 overlay used ``place_roi``'s box, 85-210 um away. The pipeline now places the ROI itself,
writes ``roi_placement.json``, and every driver that reads a batch output checks it.
"""
import json
from types import SimpleNamespace

import pytest

from ImageLynx.preprocessing import crop_roi
from ImageLynx.roi_placement import (
    ROI_RECORD_NAME,
    RoiPlacement,
    centre_to_offsets,
    centred_placement,
    check_output_roi,
    roi_record,
    write_roi_record,
)
from ImageLynx.specimens import SPECIMENS

C = pytest.importorskip("carotid_image_to_model")

SIZE = (160, 160, 160)


class _Probe:
    """Stands in for a volume: records the slices crop_roi takes, without allocating it."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(shape)

    def __getitem__(self, slices):
        return SimpleNamespace(slices=slices,
                               shape=tuple(s.stop - s.start for s in slices))


def _placement(specimen_id, centre, shape, source="z=qc_peak_slice, yx=grayscale_centroid"):
    return RoiPlacement(specimen_id=specimen_id, centre_zyx=tuple(centre), size_zyx=SIZE,
                        offsets_zyx=centre_to_offsets(centre, shape), peak_slice=centre[0],
                        source=source)


def _fake_specimen(specimen_id="WKY-A", shape=(435, 456, 507)):
    return SimpleNamespace(specimen_id=specimen_id, shape_zyx=shape)


@pytest.mark.parametrize("specimen", SPECIMENS, ids=lambda s: s.specimen_id)
def test_the_offsets_the_pipeline_sets_crop_exactly_the_placed_bounds(specimen):
    """For every legal centre on every axis of every specimen's shape, not just today's."""
    shape = specimen.shape_zyx
    for axis in range(3):
        for c in range(SIZE[axis] // 2, shape[axis] - SIZE[axis] // 2 + 1):
            centre = [e // 2 for e in shape]
            centre[axis] = c
            placement = _placement(specimen.specimen_id, centre, shape)
            oz, oy, ox = placement.offsets_zyx
            cut = crop_roi(_Probe(shape), offset_z=oz, offset_y=oy, offset_x=ox,
                           size_zyx=SIZE)
            assert cut.slices == placement.bounds, (specimen.specimen_id, axis, c)


def test_the_pipeline_uses_place_roi_when_a_size_is_given(monkeypatch):
    import ImageLynx.roi_placement as rp

    specimen = _fake_specimen()
    placed = _placement("WKY-A", (230, 240, 188), specimen.shape_zyx)
    monkeypatch.setattr(rp, "place_roi", lambda s, size: placed)
    config = C.SkeletonConfig()
    config.sub_volume_voxels = SIZE

    record = C._apply_roi_placement(config, specimen)

    assert (config.sub_volume_offset_z, config.sub_volume_offset_y,
            config.sub_volume_offset_x) == placed.offsets_zyx
    assert record["bounds_zyx"] == [[s.start, s.stop] for s in placed.bounds]
    assert record["centred"] is False


def test_roi_centred_is_the_array_centre_and_is_recorded(monkeypatch):
    import ImageLynx.roi_placement as rp

    def _must_not_place(*_):
        raise AssertionError("--roi-centred must not call place_roi")

    monkeypatch.setattr(rp, "place_roi", _must_not_place)
    specimen = _fake_specimen()
    config = C.SkeletonConfig()
    config.sub_volume_voxels = SIZE

    record = C._apply_roi_placement(config, specimen, centred=True)

    assert record["centred"] is True
    assert record["centre_zyx"] == [e // 2 for e in specimen.shape_zyx]
    cut = crop_roi(_Probe(specimen.shape_zyx), offset_z=config.sub_volume_offset_z,
                   offset_y=config.sub_volume_offset_y, offset_x=config.sub_volume_offset_x,
                   size_zyx=SIZE)
    # The same box crop_roi cut with zero offsets, which is what the batch ran until item 27.
    zero = crop_roi(_Probe(specimen.shape_zyx), size_zyx=SIZE)
    assert cut.slices == zero.slices == centred_placement(specimen, SIZE).bounds


def test_a_placement_that_fell_back_to_the_centre_raises(monkeypatch):
    import ImageLynx.roi_placement as rp

    specimen = _fake_specimen()
    fallback = _placement("WKY-A", (217, 228, 253), specimen.shape_zyx,
                          source="z=qc_peak_slice, yx=volume_centre (absent)")
    monkeypatch.setattr(rp, "place_roi", lambda s, size: fallback)
    config = C.SkeletonConfig()
    config.sub_volume_voxels = SIZE
    with pytest.raises(ValueError, match="fell back to the volume centre"):
        C._apply_roi_placement(config, specimen)


def test_hand_set_offsets_with_a_size_raise():
    config = C.SkeletonConfig()
    config.sub_volume_voxels = SIZE
    config.sub_volume_offset_y = 0.1
    with pytest.raises(ValueError, match="sub_volume_offset_zyx"):
        C._apply_roi_placement(config, _fake_specimen())


def test_no_explicit_size_leaves_the_percentage_crop_alone():
    config = C.SkeletonConfig()
    assert config.sub_volume_voxels is None
    assert C._apply_roi_placement(config, _fake_specimen()) is None
    assert (config.sub_volume_offset_z, config.sub_volume_offset_y,
            config.sub_volume_offset_x) == (0.0, 0.0, 0.0)


# --- The sidecar guard ---------------------------------------------------------------------

def test_an_output_without_a_sidecar_is_refused(tmp_path):
    specimen = _fake_specimen()
    placed = _placement("WKY-A", (230, 240, 188), specimen.shape_zyx)
    with pytest.raises(FileNotFoundError, match="predate open item 27"):
        check_output_roi(tmp_path, specimen, SIZE, placed)


def test_a_matching_sidecar_passes(tmp_path):
    specimen = _fake_specimen()
    placed = _placement("WKY-A", (230, 240, 188), specimen.shape_zyx)
    write_roi_record(tmp_path, roi_record(placed, specimen.shape_zyx, centred=False))
    record = check_output_roi(tmp_path, specimen, SIZE, placed)
    assert record["centre_zyx"] == [230, 240, 188]


def test_a_centred_output_is_refused(tmp_path):
    specimen = _fake_specimen()
    centred = centred_placement(specimen, SIZE)
    write_roi_record(tmp_path, roi_record(centred, specimen.shape_zyx, centred=True))
    with pytest.raises(ValueError, match="array centre"):
        check_output_roi(tmp_path, specimen, SIZE, centred)


def test_an_output_from_another_placement_is_refused(tmp_path):
    """E.g. a run from before item 13 moved the lateral centroid."""
    specimen = _fake_specimen()
    old = _placement("WKY-A", (230, 240, 188), specimen.shape_zyx)
    new = _placement("WKY-A", (230, 250, 180), specimen.shape_zyx)
    write_roi_record(tmp_path, roi_record(old, specimen.shape_zyx, centred=False))
    with pytest.raises(ValueError, match="placement has changed"):
        check_output_roi(tmp_path, specimen, SIZE, new)


def test_an_output_for_another_specimen_is_refused(tmp_path):
    specimen = _fake_specimen()
    placed = _placement("WKY-B", (106, 198, 174), specimen.shape_zyx)
    write_roi_record(tmp_path, roi_record(placed, specimen.shape_zyx, centred=False))
    with pytest.raises(ValueError, match="is for WKY-B"):
        check_output_roi(tmp_path, specimen, SIZE, placed)


def test_the_sidecar_is_plain_json_with_the_box(tmp_path):
    specimen = _fake_specimen()
    placed = _placement("WKY-A", (230, 240, 188), specimen.shape_zyx)
    path = write_roi_record(tmp_path, roi_record(placed, specimen.shape_zyx, centred=False))
    assert path.name == ROI_RECORD_NAME
    data = json.loads(path.read_text())
    assert data["bounds_zyx"] == [[150, 310], [160, 320], [108, 268]]
    assert data["volume_shape_zyx"] == [435, 456, 507]
