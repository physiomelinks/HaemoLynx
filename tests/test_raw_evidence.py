"""``preprocessing.raw_evidence``: the raw image read where the mask is silent.

Synthetic volumes with known answers: a bright tube broken by a dark gap, two
tubes with a dark wall between them, a loop round a pinhole and one round
tissue -- each drawn as a plasma (lumen-bright) label and as an endothelial
(wall-bright) one, since both stains must answer the same questions. The
memory-map test pins that a question reads windows, never the volume.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from haemolynx.io.crop import CropBox
from haemolynx.pipeline import default_schema, stages
from haemolynx.preprocessing import raw_evidence as module
from haemolynx.preprocessing.raw_evidence import (
    BRIGHT_THRESHOLD,
    ENDOTHELIAL,
    GRAPH_ATTRIBUTE,
    PLASMA_COLUMN,
    RawCalibration,
    RawEvidenceEndothelial,
    RawEvidencePlasma,
    calibrate_raw,
    make_raw_evidence,
    raw_evidence_from_settings,
    raw_image_for_build,
)

VOXEL = (1.0, 1.0, 1.0)
SHAPE = (24, 40, 60)
BACKGROUND = 50.0
BRIGHT = 1000.0


def _radial(shape, centre_zy):
    zz, yy, _xx = np.indices(shape)
    return np.sqrt((zz - centre_zy[0]) ** 2 + (yy - centre_zy[1]) ** 2)


def _plasma_tube(shape=SHAPE, centre_zy=(12, 20), radius=3.0, gap_x=None):
    """A lumen-bright tube along x (and its mask); *gap_x* darkens a run of x."""
    r = _radial(shape, centre_zy)
    mask = r <= radius
    raw = np.where(mask, BRIGHT, BACKGROUND)
    if gap_x is not None:
        raw[:, :, gap_x[0]:gap_x[1]] = BACKGROUND
    return raw, mask


def _endothelial_tube(shape=SHAPE, centre_zy=(12, 20), radius=3.0, wall=1.5, gap_x=None):
    """A wall-bright tube along x: the ring between *radius* and *radius* +
    *wall* is bright, the lumen and the tissue dark; the mask is the filled
    vessel out to the wall's outer face."""
    r = _radial(shape, centre_zy)
    ring = (r > radius) & (r <= radius + wall)
    mask = r <= radius + wall
    raw = np.where(ring, BRIGHT, BACKGROUND)
    if gap_x is not None:
        raw[:, :, gap_x[0]:gap_x[1]] = BACKGROUND
    return raw, mask


def _noisy(raw, seed=0, sd=10.0):
    rng = np.random.default_rng(seed)
    return np.clip(raw + rng.normal(0.0, sd, raw.shape), 0.0, None)


def _evidence(raw, mask, stain="auto"):
    calibration = calibrate_raw(raw, mask, VOXEL, stain=stain)
    assert calibration is not None
    return make_raw_evidence(raw, mask.shape, VOXEL, calibration)


def _x_path(z, y, x0, x1):
    return np.array([[z, y, x0], [z, y, x1]], dtype=float)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_calibration_reads_a_plasma_label():
    raw, mask = _plasma_tube()
    calibration = calibrate_raw(_noisy(raw), mask, VOXEL)
    assert calibration is not None
    assert calibration.stain == PLASMA_COLUMN
    assert calibration.vessel == pytest.approx(BRIGHT, abs=30)
    assert calibration.background == pytest.approx(BACKGROUND, abs=20)
    assert calibration.contrast > 0
    assert calibration.blocks >= 1


def test_calibration_reads_an_endothelial_label():
    raw, mask = _endothelial_tube()
    calibration = calibrate_raw(_noisy(raw), mask, VOXEL)
    assert calibration is not None
    assert calibration.stain == ENDOTHELIAL
    assert calibration.wall > calibration.vessel
    assert calibration.level == calibration.wall


def test_calibration_takes_the_stain_it_is_told_when_that_level_has_signal():
    raw, mask = _endothelial_tube()
    told = calibrate_raw(_noisy(raw), mask, VOXEL, stain=ENDOTHELIAL)
    assert told is not None and told.stain == ENDOTHELIAL
    # Told the lumen is the label when the lumen is dark: nothing to read.
    assert calibrate_raw(_noisy(raw), mask, VOXEL, stain=PLASMA_COLUMN) is None
    with pytest.raises(ValueError, match="stain must be one of"):
        calibrate_raw(raw, mask, VOXEL, stain="nuclear")


def test_calibration_refuses_an_image_without_contrast():
    _raw, mask = _plasma_tube()
    flat = np.full(SHAPE, BACKGROUND)
    assert calibrate_raw(flat, mask, VOXEL) is None
    assert calibrate_raw(_noisy(flat), mask, VOXEL) is None


def test_calibration_refuses_a_mismatched_shape():
    raw, mask = _plasma_tube()
    with pytest.raises(ValueError, match="differ in shape"):
        calibrate_raw(raw[:-1], mask, VOXEL)


def test_calibration_round_trips_through_its_record_and_a_stain_reading():
    raw, mask = _plasma_tube()
    calibration = calibrate_raw(raw, mask, VOXEL)
    assert RawCalibration.from_record(calibration.as_dict()) == calibration

    class Reading:  # the shape of raw_stain.StainReading
        stain, core, wall, background, background_spread, contrast, blocks = (
            PLASMA_COLUMN, 900.0, 400.0, 40.0, 5.0, 0.4, 16,
        )

    from_reading = RawCalibration.from_stain_reading(Reading())
    assert (from_reading.vessel, from_reading.wall, from_reading.background) == (900.0, 400.0, 40.0)
    assert from_reading.stain == PLASMA_COLUMN
    with pytest.raises(ValueError, match="names no stain"):
        Reading.stain = None
        RawCalibration.from_stain_reading(Reading())


# --------------------------------------------------------------------------
# signal_along
# --------------------------------------------------------------------------


@pytest.mark.parametrize("make", [_plasma_tube, _endothelial_tube])
def test_signal_along_a_vessel_with_a_gap(make):
    raw, mask = make(gap_x=(30, 33))
    evidence = _evidence(_noisy(raw), mask)
    signal = evidence.signal_along(_x_path(12, 20, 5, 55), radius_um=3.0)
    assert signal.samples > 50
    assert signal.longest_dark_um == pytest.approx(3.0, abs=1.5)
    assert signal.fraction_bright == pytest.approx(1 - 3.0 / 50.0, abs=0.06)


@pytest.mark.parametrize("make", [_plasma_tube, _endothelial_tube])
def test_signal_along_an_unbroken_vessel_and_through_background(make):
    raw, mask = make()
    evidence = _evidence(_noisy(raw), mask)
    whole = evidence.signal_along(_x_path(12, 20, 5, 55), radius_um=3.0)
    assert whole.fraction_bright == pytest.approx(1.0, abs=0.02)
    assert whole.longest_dark_um == 0.0
    tissue = evidence.signal_along(_x_path(3, 5, 5, 55), radius_um=3.0)
    assert tissue.fraction_bright == 0.0
    assert tissue.longest_dark_um == pytest.approx(50.0, abs=1.0)


def test_signal_along_outside_the_image_is_background():
    raw, mask = _plasma_tube()
    evidence = _evidence(raw, mask)
    beyond = evidence.signal_along(_x_path(12, 20, -20, -5), radius_um=3.0)
    assert beyond.fraction_bright == 0.0
    assert evidence.signal_along(np.zeros((0, 3)), radius_um=3.0).samples == 0


# --------------------------------------------------------------------------
# wall_between
# --------------------------------------------------------------------------


def test_wall_between_two_plasma_lumens_but_not_within_one():
    r = _radial(SHAPE, (12, 16))
    r2 = _radial(SHAPE, (12, 24))
    two = (r <= 3.0) | (r2 <= 3.0)  # centres 8 apart, radius 3: a two-voxel wall between
    raw = np.where(two, BRIGHT, BACKGROUND)
    evidence = _evidence(_noisy(raw), two)
    assert evidence.wall_between([12, 16, 30], [12, 24, 30]) >= 0.5

    one = _radial(SHAPE, (12, 20)) <= 5.0
    raw_one = np.where(one, BRIGHT, BACKGROUND)
    evidence_one = _evidence(_noisy(raw_one), one)
    assert evidence_one.wall_between([12, 17, 30], [12, 23, 30]) <= 0.1
    assert evidence_one.wall_between([3, 5, 30], [3, 10, 30]) == 0.0  # ends in tissue: no evidence


def test_wall_between_two_endothelial_lumens_but_not_within_one():
    r = _radial(SHAPE, (12, 14))
    r2 = _radial(SHAPE, (12, 26))
    ring = ((r > 4.0) & (r <= 5.5)) | ((r2 > 4.0) & (r2 <= 5.5))
    mask = (r <= 5.5) | (r2 <= 5.5)
    raw = np.where(ring, BRIGHT, BACKGROUND)
    evidence = _evidence(_noisy(raw), mask)
    assert isinstance(evidence, RawEvidenceEndothelial)
    assert evidence.wall_between([12, 14, 30], [12, 26, 30]) >= 0.5

    one = _radial(SHAPE, (12, 20))
    raw_one = np.where((one > 7.0) & (one <= 8.5), BRIGHT, BACKGROUND)
    evidence_one = _evidence(_noisy(raw_one), one <= 8.5)
    assert evidence_one.wall_between([12, 16, 30], [12, 24, 30]) <= 0.1


# --------------------------------------------------------------------------
# hole_is_lumen
# --------------------------------------------------------------------------


def _square_loop(z, y0, x0, side):
    return np.array(
        [[z, y0, x0], [z, y0 + side, x0], [z, y0 + side, x0 + side], [z, y0, x0 + side], [z, y0, x0]],
        dtype=float,
    )


def test_a_plasma_pinhole_is_lumen_and_a_tissue_hole_is_not():
    slab = np.zeros(SHAPE, dtype=bool)
    slab[10:15, 5:35, 10:50] = True  # a flat bright sheet of lumen
    raw = np.where(slab, BRIGHT, BACKGROUND)
    loop = _square_loop(12, 12, 20, 14)
    evidence = _evidence(_noisy(raw), slab)
    assert evidence.hole_is_lumen(loop, radius_um=2.0) >= 0.8

    hole = slab.copy()
    hole[:, 14:25, 22:33] = False  # tissue inside the loop, the loop's vessels round it
    raw_hole = np.where(hole, BRIGHT, BACKGROUND)
    assert _evidence(_noisy(raw_hole), hole).hole_is_lumen(loop, radius_um=2.0) <= 0.2


def test_an_endothelial_hole_lined_by_wall_is_tissue():
    # Outer walls of a flat vessel, so the stain calibrates; a loop drawn
    # round a hole lined by a wall ring is round tissue, one round a dark
    # interior with no wall is a pinhole.
    from scipy.ndimage import binary_erosion

    slab = np.zeros(SHAPE, dtype=bool)
    slab[8:17, 4:36, 9:51] = True
    outer = slab & ~binary_erosion(slab, iterations=2)  # a two-voxel wall round the vessel
    raw = np.where(outer, BRIGHT, BACKGROUND)
    loop = _square_loop(12, 12, 20, 14)
    evidence = _evidence(_noisy(raw), slab)
    assert isinstance(evidence, RawEvidenceEndothelial)
    assert evidence.hole_is_lumen(loop, radius_um=2.0) >= 0.7

    lined = raw.copy()
    lined[11:14, 14:25, 22:33] = BRIGHT  # a two-voxel wall ring just inside the loop
    lined[11:14, 16:23, 24:31] = BACKGROUND
    assert _evidence(_noisy(lined), slab).hole_is_lumen(loop, radius_um=2.0) <= 0.3


def test_a_hole_too_small_to_judge_is_nan():
    raw, mask = _plasma_tube()
    evidence = _evidence(raw, mask)
    assert np.isnan(evidence.hole_is_lumen(_square_loop(12, 19, 30, 2), radius_um=3.0))


# --------------------------------------------------------------------------
# Reading: windows, memory maps, crops, stats
# --------------------------------------------------------------------------


class _Watched:
    """A volume that records every slice read of it."""

    def __init__(self, array):
        self.array = array
        self.reads: list[int] = []

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, key):
        block = self.array[key]
        self.reads.append(int(np.size(block)))
        return block


def test_a_memory_mapped_raw_image_is_read_in_windows(tmp_path):
    raw, mask = _plasma_tube(gap_x=(30, 33))
    raw = _noisy(raw)
    path = tmp_path / "raw.memmap"
    stored = np.memmap(path, dtype=np.uint16, mode="w+", shape=SHAPE)
    stored[:] = raw.astype(np.uint16)
    stored.flush()
    watched = _Watched(np.memmap(path, dtype=np.uint16, mode="r", shape=SHAPE))

    in_ram = _evidence(raw.astype(np.uint16).astype(np.float32), mask)
    calibration = calibrate_raw(watched, mask, VOXEL)
    watched.reads.clear()  # the calibration's block reads are not a question's
    on_disk = make_raw_evidence(watched, mask.shape, VOXEL, calibration)
    path_um = _x_path(12, 20, 5, 55)
    assert on_disk.signal_along(path_um, 3.0) == in_ram.signal_along(path_um, 3.0)
    assert on_disk.wall_between([12, 20, 10], [12, 20, 20]) == in_ram.wall_between([12, 20, 10], [12, 20, 20])
    assert watched.reads and max(watched.reads) < raw.size / 4
    assert on_disk.stats.queries == len(watched.reads)
    assert on_disk.stats.voxels_read == sum(watched.reads)
    assert on_disk.stats.seconds > 0


def test_a_long_path_is_read_in_pieces(monkeypatch):
    raw, mask = _plasma_tube()
    evidence = _evidence(raw, mask)
    monkeypatch.setattr(module, "MAX_WINDOW_VOXELS", 200)
    before = evidence.stats.queries
    signal = evidence.signal_along(_x_path(12, 20, 5, 55), radius_um=3.0)
    assert evidence.stats.queries - before > 1
    assert signal.fraction_bright == pytest.approx(1.0, abs=0.02)


def test_a_crop_box_frames_the_raw_image_as_a_view():
    whole_shape = (30, 50, 80)
    raw, mask = _plasma_tube(whole_shape, centre_zy=(15, 25))
    box = CropBox((3, 5, 10), (27, 45, 70), whole_shape)
    cropped_mask = mask[box.slices]
    calibration = calibrate_raw(raw[box.slices], cropped_mask, VOXEL)
    evidence = make_raw_evidence(raw, cropped_mask.shape, VOXEL, calibration, box=box)
    assert evidence.shape == box.shape_zyx
    assert np.shares_memory(evidence.raw, raw)
    # Positions are in the cropped frame: the tube's centre is (12, 20) there.
    assert evidence.brightness(np.array([[12.0, 20.0, 30.0]]))[0] == pytest.approx(1.0, abs=0.02)
    assert evidence.brightness(np.array([[12.0, 20.0, 30.0]]) + np.array([3, 5, 10]))[0] < 0.5
    with pytest.raises(ValueError, match="not the image's"):
        make_raw_evidence(raw[box.slices], cropped_mask.shape, VOXEL, calibration, box=box)
    with pytest.raises(ValueError, match="not the mask's"):
        make_raw_evidence(raw, mask.shape, VOXEL, calibration, box=box)


def test_a_misaligned_raw_image_is_refused():
    raw, mask = _plasma_tube()
    calibration = calibrate_raw(raw, mask, VOXEL)
    with pytest.raises(ValueError, match="not the mask's"):
        make_raw_evidence(raw[:, :-1], mask.shape, VOXEL, calibration)
    with pytest.raises(ValueError, match="does not stand above"):
        RawEvidencePlasma(raw, VOXEL, RawCalibration(PLASMA_COLUMN, 50.0, 50.0, 50.0, 1.0, 0.0, 1))


def test_anisotropic_voxels_place_points_in_microns():
    raw, mask = _plasma_tube()
    voxel = (2.0, 0.5, 0.5)
    calibration = calibrate_raw(raw, mask, voxel)
    evidence = make_raw_evidence(raw, mask.shape, voxel, calibration)
    centre_um = np.array([[12 * 2.0, 20 * 0.5, 30 * 0.5]])
    assert evidence.brightness(centre_um)[0] == pytest.approx(1.0, abs=0.02)
    assert evidence.brightness(centre_um + [0, 0, 10.0])[0] == pytest.approx(1.0, abs=0.02)
    assert evidence.brightness(centre_um + [0, 5.0, 0])[0] < BRIGHT_THRESHOLD


# --------------------------------------------------------------------------
# From a run's settings
# --------------------------------------------------------------------------


def test_raw_image_for_build_reads_either_naming():
    assert raw_image_for_build({"fwhm_raw_tiff_path": None}) is None
    spec = raw_image_for_build({"fwhm_raw_tiff_path": "a.tif", "fwhm_raw_channel": 1})
    assert (str(spec.path), spec.channel, spec.stain, spec.setting) == ("a.tif", 1, "auto", "fwhm_raw_tiff_path")
    spec = raw_image_for_build(
        {"raw_image_path": "b.tif", "raw_image_channel": None, "raw_image_stain": ENDOTHELIAL,
         "fwhm_raw_tiff_path": "a.tif"}
    )
    assert (str(spec.path), spec.channel, spec.stain, spec.setting) == ("b.tif", None, ENDOTHELIAL, "raw_image_path")
    spec = raw_image_for_build({"endothelial_image_path": "c.tif", "endothelial_channel": 0})
    assert (spec.stain, spec.channel) == (ENDOTHELIAL, 0)


def _run_settings(tmp_path, **overrides):
    shape = (12, 30, 60)
    mask = np.zeros(shape, dtype=bool)
    mask[5:8, 13:17, 5:55] = True
    raw = np.where(mask, BRIGHT, BACKGROUND).astype(np.uint16)
    tifffile.imwrite(tmp_path / "mask.tif", mask.astype(np.uint8) * 255)
    tifffile.imwrite(tmp_path / "raw.tif", raw)
    schema = default_schema()
    settings = schema.defaults()
    settings.update(
        {
            "input_path": str(tmp_path / "mask.tif"),
            "vtk_output_prefix": tmp_path / "run" / "network",
            "plot_dir": tmp_path / "plots",
            "visualize_results": False,
            **overrides,
        }
    )
    return settings, schema


def test_build_network_keeps_the_calibration_on_the_graph(tmp_path):
    settings, schema = _run_settings(tmp_path, fwhm_raw_tiff_path=str(tmp_path / "raw.tif"))
    network = stages.build_network(settings, stages.skeletonise(settings, stages.segment(settings)), schema)
    record = network.graph.graph[GRAPH_ATTRIBUTE]
    calibration = RawCalibration.from_record(record)
    assert calibration.stain == PLASMA_COLUMN
    assert calibration.vessel == pytest.approx(BRIGHT, abs=30)
    assert calibration.background == pytest.approx(BACKGROUND, abs=20)


def test_build_network_without_a_raw_image_records_nothing(tmp_path):
    settings, schema = _run_settings(tmp_path)
    network = stages.build_network(settings, stages.skeletonise(settings, stages.segment(settings)), schema)
    assert GRAPH_ATTRIBUTE not in network.graph.graph


def test_build_network_refuses_a_raw_image_of_another_shape(tmp_path):
    settings, schema = _run_settings(tmp_path)
    tifffile.imwrite(tmp_path / "other.tif", np.full((12, 30, 61), BACKGROUND, dtype=np.uint16))
    settings["fwhm_raw_tiff_path"] = str(tmp_path / "other.tif")
    with pytest.raises(ValueError, match="same stack the segmentation was made from"):
        stages.build_network(settings, stages.skeletonise(settings, stages.segment(settings)), schema)


def test_raw_evidence_from_settings_reads_a_channel_of_a_composite(tmp_path):
    raw, mask = _plasma_tube()
    other = np.full(SHAPE, BACKGROUND)
    composite = np.stack([other, raw], axis=1).astype(np.uint16)  # (z, c, y, x), as ImageJ writes
    tifffile.imwrite(tmp_path / "composite.tif", composite, imagej=True, metadata={"axes": "ZCYX"})
    settings = {"raw_image_path": str(tmp_path / "composite.tif"), "raw_image_channel": 1}
    evidence = raw_evidence_from_settings(settings, mask, VOXEL)
    assert isinstance(evidence, RawEvidencePlasma)
    assert raw_evidence_from_settings({**settings, "raw_image_channel": 0}, mask, VOXEL) is None
