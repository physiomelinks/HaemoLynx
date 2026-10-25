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
    # Levels are read under the blur the questions sample with, so a thin
    # tube's lumen reads a little under its peak.
    assert 0.85 * BRIGHT <= calibration.vessel <= BRIGHT
    assert calibration.background == pytest.approx(BACKGROUND, abs=20)
    assert calibration.contrast > 0
    assert calibration.blocks >= 1
    assert calibration.sigma_vox == module.WINDOW_SMOOTHING_SIGMA_VOX


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
    # Told the lumen is the label when the wall is: read as told, with the
    # contrast saying the wall is the brighter.
    as_plasma = calibrate_raw(_noisy(raw), mask, VOXEL, stain=PLASMA_COLUMN)
    assert as_plasma is not None and as_plasma.stain == PLASMA_COLUMN
    assert as_plasma.contrast < 0 and as_plasma.vessel < as_plasma.wall
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
    # 1.5 um across in y: too thin for the stain reader to tell, so named.
    calibration = calibrate_raw(raw, mask, voxel, stain=PLASMA_COLUMN)
    evidence = make_raw_evidence(raw, mask.shape, voxel, calibration)
    centre_um = np.array([[12 * 2.0, 20 * 0.5, 30 * 0.5]])
    assert evidence.brightness(centre_um)[0] == pytest.approx(1.0, abs=0.02)
    assert evidence.brightness(centre_um + [0, 0, 10.0])[0] == pytest.approx(1.0, abs=0.02)
    assert evidence.brightness(centre_um + [0, 5.0, 0])[0] < BRIGHT_THRESHOLD


# --------------------------------------------------------------------------
# From a run's settings
# --------------------------------------------------------------------------


def test_raw_image_for_build_reads_the_one_raw_image():
    """The Input tab's raw image only: the names it replaced are read into it
    when a config loads (Schema.upgrade), so a settings dict never holds them."""
    assert raw_image_for_build({"raw_image_path": None}) is None
    assert raw_image_for_build({"fwhm_raw_tiff_path": "a.tif", "endothelial_image_path": "c.tif"}) is None
    spec = raw_image_for_build({"raw_image_path": "b.tif", "raw_image_channel": 1})
    assert (str(spec.path), spec.channel, spec.stain, spec.setting) == ("b.tif", 1, "auto", "raw_image_path")
    spec = raw_image_for_build(
        {"raw_image_path": "b.tif", "raw_image_channel": None, "raw_image_stain": ENDOTHELIAL}
    )
    assert (str(spec.path), spec.channel, spec.stain) == ("b.tif", None, ENDOTHELIAL)


def test_an_auto_stain_is_told_by_the_one_reader_and_kept_on_the_calibration():
    """``auto`` is answered by ``raw_stain.read_raw_stain`` -- the rule the
    Diameters stage measures by -- and the reading travels with the record."""
    from haemolynx.preprocessing.raw_stain import read_raw_stain

    raw, mask = _endothelial_tube()
    calibration = calibrate_raw(_noisy(raw), mask, VOXEL)
    told = read_raw_stain(_noisy(raw), mask, VOXEL)
    assert calibration.stain == told.stain == ENDOTHELIAL
    assert calibration.reading == told.as_dict()
    assert RawCalibration.from_record(calibration.as_dict()).reading == told.as_dict()
    assert calibrate_raw(_noisy(raw), mask, VOXEL, stain=ENDOTHELIAL).reading is None


def test_an_image_the_reader_cannot_tell_is_not_read():
    """Capillaries too thin to have a middle: the reader says it cannot tell,
    so there is no evidence rather than evidence under a guessed stain."""
    raw, mask = _plasma_tube(radius=1.0)
    assert calibrate_raw(_noisy(raw), mask, VOXEL) is None


def test_a_raw_image_that_is_not_there_is_refused_by_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="raw_image_path"):
        raw_evidence_from_settings(
            {"raw_image_path": str(tmp_path / "gone.tif")}, np.zeros(SHAPE, dtype=bool), VOXEL
        )


def _run_settings(tmp_path, **overrides):
    """A 7 x 7 voxel rod: wide enough to have a middle for the stain reader
    (3 um deep), and odd, since Lee thinning leaves nothing of a 6 x 6 one."""
    shape = (16, 30, 60)
    mask = np.zeros(shape, dtype=bool)
    mask[4:11, 11:18, 5:55] = True
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
    settings, schema = _run_settings(tmp_path, raw_image_path=str(tmp_path / "raw.tif"))
    network = stages.build_network(settings, stages.skeletonise(settings, stages.segment(settings)), schema)
    record = network.graph.graph[GRAPH_ATTRIBUTE]
    calibration = RawCalibration.from_record(record)
    assert calibration.stain == PLASMA_COLUMN
    assert 0.7 * BRIGHT <= calibration.vessel <= BRIGHT  # a 7 x 7 voxel rod, read blurred
    assert calibration.background == pytest.approx(BACKGROUND, abs=20)


def test_build_network_records_an_auto_stain_for_the_diameters_stage(tmp_path, monkeypatch):
    """One answer per run: the build's reading is the graph's raw_image_stain,
    and the Diameters stage takes it without opening the file again."""
    import haemolynx.haemodynamics.fwhm_diameter as fwhm_diameter

    settings, schema = _run_settings(tmp_path, raw_image_path=str(tmp_path / "raw.tif"))
    network = stages.build_network(settings, stages.skeletonise(settings, stages.segment(settings)), schema)
    record = network.graph.graph
    assert record["raw_image_stain"] == record[GRAPH_ATTRIBUTE]["stain"] == PLASMA_COLUMN
    assert record["raw_image_stain_reading"]["stain"] == PLASMA_COLUMN
    assert record["raw_image_stain_reading"] == record[GRAPH_ATTRIBUTE]["reading"]

    def no_load(*_args, **_kwargs):
        raise AssertionError("the Diameters stage opened the raw image again")

    monkeypatch.setattr(fwhm_diameter, "load_single_channel_tiff_volume", no_load)
    assert stages.resolve_raw_image_stain(settings, network) == (PLASMA_COLUMN, None, None)


def test_build_network_records_no_stain_for_one_the_run_names(tmp_path):
    settings, schema = _run_settings(
        tmp_path, raw_image_path=str(tmp_path / "raw.tif"), raw_image_stain=PLASMA_COLUMN
    )
    network = stages.build_network(settings, stages.skeletonise(settings, stages.segment(settings)), schema)
    assert "raw_image_stain" not in network.graph.graph
    assert network.graph.graph[GRAPH_ATTRIBUTE]["reading"] is None


def test_build_network_without_a_raw_image_records_nothing(tmp_path):
    settings, schema = _run_settings(tmp_path)
    network = stages.build_network(settings, stages.skeletonise(settings, stages.segment(settings)), schema)
    assert GRAPH_ATTRIBUTE not in network.graph.graph


def test_build_network_refuses_a_raw_image_of_another_shape(tmp_path):
    settings, schema = _run_settings(tmp_path)
    tifffile.imwrite(tmp_path / "other.tif", np.full((16, 30, 61), BACKGROUND, dtype=np.uint16))
    settings["raw_image_path"] = str(tmp_path / "other.tif")
    with pytest.raises(ValueError, match="same stack the segmentation was made from"):
        stages.build_network(settings, stages.skeletonise(settings, stages.segment(settings)), schema)


# --------------------------------------------------------------------------
# Dead ends under an endothelial label: E1 ring_along, E2 inside_walls,
# E3 walls_ahead (PLAN_dead_ends_raw.md, Phase A)
# --------------------------------------------------------------------------

WIDE = (40, 40, 70)
#: The fixtures' blur and wall, as the ring fit is told them.
PSF = (1.0, 1.0, 1.0)
WALL = 1.5


def _walls(rings, seed=0):
    """Wall voxels as an endothelial channel: blurred by a voxel, with noise."""
    from scipy.ndimage import gaussian_filter

    raw = gaussian_filter(np.where(rings, BRIGHT, BACKGROUND).astype(float), 1.0)
    return _noisy(raw, seed=seed).astype(np.float32)


def _walled_tube(lumen=4.0, centre_zy=(20, 20), x_range=(0, WIDE[2]), shape=WIDE):
    """A wall ring of thickness :data:`WALL` round a lumen of radius *lumen*,
    along x over *x_range*, and its mask out to the wall's outer face."""
    r = _radial(shape, centre_zy)
    xs = np.arange(shape[2])
    along = ((xs >= x_range[0]) & (xs < x_range[1]))[None, None, :]
    return (r > lumen) & (r <= lumen + WALL) & along, (r <= lumen + WALL) & along


def _endothelial_evidence(rings, mask, seed=0):
    raw = _walls(rings, seed=seed)
    calibration = calibrate_raw(raw, mask, VOXEL, stain=ENDOTHELIAL)
    assert calibration is not None
    return make_raw_evidence(raw, mask.shape, VOXEL, calibration)


def _x_axis(z, y, x0, x1):
    xs = np.arange(x0, x1 + 1, dtype=float)
    return np.column_stack([np.full(len(xs), float(z)), np.full(len(xs), float(y)), xs])


def test_ring_along_a_chain_in_its_own_ring():
    """A ring round the chain: every section is walled, and every ring fitted
    at the tip's five sections is centred on it -- with the blur and wall
    given, or fitted free."""
    evidence = _endothelial_evidence(*_walled_tube())
    path = _x_path(20, 20, 50, 10)  # tip first
    for given in ({"psf_sigma_zyx": PSF, "wall_um": WALL}, {}):
        support = evidence.ring_along(path, 4.0, **given)
        assert support.fraction == 1.0 and support.tip_fraction == 1.0
        assert (support.sections, support.tip_sections) == (20, 5)
        assert [r.at_um for r in support.readings] == [1.0, 3.0, 5.0, 7.0, 9.0]
        assert support.rings_seen == 5 and support.rings(module.RING_OWN) == 5
        assert support.own_ring_fraction == 1.0
        for reading in support.readings:
            assert reading.diameter_um == pytest.approx(8.0, abs=0.5)
            assert np.linalg.norm(np.subtract(reading.centre_um[:2], (20, 20))) < 0.2
            assert reading.centre_um[2] == pytest.approx(50.0 - reading.at_um)


def test_ring_along_a_strand_inside_a_neighbours_ring():
    """A strand running 2.5 um off the axis of a 12 um vessel: there is a
    vessel there (walls either side), but every ring fitted round the strand
    is the neighbour's -- centred on the other centreline."""
    evidence = _endothelial_evidence(*_walled_tube(lumen=6.0))
    strand = _x_path(20, 22.5, 50, 10)
    support = evidence.ring_along(strand, 3.5, others_um=_x_axis(20, 20, 5, 65), psf_sigma_zyx=PSF, wall_um=WALL)
    assert support.tip_fraction == 1.0
    assert support.rings_seen == 5
    assert support.rings(module.RING_OTHER) == 5 and support.own_ring_fraction == 0.0
    assert all(r.diameter_um == pytest.approx(12.0, abs=0.5) for r in support.readings)


def test_ring_along_through_tissue_finds_no_ring():
    rings, mask = _walled_tube()
    evidence = _endothelial_evidence(rings, mask)
    support = evidence.ring_along(_x_path(10, 32, 50, 10), 3.0, psf_sigma_zyx=PSF, wall_um=WALL)
    assert support.fraction == 0.0 and support.tip_fraction == 0.0
    assert support.rings_seen == 0
    assert evidence.ring_along(np.zeros((1, 3)), 3.0).sections == 0


def test_ring_along_fits_as_on_the_whole_volume_and_reads_the_calibrations_rings():
    """The fit reads a window round its section and gives exactly what
    ``fit_ring_section`` gives on the whole volume; without a blur and wall
    of its own it takes the calibration's ring fields."""
    from haemolynx.haemodynamics.endothelial import fit_ring_section
    from haemolynx.haemodynamics.fwhm_diameter import _arc_length_parameterize

    evidence = _endothelial_evidence(*_walled_tube())
    path = module._distinct(_x_path(20, 20, 50, 10))
    s, total = _arc_length_parameterize(path)
    whole, _why = fit_ring_section(
        evidence.raw, path, s, total, 3.0, VOXEL, guide_um=8.0, psf_sigma_zyx=PSF, wall_um=WALL,
    )
    support = evidence.ring_along(path, 4.0, psf_sigma_zyx=PSF, wall_um=WALL)
    assert support.readings[1].diameter_um == whole.diameter_um
    assert support.readings[1].contrast == whole.contrast

    evidence.calibration = evidence.calibration.with_rings(PSF, WALL, 900.0)
    assert evidence.ring_along(path, 4.0).readings == support.readings


def test_inside_walls_tells_own_lumen_shared_lumen_wall_and_tissue():
    rings, mask = _walled_tube(lumen=6.0)
    evidence = _endothelial_evidence(rings, mask)
    axis = _x_axis(20, 20, 5, 65)
    on_axis = evidence.inside_walls(_x_axis(20, 20, 20, 40)[::5], 6.0)
    assert on_axis.lumen.all() and not on_axis.wall.any() and not on_axis.shared.any()
    assert on_axis.lumen_fraction == 1.0
    # A strand off the axis shares the vessel's lumen with its centreline.
    strand = evidence.inside_walls(_x_axis(20, 22.5, 20, 40)[::5], 3.5, others_um=axis)
    assert strand.lumen.all() and strand.shared.all() and strand.shared_fraction == 1.0
    on_wall = evidence.inside_walls(np.array([[20.0, 26.7, 30.0]]), 2.0)
    assert on_wall.wall.all() and not on_wall.lumen.any()
    tissue = evidence.inside_walls(np.array([[6.0, 6.0, 30.0], [20.0, 20.0, 500.0]]), 3.0, others_um=axis)
    assert not (tissue.lumen.any() or tissue.wall.any() or tissue.shared.any())


def test_inside_walls_keeps_two_vessels_lumens_apart():
    """Two vessels side by side, a wall each between them: a point in one
    shares no lumen with the other's centreline."""
    first, first_mask = _walled_tube(lumen=4.0, centre_zy=(20, 13))
    second, second_mask = _walled_tube(lumen=4.0, centre_zy=(20, 26))
    evidence = _endothelial_evidence(first | second, first_mask | second_mask)
    answer = evidence.inside_walls(
        _x_axis(20, 13, 20, 40)[::5], 4.0, others_um=_x_axis(20, 26, 5, 65), reach_um=15.0
    )
    assert answer.lumen.all() and not answer.shared.any()


def test_walls_running_on_past_the_tip():
    evidence = _endothelial_evidence(*_walled_tube())
    ahead = evidence.walls_ahead([20.0, 20.0, 20.0], [0.0, 0.0, 1.0], 4.0)
    assert ahead.stop == module.AHEAD_RUNS_ON
    assert ahead.run_um == module.AHEAD_LIMIT_UM and ahead.gap_um == 0.0 and ahead.gap_stretches == 0
    assert (ahead.start_um, ahead.step_um, len(ahead.profile)) == (4.0, 1.0, 30)
    assert ahead.meets_um is None and ahead.face_um is None
    # Running on to the image face: the line leaves the image 29 um out.
    to_face = evidence.walls_ahead([20.0, 20.0, 40.0], [0.0, 0.0, 1.0], 4.0)
    assert to_face.stop == module.AHEAD_FACE and to_face.face_um == pytest.approx(29.0)
    assert to_face.run_um == pytest.approx(25.0) and to_face.gap_um == 0.0


def test_walls_stopping_at_the_tip():
    evidence = _endothelial_evidence(*_walled_tube(x_range=(0, 40)))
    ahead = evidence.walls_ahead([20.0, 20.0, 35.0], [0.0, 0.0, 1.0], 4.0)
    assert ahead.stop == module.AHEAD_ENDS
    assert ahead.run_um <= 3.0
    assert ahead.gap_um >= 25.0 and ahead.gap_stretches == 1
    assert max(ahead.profile[3:]) < BRIGHT_THRESHOLD


def test_walls_ahead_into_another_vessel_with_and_without_a_gap():
    """A vessel the line meets: reached with the walls running on all the
    way, or across a stretch with no walls (a vessel the segmentation broke
    and the raw image too, or none)."""
    rings, mask = _walled_tube(x_range=(0, 52))
    zz, _yy, xx = np.indices(WIDE)
    across = np.sqrt((zz - 20.0) ** 2 + (xx - 57.0) ** 2)  # a vessel along y at x = 57
    cross_rings, cross_mask = (across > 4.0) & (across <= 4.0 + WALL), across <= 4.0 + WALL
    others = np.column_stack([np.full(31, 20.0), np.arange(5, 36, dtype=float), np.full(31, 57.0)])
    evidence = _endothelial_evidence(rings | cross_rings, mask | cross_mask)
    joined = evidence.walls_ahead([20.0, 20.0, 30.0], [0.0, 0.0, 1.0], 4.0, others_um=others, others_radius_um=4.0)
    assert joined.stop == module.AHEAD_MEETS_VESSEL
    assert joined.meets_um == pytest.approx(23.0) and joined.gap_um == 0.0
    assert joined.run_um == pytest.approx(18.0)  # 5 to 22 um past the tip

    gapped_rings = rings.copy()
    gapped_rings[:, :, 40:46] = False
    broken = _endothelial_evidence(gapped_rings | cross_rings, mask | cross_mask)
    gap = broken.walls_ahead([20.0, 20.0, 30.0], [0.0, 0.0, 1.0], 4.0, others_um=others, others_radius_um=4.0)
    assert gap.stop == module.AHEAD_ENDS and gap.meets_um == pytest.approx(23.0)
    assert gap.gap_stretches == 1 and gap.gap_um == pytest.approx(6.0, abs=2.0)
    with pytest.raises(ValueError, match="heading"):
        broken.walls_ahead([20.0, 20.0, 30.0], [0.0, 0.0, 0.0], 4.0)


BENT = (40, 70, 80)


def _bent_centreline():
    """Along x at (z, y) = (20, 12) to x = 30, a quarter turn of radius 25 um
    to heading +y at x = 55, then on along y to the image's face."""
    straight = np.column_stack([np.full(301, 20.0), np.full(301, 12.0), np.linspace(0.0, 30.0, 301)])
    t = np.linspace(0.0, np.pi / 2, 400)
    arc = np.column_stack([np.full(len(t), 20.0), 37.0 - 25.0 * np.cos(t), 30.0 + 25.0 * np.sin(t)])
    on = np.column_stack([np.full(151, 20.0), np.linspace(37.0, 69.0, 151), np.full(151, 55.0)])
    return np.vstack([straight, arc, on])


def _walled_along(centreline, lumen=3.0, shape=BENT):
    """A wall ring round a lumen of radius *lumen* along any centreline."""
    from scipy.spatial import cKDTree

    distance = cKDTree(centreline).query(np.indices(shape).reshape(3, -1).T)[0].reshape(shape)
    return (distance > lumen) & (distance <= lumen + WALL), distance <= lumen + WALL


def test_the_route_follows_a_vessel_round_its_turn():
    """E4 follows the walls round a quarter turn, staying within a voxel of
    the centreline, where E3's straight line leaves them."""
    from scipy.spatial import cKDTree

    centreline = _bent_centreline()
    evidence = _endothelial_evidence(*_walled_along(centreline))
    tip, heading = [20.0, 12.0, 22.0], [0.0, 0.0, 1.0]
    straight = evidence.walls_ahead(tip, heading, 3.0)
    route = evidence.route_ahead(tip, heading, 3.0)
    assert straight.stop == module.AHEAD_ENDS and straight.run_um <= 22.0
    assert route.stop == module.AHEAD_RUNS_ON and route.run_um == module.AHEAD_LIMIT_UM
    assert route.gap_stretches == 0 and len(route.profile) == 30
    assert cKDTree(centreline).query(np.asarray(route.path_um[1:]))[0].max() < 1.5
    assert route.path_um[0] == tuple(tip)
    assert 15.0 < route.turn_deg <= route.bend_deg < 40.0
    assert route.reach_um == pytest.approx(np.linalg.norm(np.subtract(route.path_um[-1], tip)))
    # The turn limit holds, give or take the re-centring across the route.
    assert evidence.route_ahead(tip, heading, 3.0, max_turn_deg=20.0).bend_deg <= 20.0 + 5.0


def test_the_route_round_a_turn_into_another_vessel_or_out_of_the_image():
    centreline = _bent_centreline()
    evidence = _endothelial_evidence(*_walled_along(centreline))
    tip, heading = [20.0, 12.0, 22.0], [0.0, 0.0, 1.0]
    beyond = centreline[centreline[:, 1] > 50.0]  # as if another vessel's centreline
    # A quarter turn is past the default limit (60 deg off the heading): let it turn.
    joined = evidence.route_ahead(tip, heading, 3.0, others_um=beyond, others_radius_um=3.0, limit_um=80.0,
                                  max_turn_deg=100.0)
    assert joined.stop == module.AHEAD_MEETS_VESSEL
    # From 3 um past the tip: 5 um straight, the quarter turn, 13 um on.
    assert joined.meets_um == pytest.approx(5.0 + 25.0 * np.pi / 2 + 13.0, abs=3.0)
    assert joined.run_um == joined.meets_um and joined.gap_um == 0.0
    to_face = evidence.route_ahead(tip, heading, 3.0, limit_um=80.0, max_turn_deg=100.0)
    assert to_face.stop == module.AHEAD_FACE
    assert to_face.path_um[-1][1] > 64.0  # stopped heading out of the y face
    # At the default limit it cuts the corner, no more than about 60 deg off.
    assert evidence.route_ahead(tip, heading, 3.0, limit_um=80.0).bend_deg <= module.ROUTE_MAX_TURN_DEG + 5.0


def test_the_route_without_turns_or_gaps_is_the_straight_line():
    """E4 with no cone and no gap allowed reads E3's line: same run, same stop."""
    centreline = _bent_centreline()
    bent = _endothelial_evidence(*_walled_along(centreline))
    whole = _endothelial_evidence(*_walled_tube())
    stopping = _endothelial_evidence(*_walled_tube(x_range=(0, 40)))
    asks = [
        (bent, [20.0, 12.0, 22.0], 3.0),
        (whole, [20.0, 20.0, 20.0], 4.0),
        (whole, [20.0, 20.0, 40.0], 4.0),
        (stopping, [20.0, 20.0, 35.0], 4.0),
    ]
    for evidence, tip, radius in asks:
        line = evidence.walls_ahead(tip, [0.0, 0.0, 1.0], radius)
        route = evidence.route_ahead(tip, [0.0, 0.0, 1.0], radius, cone_deg=0.0, max_gap_um=0.0)
        assert (route.stop, route.run_um) == (line.stop, line.run_um)


def test_the_route_crosses_a_short_dropout_and_stops_where_the_walls_stop():
    from scipy.ndimage import gaussian_filter

    line = np.column_stack([np.full(801, 20.0), np.full(801, 35.0), np.linspace(0.0, 79.0, 801)])
    rings, mask = _walled_along(line)
    level = np.where(rings, BRIGHT, BACKGROUND)
    level[:, :, 40:44] = np.where(rings[:, :, 40:44], 0.25 * BRIGHT, BACKGROUND)  # the stain dims for 4 um
    raw = _noisy(gaussian_filter(level, 1.0)).astype(np.float32)
    evidence = make_raw_evidence(raw, BENT, VOXEL, calibrate_raw(raw, mask, VOXEL, stain=ENDOTHELIAL))
    tip, heading = [20.0, 35.0, 25.0], [0.0, 0.0, 1.0]
    assert evidence.walls_ahead(tip, heading, 3.0).stop == module.AHEAD_ENDS
    route = evidence.route_ahead(tip, heading, 3.0)
    assert route.stop == module.AHEAD_RUNS_ON
    assert route.gap_stretches == 1 and 0.0 < route.gap_um <= module.ROUTE_MAX_GAP_UM
    assert abs(route.path_um[-1][1] - 35.0) < 1.5  # straight on, not round the dim stretch's edge

    ending = _endothelial_evidence(*_walled_tube(x_range=(0, 40)))
    stops = ending.route_ahead([20.0, 20.0, 35.0], heading, 4.0)
    assert stops.stop == module.AHEAD_ENDS and stops.run_um <= 3.0
    with pytest.raises(ValueError, match="heading"):
        ending.route_ahead([20.0, 20.0, 35.0], [0.0, 0.0, 0.0], 4.0)


def test_wall_primitives_read_a_memory_map_in_windows_as_in_ram(tmp_path):
    """The low-RAM option's memory map gives the same answers, read in
    windows, none of them a quarter of the volume."""
    shape = (40, 40, 240)
    rings, mask = _walled_tube(lumen=6.0, x_range=(0, 240), shape=shape)
    raw = _walls(rings).astype(np.uint16)
    path = tmp_path / "walls.memmap"
    stored = np.memmap(path, dtype=np.uint16, mode="w+", shape=shape)
    stored[:] = raw
    stored.flush()
    watched = _Watched(np.memmap(path, dtype=np.uint16, mode="r", shape=shape))
    calibration = calibrate_raw(raw.astype(np.float32), mask, VOXEL, stain=ENDOTHELIAL)
    in_ram = make_raw_evidence(raw.astype(np.float32), shape, VOXEL, calibration)
    on_disk = make_raw_evidence(watched, shape, VOXEL, calibration)
    strand, axis = _x_path(20, 22.5, 140, 100), _x_axis(20, 20, 5, 235)
    asks = {
        "ring_along": lambda e: e.ring_along(strand, 3.5, others_um=axis, psf_sigma_zyx=PSF, wall_um=WALL),
        "walls_ahead": lambda e: e.walls_ahead([20.0, 20.0, 100.0], [0.0, 0.0, 1.0], 6.0),
        "route_ahead": lambda e: e.route_ahead([20.0, 20.0, 100.0], [0.0, 0.0, 1.0], 6.0),
    }
    for name, ask in asks.items():
        assert ask(on_disk) == ask(in_ram), name
    points = _x_axis(20, 22.5, 110, 130)[::5]
    disk, ram = on_disk.inside_walls(points, 3.5, others_um=axis), in_ram.inside_walls(points, 3.5, others_um=axis)
    for field in ("lumen", "wall", "shared"):
        assert np.array_equal(getattr(disk, field), getattr(ram, field)), field
    assert watched.reads and max(watched.reads) < raw.size / 4
    assert on_disk.stats.queries == len(watched.reads)


def test_the_calibrations_ring_fields_round_trip_and_old_records_read():
    raw, mask = _endothelial_tube()
    calibration = calibrate_raw(_noisy(raw), mask, VOXEL)
    old = calibration.as_dict()
    for name in ("ring_psf_sigma_zyx", "ring_wall_um", "ring_wall_level"):
        old.pop(name)
    read = RawCalibration.from_record(old)
    assert read == calibration
    assert (read.ring_psf_sigma_zyx, read.ring_wall_um, read.ring_wall_level) == (None, None, None)
    ringed = calibration.with_rings([2.9, 1.2, 1.2], 1.2, 480.0)
    assert ringed.ring_psf_sigma_zyx == (2.9, 1.2, 1.2) and ringed.ring_wall_um == 1.2
    assert RawCalibration.from_record(ringed.as_dict()) == ringed
    assert "wall 1.20 um" in ringed.describe() and "rings" not in calibration.describe()


def test_ring_calibration_fills_the_fields_only_when_the_rings_calibrate(monkeypatch):
    """The image's own rings: the blur and wall ``calibrate_from_rings``
    gives, and the median wall level of the fits its gate passes."""
    from haemolynx.haemodynamics import endothelial
    from haemolynx.haemodynamics.endothelial import RingCalibrationFits, RingFit

    def ring(level, contrast):
        return RingFit(
            diameter_um=8.0, wall_um=1.2, contrast=contrast, resolution=4.0, wall_coverage=1.0,
            centre_offset_um=0.1, aspect=1.0, sigma_across_um=1.1, sigma_other_um=2.9, other_axis_z=0.8,
            wall_level=level,
        )

    fits = RingCalibrationFits(None, 4, 2, ((
        (ring(400.0, 10.0), "ok"), (ring(600.0, 10.0), "ok")), ((ring(5000.0, 1.0), "ok"), (None, "no_dark_lumen")),
    ))
    monkeypatch.setattr(endothelial, "ring_calibration_fits", lambda *a, **k: fits)
    monkeypatch.setattr(endothelial, "calibrate_from_rings", lambda *a, **k: ((2.9, 1.1, 1.1), 1.2, {}))
    evidence = _endothelial_evidence(*_walled_tube())
    calibrated = module.ring_calibration(object(), evidence)
    assert calibrated.ring_psf_sigma_zyx == (2.9, 1.1, 1.1) and calibrated.ring_wall_um == 1.2
    assert calibrated.ring_wall_level == 500.0  # the low-contrast ring is left out
    assert calibrated.vessel == evidence.calibration.vessel

    monkeypatch.setattr(endothelial, "calibrate_from_rings", lambda *a, **k: (None, None, {"reason": "few"}))
    assert module.ring_calibration(object(), evidence) is evidence.calibration


def test_raw_evidence_from_settings_reads_a_channel_of_a_composite(tmp_path):
    raw, mask = _plasma_tube()
    other = np.full(SHAPE, BACKGROUND)
    composite = np.stack([other, raw], axis=1).astype(np.uint16)  # (z, c, y, x), as ImageJ writes
    tifffile.imwrite(tmp_path / "composite.tif", composite, imagej=True, metadata={"axes": "ZCYX"})
    settings = {"raw_image_path": str(tmp_path / "composite.tif"), "raw_image_channel": 1}
    evidence = raw_evidence_from_settings(settings, mask, VOXEL)
    assert isinstance(evidence, RawEvidencePlasma)
    assert raw_evidence_from_settings({**settings, "raw_image_channel": 0}, mask, VOXEL) is None


def test_the_build_log_says_what_the_raw_image_cost(tmp_path, caplog):
    """The seconds the channel took to load and calibrate, and the RAM it
    holds (``PLAN_raw_confirmation.md``, D6): read in RAM, the whole file's
    channel; memory-mapped, none."""
    raw, mask = _plasma_tube()
    tifffile.imwrite(tmp_path / "raw.tif", raw.astype(np.uint16))
    settings = {"raw_image_path": str(tmp_path / "raw.tif"), "raw_image_stain": PLASMA_COLUMN}
    with caplog.at_level("INFO", logger=module.__name__):
        evidence = raw_evidence_from_settings(settings, mask, VOXEL)
    assert module.resident_bytes(evidence.raw) == raw.size * 2
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Raw evidence from raw.tif"))
    assert "loaded and calibrated in " in line and f"{raw.size * 2 / 2**20:.0f} MB of it in RAM" in line
    caplog.clear()
    with caplog.at_level("INFO", logger=module.__name__):
        raw_evidence_from_settings(
            {**settings, "use_memmap_loading": True, "memmap_directory": str(tmp_path / "memmaps")}, mask, VOXEL
        )
    assert any(r.getMessage().endswith("memory-mapped.") for r in caplog.records)


def test_resident_bytes_counts_the_array_a_view_keeps(tmp_path):
    """A crop or a transpose keeps its whole array in RAM; a memory map
    keeps none of it; anything else holds no volume."""
    whole = np.zeros((4, 6, 8), dtype=np.uint16)
    assert module.resident_bytes(whole) == whole.nbytes
    assert module.resident_bytes(whole.transpose(2, 1, 0)[1:3]) == whole.nbytes
    mapped = np.lib.format.open_memmap(tmp_path / "v.npy", mode="w+", dtype=np.uint16, shape=(4, 6, 8))
    assert module.resident_bytes(mapped) == 0
    assert module.resident_bytes(mapped[1:3, ::2]) == 0
    assert module.resident_bytes(None) == 0
