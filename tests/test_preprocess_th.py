"""Tests for preprocess_th.py.

Each test encodes one of the defects found in th_glomus_preprocessing_review.md,
so that reintroducing it fails here rather than silently in a cohort run.

Run with:  python3 -m pytest test_preprocess_th.py -v
"""
import json
import os
import subprocess
import sys
from argparse import Namespace

import h5py
import numpy as np
import pytest
import tifffile

_PREPROCESSING = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "preprocessing")
sys.path.insert(0, _PREPROCESSING)

import preprocess_th as pth

ILASTIK_PY = os.path.expanduser(
    "~/Desktop/ilastik-1.4.1rc2-gpu-Linux/bin/python")


# ------------------------------------------------------------------ fixtures

def doughnut_volume(shape=(24, 64, 64), n_cells=40, seed=0):
    """Synthetic glomus cells: bright cytoplasmic shell, dark nuclear core."""
    rng = np.random.default_rng(seed)
    vol = np.full(shape, 100.0, dtype=np.float32)
    zz, yy, xx = np.mgrid[-5:6, -5:6, -5:6]
    r = np.sqrt(zz ** 2 + yy ** 2 + xx ** 2)
    cell = np.where(r <= 4.0, 3000.0, 0.0)     # soma
    cell[r <= 1.8] = 300.0                     # dark nucleus
    for _ in range(n_cells):
        z = rng.integers(6, shape[0] - 6)
        y = rng.integers(6, shape[1] - 6)
        x = rng.integers(6, shape[2] - 6)
        sub = vol[z - 5:z + 6, y - 5:y + 6, x - 5:x + 6]
        np.maximum(sub, cell, out=sub)
    return vol + rng.normal(0, 15, shape).astype(np.float32)


def sparse_tissue_volume(shape=(16, 64, 64)):
    """Mostly empty field with a bright blob, as the real acquisitions are."""
    vol = np.full(shape, 80.0, dtype=np.float32)
    vol[4:12, 20:44, 20:44] = 4000.0
    return vol


def write_zcyx_tiff(path, th, lectin):
    """Two-channel ZCYX stack, matching the CB3 acquisition layout."""
    stack = np.stack([lectin, th], axis=1).astype(np.uint16)
    tifffile.imwrite(path, stack, imagej=True,
                     metadata={"axes": "ZCYX", "spacing": 1.8639, "unit": "um"})


def default_args(**over):
    a = Namespace(
        channel=pth.TH_CHANNEL, with_vessel_channel=False, vessel_channel=0,
        z_correct="none", force_z_correct=False, rolling_ball=0,
        remove_outliers=0, saturated=0.35, anchors=None,
        tissue_smooth=pth.DEF_TISSUE_SMOOTH,
        dog_sigmas=list(pth.DEF_DOG_SIGMAS), split_dog=False, workers=2,
        voxel=list(pth.VOXEL_ZYX), diagnose=False, save_tif=False,
        output_dir=None)
    for k, v in over.items():
        setattr(a, k, v)
    return a


# ------------------------------------------------------- R1: ilastik axistags

def test_axistags_use_integer_typeflags_and_channel_last_order(tmp_path):
    """vigra requires an integer 'typeFlags'; a 'type' string raises KeyError.

    The original TH script wrote {"key","type","description"}, which ilastik
    cannot parse, so it silently fell back to guessing axes from shape.
    """
    out = tmp_path / "t.h5"
    ch = [np.zeros((4, 8, 8), np.float32) for _ in range(2)]
    pth.write_h5(str(out), ch, ["grayscale", "soma_dog_signed"], pth.VOXEL_ZYX)

    with h5py.File(out, "r") as f:
        assert "data" in f, "downstream tools open the dataset named 'data'"
        tags = json.loads(f["data"].attrs["axistags"])

    keys = [a["key"] for a in tags["axes"]]
    assert keys == list("zyxc"), (
        f"axes must be channel-last zyxc, got {keys}. prob_to_mask.py slices "
        "arr[..., channel], so a channel-second layout selects the wrong axis.")
    for axis in tags["axes"]:
        assert isinstance(axis["typeFlags"], int), (
            f"axis {axis['key']} has no integer typeFlags; vigra raises "
            "KeyError: 'typeFlags' on the string form")


@pytest.mark.skipif(not os.path.exists(ILASTIK_PY),
                    reason="ilastik bundle not installed")
def test_axistags_actually_parse_in_ilastiks_own_vigra(tmp_path):
    out = tmp_path / "t.h5"
    ch = [np.zeros((4, 8, 8), np.float32) for _ in range(2)]
    pth.write_h5(str(out), ch, ["grayscale", "soma_dog_signed"], pth.VOXEL_ZYX)
    with h5py.File(out, "r") as f:
        tags = f["data"].attrs["axistags"]

    proc = subprocess.run(
        [ILASTIK_PY, "-c",
         "import vigra,sys;at=vigra.AxisTags.fromJSON(sys.argv[1]);"
         "print(''.join(a.key for a in at))", tags],
        capture_output=True, text=True)
    assert proc.returncode == 0, f"vigra rejected the axistags: {proc.stderr}"
    assert proc.stdout.strip() == "zyxc"


# ------------------------------------------------- R2: no histogram matching

def test_multiplicative_correction_refuses_where_the_profile_is_a_hump():
    """A hump is the extent of the tissue block, not photobleaching.

    Correcting it amplifies the sparse end slices. WKY-B and WKY-C both show
    this shape, and WKY-C's TH signal rises five-fold with depth.
    """
    vol = np.ones((20, 8, 8), np.float32)
    with pytest.raises(ValueError, match="not 'monotonic decay'"):
        pth.multiplicative_z_correction(vol, "hump / tissue-extent dominated")
    with pytest.raises(ValueError, match="tissue block"):
        pth.multiplicative_z_correction(vol, "inverse decay (rises with z)")


def test_multiplicative_correction_runs_on_genuine_decay_and_is_bounded():
    n = 40
    vol = np.zeros((n, 16, 16), np.float32)
    for z in range(n):
        vol[z] = 1000.0 * np.exp(-z / 15.0)
    out, stats = pth.multiplicative_z_correction(vol, "monotonic decay")
    p99 = np.array([np.percentile(out[z], 99.0) for z in range(n)])
    assert p99.max() / p99.min() < 1.15, (
        "a multiplicative correction should flatten a genuine exponential decay")
    assert stats["gain_max"] <= 4.0 and stats["gain_min"] >= 0.25, (
        "gains must be capped so near-empty slices are left alone, not amplified")


def test_module_offers_no_histogram_matching():
    src = open(pth.__file__).read()
    assert "match_histograms" not in src, (
        "histogram matching forces every slice to the same intensity "
        "distribution, which hard-codes TH density to be uniform in z")


# ------------------------------------------------------- R4: signed DoG channel

def test_soma_dog_keeps_the_negative_core_signal():
    """The nuclear core is where the DoG is negative.

    Clipping at zero mapped 99.8% of cores and 55.8% of background to the same
    value, so the channel could not separate 'inside the nucleus' from
    'outside the cell'.
    """
    vol = doughnut_volume()
    mask = np.ones(vol.shape, bool)
    (dog,), stats = pth.soma_dog(vol, (1.0, 3.0), mask, 0.35)

    assert dog.min() < -0.05, (
        f"DoG channel has no negative values (min={dog.min():.3f}); the "
        "nuclear core signal has been clipped away")
    assert stats["fraction_negative"] > 0.01
    assert dog.max() > 0.05, "DoG channel has lost the positive ring response"


def test_soma_dog_separates_core_from_background():
    """Core and background must not map to the same value."""
    vol = doughnut_volume()
    mask = np.ones(vol.shape, bool)
    (dog,), _ = pth.soma_dog(vol, (1.0, 3.0), mask, 0.35)

    bright = vol > np.percentile(vol, 90)
    core = (vol < np.percentile(vol, 60)) & _dilate(bright)
    background = vol < np.percentile(vol, 30)
    background &= ~_dilate(bright)
    if core.sum() < 20 or background.sum() < 20:
        pytest.skip("synthetic volume did not yield enough core/background voxels")

    assert dog[core].mean() < dog[background].mean() - 0.02, (
        f"core mean {dog[core].mean():.3f} is not separated from background "
        f"mean {dog[background].mean():.3f}")


def _dilate(m, it=3):
    from scipy.ndimage import binary_dilation
    return binary_dilation(m, iterations=it)


def test_split_dog_returns_two_non_negative_channels():
    vol = doughnut_volume()
    mask = np.ones(vol.shape, bool)
    chans, _ = pth.soma_dog(vol, (1.0, 3.0), mask, 0.35, split=True)
    assert len(chans) == 2
    for c in chans:
        assert c.min() >= 0.0
    assert chans[1].max() > 0.05, "the negative part carries the core signal"


def test_soma_dog_rejects_non_increasing_sigmas():
    vol = doughnut_volume(shape=(16, 24, 24), n_cells=2)
    with pytest.raises(ValueError, match="increasing"):
        pth.soma_dog(vol, (3.0, 1.0), np.ones(vol.shape, bool), 0.35)


# ------------------------------------------- R5: anchors come from tissue only

def test_normalisation_anchors_do_not_track_how_much_empty_frame_was_acquired():
    """The defect: a whole-volume anchor is set partly by the black surround.

    Measured occupancy across three WKY volumes was 16.1%, 17.4% and 28.9%,
    a 1.8x spread, which moved the anchor by 1.22x to 1.44x between specimens
    of the same group. Widening the field must not change the gain.
    """
    rng = np.random.default_rng(0)
    core = np.full((16, 48, 48), 80.0, np.float32)
    core[4:12, 12:36, 12:36] = rng.normal(4000, 600, (8, 24, 24))
    wide = np.full((16, 96, 96), 80.0, np.float32)
    wide[:, 24:72, 24:72] = core

    m_core, occ_core = pth.tissue_mask(core)
    m_wide, occ_wide = pth.tissue_mask(wide)
    assert occ_wide < occ_core / 2, "the wider field must be emptier"

    whole = [pth.percentile_anchors(v, 0.35)[1] for v in (core, wide)]
    tissue = [pth.percentile_anchors(v[m], 0.35)[1]
              for v, m in ((core, m_core), (wide, m_wide))]

    assert abs(tissue[1] - tissue[0]) / tissue[0] < 0.05, (
        f"tissue anchors moved {tissue} when only the empty surround changed")
    assert abs(whole[1] - whole[0]) / whole[0] > abs(
        tissue[1] - tissue[0]) / tissue[0], (
        f"whole-volume anchors {whole} should be the more frame-dependent pair")


def test_tissue_mask_selects_the_blob_and_rejects_the_empty_corner():
    vol = sparse_tissue_volume()
    mask, occ = pth.tissue_mask(vol)
    assert 0.02 < occ < 0.5, f"tissue mask occupancy implausible: {occ}"
    assert mask[6, 30, 30], "the bright blob must be inside the tissue mask"
    assert not mask[0, 0, 0], "the empty corner must be outside the tissue mask"


def test_tissue_mask_falls_back_rather_than_anchoring_on_nothing():
    flat = np.full((8, 16, 16), 50.0, np.float32)
    mask, occ = pth.tissue_mask(flat)
    assert mask.all(), "a degenerate mask must fall back to the whole volume"


def test_process_volume_normalises_against_tissue_not_the_whole_volume(tmp_path):
    """The end-to-end version of the anchor test.

    Exercises the choice process_volume actually makes, rather than the two
    helpers in isolation, so that reverting to a whole-volume anchor fails here.
    """
    rng = np.random.default_rng(1)
    th = np.full((16, 64, 64), 80.0, np.float32)
    th[4:12, 20:44, 20:44] = rng.normal(4000, 700, (8, 24, 24))
    p = tmp_path / "v.tif"
    write_zcyx_tiff(p, th, th * 0.5)

    qc = pth.process_volume(str(p), default_args(output_dir=str(tmp_path / "o")))
    used = qc["normalisation_anchors"]
    whole = qc["normalisation_anchors_whole_volume"]

    assert qc["anchor_source"] == "tissue"
    # The HIGH anchor sets the gain, and is the one measured to move 1.22x to
    # 1.44x between specimens on framing alone. The low anchor is the
    # background floor by design, and is the same either way.
    assert abs(used[1] - whole[1]) / whole[1] > 0.02, (
        f"high anchor {used[1]:.0f} is indistinguishable from the whole-volume "
        f"one {whole[1]:.0f}; the tissue mask is not being used")
    assert used[1] > whole[1], (
        "excluding empty frame should raise the high anchor, not lower it")


def test_supplied_cohort_anchors_override_per_volume_ones(tmp_path):
    th = sparse_tissue_volume()
    p = tmp_path / "v.tif"
    write_zcyx_tiff(p, th, th * 0.5)
    args = default_args(output_dir=str(tmp_path / "out"), anchors=[100.0, 2000.0])
    qc = pth.process_volume(str(p), args)
    assert qc["anchor_source"] == "supplied"
    assert qc["normalisation_anchors"] == [100.0, 2000.0]
    assert "normalisation_anchors_whole_volume" in qc, (
        "QC must record what the per-volume anchor would have been")


# ---------------------------------------------- R3: channel extraction and IO

def test_th_channel_defaults_to_index_one_and_is_extracted_correctly(tmp_path):
    """Channel 0 is lectin, verified byte-identical to C1-*_vessels.tif."""
    assert pth.TH_CHANNEL == 1
    th = sparse_tissue_volume()
    lectin = np.full_like(th, 7.0)
    p = tmp_path / "v.tif"
    write_zcyx_tiff(p, th, lectin)

    vol, n_ch, axes = pth.read_vessel_channel(str(p), pth.TH_CHANNEL)
    assert n_ch == 2 and "C" in axes
    assert vol.shape == th.shape
    assert vol.max() > 100, "extracted the flat lectin channel instead of TH"


def test_four_dimensional_input_is_handled_not_rejected(tmp_path):
    """The original script called io.imread and raised on the real files."""
    th = sparse_tissue_volume()
    p = tmp_path / "v.tif"
    write_zcyx_tiff(p, th, th * 0.5)
    args = default_args(output_dir=str(tmp_path / "out"))
    qc = pth.process_volume(str(p), args)
    assert qc["n_channels_in_file"] == 2
    assert os.path.exists(qc["output_h5"])


# -------------------------------------------------------------- R6: no tiling

def test_output_is_one_untiled_volume_at_full_extent(tmp_path):
    """Reflect padding to a 256 grid made 52% of WKY-B a mirrored duplicate."""
    th = sparse_tissue_volume(shape=(12, 50, 70))
    p = tmp_path / "v.tif"
    write_zcyx_tiff(p, th, th * 0.5)
    outdir = tmp_path / "out"
    args = default_args(output_dir=str(outdir))
    qc = pth.process_volume(str(p), args)

    tiles = [f for f in os.listdir(outdir) if f.endswith(".h5")]
    assert len(tiles) == 1, f"expected one untiled output, got {tiles}"
    with h5py.File(qc["output_h5"], "r") as f:
        assert f["data"].shape[:3] == th.shape, (
            "output extent must match the input; no padding to a tile grid")


def test_module_does_no_reflect_padding():
    src = open(pth.__file__).read()
    assert "reflect" not in src, (
        "reflect padding fabricates mirrored tissue that the classifier will "
        "label as real")


# ---------------------------------------------------------------- R7, C4: QC

def test_qc_json_records_provenance(tmp_path):
    th = sparse_tissue_volume()
    p = tmp_path / "v.tif"
    write_zcyx_tiff(p, th, th * 0.5)
    outdir = tmp_path / "out"
    qc = pth.process_volume(str(p), default_args(output_dir=str(outdir)))
    written = json.load(open(outdir / "v_TH_qc.json"))
    for key in ("z_profile", "tissue_occupancy", "normalisation_anchors",
                "normalisation_anchors_whole_volume", "dog_sigmas_px",
                "channel_names", "parameters", "z_correction"):
        assert key in written, f"QC is missing {key}"
    assert written["z_correction"] is None, (
        "the default must record that no depth correction was applied")


def test_module_applies_no_blanket_median_filter():
    src = open(pth.__file__).read()
    assert "median_filter" not in src, (
        "a 3x3x3 median costs 18% of the doughnut contrast; use "
        "remove_outliers, which touches only impulse noise")
