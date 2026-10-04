"""Vessels of known width planted in a raw image: haemolynx.haemodynamics.fwhm_planted.

Real vessels here are the diameter benchmark's (``test_raw_section_diameter``):
a plasma-filled lumen, a blur several times longer in z than across, a few
photons per voxel and shot noise.
"""
from __future__ import annotations

import inspect

import networkx as nx
import numpy as np
import pytest

from haemolynx.haemodynamics import automated, fwhm_planted
from haemolynx.haemodynamics.fwhm_planted import (
    DEFAULT_OPTICS_PSF_UM,
    draw_vessel,
    image_noise_gain,
    optics_psf,
    plant_vessels,
    planted_width_report,
)
from haemolynx.pipeline.schema import SCHEMA
from test_raw_section_diameter import EFFECTIVE_PSF, PSF, VOXEL, _graph, _render  # noqa: E402

#: Five vessels that do not cross, with room beside each to plant in.
TUBES = [
    (3.0, (0, 0, 1), (0, -42, 0)),
    (4.0, (0, 0, 1), (10, -18, 0)),
    (5.0, (1, 0, 1), (0, 8, 0)),
    (8.0, (1, 0.05, 0.05), (0, 34, -32)),
    (6.0, (0, 0, 1), (-12, 50, 20)),
]


@pytest.fixture(scope="module")
def scene():
    raw, mask, lines = _render(TUBES, shape=(60, 128, 128), peak_counts=6.0, seed=2)
    truth = np.array([d for d, _direction, _shift in TUBES])
    return raw, mask, lines, truth


def _fwhm(graph, volume, **overrides):
    """FWHM at the schema's defaults, its blur held at the image PSF."""
    valid = set(inspect.signature(automated.measure_edge_diameters_fwhm_from_raw_tiff).parameters)
    kwargs = {
        field.name[5:]: field.default
        for field in SCHEMA
        if field.name.startswith("fwhm_") and field.name[5:] in valid
    }
    kwargs.update(overrides)
    automated.measure_edge_diameters_fwhm_from_raw_tiff(
        graph, raw_volume=volume, voxel_size_zyx=VOXEL, store_profile_debug=False,
        profile_psf_sigma_zyx=EFFECTIVE_PSF, **kwargs,
    )


# --- drawing ------------------------------------------------------------------


def test_the_optics_are_the_image_blur_without_the_voxels_own_box():
    assert optics_psf(EFFECTIVE_PSF, VOXEL) == pytest.approx(PSF, rel=1e-6)
    assert optics_psf(None, VOXEL) == DEFAULT_OPTICS_PSF_UM


def _drawn(peak=10.0, noise_gain=0.0, seed=0):
    volume = np.zeros((40, 60, 60), dtype=np.float32)
    line = np.array([[20.0, 30.0, x] for x in np.arange(10.0, 51.0)])
    drawn = draw_vessel(
        volume, line, 6.0, (1.0, 1.0, 1.0), (0.4, 0.4, 0.4),
        peak=peak, noise_gain=noise_gain, rng=np.random.default_rng(seed),
    )
    return volume, line, drawn


def test_a_drawn_vessel_has_its_width_and_brightness():
    volume, line, drawn = _drawn()
    box, lumen = drawn

    along = volume[20, 30, 15:46]
    assert np.median(along) == pytest.approx(10.0, rel=0.02)
    # A cylinder 40 um long with round ends: pi r^2 L + 4/3 pi r^3.
    assert np.count_nonzero(lumen) == pytest.approx(np.pi * 9 * 40 + 4 / 3 * np.pi * 27, rel=0.1)
    across = volume[20, :, 30]
    assert np.count_nonzero(across >= 5.0) == pytest.approx(6, abs=1)  # its half-maximum width
    untouched = np.ones(volume.shape, dtype=bool)
    untouched[box] = False
    assert not volume[untouched].any()


def test_a_drawn_vessels_noise_follows_the_gain():
    quiet, _line, _drawn_quiet = _drawn()
    noisy, _line, _drawn_noisy = _drawn(noise_gain=3.0)
    residual = (noisy - quiet)[20, 29:32, 15:46]
    assert np.var(residual) == pytest.approx(3.0 * 10.0, rel=0.3)


def test_the_noise_gain_reads_a_photon_counts():
    rng = np.random.default_rng(0)
    lines = [np.array([[z, 20.0, x] for x in np.arange(5.0, 55.0)]) for z in (5.0, 10.0, 15.0)]
    volume = rng.poisson(40.0, size=(20, 40, 60)).astype(np.float32)
    assert image_noise_gain(volume, lines, (1.0, 1.0, 1.0)) == pytest.approx(1.0, rel=0.25)
    assert image_noise_gain(3.0 * volume, lines, (1.0, 1.0, 1.0)) == pytest.approx(3.0, rel=0.25)


# --- planting -----------------------------------------------------------------


def test_planted_vessels_keep_clear_of_the_real_ones(scene):
    raw, mask, lines, truth = scene
    G = _graph(lines, list(truth * 1.1))
    before = raw.copy()

    planted = plant_vessels(
        raw, G, list(G.edges(keys=True)), mask, VOXEL,
        image_psf_um=EFFECTIVE_PSF, rng=np.random.default_rng(0),
    )

    assert planted is not None
    assert planted.probe.number_of_edges() >= len(TUBES)
    np.testing.assert_array_equal(raw, before)  # drawn in a copy
    vessel_points = np.argwhere(mask) * np.asarray(VOXEL)
    for _u, _v, data in planted.probe.edges(data=True):
        line = np.asarray(data["voxels"])
        nearest = min(np.min(np.linalg.norm(vessel_points - p, axis=1)) for p in line[:: len(line) // 4])
        clearance = fwhm_planted.PLANTED_CLEARANCE_MULTIPLE * data["planted_diameter_um"]
        assert nearest >= clearance
        guide = data["edt_diameter_um"]
        assert (1 - fwhm_planted.PLANTED_WIDTH_SPREAD) * guide <= data["planted_diameter_um"]
        assert data["planted_diameter_um"] <= (1 + fwhm_planted.PLANTED_WIDTH_SPREAD) * guide


def test_planted_vessels_are_read_about_as_well_as_real_ones(scene):
    """What makes them worth scoring by: FWHM's error on them is the size of
    its error on the real vessels beside them."""
    raw, mask, lines, truth = scene
    G = _graph(lines, list(truth * 1.1))
    planted = plant_vessels(
        raw, G, list(G.edges(keys=True)), mask, VOXEL,
        image_psf_um=EFFECTIVE_PSF, rng=np.random.default_rng(0),
    )

    _fwhm(G, raw)
    probe = planted.probe.copy()
    _fwhm(probe, planted.volume)

    real = np.array([d["fwhm_diameter_um"] for _u, _v, d in G.edges(data=True)])
    real_error = float(np.mean(np.abs(real - truth) / truth))
    planted_error = planted_width_report(probe)["relative_error"]
    assert abs(planted_error - real_error) < 0.05, (planted_error, real_error)


def test_a_planted_gaussian_fit_reads_narrow_as_a_real_one_does(scene):
    """The planted vessels tell the profile models apart the way the real
    ones do: a Gaussian reads a plasma-filled lumen narrow."""
    raw, mask, lines, truth = scene
    G = _graph(lines, list(truth * 1.1))
    planted = plant_vessels(
        raw, G, list(G.edges(keys=True)), mask, VOXEL,
        image_psf_um=EFFECTIVE_PSF, rng=np.random.default_rng(0),
    )
    errors = {}
    for model in ("blurred_lumen", "gaussian"):
        probe = planted.probe.copy()
        _fwhm(probe, planted.volume, profile_model=model)
        errors[model] = planted_width_report(probe)["relative_error"]
    assert errors["gaussian"] > errors["blurred_lumen"] + 0.03, errors


def test_the_width_report_counts_an_unmeasured_vessel_as_all_wrong():
    probe = nx.MultiGraph()
    probe.add_edge(0, 1, planted_diameter_um=4.0, fwhm_diameter_um=4.4)  # 10% off
    probe.add_edge(2, 3, planted_diameter_um=5.0, fwhm_diameter_um=5.0)  # right
    probe.add_edge(4, 5, planted_diameter_um=6.0)  # not measured
    probe.add_edge(6, 7, planted_diameter_um=2.0, fwhm_diameter_um=9.0)  # 350% off: counts as 1

    report = planted_width_report(probe)

    assert report["planted"] == 4 and report["planted_measured"] == 3
    assert report["measured_fraction"] == pytest.approx(0.75)
    assert report["relative_error"] == pytest.approx((0.1 + 0.0 + 1.0 + 1.0) / 4)
    assert report["relative_error_measured"] == pytest.approx(0.1)
