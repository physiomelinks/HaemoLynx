"""FWHM vessel diameters on plasma-labelled vessels: the lumen model and the
first-ray cap that used to read a 25um vessel as 3.5um."""
from __future__ import annotations

import inspect

import networkx as nx
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from haemolynx.haemodynamics.automated import (
    _GAUSSIAN_FWHM_FROM_SIGMA,
    _blurred_lumen_1d,
    _blurred_lumen_fwhm,
    _fwhm_gaussian_fit_with_diagnostics,
    _lumen_fwhm_fit_with_diagnostics,
    measure_edge_diameters_fwhm_from_raw_tiff,
)
from haemolynx.pipeline import default_schema

SCHEMA = default_schema()
VOXEL_SIZE_ZYX = (1.0, 0.5, 0.5)


# --- the model -----------------------------------------------------------------


def test_a_wide_lumens_half_maximum_width_is_the_lumen_itself():
    assert _blurred_lumen_fwhm(20.0, 0.7) == pytest.approx(20.0, rel=1e-6)


def test_a_sub_resolution_lumens_half_maximum_width_is_the_blurs():
    """So a capillary below the resolution measures what the Gaussian fit
    always measured for it."""
    assert _blurred_lumen_fwhm(0.0, 1.0) == pytest.approx(_GAUSSIAN_FWHM_FROM_SIGMA)
    assert _blurred_lumen_fwhm(1e-4, 1.0) == pytest.approx(_GAUSSIAN_FWHM_FROM_SIGMA, rel=1e-4)


def test_the_half_maximum_width_grows_with_the_lumen():
    widths = [_blurred_lumen_fwhm(w, 1.0) for w in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)]
    assert widths == sorted(widths)
    assert all(fwhm >= w for fwhm, w in zip(widths, (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)))


def _profile(model, *params, step=0.25, half=40.0):
    x = np.arange(-half, half + step / 2, step)
    return x, model(x, *params)


def test_a_flat_topped_profile_is_measured_at_its_lumen_width():
    """What a plasma-filled vessel looks like across: a filled column with
    blurred edges. The Gaussian fit this replaces reads it ~10% narrow."""
    x, y = _profile(_blurred_lumen_1d, 0.2, 0.8, 0.0, 20.0, 0.7)

    lumen, centre, r2 = _lumen_fwhm_fit_with_diagnostics(x, y)
    gaussian, _centre, _r2 = _fwhm_gaussian_fit_with_diagnostics(x, y)

    assert lumen == pytest.approx(20.0, rel=0.01)
    assert centre == pytest.approx(0.0, abs=0.05)
    assert r2 > 0.999
    assert gaussian < 0.95 * 20.0


def test_a_gaussian_profile_still_measures_its_gaussian_fwhm():
    sigma = 2.0
    x = np.arange(-30.0, 30.0, 0.25)
    y = 0.1 + np.exp(-0.5 * (x / sigma) ** 2)

    lumen, _centre, r2 = _lumen_fwhm_fit_with_diagnostics(x, y)

    assert lumen == pytest.approx(_GAUSSIAN_FWHM_FROM_SIGMA * sigma, rel=0.03)
    assert r2 > 0.99


def test_an_off_centre_lumen_reports_its_own_centre():
    x, y = _profile(_blurred_lumen_1d, 0.1, 1.0, 3.0, 12.0, 0.8)
    _fwhm, centre, _r2 = _lumen_fwhm_fit_with_diagnostics(x, y)
    assert centre == pytest.approx(3.0, abs=0.05)


# --- end to end, with the pipeline's own defaults --------------------------------


def _pipeline_defaults(**overrides):
    parameters = inspect.signature(measure_edge_diameters_fwhm_from_raw_tiff).parameters
    kwargs = {
        setting.name[len("fwhm_"):]: setting.default
        for setting in SCHEMA
        if setting.name.startswith("fwhm_")
        and setting.name != "fwhm_raw_tiff_path"  # passed explicitly
        and setting.name[len("fwhm_"):] in parameters
    }
    kwargs.update(overrides)
    return kwargs


def _plasma_vessel(diameter_um, *, seed=0):
    """A straight plasma-labelled vessel along x: a filled column with dark
    red-cell shadows, blurred and noisy, on a dimmer tissue background."""
    rng = np.random.default_rng(seed)
    nz, ny, nx_ = 40, int(3 * diameter_um / VOXEL_SIZE_ZYX[1]) + 60, 200
    zz, yy = np.meshgrid(
        np.arange(nz) * VOXEL_SIZE_ZYX[0], np.arange(ny) * VOXEL_SIZE_ZYX[1], indexing="ij"
    )
    cz, cy = nz * VOXEL_SIZE_ZYX[0] / 2, ny * VOXEL_SIZE_ZYX[1] / 2
    lumen = (np.hypot(zz - cz, yy - cy) <= diameter_um / 2)[..., None] * np.ones(nx_)
    cells = gaussian_filter((rng.random(lumen.shape) < 0.02).astype(float), sigma=(2, 4, 4))
    signal = lumen * (1.0 - 0.6 * cells / max(cells.max(), 1e-9))
    blur = [0.6 / v for v in VOXEL_SIZE_ZYX]
    image = 0.2 + 0.8 * gaussian_filter(signal, sigma=blur) + rng.normal(0, 0.04, lumen.shape)
    xs = np.arange(10, nx_ - 10) * VOXEL_SIZE_ZYX[2]
    centreline = np.stack([np.full_like(xs, cz), np.full_like(xs, cy), xs], axis=1)
    graph = nx.MultiGraph()
    graph.add_node(0, pos=centreline[0])
    graph.add_node(1, pos=centreline[-1])
    # The per-edge starting guess the pipeline seeds FWHM with (EDT, from the mask).
    graph.add_edge(0, 1, key=0, voxels=centreline.tolist(), edt_diameter_um=float(diameter_um))
    return graph, image.astype(np.float32), cy


@pytest.mark.parametrize("diameter_um", [4.0, 12.0, 25.0])
def test_a_plasma_vessel_is_measured_at_its_own_diameter(diameter_um):
    """Regression: the first transverse ray was capped by the distance to
    "non-local" points of the vessel's own straight centreline -- ~5um with
    the defaults, whatever the vessel -- so a 25um vessel was sampled from
    inside its own lumen and measured 3.5um, and most 15um vessels not at all."""
    graph, image, _cy = _plasma_vessel(diameter_um)

    measure_edge_diameters_fwhm_from_raw_tiff(
        graph,
        raw_tiff_path="not-read.tif",
        voxel_size_zyx=VOXEL_SIZE_ZYX,
        raw_volume=image,
        **_pipeline_defaults(sample_spacing_along_edge_um=8.0),
    )

    data = graph[0][1][0]
    assert data["fwhm_status"] == "measured"
    assert data["fwhm_diameter_um"] == pytest.approx(diameter_um, rel=0.08)
    assert len(data["fwhm_diameter_samples_um"]) >= 5


def test_the_drawn_lines_span_the_measured_diameter_across_the_vessel():
    """What the viewer draws: the width each sample measured, centred on the
    vessel -- not the sampled ray, which is several times wider on purpose."""
    diameter_um = 20.0
    graph, image, cy = _plasma_vessel(diameter_um)

    measure_edge_diameters_fwhm_from_raw_tiff(
        graph,
        raw_tiff_path="not-read.tif",
        voxel_size_zyx=VOXEL_SIZE_ZYX,
        raw_volume=image,
        store_profile_debug=True,
        **_pipeline_defaults(sample_spacing_along_edge_um=8.0),
    )

    data = graph[0][1][0]
    measured = [np.asarray(line) for line in data["fwhm_measured_lines_phys"]]
    sampled = [np.asarray(line) for line in data["fwhm_profile_lines_phys"]]
    assert len(measured) == len(sampled) == len(data["fwhm_diameter_samples_um"])
    for line, diameter in zip(measured, data["fwhm_diameter_samples_um"]):
        assert np.linalg.norm(line[1] - line[0]) == pytest.approx(diameter, rel=1e-9)
        # Across the vessel (in y, at fixed x) and centred on it.
        assert line[0][2] == pytest.approx(line[1][2], abs=1e-6)
        assert 0.5 * (line[0][1] + line[1][1]) == pytest.approx(cy, abs=1.0)
        assert min(line[0][1], line[1][1]) == pytest.approx(cy - diameter_um / 2, abs=1.5)
    assert all(np.linalg.norm(s[-1] - s[0]) >= 2.0 * diameter_um for s in sampled)


def test_the_pipeline_defaults_fit_plasma_vessels():
    by_name = {setting.name: setting for setting in SCHEMA}
    assert by_name["fwhm_profile_model"].default == "blurred_lumen"
    assert set(by_name["fwhm_profile_model"].choices) == {"blurred_lumen", "gaussian"}
    # A plasma-filled lumen *is* a plateau: on by default, this threw away
    # the best samples of every wide vessel.
    assert by_name["fwhm_reject_samples_with_plateau_shape"].default is False


def test_an_unknown_profile_model_is_refused():
    graph, image, _cy = _plasma_vessel(4.0)
    with pytest.raises(ValueError, match="profile_model"):
        measure_edge_diameters_fwhm_from_raw_tiff(
            graph,
            raw_tiff_path="not-read.tif",
            voxel_size_zyx=VOXEL_SIZE_ZYX,
            raw_volume=image,
            **_pipeline_defaults(profile_model="tophat"),
        )
