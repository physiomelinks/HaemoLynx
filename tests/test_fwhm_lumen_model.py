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


# --- a dim, photon-limited lumen channel -----------------------------------------


_DIM_VOXEL_SIZE_ZYX = (1.0, 0.98, 0.98)


def _dim_lumen_vessel(diameter_um, *, seed):
    """About one photon per lumen pixel, each photon reading 25 grey levels --
    the statistics of a real dim dextran channel: a single transverse line
    across it is mostly isolated one-pixel spikes."""
    rng = np.random.default_rng(seed)
    vox = _DIM_VOXEL_SIZE_ZYX
    nz, ny, nx_ = 30, int(4 * diameter_um / vox[1]) + 40, 160
    zz, yy = np.meshgrid(np.arange(nz) * vox[0], np.arange(ny) * vox[1], indexing="ij")
    cz, cy = nz * vox[0] / 2, ny * vox[1] / 2
    lumen = (np.hypot(zz - cz, yy - cy) <= diameter_um / 2)[..., None] * np.ones(nx_)
    expected = 0.08 + 0.8 * gaussian_filter(lumen, sigma=[0.5 / v for v in vox])
    image = (25.0 * rng.poisson(expected)).astype(np.float32)
    xs = np.arange(8, nx_ - 8) * vox[2]
    centreline = np.stack([np.full_like(xs, cz), np.full_like(xs, cy), xs], axis=1)
    graph = nx.MultiGraph()
    graph.add_node(0, pos=centreline[0])
    graph.add_node(1, pos=centreline[-1])
    graph.add_edge(0, 1, key=0, voxels=centreline.tolist())
    return graph, image


def _measure_dim(diameter_um, seed, **overrides):
    graph, image = _dim_lumen_vessel(diameter_um, seed=seed)
    measure_edge_diameters_fwhm_from_raw_tiff(
        graph,
        raw_tiff_path="not-read.tif",
        voxel_size_zyx=_DIM_VOXEL_SIZE_ZYX,
        raw_volume=image,
        **_pipeline_defaults(
            sample_spacing_along_edge_um=4.0, diameter_guess_um=4.0, **overrides
        ),
    )
    return graph[0][1][0].get("fwhm_diameter_um")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_dim_lumen_is_measured_at_its_width_not_a_noise_spikes(seed):
    """Regression, from a real run: on a dim dextran channel every vessel
    measured 1-2um -- one line across it is mostly one-pixel spikes, the
    central-lobe clip cut at the first one, and the fit measured that spike.
    Averaging along the vessel, a two-pixel floor and a clip that smooths
    over two pixels measure the vessel instead."""
    diameter_um = 6.0

    old = _measure_dim(
        diameter_um, seed,
        longitudinal_average_um=0.0, min_diameter_pixels=0.0,
        clip_decision_smoothing_um=1.0,
    )
    new = _measure_dim(diameter_um, seed)

    assert old is not None and old < 0.6 * diameter_um
    # About one photon per pixel: still noisy, but a vessel, not a spike.
    assert new == pytest.approx(diameter_um, rel=0.25)


def test_averaging_along_the_vessel_keeps_the_profile_and_cuts_the_noise():
    from haemolynx.haemodynamics.automated import _sample_transverse_profile

    graph, clean, _cy = _plasma_vessel(10.0)
    rng = np.random.default_rng(3)
    noisy = clean + rng.normal(0.0, 0.3, clean.shape).astype(np.float32)
    centreline = np.asarray(graph[0][1][0]["voxels"])
    centre = centreline[len(centreline) // 2]
    labels = np.zeros(clean.shape, dtype=np.int32)

    def profile(image, average_um):
        return _sample_transverse_profile(
            image, labels, centre, np.array([0.0, 0.0, 1.0]), 1, 20.0, 0.25,
            VOXEL_SIZE_ZYX, background_label=0, junction_label=None,
            longitudinal_average_um=average_um,
        )

    offsets, single = profile(noisy, 0.0)
    offsets_avg, averaged = profile(noisy, 6.0)
    _offsets, truth = profile(clean, 6.0)

    np.testing.assert_array_equal(offsets, offsets_avg)
    assert np.std(averaged - truth) < 0.5 * np.std(single - truth)


def test_a_width_under_two_pixels_is_rejected(monkeypatch):
    """No fit narrower than two pixels is a resolved vessel; on real data it
    is almost always a single noise spike."""
    from haemolynx.haemodynamics import automated

    monkeypatch.setattr(
        automated, "_lumen_fwhm_fit_with_diagnostics",
        lambda *args, **kwargs: (1.5, 0.0, 0.99),
    )
    assert _measure_dim(6.0, 0) is None  # 1.5um < 2 x 0.98um
    assert _measure_dim(6.0, 0, min_diameter_pixels=0.0) == pytest.approx(1.5)


def test_the_noise_defaults_are_the_pipeline_defaults():
    by_name = {setting.name: setting for setting in SCHEMA}
    assert by_name["fwhm_longitudinal_average_um"].default == 4.0
    assert by_name["fwhm_min_diameter_pixels"].default == 2.0


# --- where another vessel actually is ---------------------------------------------


def _ray(vessel_mask, start_y=10):
    """Distance a +y ray from (z=2, y=start_y, x=5) travels, 1um voxels."""
    from haemolynx.haemodynamics.automated import _max_extent_along_ray

    labels = np.zeros(vessel_mask.shape, dtype=np.int32)
    return _max_extent_along_ray(
        np.array([2.0, float(start_y), 5.0]), np.array([0.0, 1.0, 0.0]), 1, labels,
        30.0, (1.0, 1.0, 1.0), 0.25, background_label=0, junction_label=None,
        allow_junction_crossing=False, vessel_mask=vessel_mask,
    )


def _two_vessels(gap_voxels):
    """Own vessel y 8..12, background, then a neighbour from 13 + gap."""
    mask = np.zeros((5, 60, 10), dtype=bool)
    mask[:, 8:13, :] = True
    mask[:, 13 + gap_voxels:20 + gap_voxels, :] = True
    return mask


def test_a_ray_stops_where_it_enters_a_neighbouring_vessel():
    """The one case a line may be shorter than 3x the width: something is in
    the way. Read from the segmentation, not guessed from other centrelines."""
    distance = _ray(_two_vessels(gap_voxels=4))
    # Own vessel ends at y=12.5, the neighbour starts at y=16.5: the ray from
    # y=10 ends just before it.
    assert 5.5 <= distance <= 6.5


def test_a_sub_voxel_gap_is_the_masks_pixelation_not_a_second_vessel():
    """Two regions touching (no voxel of background between) are one vessel as
    far as the mask can tell; the ray carries on through."""
    assert _ray(_two_vessels(gap_voxels=0)) == pytest.approx(30.0)


def test_a_ray_starting_outside_the_mask_ignores_it():
    """A centreline off its own segmentation cannot tell its own vessel from
    a neighbour, so the mask must not stop it."""
    assert _ray(_two_vessels(gap_voxels=4), start_y=4) == pytest.approx(30.0)


def test_a_ray_with_nothing_in_the_way_runs_its_full_length():
    mask = np.zeros((5, 60, 10), dtype=bool)
    mask[:, 8:13, :] = True
    assert _ray(mask) == pytest.approx(30.0)


# --- the 3x rule ---------------------------------------------------------------------


def _measure_with_fixed_fit(monkeypatch, width_um, **overrides):
    """A plain vessel measured by a fit that reports *width_um* on the first,
    12um line and fails on any wider one -- the widening loop then keeps the
    12um pass, as it does when a wider pass fails a quality gate."""
    from haemolynx.haemodynamics import automated

    def fit(positions, _intensities, **_kwargs):
        if float(np.ptp(positions)) <= 12.1:
            return width_um, 0.0, 0.99
        return None, None, None

    monkeypatch.setattr(automated, "_lumen_fwhm_fit_with_diagnostics", fit)
    graph, image, _cy = _plasma_vessel(4.0)
    summary = measure_edge_diameters_fwhm_from_raw_tiff(
        graph,
        raw_tiff_path="not-read.tif",
        voxel_size_zyx=VOXEL_SIZE_ZYX,
        raw_volume=image,
        **_pipeline_defaults(
            sample_spacing_along_edge_um=10.0,
            transverse_half_extent_um=6.0,
            diameter_guess_um=1.0,
            max_transverse_widen_passes=0,
            clip_profile_to_single_vessel=False,
            **overrides,
        ),
    )
    return graph[0][1][0].get("fwhm_diameter_um"), summary


def test_a_sample_whose_line_never_reached_three_widths_is_dropped(monkeypatch):
    """A 12um line cannot measure a 6um width by FWHM: at most 3um of it on
    each side is background. The widening loop used to keep such a pass
    whenever a wider one failed a gate -- a third of accepted samples on a
    real dim run, down to a line exactly as wide as its own measurement."""
    width, summary = _measure_with_fixed_fit(monkeypatch, 6.0)
    assert width is None
    assert summary["samples_rejected_short_line"] > 0

    width, summary = _measure_with_fixed_fit(monkeypatch, 3.9)  # 12um >= 3 x 3.9
    assert width == pytest.approx(3.9)
    assert summary["samples_rejected_short_line"] == 0


def test_a_short_line_is_kept_when_another_vessel_stopped_it(monkeypatch):
    graph, image, cy = _plasma_vessel(4.0)
    mask = np.zeros(image.shape, dtype=bool)
    iy = lambda y_um: int(round(y_um / VOXEL_SIZE_ZYX[1]))  # noqa: E731
    mask[:, iy(cy - 2.0):iy(cy + 2.0) + 1, :] = True  # the vessel itself
    mask[:, iy(cy + 5.0):iy(cy + 8.0), :] = True  # a neighbour 3um away

    width, _summary = _measure_with_fixed_fit(monkeypatch, 6.0, vessel_mask=mask)

    assert width == pytest.approx(6.0)
