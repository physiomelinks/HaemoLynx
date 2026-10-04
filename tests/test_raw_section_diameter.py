"""The raw cross-section fallback: a vessel's width from the raw image's own
section, fitted as a blurred lumen where FWHM fails.

Synthetic vessels are built like a dim two-photon channel: a plasma-filled
lumen, a blur several times longer in z than across, a few photons per
voxel and shot noise.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from haemolynx.haemodynamics import raw_section
from haemolynx.haemodynamics.raw_section import (
    estimate_psf_sigma,
    measure_edge_diameters_from_raw_sections,
)

VOXEL = (1.0, 0.98, 0.98)  # z, y, x: the real run's
PSF = (1.2, 0.35, 0.35)  # the optics'
#: What the fit sees: the optics, and the voxels' own box on top.
EFFECTIVE_PSF = tuple(float(np.hypot(s, v / np.sqrt(12.0))) for s, v in zip(PSF, VOXEL))


def _render(tubes, shape=(36, 44, 44), *, peak_counts=6.0, background=0.2, seed=0):
    """A noisy raw volume of *tubes*: (diameter, direction, centre offset um).

    Returns the volume, its mask (lumen > half) and each tube's centreline."""
    rng = np.random.default_rng(seed)
    spacing = np.asarray(VOXEL)
    points = np.stack(np.indices(shape), axis=-1) * spacing
    middle = np.asarray(shape) * spacing / 2
    lumen = np.zeros(shape)
    lines = []
    sub = (np.arange(2) + 0.5) / 2 - 0.5
    for diameter, direction, shift in tubes:
        d = np.asarray(direction, dtype=float)
        d /= np.linalg.norm(d)
        centre = middle + np.asarray(shift, dtype=float)
        occupancy = np.zeros(shape)
        for dz in sub:
            for dy in sub:
                for dx in sub:
                    rel = points + np.array([dz, dy, dx]) * spacing - centre
                    along = rel @ d
                    across = rel - along[..., None] * d
                    occupancy += np.linalg.norm(across, axis=-1) <= diameter / 2
        lumen = np.maximum(lumen, occupancy / 8.0)
        t = np.linspace(-14.0, 14.0, 29)
        lines.append(centre + t[:, None] * d)
    blurred = gaussian_filter(lumen, sigma=np.asarray(PSF) / spacing)
    raw = rng.poisson(background + peak_counts * blurred).astype(np.float32)
    return raw, lumen > 0.5, lines


def _graph(lines, guides):
    graph = nx.MultiGraph()
    for index, (line, guide) in enumerate(zip(lines, guides)):
        graph.add_edge(
            2 * index, 2 * index + 1, key=0, voxels=[tuple(p) for p in line],
            length=float(np.linalg.norm(line[-1] - line[0])), edt_diameter_um=guide,
        )
    return graph


def _measure(graph, raw, mask=None, **kwargs):
    kwargs.setdefault("psf_sigma_zyx", EFFECTIVE_PSF)
    kwargs.setdefault("branch_endpoint_exclusion_um", 0.0)
    return measure_edge_diameters_from_raw_sections(
        graph, raw_volume=raw, voxel_size_zyx=VOXEL, vessel_mask=mask, **kwargs
    )


@pytest.mark.parametrize("diameter", [4.0, 8.0])
@pytest.mark.parametrize("direction", [(0, 0, 1), (1, 1, 1)], ids=["along_x", "oblique"])
def test_a_vessel_is_read_at_its_width(diameter, direction):
    raw, mask, lines = _render([(diameter, direction, (0, 0, 0))], seed=int(diameter))
    graph = _graph(lines, [diameter * 1.2])

    _measure(graph, raw, mask)

    data = graph[0][1][0]
    assert data["raw_section_status"] == "measured"
    assert data["raw_section_diameter_um"] == pytest.approx(diameter, rel=0.12)


def test_a_vessel_running_along_z_is_measured():
    """A line in the y-x plane is not across a z-running vessel, so FWHM
    never measures one; its section is an ordinary disc."""
    raw, mask, lines = _render([(5.0, (1, 0.05, 0.05), (0, 0, 0))], seed=3)
    graph = _graph(lines, [5.0])

    _measure(graph, raw, mask)

    assert graph[0][1][0]["raw_section_diameter_um"] == pytest.approx(5.0, rel=0.12)


def test_a_uint16_stack_is_read_as_intensity_not_truncated():
    """Regression: map_coordinates returns the volume's own dtype, so a uint16
    stack came back rounded to whole counts, with nothing marking where a
    section left the volume."""
    raw, mask, lines = _render([(6.0, (0, 0, 1), (0, 0, 0))], seed=5)
    graph = _graph(lines, [6.0])

    _measure(graph, raw.astype(np.uint16), mask)

    assert graph[0][1][0]["raw_section_diameter_um"] == pytest.approx(6.0, rel=0.12)


def test_a_neighbouring_vessel_is_left_out_of_the_fit():
    """Two vessels a micron apart: the mask tells the fit which is which."""
    raw, mask, lines = _render(
        [(4.0, (0, 0, 1), (0, -3.5, 0)), (8.0, (0, 0, 1), (0, 3.5, 0))], seed=7
    )
    graph = _graph(lines[:1], [4.0])

    _measure(graph, raw, mask)

    assert graph[0][1][0]["raw_section_diameter_um"] == pytest.approx(4.0, rel=0.15)


def test_texture_with_no_lumen_in_it_is_not_given_a_width():
    """Speckled tissue fluorescence, no vessel: a lumen fitted to a blob of
    it stands only a fluctuation or two above the speckle round it, and not
    at two readings along the vessel, so it is not a width. Judged against
    what the fit left unexplained instead, 57% of such readings passed."""
    rng = np.random.default_rng(11)
    shape = (36, 44, 44)
    field = gaussian_filter(rng.normal(size=shape), sigma=(1.2, 1.3, 1.3))
    raw = rng.poisson(6.0 * np.clip(field / field.std() + 1.0, 0.0, None)).astype(np.float32)
    middle = np.asarray(shape) * np.asarray(VOXEL) / 2
    line = middle + np.linspace(-14.0, 14.0, 29)[:, None] * np.array([0.0, 0.0, 1.0])
    graph = _graph([line], [5.0])

    summary = _measure(graph, raw)

    data = graph[0][1][0]
    assert "raw_section_diameter_um" not in data
    assert data["raw_section_status"].startswith("failed:")
    assert summary["edges_measured"] == 0


def test_only_the_edges_asked_for_are_read():
    raw, mask, lines = _render(
        [(5.0, (0, 0, 1), (0, -6.0, 0)), (5.0, (0, 0, 1), (0, 6.0, 0))], seed=13
    )
    graph = _graph(lines, [5.0, 5.0])

    _measure(graph, raw, mask, edges=[(2, 3, 0)])

    assert "raw_section_status" not in graph[0][1][0]
    assert graph[2][3][0]["raw_section_status"] == "measured"


def test_the_blur_is_estimated_from_wide_vessels():
    """Wide vessels show their edges' blur apart from their width; sections
    tilted towards z show the axial blur, the others the lateral one."""
    raw, mask, lines = _render(
        [(9.0, (0, 0, 1), (0, -9.0, 0)), (9.0, (0, 1, 0), (0, 0, 9.0)), (9.0, (1, 0.3, 0.2), (0, 6.0, -9.0))],
        shape=(40, 56, 56), peak_counts=8.0, seed=17,
    )
    # Six edges per vessel: the estimate reads one section at each's middle.
    pieces = [
        line[chunk[0]:chunk[-1] + 2]
        for line in lines
        for chunk in np.array_split(np.arange(len(line) - 1), 6)
    ]
    graph = _graph(pieces, [9.0] * len(pieces))

    psf, details = estimate_psf_sigma(graph, raw, VOXEL, vessel_mask=mask)

    assert psf is not None, details
    sigma_z, sigma_y, _sigma_x = psf
    assert sigma_z == pytest.approx(EFFECTIVE_PSF[0], rel=0.35)
    assert sigma_y == pytest.approx(EFFECTIVE_PSF[1], rel=0.5)
    assert sigma_z > 1.5 * sigma_y


def _sections_fitted_by(monkeypatch):
    """Record the first point of each edge whose section the blur estimate fits."""
    fitted = []

    def record(_raw, poly, *_args, **_kwargs):
        fitted.append(tuple(poly[0]))
        return None, "low_contrast"

    monkeypatch.setattr(raw_section, "fit_section", record)
    return fitted


def test_the_blur_is_estimated_only_from_the_edges_given(monkeypatch):
    """What a run passes when its haemodynamics will remove a branch without
    an inlet and an outlet: a vessel about to go must not shape the blur the
    kept ones are fitted with. Either way round names an edge."""
    lines = [np.array([[10.0, 10.0 + 10.0 * i, x] for x in np.linspace(0.0, 20.0, 5)])
             for i in range(4)]
    graph = _graph(lines, [9.0] * 4)
    raw = np.zeros((4, 4, 4), dtype=np.float32)
    fitted = _sections_fitted_by(monkeypatch)

    psf, details = estimate_psf_sigma(graph, raw, VOXEL, edges=[(0, 1, 0), (5, 4, 0)])

    assert psf is None
    assert details["sections_tried"] == 2
    assert sorted(fitted) == sorted([tuple(lines[0][0]), tuple(lines[2][0])])

    fitted.clear()
    estimate_psf_sigma(graph, raw, VOXEL)
    assert len(fitted) == 4  # no edges given: every wide vessel


def test_the_fallbacks_own_blur_estimate_samples_the_calibration_edges(monkeypatch):
    lines = [np.array([[10.0, 10.0 + 10.0 * i, x] for x in np.linspace(0.0, 20.0, 5)])
             for i in range(3)]
    graph = _graph(lines, [9.0] * 3)
    fitted = _sections_fitted_by(monkeypatch)

    summary = _measure(
        graph, np.zeros((4, 4, 4), dtype=np.float32), psf_sigma_zyx=None,
        calibration_edges=[(2, 3, 0)],
    )

    assert summary["skipped"] is True
    assert fitted == [tuple(lines[1][0])]


def test_without_wide_vessels_to_estimate_the_blur_nothing_is_measured():
    """A PSF guessed wrong reads small vessels far off (1.5x too wide reads a
    3 um vessel ~0.65 of its width), so with no way to estimate it the
    fallback does nothing rather than guess."""
    raw, mask, lines = _render([(3.0, (0, 0, 1), (0, 0, 0))], seed=19)
    graph = _graph(lines, [3.0])

    summary = _measure(graph, raw, mask, psf_sigma_zyx=None)

    assert summary["skipped"] is True
    assert "blur" in summary["reason"]
    assert "raw_section_status" not in graph[0][1][0]


def test_the_psf_used_is_recorded_on_the_graph():
    raw, mask, lines = _render([(5.0, (0, 0, 1), (0, 0, 0))], seed=23)
    graph = _graph(lines, [5.0])

    _measure(graph, raw, mask)

    assert graph.graph["raw_section_psf_sigma_zyx"] == pytest.approx(EFFECTIVE_PSF)
    assert graph.graph["raw_section_psf_source"] == "set"


def test_remeasuring_clears_an_earlier_width():
    rng = np.random.default_rng(29)
    raw = rng.poisson(0.2, size=(36, 44, 44)).astype(np.float32)  # nothing there
    middle = np.asarray(raw.shape) * np.asarray(VOXEL) / 2
    line = middle + np.linspace(-14.0, 14.0, 29)[:, None] * np.array([0.0, 0.0, 1.0])
    graph = _graph([line], [5.0])
    graph[0][1][0]["raw_section_diameter_um"] = 5.0

    _measure(graph, raw)

    assert "raw_section_diameter_um" not in graph[0][1][0]


def test_a_lumen_at_only_one_reading_is_not_a_vessel_width():
    """A blob of speckle can pass the contrast gate at one place; a lumen
    passes it all along the vessel, so one accepted reading is not enough."""
    raw, mask, lines = _render([(5.0, (0, 0, 1), (0, 0, 0))], seed=31)
    x = np.arange(raw.shape[2]) * VOXEL[2]
    middle = raw.shape[2] * VOXEL[2] / 2
    away = np.abs(x - middle) > 3.0  # the vessel only within 3 um of the middle reading
    rng = np.random.default_rng(31)
    raw[:, :, away] = rng.poisson(0.2, size=raw[:, :, away].shape)
    mask[:, :, away] = False
    graph = _graph(lines, [5.0])

    _measure(graph, raw, mask, sample_spacing_along_edge_um=7.0)

    data = graph[0][1][0]
    assert "raw_section_diameter_um" not in data
    assert data["raw_section_status"] == "failed:too_few_readings"
    assert raw_section.MIN_ACCEPTED_READINGS == 2


def test_a_vessel_fwhm_cannot_read_is_read_by_the_fallback_in_a_run(tmp_path):
    """End to end through assign_edge_diameters: where FWHM gives a vessel no
    width -- here made to, by asking for more accepted samples than the
    vessel has -- the raw section measures it before the mask or the table
    are reached. (This used a z-running vessel, which FWHM could not measure
    until its fold-back cap stopped taking every point above or below a
    sample as a fold.)"""
    import tifffile

    from haemolynx.haemodynamics import HaemodynamicsApplyConfig, assign_edge_diameters
    from haemolynx.haemodynamics.poiseuille import DIAMETER_SOURCE_RAW_SECTION

    raw, mask, lines = _render([(5.0, (1, 0.05, 0.05), (0, 0, 0))], seed=37)
    path = tmp_path / "raw.tif"
    tifffile.imwrite(path, raw)
    graph = _graph(lines, [5.0])
    graph[0][1][0]["branch_order"] = "B01"
    graph.graph["image_voxel_size_zyx"] = VOXEL
    from haemolynx.pipeline import default_schema

    schema = default_schema()
    defaults = {setting.name: setting.default for setting in schema}
    config = HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": {"B01": 9.0}},
        fwhm={
            **schema.section_values(defaults, "FWHM diameter measurement"),
            "use_fwhm_edge_diameters": True,
            "do_fwhm_measurement": True,
            "fwhm_raw_tiff_path": path,
            "fwhm_branch_endpoint_exclusion_um": 0.0,
            "fwhm_min_accepted_samples": 99,
            "use_raw_section_fallback": True,
            "raw_section_psf_sigma_xy_um": EFFECTIVE_PSF[1],
            "raw_section_psf_sigma_z_um": EFFECTIVE_PSF[0],
        },
        voxel_size_zyx=VOXEL,
    )

    graph, summary, _raw = assign_edge_diameters(graph, config, mask_volume=mask)

    data = graph[0][1][0]
    assert "fwhm_diameter_um" not in data
    assert data["diameter_source"] == DIAMETER_SOURCE_RAW_SECTION
    assert data["diameter_um"] == pytest.approx(5.0, rel=0.12)
    assert summary["raw_section"]["edges_measured"] == 1
