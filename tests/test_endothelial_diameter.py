"""The endothelial internal diameter: the lumen inside an endothelial stain's
wall, the run's alternative to FWHM, and the per-vessel diameter basis it
brings with it.

Synthetic vessels are hollow rings -- a dark lumen of known diameter inside a
thin bright wall -- seen through a blur longer along z than across, with shot
noise at the brightness of a real claudin-5 channel.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from haemolynx.haemodynamics import endothelial
from haemolynx.haemodynamics.endothelial import (
    calibrate_from_rings,
    fit_ring_section,
    measure_edge_diameters_from_endothelium,
)
from haemolynx.haemodynamics.automated import _arc_length_parameterize

VOXEL = (1.0, 0.98, 0.98)
OPTICS = (2.0, 0.5, 0.5)
#: What a fit sees: the optics, and the voxels' own box on top.
PSF = tuple(float(np.hypot(s, v / np.sqrt(12.0))) for s, v in zip(OPTICS, VOXEL))
WALL = 1.0


def _ring(inner_diameter, direction, *, wall=WALL, shape=(34, 44, 44), shift=(0.0, 0.0, 0.0),
          solid=False, half=False, seed=0, peak=300.0, background=5.0):
    """A noisy endothelial stack: a hollow vessel (or, *solid*, a filled one;
    or, *half*, a wall on one side only) and its centreline."""
    rng = np.random.default_rng(seed)
    spacing = np.asarray(VOXEL)
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    points = np.stack(np.indices(shape), axis=-1) * spacing
    centre = np.asarray(shape) * spacing / 2 + np.asarray(shift)
    occupancy = np.zeros(shape)
    sub = (np.arange(2) + 0.5) / 2 - 0.5
    for dz in sub:
        for dy in sub:
            for dx in sub:
                rel = points + np.array([dz, dy, dx]) * spacing - centre
                across = rel - (rel @ d)[..., None] * d
                r = np.linalg.norm(across, axis=-1)
                inside = r <= inner_diameter / 2 + wall
                if not solid:
                    inside &= r >= inner_diameter / 2
                if half:
                    reference = np.cross(d, [1.0, 0.0, 0.0])
                    if np.linalg.norm(reference) < 1e-6:
                        reference = np.array([0.0, 1.0, 0.0])
                    inside &= across @ reference > 0
                occupancy += inside
    blurred = gaussian_filter(occupancy / 8.0, np.asarray(OPTICS) / spacing)
    volume = rng.poisson(background + peak * blurred).astype(np.float32)
    line = centre + np.linspace(-12.0, 12.0, 25)[:, None] * d
    return volume, line


def _graph(lines, guide):
    graph = nx.MultiGraph()
    for index, line in enumerate(lines):
        graph.add_edge(2 * index, 2 * index + 1, key=0, voxels=[tuple(p) for p in line],
                       length=24.0, edt_diameter_um=guide)
    return graph


def _fit(volume, line, guide, **kwargs):
    s, total = _arc_length_parameterize(np.asarray(line))
    kwargs.setdefault("psf_sigma_zyx", PSF)
    kwargs.setdefault("wall_um", WALL)
    return fit_ring_section(volume, np.asarray(line), s, total, total / 2, VOXEL, guide_um=guide, **kwargs)


def _gate(fit):
    return endothelial._gate(fit, min_contrast=endothelial.ENDOTHELIAL_MIN_RING_CONTRAST, min_diameter_um=2.0)


# --- the ring fit -------------------------------------------------------------------


@pytest.mark.parametrize("inner", [6.0, 9.0])
@pytest.mark.parametrize("direction", [(0, 0, 1), (1, 1, 1)], ids=["in_plane", "oblique"])
def test_the_lumen_inside_the_wall_is_read_at_its_width(inner, direction):
    volume, line = _ring(inner, direction, seed=int(inner))

    fit, why = _fit(volume, line, 0.7 * inner)  # a plasma mask reads inside the wall

    assert why == "ok" and _gate(fit) == "ok"
    assert fit.diameter_um == pytest.approx(inner, rel=0.06)


def test_a_vessel_running_along_z_is_read_in_its_sharp_plane():
    volume, line = _ring(5.0, (1, 0.05, 0.05), seed=3)

    fit, _why = _fit(volume, line, 4.0)

    assert _gate(fit) == "ok"
    assert fit.diameter_um == pytest.approx(5.0, rel=0.06)


def test_under_a_long_z_blur_the_lumen_is_round_and_set_by_the_sharp_axis():
    """In the section of a vessel lying in the image plane, z runs along one
    axis and blurs it four times as much as the other; the lumen's width
    there is the image's guess, so it is fitted round."""
    volume, line = _ring(6.0, (0, 0, 1), seed=5)

    fit, _why = _fit(volume, line, 4.0)

    assert fit.sigma_other_um >= endothelial.ROUND_LUMEN_BLUR_RATIO * fit.sigma_across_um
    assert fit.aspect == pytest.approx(1.0)


def test_a_filled_vessel_has_no_lumen_to_read():
    """An unresolved small vessel is bright right through: no dark lumen, so
    no reading, and the vessel falls back to the next source."""
    volume, line = _ring(3.0, (1, 1, 1), solid=True, seed=7)

    fit, why = _fit(volume, line, 3.0)

    assert fit is None or _gate(fit) != "ok"


def test_a_wall_on_one_side_only_is_not_a_ring():
    """A junction, a neighbour or a merged structure fits a ring too, with its
    wall on one side: the wall must be seen most of the way round."""
    volume, line = _ring(7.0, (0, 1, 1), half=True, seed=9)

    fit, why = _fit(volume, line, 5.0)

    assert fit is None or _gate(fit) != "ok"
    if fit is not None:
        assert fit.wall_coverage < endothelial.MIN_WALL_COVERAGE


def test_a_free_wall_trades_its_thickness_against_the_lumen():
    """Why the wall is held fixed: across the image plane a thin wall's
    thickness and the blur look alike, so a wall fitted free comes out wider
    than it is -- and the lumen narrower -- where the fixed one does not."""
    volume, line = _ring(9.0, (0, 0, 1), seed=11)

    free, _ = _fit(volume, line, 6.0, wall_um=None, psf_sigma_zyx=tuple(0.7 * s for s in PSF))
    fixed, _ = _fit(volume, line, 6.0)

    assert free.wall_um > WALL
    assert abs(fixed.diameter_um - 9.0) < abs(free.diameter_um - 9.0)


# --- calibration and measurement -----------------------------------------------------


def test_the_blur_and_wall_are_estimated_from_wide_vessels():
    shape = (44, 60, 60)
    rings = [((0, 0, 1), (0, -9.0, 0)), ((0, 1, 0), (0, 0, 9.0)), ((1, 0.3, 0.2), (0, 7.0, -9.0))]
    volume = np.zeros(shape, dtype=np.float32)
    lines = []
    for index, (direction, shift) in enumerate(rings):
        part, line = _ring(9.0, direction, shape=shape, shift=shift, seed=20 + index, background=0.0)
        volume = np.maximum(volume, part)
        lines += [line[i * 4:(i + 1) * 4 + 1] for i in range(6)]  # six edges each, one section apiece
    graph = _graph(lines, 9.0)

    psf, wall, details = calibrate_from_rings(graph, volume, VOXEL)

    assert psf is not None, details
    sigma_z, sigma_y, _ = psf
    assert sigma_z > 2 * sigma_y  # the blur is longer along z
    assert sigma_z == pytest.approx(PSF[0], rel=0.4)
    assert wall == pytest.approx(WALL, abs=0.5)


def test_an_edge_needs_two_readings_and_records_its_widths():
    volume, line = _ring(7.0, (1, 1, 1), seed=13)
    graph = _graph([line], 5.0)

    summary = measure_edge_diameters_from_endothelium(
        graph, endothelial_volume=volume, voxel_size_zyx=VOXEL, psf_sigma_zyx=PSF, wall_um=WALL,
        branch_endpoint_exclusion_um=0.0,
    )

    data = graph[0][1][0]
    assert summary["edges_measured"] == 1
    assert data["endothelial_status"] == "measured"
    assert len(data["endothelial_diameter_samples_um"]) >= endothelial.MIN_ACCEPTED_READINGS
    assert data["endothelial_diameter_um"] == pytest.approx(7.0, rel=0.06)
    assert graph.graph["endothelial_wall_um"] == WALL
    assert graph.graph["endothelial_psf_source"] == "set"


def test_a_vessel_it_cannot_read_is_left_for_the_next_source():
    volume, line = _ring(2.5, (0, 0, 1), solid=True, seed=17)
    graph = _graph([line], 2.5)

    measure_edge_diameters_from_endothelium(
        graph, endothelial_volume=volume, voxel_size_zyx=VOXEL, psf_sigma_zyx=PSF, wall_um=WALL,
        branch_endpoint_exclusion_um=0.0,
    )

    assert "endothelial_diameter_um" not in graph[0][1][0]
    assert graph[0][1][0]["endothelial_status"].startswith("failed:")


# --- the chain and the per-vessel basis ----------------------------------------------


def test_the_endothelial_width_comes_first_and_carries_its_own_basis():
    from haemolynx.haemodynamics.poiseuille import (
        DIAMETER_SOURCE_EDT,
        DIAMETER_SOURCE_ENDOTHELIAL,
        EDGE_DIAMETER_BASIS,
        stamp_edge_diameters,
    )

    graph = nx.MultiGraph()
    graph.add_edge(0, 1, key=0, branch_order="B01", length=100.0,
                   endothelial_diameter_um=8.0, edt_diameter_um=5.0)
    graph.add_edge(1, 2, key=0, branch_order="B01", length=100.0, edt_diameter_um=5.0,
                   **{EDGE_DIAMETER_BASIS: "anatomical"})  # stale, from an earlier stamping

    counts = stamp_edge_diameters(graph, {"B01": 4.0}, use_edt_fallback=True, use_endothelial=True)

    first, second = graph[0][1][0], graph[1][2][0]
    assert (first["diameter_source"], first["diameter_um"]) == (DIAMETER_SOURCE_ENDOTHELIAL, 8.0)
    assert first[EDGE_DIAMETER_BASIS] == "anatomical"
    assert second["diameter_source"] == DIAMETER_SOURCE_EDT
    assert EDGE_DIAMETER_BASIS not in second
    assert counts["endothelial"] == 1


def test_a_hand_set_diameter_is_on_the_runs_basis():
    from haemolynx.haemodynamics.poiseuille import EDGE_DIAMETER_BASIS, set_edge_diameter_override

    data = {EDGE_DIAMETER_BASIS: "anatomical"}
    set_edge_diameter_override(data, 6.0)

    assert EDGE_DIAMETER_BASIS not in data


def test_a_vessel_measured_wall_to_wall_takes_the_anatomical_form_of_the_law():
    """One network, two bases: an endothelial edge's resistance is what a run
    on the anatomical basis would give it, beside a plasma-column edge's."""
    from haemolynx.haemodynamics.poiseuille import EDGE_DIAMETER_BASIS, PoiseuilleModel

    def network():
        graph = nx.MultiGraph()
        graph.add_edge(0, 1, key=0, branch_order="B01", length=100.0, diameter_um=6.0,
                       **{EDGE_DIAMETER_BASIS: "anatomical"})
        graph.add_edge(1, 2, key=0, branch_order="B01", length=100.0, diameter_um=6.0)
        return graph

    mixed, _ = PoiseuilleModel(40.0, 100.0).set_poiseuille_resistances(network(), {"B01": 6.0})
    plasma = PoiseuilleModel(40.0, 100.0, diameter_basis="plasma_column")
    anatomical = PoiseuilleModel(40.0, 100.0, diameter_basis="anatomical")

    assert mixed[0][1][0]["resistance"] == pytest.approx(anatomical.resistance_of_uniform_segment(100.0, 6.0))
    assert mixed[1][2][0]["resistance"] == pytest.approx(plasma.resistance_of_uniform_segment(100.0, 6.0))
    assert mixed[0][1][0]["resistance"] > mixed[1][2][0]["resistance"]


def test_a_constricted_vessel_keeps_its_own_basis():
    from haemolynx.haemodynamics.constriction import integrated_resistance
    from haemolynx.haemodynamics.constriction_strategy import set_resistances_for_constriction_strategy
    from haemolynx.haemodynamics.poiseuille import EDGE_DIAMETER_BASIS

    graph = nx.MultiGraph()
    graph.add_edge(0, 1, key=0, branch_order="B01", length=100.0, diameter_um=6.0,
                   **{EDGE_DIAMETER_BASIS: "anatomical"})

    set_resistances_for_constriction_strategy(
        graph, diameter_by_branch_order={"B01": 6.0}, constriction_factor_by_branch_order={},
        use_pericyte_mask_constriction=False, prefer_edge_fwhm_baseline=False,
        constriction_length=40.0, constriction_spacing=100.0, default_constriction_factor=1.0,
        use_probabilistic_constriction=False, seed=1,
    )

    expected = integrated_resistance(
        length=100.0, d1=6.0, d2=6.0, constriction_centers=[], constriction_length=40.0,
        num_points=1000, diameter_basis="anatomical",
    )
    assert graph[0][1][0]["resistance"] == pytest.approx(expected, rel=1e-3)


# --- reading the stain, and the run --------------------------------------------------


def _composite(path, plasma, endothelium):
    import tifffile

    stack = np.stack([plasma, endothelium], axis=1).astype(np.uint16)  # z, c, y, x
    tifffile.imwrite(path, stack, imagej=True, metadata={"axes": "ZCYX"})


def test_one_channel_of_a_composite_is_read(tmp_path):
    from haemolynx.haemodynamics.automated import load_single_channel_tiff_volume

    rng = np.random.default_rng(0)
    plasma = rng.integers(0, 50, size=(6, 8, 9))
    wall = rng.integers(100, 500, size=(6, 8, 9))
    _composite(tmp_path / "composite.tif", plasma, wall)

    read = load_single_channel_tiff_volume(tmp_path / "composite.tif", channel=1)

    assert read.dtype == np.float32
    np.testing.assert_array_equal(read, wall)
    with pytest.raises(ValueError, match="has 2 channels"):
        load_single_channel_tiff_volume(tmp_path / "composite.tif", channel=2)


def test_endothelial_and_fwhm_diameters_are_alternatives():
    from haemolynx.pipeline.checks import check_one_primary_diameter_measurement

    both = check_one_primary_diameter_measurement(
        {"use_endothelial_diameters": True, "use_fwhm_edge_diameters": True}
    )
    one = check_one_primary_diameter_measurement(
        {"use_endothelial_diameters": True, "use_fwhm_edge_diameters": False}
    )

    assert not both.ok and one.ok


def test_the_endothelial_option_is_off_by_default():
    from haemolynx.pipeline import default_schema

    schema = default_schema()
    assert schema["use_endothelial_diameters"].default is False
    assert schema["endothelial_channel"].default is None


def test_a_run_reads_its_internal_diameters_from_the_stain(tmp_path):
    """End to end through assign_edge_diameters: channel 1 of a composite,
    the mask estimate as each vessel's guide and fallback, the endothelial
    width first and on its own basis."""
    from haemolynx.haemodynamics import HaemodynamicsApplyConfig, assign_edge_diameters
    from haemolynx.haemodynamics.poiseuille import DIAMETER_SOURCE_ENDOTHELIAL, EDGE_DIAMETER_BASIS

    volume, line = _ring(7.0, (1, 1, 1), seed=19)
    _composite(tmp_path / "composite.tif", np.zeros_like(volume), volume)
    graph = _graph([line], 5.0)
    graph[0][1][0]["branch_order"] = "B01"
    graph.graph["image_voxel_size_zyx"] = VOXEL
    config = HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": {"B01": 4.0}},
        endothelial={
            "use_endothelial_diameters": True,
            "endothelial_image_path": tmp_path / "composite.tif",
            "endothelial_channel": 1,
            "endothelial_wall_thickness_um": WALL,
            "endothelial_psf_sigma_xy_um": PSF[1],
            "endothelial_psf_sigma_z_um": PSF[0],
        },
        edt={"edt_junction_proximity_exclusion_um": 0.0},
        voxel_size_zyx=VOXEL,
    )

    graph, summary, _raw = assign_edge_diameters(graph, config)

    data = graph[0][1][0]
    assert summary["endothelial"]["edges_measured"] == 1
    assert data["diameter_source"] == DIAMETER_SOURCE_ENDOTHELIAL
    assert data[EDGE_DIAMETER_BASIS] == "anatomical"
    assert data["diameter_um"] == pytest.approx(7.0, rel=0.06)


# --- more than one process ----------------------------------------------------------


def _three_rings():
    shape = (34, 60, 60)
    volume = np.zeros(shape, dtype=np.float32)
    lines = []
    for index, (direction, shift) in enumerate((((1, 1, 1), (0, -12.0, 0)), ((0, 1, 1), (0, 0, 0)), ((1, 0.05, 0.05), (0, 12.0, 8.0)))):
        part, line = _ring(6.0, direction, shape=shape, shift=shift, seed=30 + index, background=0.0)
        volume = np.maximum(volume, part)
        lines.append(line)
    return volume, lines


def test_worker_processes_give_the_same_widths_as_one(monkeypatch):
    from haemolynx.haemodynamics import sections

    volume, lines = _three_rings()
    widths = []
    for workers in (1, 2):
        monkeypatch.setattr(sections, "MIN_EDGES_FOR_WORKERS", 1)
        graph = _graph(lines, 5.0)
        measure_edge_diameters_from_endothelium(
            graph, endothelial_volume=volume, voxel_size_zyx=VOXEL, psf_sigma_zyx=PSF,
            wall_um=WALL, branch_endpoint_exclusion_um=0.0, workers=workers,
        )
        widths.append([graph[2 * i][2 * i + 1][0].get("endothelial_diameter_samples_um") for i in range(3)])

    assert widths[0] == widths[1]
    assert all(w for w in widths[0])


def test_when_workers_cannot_start_it_measures_in_this_process(monkeypatch, caplog):
    import concurrent.futures

    from haemolynx.haemodynamics import sections

    def refuse(*_args, **_kwargs):
        raise OSError("no processes here")

    monkeypatch.setattr(sections, "MIN_EDGES_FOR_WORKERS", 1)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", refuse)
    volume, lines = _three_rings()
    graph = _graph(lines, 5.0)

    measure_edge_diameters_from_endothelium(
        graph, endothelial_volume=volume, voxel_size_zyx=VOXEL, psf_sigma_zyx=PSF,
        wall_um=WALL, branch_endpoint_exclusion_um=0.0, workers=4,
    )

    assert graph[0][1][0].get("endothelial_diameter_um")
    assert "measuring in this one" in caplog.text


def test_the_default_worker_count_leaves_a_core_and_stops_at_eight(monkeypatch):
    import os

    from haemolynx.haemodynamics import sections

    for cores, expected in ((20, 8), (4, 3), (1, 1), (None, 1)):
        monkeypatch.setattr(os, "cpu_count", lambda cores=cores: cores)
        assert sections.default_worker_count() == expected
