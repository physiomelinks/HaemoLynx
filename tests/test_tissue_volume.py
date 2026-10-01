"""The tissue's own volume, measured from the raw image, and the vessel density
that divides by it.

Most of a stack can be empty space at a lower background than the tissue's;
vessel density per *tissue* volume needs that space left out. These pin the
measurement against shapes whose volume is known exactly, and the pipeline
wiring that feeds it into the statistics.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import tifffile

from haemolynx.pipeline import default_schema, stages
from haemolynx.pipeline.checks import check_tissue_volume_image, preflight, tissue_raw_image
from haemolynx.pipeline.citations import (
    IMAGE_ANALYSIS_CITATIONS,
    render_citations,
    used_citations,
)
from haemolynx.statistics import (
    TISSUE_THRESHOLD_METHODS,
    compute_comprehensive_vessel_statistics,
    compute_vessel_density,
    measure_tissue_volume,
)
from haemolynx.statistics.tissue_volume import MIN_CLASS_SEPARATION

SCHEMA = default_schema()

# Coarse z, fine x and y -- three blocks' worth of anisotropy, so a z/x mix-up
# in the working grid or the vertex mapping cannot pass.
SLAB_VOXEL_SIZE_ZYX = (2.0, 0.5, 0.5)
SLAB_SHAPE = (40, 200, 200)
EMPTY, TISSUE, VESSEL = 20.0, 100.0, 1000.0


def _slab(noise: float = 0.0, seed: int = 0):
    """A tissue slab 50 x 70 x 100 um in an 80 x 100 x 100 um image.

    Voxel centres are at index x spacing, so the slab's faces are voxel faces:
    z from 9 to 59 um, y from the image face (-0.25) to 69.75 um, x through
    the whole image. Bright vessel columns run through it, 9% of its volume.
    Returns ``(raw, vessel_mask, tissue_mask)``.
    """
    z, y, x = np.indices(SLAB_SHAPE)
    zc, yc, xc = (index * spacing for index, spacing in zip((z, y, x), SLAB_VOXEL_SIZE_ZYX))
    tissue = (zc >= 10) & (zc < 60) & (yc < 70)
    vessel = tissue & ((yc % 10) < 3) & ((xc % 10) < 3)
    raw = np.where(tissue, TISSUE, EMPTY)
    if noise:
        raw = raw + np.random.default_rng(seed).normal(0.0, noise, SLAB_SHAPE)
    raw[vessel] = VESSEL
    return raw, vessel, tissue


SLAB_VOLUME = 50.0 * 70.0 * 100.0
#: The slab's own faces: top and bottom (70 x 100) and its y face (50 x 100).
SLAB_SURFACE = 2 * 70.0 * 100.0 + 50.0 * 100.0
#: Where the image cuts it: both x faces (50 x 70) and the y = 0 face (50 x 100).
SLAB_CUT = 2 * 50.0 * 70.0 + 50.0 * 100.0


# --- the measurement ----------------------------------------------------------


@pytest.mark.parametrize("method", ["otsu", "li", "triangle"])
def test_a_slab_cut_by_the_image_measures_its_own_volume(method):
    raw, vessel, _tissue = _slab(noise=8.0)
    tissue = measure_tissue_volume(
        raw, SLAB_VOXEL_SIZE_ZYX, vessel_mask=vessel, threshold_method=method
    )
    # Smoothing rounds the slab's two convex edges a little (about sigma
    # squared per micron of edge); on real tissue that is a far smaller share.
    assert tissue.volume_um3 == pytest.approx(SLAB_VOLUME, rel=0.02)
    assert tissue.surface_area_um2 == pytest.approx(SLAB_SURFACE, rel=0.03)
    assert tissue.image_volume_um3 == pytest.approx(80.0 * 100.0 * 100.0)
    assert tissue.tissue_fraction == pytest.approx(SLAB_VOLUME / tissue.image_volume_um3, rel=0.02)
    assert tissue.component_count == 1
    # Every method finds the same two classes, so the same surface.
    assert tissue.background_level == pytest.approx(EMPTY, abs=1.0)
    assert tissue.tissue_level == pytest.approx(TISSUE, abs=1.0)
    assert tissue.level == pytest.approx(0.5 * (EMPTY + TISSUE), abs=1.0)


def test_the_surface_lands_on_the_slab_faces_and_closes_on_the_image_faces():
    raw, vessel, _tissue = _slab()
    tissue = measure_tissue_volume(
        raw, SLAB_VOXEL_SIZE_ZYX, vessel_mask=vessel, smoothing_sigma_um=0.0
    )
    z, y, x = tissue.vertices.T
    # Marching cubes places vertices in float32.
    assert (z.min(), z.max()) == pytest.approx((9.0, 59.0), abs=1e-4)
    assert (y.min(), y.max()) == pytest.approx((-0.25, 69.75), abs=1e-4)
    # Cut by both x faces: the caps sit exactly on them.
    assert (x.min(), x.max()) == pytest.approx((-0.25, 99.75), abs=1e-4)
    assert tissue.volume_um3 == pytest.approx(SLAB_VOLUME, rel=0.005)
    assert tissue.cut_face_area_um2 == pytest.approx(SLAB_CUT, rel=0.1)
    assert tissue.cut_face_area_um2 + tissue.surface_area_um2 == pytest.approx(
        SLAB_CUT + SLAB_SURFACE, rel=0.03
    )


def test_vessels_inside_the_tissue_do_not_move_its_surface():
    """Read as the tissue's own level, vessel voxels neither drag a threshold
    up nor leave an edge block to its empty half."""
    raw, vessel, _tissue = _slab()
    without_vessels = np.where(vessel, TISSUE, raw)
    reference = measure_tissue_volume(without_vessels, SLAB_VOXEL_SIZE_ZYX)
    with_vessels = measure_tissue_volume(raw, SLAB_VOXEL_SIZE_ZYX, vessel_mask=vessel)
    assert with_vessels.volume_um3 == pytest.approx(reference.volume_um3, rel=1e-6)
    # Every vessel voxel, the ones in the blocks the surface crosses included.
    assert with_vessels.vessel_volume_um3 == pytest.approx(
        vessel.sum() * np.prod(SLAB_VOXEL_SIZE_ZYX)
    )


def test_without_the_vessel_mask_otsu_splits_vessels_from_everything_else():
    """Why the segmentation is passed in: bright vessels are their own class."""
    raw, vessel, _tissue = _slab()
    blind = measure_tissue_volume(raw, SLAB_VOXEL_SIZE_ZYX, threshold_method="otsu")
    masked = measure_tissue_volume(raw, SLAB_VOXEL_SIZE_ZYX, vessel_mask=vessel)
    assert blind.volume_um3 < 0.75 * SLAB_VOLUME
    assert masked.volume_um3 == pytest.approx(SLAB_VOLUME, rel=0.02)


def test_tissue_filling_the_image_is_the_whole_image_even_in_partial_blocks():
    """41 x 37 x 23 voxels do not divide into the working blocks; the closed
    surface still caps exactly on every face."""
    shape, spacing = (23, 37, 41), (1.5, 0.7, 0.3)
    raw = np.full(shape, TISSUE)
    raw[0, 0, 0] = EMPTY  # one dim voxel, so there are two classes to split
    tissue = measure_tissue_volume(
        raw, spacing, threshold_method="manual", manual_threshold=0.5 * (EMPTY + TISSUE),
        smoothing_sigma_um=2.0,
    )
    image_volume = float(np.prod([n * s for n, s in zip(shape, spacing)]))
    assert tissue.volume_um3 == pytest.approx(image_volume, rel=1e-3)
    assert tissue.voxel_volume_um3 == pytest.approx(image_volume)
    assert tissue.surface_area_um2 == pytest.approx(0.0, abs=1e-6 * image_volume)


def test_a_sphere_inside_the_image_measures_its_volume():
    shape, spacing, radius = (60, 60, 60), (1.0, 1.0, 1.0), 20.0
    z, y, x = np.indices(shape)
    inside = (z - 29.5) ** 2 + (y - 29.5) ** 2 + (x - 29.5) ** 2 <= radius**2
    raw = np.where(inside, TISSUE, EMPTY)
    # A little smoothing: through a bare voxelised edge marching cubes leaves
    # a staircase, whose area runs high even where its volume does not.
    tissue = measure_tissue_volume(raw, spacing, working_voxel_size_um=1.0, smoothing_sigma_um=1.0)
    assert tissue.volume_um3 == pytest.approx(4.0 / 3.0 * np.pi * radius**3, rel=0.02)
    assert tissue.surface_area_um2 == pytest.approx(4.0 * np.pi * radius**2, rel=0.03)
    assert tissue.cut_face_area_um2 == 0.0


def test_specks_in_the_empty_space_are_dropped_unless_asked_to_keep_them():
    raw, vessel, _tissue = _slab()
    # 12 um of debris 10 um beyond the slab's y face: 0.5% of the slab.
    raw[15:21, 160:184, 80:104] = TISSUE
    options = dict(vessel_mask=vessel, smoothing_sigma_um=2.0)
    kept = measure_tissue_volume(raw, SLAB_VOXEL_SIZE_ZYX, min_component_fraction=0.0, **options)
    dropped = measure_tissue_volume(raw, SLAB_VOXEL_SIZE_ZYX, **options)
    assert kept.component_count == 2
    assert dropped.component_count == 1
    assert dropped.volume_um3 == pytest.approx(SLAB_VOLUME, rel=0.02)
    assert kept.volume_um3 > dropped.volume_um3


def test_an_enclosed_dark_cavity_is_tissue_only_when_holes_are_filled():
    shape, spacing = (40, 40, 40), (1.0, 1.0, 1.0)
    raw = np.full(shape, EMPTY)
    raw[5:35, 5:35, 5:35] = TISSUE
    raw[15:25, 15:25, 15:25] = EMPTY
    options = dict(working_voxel_size_um=1.0, smoothing_sigma_um=0.0)
    filled = measure_tissue_volume(raw, spacing, fill_holes=True, **options)
    hollow = measure_tissue_volume(raw, spacing, fill_holes=False, **options)
    assert filled.volume_um3 == pytest.approx(30.0**3, rel=0.01)
    assert hollow.volume_um3 == pytest.approx(30.0**3 - 10.0**3, rel=0.02)


def test_a_manual_threshold_is_the_surface_level_itself():
    raw, vessel, _tissue = _slab()
    tissue = measure_tissue_volume(
        raw, SLAB_VOXEL_SIZE_ZYX, vessel_mask=vessel, threshold_method="manual",
        manual_threshold=30.0,
    )
    assert tissue.level == 30.0
    assert tissue.split == 30.0
    # Nearer the empty-space level than halfway: the surface sits further out.
    halfway = measure_tissue_volume(raw, SLAB_VOXEL_SIZE_ZYX, vessel_mask=vessel)
    assert tissue.volume_um3 > halfway.volume_um3


def test_no_tissue_above_a_manual_threshold_is_no_volume_and_no_surface():
    raw, _vessel, _tissue = _slab()
    tissue = measure_tissue_volume(
        raw, SLAB_VOXEL_SIZE_ZYX, threshold_method="manual", manual_threshold=10 * VESSEL
    )
    assert tissue.volume_um3 == 0.0
    assert tissue.voxel_volume_um3 == 0.0
    assert len(tissue.faces) == 0


def test_an_image_with_no_empty_space_warns_that_the_split_may_cut_the_tissue(caplog):
    raw = np.random.default_rng(1).normal(TISSUE, 5.0, (20, 40, 40))
    with caplog.at_level(logging.WARNING, logger="haemolynx.statistics.tissue_volume"):
        tissue = measure_tissue_volume(raw, (1.0, 1.0, 1.0))
    assert tissue.class_separation < MIN_CLASS_SEPARATION
    assert "barely differ" in caplog.text


def test_a_channel_dark_outside_the_vessels_warns_that_it_found_their_blur(caplog):
    """A dextran channel is black everywhere but the vessels: the 'tissue' an
    automatic split finds there is the vessels' halo, mostly vessel."""
    from scipy import ndimage

    from haemolynx.statistics.tissue_volume import MAX_PLAUSIBLE_VESSEL_FRACTION

    raw, vessel, _tissue = _slab()
    dextran = ndimage.gaussian_filter(np.where(vessel, VESSEL, 0.0), 1.0)
    with caplog.at_level(logging.WARNING, logger="haemolynx.statistics.tissue_volume"):
        found = measure_tissue_volume(dextran, SLAB_VOXEL_SIZE_ZYX, vessel_mask=vessel)
    assert found.vessel_volume_um3 > MAX_PLAUSIBLE_VESSEL_FRACTION * found.volume_um3
    assert "seems to show no tissue background" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="haemolynx.statistics.tissue_volume"):
        measure_tissue_volume(raw, SLAB_VOXEL_SIZE_ZYX, vessel_mask=vessel)
    assert "no tissue background" not in caplog.text


def test_a_memory_mapped_raw_volume_measures_the_same(tmp_path):
    raw, vessel, _tissue = _slab()
    mapped = np.memmap(tmp_path / "raw.memmap", dtype=np.float32, mode="w+", shape=raw.shape)
    mapped[:] = raw
    in_memory = measure_tissue_volume(raw.astype(np.float32), SLAB_VOXEL_SIZE_ZYX, vessel_mask=vessel)
    on_disk = measure_tissue_volume(mapped, SLAB_VOXEL_SIZE_ZYX, vessel_mask=vessel)
    assert on_disk.volume_um3 == pytest.approx(in_memory.volume_um3)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"vessel_mask": np.zeros((2, 2, 2))}, "segmentation's own grid"),
        ({"threshold_method": "huang"}, "threshold_method"),
        ({"threshold_method": "manual"}, "manual_threshold"),
    ],
)
def test_bad_inputs_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        measure_tissue_volume(np.zeros((4, 4, 4)), (1.0, 1.0, 1.0), **kwargs)


def test_the_schema_offers_every_threshold_method_and_no_other():
    assert SCHEMA["tissue_threshold_method"].choices == TISSUE_THRESHOLD_METHODS


# --- vessel density -----------------------------------------------------------


def _one_vessel_graph(length: float = 59.0) -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.add_node(0, pos=np.asarray([10.0, 20.0, 0.0]))
    G.add_node(1, pos=np.asarray([20.0, 30.0, length]))
    G.add_edge(0, 1, key=0, length=length, branch_order="B01")
    return G


def test_vessel_density_in_tissue_divides_by_the_measured_tissue():
    G = _one_vessel_graph()
    density = compute_vessel_density(
        G, nx.get_node_attributes(G, "pos"), (1.0, 1.0, 1.0), (10, 10, 10), True,
        tissue_volume_um3=1000.0, vessel_volume_um3=50.0,
    )
    assert density["Vessel Density in Tissue (microns/micron³)"] == pytest.approx(59.0 / 1000.0)
    assert density["Tissue Volume (micron³)"] == 1000.0
    assert density["Vascular Volume Fraction in Tissue"] == pytest.approx(0.05)
    assert density["Tissue Volume Source"] == "tissue surface measured from the raw image"


def test_without_a_measured_tissue_density_says_it_used_the_networks_box():
    G = _one_vessel_graph()
    density = compute_vessel_density(
        G, nx.get_node_attributes(G, "pos"), (1.0, 1.0, 1.0), (10, 10, 10), True
    )
    box = density["Vessel-Occupied Volume (micron³)"]
    assert density["Vessel Density in Tissue (microns/micron³)"] == pytest.approx(59.0 / box)
    assert density["Tissue Volume Source"] == "bounding box of the network's nodes"
    assert "Tissue Volume (micron³)" not in density
    assert "Vascular Volume Fraction in Tissue" not in density


def test_the_comprehensive_report_passes_the_tissue_to_vessel_density():
    G = _one_vessel_graph()
    stats = compute_comprehensive_vessel_statistics(
        G, node_positions=nx.get_node_attributes(G, "pos"),
        enabled_measures=frozenset({"vessel_density"}), tissue_volume_um3=2000.0,
    )
    assert stats["Vessel Density in Tissue (microns/micron³)"] == pytest.approx(59.0 / 2000.0)


# --- the pipeline --------------------------------------------------------------

#: A 40 x 60 x 60 um image whose tissue is the y < 40 um half, a vessel through it.
STAGE_SHAPE = (20, 60, 60)
STAGE_VOXEL_SIZE_ZYX = (2.0, 1.0, 1.0)
STAGE_TISSUE_VOLUME = 40.0 * 40.0 * 60.0


def _stage_inputs(tmp_path: Path, **overrides):
    z, y, x = np.indices(STAGE_SHAPE)
    tissue = y < 40
    vessel = tissue & (np.abs(y - 20) <= 1) & (np.abs(z - 10) <= 1)
    raw = np.where(tissue, 100, 10).astype(np.uint16)
    raw[vessel] = 2000
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(raw_path, raw)
    settings = SCHEMA.defaults()
    settings.update(
        input_path=tmp_path / "seg.tif",
        statistics=True,
        run_haemodynamics=False,
        vtk_export=False,
        visualize_vtk=False,
        visualize_results=False,
        export_citations=False,
        measure_tissue_volume=True,
        tissue_raw_tiff_path=str(raw_path),
    )
    settings.update(overrides)
    volume = stages.SkeletonisedVolume(
        image=vessel.astype(np.uint8),
        skeleton=np.zeros(STAGE_SHAPE, dtype=bool),
        voxel_size_xyz=tuple(reversed(STAGE_VOXEL_SIZE_ZYX)),
        voxel_size_zyx=STAGE_VOXEL_SIZE_ZYX,
        output_dir=tmp_path,
    )
    G = _one_vessel_graph()
    network = stages.VesselNetwork(graph=G, volume=volume)
    return settings, network, G, vessel


def _csv_rows(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        return {row["Metric"]: row["Value"] for row in csv.DictReader(handle)}


def test_the_export_stage_divides_vessel_density_by_the_tissue_it_measured(tmp_path):
    settings, network, G, vessel = _stage_inputs(tmp_path)
    solution = stages.Solution()
    stages.export_results(settings, network, stages.HaemodynamicModel(graph=G), solution)

    # The CSV writes six significant figures.
    statistics_rows = _csv_rows(tmp_path / "seg_statistics.csv")
    tissue_volume = float(statistics_rows["Tissue Volume"])
    assert tissue_volume == pytest.approx(STAGE_TISSUE_VOLUME, rel=1e-3)
    assert float(statistics_rows["Vessel Density in Tissue"]) == pytest.approx(
        59.0 / tissue_volume, rel=1e-5
    )
    assert float(statistics_rows["Vascular Volume Fraction in Tissue"]) == pytest.approx(
        vessel.sum() * np.prod(STAGE_VOXEL_SIZE_ZYX) / tissue_volume, rel=1e-5
    )
    # The whole-image density is still there beside it, and smaller.
    assert float(statistics_rows["Vessel Density in Whole Image"]) < float(
        statistics_rows["Vessel Density in Tissue"]
    )

    tissue_rows = _csv_rows(tmp_path / "seg_tissue_volume.csv")
    assert float(tissue_rows["Tissue Volume"]) == pytest.approx(tissue_volume, rel=1e-5)
    assert solution.tissue is not None
    assert solution.tissue.volume_um3 == pytest.approx(tissue_volume)


def test_the_tissue_surface_file_holds_the_measured_mesh(tmp_path):
    import pyvista as pv

    settings, network, G, _vessel = _stage_inputs(tmp_path, statistics=False)
    solution = stages.Solution()
    stages.export_results(settings, network, stages.HaemodynamicModel(graph=G), solution)

    mesh = pv.read(tmp_path / "seg_tissue_surface.vtp")
    assert mesh.n_points == len(solution.tissue.vertices)
    assert mesh.n_cells == len(solution.tissue.faces)
    # Written in node pos's own (z, y, x) order, as the vessels' .vtp is.
    assert np.asarray(mesh.points) == pytest.approx(solution.tissue.vertices)
    assert mesh.volume == pytest.approx(solution.tissue.volume_um3, rel=1e-6)
    # Statistics off: the tissue is still measured and written, and no report is.
    assert not (tmp_path / "seg_statistics.csv").exists()


def test_the_export_stage_measures_nothing_with_the_setting_off(tmp_path):
    settings, network, G, _vessel = _stage_inputs(tmp_path, measure_tissue_volume=False)
    solution = stages.Solution()
    stages.export_results(settings, network, stages.HaemodynamicModel(graph=G), solution)
    assert solution.tissue is None
    assert not (tmp_path / "seg_tissue_volume.csv").exists()
    rows = _csv_rows(tmp_path / "seg_statistics.csv")
    assert rows["Tissue Volume Source"] == "bounding box of the network's nodes"


def test_one_network_is_measured_once_and_again_when_a_setting_changes(tmp_path, monkeypatch):
    settings, network, _G, _vessel = _stage_inputs(tmp_path)
    calls = []
    real = stages.statistics.measure_tissue_volume

    def counting(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(stages.statistics, "measure_tissue_volume", counting)
    first = stages.tissue_measurement(settings, network)
    assert stages.tissue_measurement(settings, network) is first
    assert len(calls) == 1

    changed = {**settings, "tissue_smoothing_sigma_um": 1.0}
    assert stages.tissue_measurement(changed, network) is not first
    assert len(calls) == 2

    # A resumed run with a new segmentation is measured against that one.
    resegmented = stages.VesselNetwork(
        graph=network.graph,
        volume=stages.SkeletonisedVolume(
            image=np.zeros(STAGE_SHAPE, dtype=np.uint8),
            skeleton=network.volume.skeleton,
            voxel_size_xyz=network.volume.voxel_size_xyz,
            voxel_size_zyx=network.volume.voxel_size_zyx,
            output_dir=tmp_path,
        ),
        tissue=network.tissue,
    )
    stages.tissue_measurement(changed, resegmented)
    assert len(calls) == 3


def test_an_ilastik_label_segmentation_reads_its_minority_label_as_vessel(tmp_path):
    """A Simple Segmentation is labels 1 and 2, not 0 and 1: the vessels are
    what skeletonisation took as foreground, not every non-zero voxel."""
    settings, network, _G, vessel = _stage_inputs(tmp_path)
    binary = stages.tissue_measurement(settings, network)
    labelled = stages.VesselNetwork(
        graph=network.graph,
        volume=stages.SkeletonisedVolume(
            image=np.where(vessel, 1, 2).astype(np.uint8),
            skeleton=network.volume.skeleton,
            voxel_size_xyz=network.volume.voxel_size_xyz,
            voxel_size_zyx=network.volume.voxel_size_zyx,
            output_dir=tmp_path,
        ),
    )
    from_labels = stages.tissue_measurement(settings, labelled)
    assert from_labels.vessel_volume_um3 == pytest.approx(binary.vessel_volume_um3)
    assert from_labels.volume_um3 == pytest.approx(binary.volume_um3)


def test_a_perturbation_divides_by_the_same_tissue_as_the_baseline(tmp_path):
    settings, network, _G, _vessel = _stage_inputs(tmp_path)
    tissue = stages.tissue_measurement(settings, network)
    assert stages.tissue_statistics_arguments(tissue) == {
        "tissue_volume_um3": tissue.volume_um3,
        "vessel_volume_um3": tissue.vessel_volume_um3,
    }
    assert stages.tissue_statistics_arguments(None) == {}


def test_a_memory_mapped_load_is_measured_and_its_file_released(tmp_path):
    memmaps = tmp_path / "memmaps"
    memmaps.mkdir()
    settings, network, _G, _vessel = _stage_inputs(
        tmp_path, use_memmap_loading=True, memmap_directory=str(memmaps)
    )
    tissue = stages.tissue_measurement(settings, network)
    assert tissue.volume_um3 == pytest.approx(STAGE_TISSUE_VOLUME, rel=1e-3)
    assert list(memmaps.iterdir()) == []


# --- which raw image, and preflight -------------------------------------------


def test_the_tissue_shares_the_fwhm_raw_image_and_channel_when_it_names_none():
    settings = {"fwhm_raw_tiff_path": "raw.tif", "fwhm_raw_channel": 1}
    assert tissue_raw_image(settings) == (
        "raw.tif", 1, "fwhm_raw_tiff_path", "fwhm_raw_channel"
    )
    # Another channel of the same file.
    assert tissue_raw_image({**settings, "tissue_raw_channel": 2}) == (
        "raw.tif", 2, "fwhm_raw_tiff_path", "tissue_raw_channel"
    )
    # Its own image wins, with its own channel.
    assert tissue_raw_image({**settings, "tissue_raw_tiff_path": "tissue.tif"}) == (
        "tissue.tif", None, "tissue_raw_tiff_path", "tissue_raw_channel"
    )


def test_preflight_refuses_a_tissue_measurement_with_no_raw_image():
    report = check_tissue_volume_image({"measure_tissue_volume": True})
    assert len(report.errors) == 1
    assert "tissue_raw_tiff_path" in report.errors[0]
    assert not check_tissue_volume_image({"measure_tissue_volume": False}).errors


def test_preflight_refuses_a_missing_fwhm_image_the_tissue_would_share(tmp_path):
    report = check_tissue_volume_image(
        {"measure_tissue_volume": True, "fwhm_raw_tiff_path": str(tmp_path / "gone.tif")}
    )
    assert len(report.errors) == 1
    assert "gone.tif" in report.errors[0]


def test_preflight_asks_for_a_channel_of_a_multi_channel_tissue_image(tmp_path):
    path = tmp_path / "composite.tif"
    tifffile.imwrite(
        path, np.zeros((3, 2, 8, 8), dtype=np.uint8), imagej=True, metadata={"axes": "ZCYX"}
    )
    settings = {"measure_tissue_volume": True, "tissue_raw_tiff_path": str(path)}
    errors = check_tissue_volume_image(settings).errors
    assert len(errors) == 1 and "tissue_raw_channel" in errors[0]
    assert not check_tissue_volume_image({**settings, "tissue_raw_channel": 1}).errors


def test_full_preflight_runs_the_tissue_check(tmp_path):
    settings, _network, _G, _vessel = _stage_inputs(tmp_path, tissue_raw_tiff_path=None)
    settings["input_path"] = tmp_path / "raw.tif"
    errors = preflight(settings, SCHEMA).errors
    assert any("measure_tissue_volume needs the raw image" in error for error in errors)


# --- citations -----------------------------------------------------------------


@pytest.mark.parametrize("method", ["otsu", "li", "triangle", "manual"])
def test_the_tissue_surface_and_only_its_own_threshold_are_cited(method):
    settings = {"measure_tissue_volume": True, "tissue_threshold_method": method}
    names = {citation.name for citation in used_citations(settings, IMAGE_ANALYSIS_CITATIONS)}
    assert "Tissue surface (measure_tissue_volume)" in names
    thresholds = {name for name in names if "threshold" in name.lower()}
    if method == "manual":
        assert thresholds == set()
    else:
        assert len(thresholds) == 1 and f"'{method}'" in thresholds.pop()
    assert "Image analysis methods" in render_citations(settings)


def test_a_run_without_the_tissue_cites_no_image_analysis_method():
    settings = {"measure_tissue_volume": False}
    assert used_citations(settings, IMAGE_ANALYSIS_CITATIONS) == []
    assert "Image analysis methods" not in render_citations(settings)


# --- the panel -------------------------------------------------------------------


def test_the_tissue_rows_sit_on_additional_measurements_under_their_own_toggle():
    from haemolynx.gui.form import CHANNEL_SETTINGS, visible_statistics_settings
    from haemolynx.gui.layout import layout_for
    from haemolynx.gui.tabs import tabs_for

    tab = next(t for t in tabs_for(SCHEMA) if t.stage.title == "9. Additional measurements")
    names = [field.name for field in tab.fields]
    layout = layout_for(tab.stage.call, names, SCHEMA)
    box = layout.box("statistics_and_measurements")
    assert box.rows[box.rows.index("measure_tissue_volume"):] == (
        "measure_tissue_volume", "tissue_raw_tiff_path", "tissue_raw_channel",
    )
    advanced = layout.disclosure_of("tissue_threshold_method")
    assert advanced.anchor == "measure_tissue_volume"
    assert set(advanced.names) == {
        "tissue_threshold_method", "tissue_manual_threshold", "tissue_smoothing_sigma_um",
        "tissue_working_voxel_size_um", "tissue_min_component_fraction", "tissue_fill_holes",
    }
    # A drop-down of the raw image's own channels.
    assert CHANNEL_SETTINGS["tissue_raw_channel"] == "tissue_raw_tiff_path"

    off = visible_statistics_settings(SCHEMA, {"measure_tissue_volume": False})
    on = visible_statistics_settings(SCHEMA, {"measure_tissue_volume": True})
    manual = visible_statistics_settings(
        SCHEMA, {"measure_tissue_volume": True, "tissue_threshold_method": "manual"}
    )
    assert "measure_tissue_volume" in off and "tissue_raw_tiff_path" not in off
    assert {"tissue_raw_tiff_path", "tissue_raw_channel", "tissue_threshold_method"} <= on
    assert "tissue_manual_threshold" not in on
    assert "tissue_manual_threshold" in manual


def test_the_3d_distance_block_still_ends_right_before_statistics():
    names = SCHEMA.names
    assert names.index("measure_tissue_volume") > names.index("statistics")
    assert names[names.index("statistics") - 1] == "measurement_3d_reference_h5_dataset_name"
