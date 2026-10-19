"""Downsample on import: the two Input tab choices, the reduction itself, and
every volume a run reads brought onto the one grid.

The reduction is pure arithmetic (``io.downsample``), pinned here block for
block. The stage tests then check what it is for: a run on a downsampled
segmented image is the same run on a coarser grid, in the same microns, and
every volume read alongside the image -- the vessel masks, the raw image, the
pericyte mask -- lands on the same grid, so each still lies on the vessels it
belongs to.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pytest
import tifffile

from haemolynx.io import clear_vessel_mask_cache, load_and_validate_vessel_masks
from haemolynx.io.crop import GRAPH_ATTRIBUTE as CROP_GRAPH_ATTRIBUTE
from haemolynx.io.crop import CropBox
from haemolynx.io.downsample import (
    DOWNSAMPLE_CHOICES,
    DOWNSAMPLE_FACTORS,
    GRAPH_ATTRIBUTE,
    RAW_SETTING,
    SEGMENTED_SETTING,
    Downsample,
    axis_downsample_factors,
    block_reduce_volume,
    downsample_factor,
    downsample_for,
    downsample_from_settings,
    downsample_raw_volume,
    downsample_volume,
    graph_downsample,
    graph_raw_downsample_factor,
    mask_volume_like_graph,
    raw_volume_like_graph,
    record_on_graph,
    spread_onto_grid,
)
from haemolynx.pipeline import default_schema
from haemolynx.pipeline import stages

SCHEMA = default_schema()


# --- the choices -------------------------------------------------------------------


def test_the_choices_are_none_and_the_four_powers_of_two():
    assert DOWNSAMPLE_CHOICES == ("none", "2x", "4x", "8x", "16x")
    assert DOWNSAMPLE_FACTORS == (1, 2, 4, 8, 16)
    assert [downsample_factor(c) for c in DOWNSAMPLE_CHOICES] == list(DOWNSAMPLE_FACTORS)


@pytest.mark.parametrize("value, factor", [(None, 1), ("none", 1), ("NONE", 1), (1, 1), ("4x", 4), ("4", 4), (4, 4), (16, 16)])
def test_a_choice_reads_as_its_factor(value, factor):
    assert downsample_factor(value) == factor


@pytest.mark.parametrize("value", ["3x", "big", 0, -2, 2.5, True])
def test_anything_else_is_refused(value):
    with pytest.raises(ValueError):
        downsample_factor(value)


def test_the_optimiser_shares_the_one_rule_for_the_axes():
    from haemolynx.optimisation import search

    assert search.axis_downsample_factors is axis_downsample_factors
    assert search.DOWNSAMPLE_FACTORS == DOWNSAMPLE_FACTORS


def test_a_coarser_axis_is_reduced_by_less_so_the_grid_stays_near_isotropic():
    # z voxels four times the size of x and y: 4x reduces z not at all, 8x by 2.
    assert axis_downsample_factors(4, (2.0, 0.5, 0.5)) == (1, 4, 4)
    assert axis_downsample_factors(8, (2.0, 0.5, 0.5)) == (2, 8, 8)
    assert axis_downsample_factors(2, (1.0, 1.0, 1.0)) == (2, 2, 2)
    assert axis_downsample_factors(1, (2.0, 0.5, 0.5)) == (1, 1, 1)


# --- the reduction -----------------------------------------------------------------


def test_block_max_keeps_a_lone_vessel_voxel_and_nothing_appears_from_nowhere():
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[1, 1, 1] = True  # plain striding mask[::4, ::4, ::4] would skip it
    reduced = block_reduce_volume(mask, (4, 4, 4), "max")
    assert reduced.shape == (2, 2, 2)
    assert reduced.dtype == bool
    assert reduced[0, 0, 0] and reduced.sum() == 1


def test_block_mean_is_the_exact_mean_of_each_block_in_the_images_dtype():
    image = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4)
    reduced = block_reduce_volume(image, (2, 2, 2), "mean")
    assert reduced.shape == (2, 2, 2)
    assert reduced.dtype == np.uint16
    expected = np.rint(image.reshape(2, 2, 2, 2, 2, 2).mean(axis=(1, 3, 5))).astype(np.uint16)
    assert np.array_equal(reduced, expected)
    # A float image stays float, its means exact.
    floats = block_reduce_volume(image.astype(np.float32), (2, 2, 2), "mean")
    assert floats.dtype == np.float32
    assert np.allclose(floats, image.reshape(2, 2, 2, 2, 2, 2).mean(axis=(1, 3, 5)))


def test_a_block_past_the_edge_is_reduced_over_the_voxels_it_has():
    # 5 voxels along x in blocks of 2: 3 coarse voxels, the last from one voxel.
    image = np.zeros((1, 1, 5), dtype=np.float64)
    image[0, 0] = [0, 2, 4, 6, 100]
    assert np.array_equal(block_reduce_volume(image, (1, 1, 2), "mean")[0, 0], [1, 5, 100])
    mask = np.zeros((3, 1, 1), dtype=bool)
    mask[2] = True
    assert np.array_equal(block_reduce_volume(mask, (2, 1, 1), "max")[:, 0, 0], [False, True])


def test_blocks_of_one_leave_the_volume_itself():
    volume = np.ones((3, 3, 3), dtype=np.uint8)
    assert block_reduce_volume(volume, (1, 1, 1), "max") is volume
    assert downsample_volume(volume, None, reduce="max") is volume
    assert downsample_volume(volume, downsample_for(1, volume.shape, (1, 1, 1)), reduce="max") is volume


def test_with_the_low_ram_option_the_reduction_is_disk_backed_and_identical(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 1000, size=(6, 9, 11)).astype(np.uint16)
    in_ram = block_reduce_volume(image, (2, 3, 4), "mean")
    on_disk = block_reduce_volume(image, (2, 3, 4), "mean", use_memmap=True, memmap_directory=str(tmp_path))
    assert isinstance(on_disk, np.memmap)
    assert str(tmp_path) in str(on_disk.filename)
    assert np.array_equal(np.asarray(on_disk), in_ram)
    assert on_disk.shape == (3, 3, 3)


def test_the_downsample_describes_itself_and_scales_the_voxel_size():
    down = downsample_for(4, (10, 100, 100), (2.0, 0.5, 0.5))
    assert down.axis_factors_zyx == (1, 4, 4)
    assert down.shape_zyx == (10, 25, 25)
    assert down.voxel_size_zyx == (2.0, 2.0, 2.0)
    assert down.scale_voxel_size_xyz((0.5, 0.5, 2.0)) == (2.0, 2.0, 2.0)
    assert down.source_voxel_size_xyz == (0.5, 0.5, 2.0)
    assert not down.is_identity
    assert down.describe() == "4x: every 4 x 4 x 1 voxels (x, y, z) become one; 100 x 100 x 10 -> 25 x 25 x 10 voxels"
    assert Downsample.from_record(down.to_record()) == down


def test_a_volume_of_another_shape_is_never_reduced_as_this_one():
    down = downsample_for(2, (4, 4, 4), (1, 1, 1))
    with pytest.raises(ValueError, match="not on one grid"):
        downsample_volume(np.zeros((4, 4, 5)), down, reduce="max")


def test_from_settings_none_means_no_reduction():
    assert downsample_from_settings({SEGMENTED_SETTING: "none"}, (4, 4, 4), (1, 1, 1)) is None
    assert downsample_from_settings({}, (4, 4, 4), (1, 1, 1)) is None
    # 2x on a z axis three times coarser reduces z by nothing, the rest by 2.
    down = downsample_from_settings({SEGMENTED_SETTING: "2x"}, (4, 4, 4), (3.0, 1.0, 1.0))
    assert down is not None and down.axis_factors_zyx == (1, 2, 2)


# --- the raw image onto the segmentation's grid --------------------------------------


def _ramp(shape=(4, 8, 8)) -> np.ndarray:
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)


def test_the_raw_image_is_reduced_by_the_segmented_images_factor_when_the_two_agree():
    raw = _ramp()
    down = downsample_for(2, raw.shape, (1, 1, 1))
    assert np.array_equal(downsample_raw_volume(raw, down, 2), block_reduce_volume(raw, (2, 2, 2), "mean"))


def test_a_finer_raw_choice_is_read_at_the_segmented_images_grid_and_says_so(caplog):
    raw = _ramp()
    down = downsample_for(4, raw.shape, (1, 1, 1))
    with caplog.at_level(logging.INFO, logger="haemolynx.io.downsample"):
        reduced = downsample_raw_volume(raw, down, 1)
    assert reduced.shape == (1, 2, 2)
    assert np.array_equal(reduced, block_reduce_volume(raw, (4, 4, 4), "mean"))
    assert any("read at 4x" in r.getMessage() for r in caplog.records)


def test_a_coarser_raw_choice_is_reduced_by_its_own_blocks_and_spread_over_the_grid():
    raw = _ramp((4, 8, 8))
    down = downsample_for(2, raw.shape, (1, 1, 1))  # the network's grid: 2 x 4 x 4
    spread = downsample_raw_volume(raw, down, 4)
    assert spread.shape == down.shape_zyx == (2, 4, 4)
    coarse = block_reduce_volume(raw, (4, 4, 4), "mean")  # 1 x 2 x 2
    # Each 2x voxel takes the 4x voxel its block lies in: every 2 x 2 x 2 of
    # them share one value, the whole image being one 4x block deep.
    assert np.array_equal(spread, np.repeat(np.repeat(coarse, 2, axis=1), 2, axis=2).repeat(2, axis=0))


def test_spreading_onto_a_grid_whose_blocks_do_not_divide_the_coarse_ones():
    # Source 10 voxels; fine blocks of 3 (4 fine voxels), coarse blocks of 5
    # (2 coarse voxels): fine block centres 1.5, 4.5, 7.5, 10.5 -> coarse 0, 0, 1, 1.
    coarse = np.array([[[10, 20]]], dtype=np.int32)
    spread = spread_onto_grid(coarse, (1, 1, 5), (1, 1, 3), (1, 1, 4))
    assert np.array_equal(spread[0, 0], [10, 10, 20, 20])


def test_without_a_grid_a_raw_factor_alone_is_refused_and_no_factor_changes_nothing():
    raw = _ramp()
    assert downsample_raw_volume(raw, None, 1) is raw
    assert downsample_raw_volume(raw, downsample_for(1, raw.shape, (1, 1, 1)), 1) is raw
    with pytest.raises(ValueError, match="raw_image_downsample is 2x"):
        downsample_raw_volume(raw, None, 2)


# --- the run: the image and every volume beside it, on one grid ----------------------

#: (z, y, x): a stack of 1 um voxels with one rod along x, 5 voxels thick: odd
#: as read and, its blocks of 2 starting on an even index, 3 thick reduced 2x --
#: Lee thinning leaves nothing of an even-width rod.
STACK = (12, 40, 72)


def _rod() -> np.ndarray:
    rod = np.zeros(STACK, dtype=bool)
    rod[4:9, 20:25, 6:66] = True
    return rod


def _write(path, mask) -> str:
    tifffile.imwrite(path, np.asarray(mask, dtype=np.uint8) * 255)
    return str(path)


def _run_settings(tmp_path, **overrides) -> dict:
    values = SCHEMA.defaults()
    values.update(
        {
            "input_path": _write(tmp_path / "mask.tif", _rod()),
            "vtk_output_prefix": tmp_path / "run" / "network",
            "plot_dir": tmp_path / "plots",
            "visualize_results": False,
            SEGMENTED_SETTING: "2x",
            **overrides,
        }
    )
    return values


def test_the_run_is_on_the_reduced_grid_in_the_same_microns(tmp_path):
    settings = _run_settings(tmp_path)

    volume = stages.skeletonise(settings, stages.segment(settings))

    assert volume.image.shape == (6, 20, 36)
    assert np.array_equal(volume.image != 0, block_reduce_volume(_rod(), (2, 2, 2), "max"))
    assert volume.skeleton.shape == (6, 20, 36)
    assert volume.skeleton.any()
    assert volume.voxel_size_xyz == (2.0, 2.0, 2.0)
    assert volume.voxel_size_zyx == (2.0, 2.0, 2.0)
    assert volume.downsample == Downsample(2, (2, 2, 2), STACK, (1.0, 1.0, 1.0))
    assert volume.crop is None


def test_without_the_choice_the_same_stack_is_analysed_as_read(tmp_path):
    settings = _run_settings(tmp_path, **{SEGMENTED_SETTING: "none"})

    volume = stages.skeletonise(settings, stages.segment(settings))

    assert volume.image.shape == STACK
    assert volume.voxel_size_xyz == (1.0, 1.0, 1.0)
    assert volume.downsample is None


def test_the_voxel_size_override_is_the_images_as_read_and_is_scaled_too(tmp_path):
    settings = _run_settings(
        tmp_path, voxel_size_policy="override", voxel_size_override_xyz=(0.5, 0.5, 2.0),
        **{SEGMENTED_SETTING: "4x"},
    )

    volume = stages.skeletonise(settings, stages.segment(settings))

    # z voxels four times coarser: 4x reduces z by nothing.
    assert volume.downsample.axis_factors_zyx == (1, 4, 4)
    assert volume.image.shape == (12, 10, 18)
    assert volume.voxel_size_xyz == (2.0, 2.0, 2.0)


def test_the_crop_box_is_cut_first_then_the_box_is_reduced(tmp_path):
    settings = _run_settings(
        tmp_path, crop_on_import=True, crop_method="coordinates",
        crop_start_xyz=(0, 10, 0), crop_end_xyz=(35, 29, 11),
    )

    volume = stages.skeletonise(settings, stages.segment(settings))

    assert volume.crop == CropBox((0, 10, 0), (12, 30, 36), STACK)
    assert volume.image.shape == (6, 10, 18)
    assert np.array_equal(volume.image != 0, block_reduce_volume(_rod()[:, 10:30, :36], (2, 2, 2), "max"))
    assert volume.downsample.source_shape_zyx == (12, 20, 36)


def test_the_network_is_built_on_the_grid_and_records_it(tmp_path):
    settings = _run_settings(tmp_path)
    volume = stages.skeletonise(settings, stages.segment(settings))

    network = stages.build_network(settings, volume, SCHEMA)

    G = network.graph
    assert G.number_of_edges() >= 1
    positions = np.array([G.nodes[n]["pos"] for n in G.nodes], dtype=float)
    # Microns, as on the full grid: the rod runs along x at y ~ 22 um, z ~ 6 um.
    assert (np.abs(positions[:, 1] - 22) <= 3).all()
    assert (np.abs(positions[:, 0] - 6) <= 3).all()
    assert positions[:, 2].max() - positions[:, 2].min() > 40
    assert G.graph["image_voxel_size_zyx"] == (2.0, 2.0, 2.0)
    record = G.graph[GRAPH_ATTRIBUTE]
    assert record["factor"] == 2 and record["axis_factors_zyx"] == (2, 2, 2)
    assert record["source_shape_zyx"] == STACK and record["shape_zyx"] == (6, 20, 36)
    assert record["raw_factor"] == 1
    assert graph_downsample(G) == volume.downsample
    assert CROP_GRAPH_ATTRIBUTE not in G.graph


def test_a_network_built_on_the_image_as_read_records_nothing(tmp_path):
    settings = _run_settings(tmp_path, **{SEGMENTED_SETTING: "none"})
    volume = stages.skeletonise(settings, stages.segment(settings))

    network = stages.build_network(settings, volume, SCHEMA)

    assert GRAPH_ATTRIBUTE not in network.graph.graph
    assert graph_downsample(network.graph) is None
    assert graph_raw_downsample_factor(network.graph) == 1


def test_the_crop_offset_is_measured_in_the_images_own_voxels(tmp_path):
    settings = _run_settings(
        tmp_path, crop_on_import=True, crop_method="coordinates",
        crop_start_xyz=(0, 10, 0), crop_end_xyz=(35, 29, 11),
    )
    volume = stages.skeletonise(settings, stages.segment(settings))

    network = stages.build_network(settings, volume, SCHEMA)

    # The box starts 10 voxels of 1 um down y: 10 um, not 10 of the run's 2 um voxels.
    assert network.graph.graph[CROP_GRAPH_ATTRIBUTE]["offset_um_zyx"] == (0.0, 10.0, 0.0)


def test_the_vessel_masks_are_reduced_by_the_same_blocks_as_the_image(tmp_path):
    clear_vessel_mask_cache()
    arteriole = _rod()
    venule = np.zeros(STACK, dtype=bool)
    venule[1:3, 30:33, 50:55] = True
    settings = _run_settings(
        tmp_path,
        automated_vessel_assignment=True,
        use_large_vessel_masks=True,
        large_arteriole_mask_path=_write(tmp_path / "arteriole.tif", arteriole),
        large_venule_mask_path=_write(tmp_path / "venule.tif", venule),
        large_vessel_mask_dilation_microns=0.0,
        large_vessel_min_component_volume_um3=0.0,
        large_vessel_swap_minority_components=False,
        large_vessel_remove_small_opposite_attached_components=False,
        exclude_smaller_overlapping_volumes=False,
    )
    volume = stages.skeletonise(settings, stages.segment(settings))

    try:
        network = stages.build_network(settings, volume, SCHEMA)
    finally:
        clear_vessel_mask_cache()

    # The arteriole mask lies exactly on the image's rod, as it did unreduced.
    assert np.array_equal(network.large_arteriole_mask, block_reduce_volume(arteriole, (2, 2, 2), "max"))
    assert np.array_equal(network.large_arteriole_mask, volume.image != 0)
    assert np.array_equal(network.large_venule_mask, block_reduce_volume(venule, (2, 2, 2), "max"))
    # A mask without a voxel size of its own took the image's as read, and
    # is now on the run's grid.
    assert network.graph.graph["large_arteriole_mask_voxel_size_xyz"] == (2.0, 2.0, 2.0)


def test_the_small_vessel_masks_are_reduced_the_same_way(tmp_path):
    clear_vessel_mask_cache()
    arteriole = _rod()
    venule = np.zeros(STACK, dtype=bool)
    venule[1:3, 30:33, 50:55] = True
    settings = _run_settings(
        tmp_path,
        automated_vessel_assignment=True,
        use_small_vessel_masks_for_boundary_assignment=True,
        small_arteriole_mask_path=_write(tmp_path / "small_arteriole.tif", arteriole),
        small_venule_mask_path=_write(tmp_path / "small_venule.tif", venule),
        small_vessel_mask_dilation_microns=0.0,
        small_vessel_min_component_volume_um3=0.0,
        small_vessel_swap_minority_components=False,
    )
    volume = stages.skeletonise(settings, stages.segment(settings))

    try:
        network = stages.build_network(settings, volume, SCHEMA)
    finally:
        clear_vessel_mask_cache()

    assert np.array_equal(network.small_arteriole_mask, volume.image != 0)
    assert np.array_equal(network.small_venule_mask, block_reduce_volume(venule, (2, 2, 2), "max"))


def _mask_arguments(tmp_path, arteriole, venule, downsample) -> dict:
    return dict(
        mask_role="large",
        enabled=True,
        use_ilastik=False,
        arteriole_mask_path=_write(tmp_path / "a.tif", arteriole),
        venule_mask_path=_write(tmp_path / "v.tif", venule),
        image_shape=downsample.shape_zyx,
        main_voxel_size_xyz=downsample.scale_voxel_size_xyz((1.0, 1.0, 1.0)),
        downsample=downsample,
    )


def test_a_vessel_mask_of_another_whole_shape_is_refused_though_reduced_it_would_fit(tmp_path):
    """Reduced first, a mask one voxel longer than the image comes out the
    same coarse shape and would pass the shape check, every voxel of it a
    little off the image."""
    down = downsample_for(2, STACK, (1.0, 1.0, 1.0))
    mask = np.zeros((STACK[0], STACK[1], STACK[2] + 1), dtype=bool)
    mask[4:9, 20:25, 6:66] = True
    clear_vessel_mask_cache()

    try:
        with pytest.raises(ValueError, match="before segmented_image_downsample"):
            load_and_validate_vessel_masks(**_mask_arguments(tmp_path, mask, mask, down))
    finally:
        clear_vessel_mask_cache()


def test_masks_remembered_at_one_factor_are_not_handed_out_for_another(tmp_path):
    arteriole = _rod()
    venule = np.zeros(STACK, dtype=bool)
    venule[1:3, 30:33, 50:55] = True
    two, four = downsample_for(2, STACK, (1.0, 1.0, 1.0)), downsample_for(4, STACK, (1.0, 1.0, 1.0))
    clear_vessel_mask_cache()

    try:
        a2, *_ = load_and_validate_vessel_masks(**_mask_arguments(tmp_path, arteriole, venule, two))
        a4, *_ = load_and_validate_vessel_masks(**_mask_arguments(tmp_path, arteriole, venule, four))
    finally:
        clear_vessel_mask_cache()

    assert a2.shape == (6, 20, 36) and a4.shape == (3, 10, 18)
    assert np.array_equal(a4, block_reduce_volume(arteriole, (4, 4, 4), "max"))


def test_a_saved_skeleton_from_another_factor_is_not_reused(tmp_path):
    settings = _run_settings(tmp_path)
    stages.skeletonise(settings, stages.segment(settings))

    resumed = _run_settings(tmp_path, do_skeletonize=False, **{SEGMENTED_SETTING: "4x"})
    with pytest.raises(ValueError, match="segmented_image_downsample"):
        stages.skeletonise(resumed, stages.segment(resumed))


def test_a_saved_skeleton_at_the_same_factor_is_reused_on_the_reduced_grid(tmp_path):
    settings = _run_settings(tmp_path)
    made = stages.skeletonise(settings, stages.segment(settings))

    resumed = _run_settings(tmp_path, do_skeletonize=False)
    loaded = stages.skeletonise(resumed, stages.segment(resumed))

    assert np.array_equal(loaded.skeleton, made.skeleton)
    assert loaded.image.shape == made.image.shape
    assert loaded.voxel_size_xyz == (2.0, 2.0, 2.0)
    assert loaded.downsample == made.downsample


def test_the_public_loader_hands_back_the_reduced_image_with_the_runs_voxel_size(tmp_path):
    settings = _run_settings(
        tmp_path, voxel_size_policy="override", voxel_size_override_xyz=(0.5, 0.5, 0.5)
    )

    image, voxel_size, status = stages.load_volume_for_skeletonise(settings, "tif")

    assert image.shape == (6, 20, 36)
    assert voxel_size == (1.0, 1.0, 1.0)  # the 0.5 um override, on the 2x grid
    assert status["source"] == stages.RUN_GRID_SOURCE
    assert Downsample.from_record(status["downsample"]).factor == 2
    # What every caller then reads it with: the grid's, not the override again.
    assert stages.run_voxel_size_xyz(settings, voxel_size, status) == ((1.0, 1.0, 1.0), "manual_override")
    # And a plain load resolves as it always did.
    plain = stages.run_voxel_size_xyz(settings, (3.0, 3.0, 3.0), {"status": "complete"})
    assert plain == ((0.5, 0.5, 0.5), "manual_override")


def _raw_settings(tmp_path, raw: np.ndarray, **overrides) -> dict:
    tifffile.imwrite(tmp_path / "raw.tif", raw)
    settings = {
        "raw_image_path": str(tmp_path / "raw.tif"),
        "raw_image_channel": None,
        "raw_image_stain": "plasma_column",
        "image_axis_order": "zyx",
        "use_memmap_loading": False,
        "memmap_directory": None,
        RAW_SETTING: "none",
    }
    settings.update(overrides)
    return settings


def test_the_raw_evidence_reads_the_whole_raw_file_onto_the_reduced_grid(tmp_path, monkeypatch):
    """The network build's raw-image calibration reads the whole raw file; it
    is reduced by the segmented image's blocks (and the raw choice) before the
    mask sees it, so the two are compared voxel for voxel on one grid."""
    from haemolynx.preprocessing import raw_evidence

    seen = {}

    def fake_calibrate(raw, mask, voxel_size_zyx, *, stain):
        seen.update(raw=np.asarray(raw), mask=mask, voxel=tuple(voxel_size_zyx), stain=stain)
        return None  # nothing to read: the evidence is None, the shapes are what matter

    monkeypatch.setattr(raw_evidence, "calibrate_raw", fake_calibrate)
    raw = np.arange(np.prod(STACK), dtype=np.uint16).reshape(STACK)
    down = downsample_for(2, STACK, (1.0, 1.0, 1.0))
    mask = block_reduce_volume(_rod(), (2, 2, 2), "max")

    assert raw_evidence.raw_evidence_from_settings(
        _raw_settings(tmp_path, raw), mask, (2.0, 2.0, 2.0), downsample=down
    ) is None
    assert seen["raw"].shape == mask.shape == (6, 20, 36)
    assert np.array_equal(seen["raw"], block_reduce_volume(raw, (2, 2, 2), "mean"))
    assert seen["voxel"] == (2.0, 2.0, 2.0) and seen["stain"] == "plasma_column"

    # The raw choice coarser than the image's: reduced 4x, spread over the 2x grid.
    raw_evidence.raw_evidence_from_settings(
        _raw_settings(tmp_path, raw, **{RAW_SETTING: "4x"}), mask, (2.0, 2.0, 2.0), downsample=down
    )
    assert seen["raw"].shape == mask.shape
    coarse = block_reduce_volume(raw, (4, 4, 4), "mean")
    assert np.array_equal(seen["raw"][::2, ::2, ::2], coarse)

    # A raw file that is not the whole image cannot be the mask's stack.
    with pytest.raises(ValueError, match="not the image's"):
        raw_evidence.raw_evidence_from_settings(
            _raw_settings(tmp_path, raw[:, :, :-1]), mask, (2.0, 2.0, 2.0), downsample=down
        )

    # The segmented image not reduced, the raw one 2x: spread over the mask's own grid.
    raw_evidence.raw_evidence_from_settings(
        _raw_settings(tmp_path, raw, **{RAW_SETTING: "2x"}), _rod(), (1.0, 1.0, 1.0), downsample=None
    )
    assert seen["raw"].shape == STACK
    assert np.array_equal(seen["raw"][::2, ::2, ::2], block_reduce_volume(raw, (2, 2, 2), "mean"))


def test_the_build_reads_the_raw_image_and_records_the_raw_factor(tmp_path):
    rng = np.random.default_rng(1)
    raw = (rng.uniform(50, 100, size=STACK) + 400 * _rod()).astype(np.uint16)
    tifffile.imwrite(tmp_path / "raw.tif", raw)
    settings = _run_settings(
        tmp_path, raw_image_path=str(tmp_path / "raw.tif"), raw_image_stain="plasma_column",
        **{RAW_SETTING: "4x"},
    )
    volume = stages.skeletonise(settings, stages.segment(settings))

    network = stages.build_network(settings, volume, SCHEMA)

    record = network.graph.graph[GRAPH_ATTRIBUTE]
    assert record["factor"] == 2 and record["raw_factor"] == 4
    assert record["raw_axis_factors_zyx"] == (4, 4, 4)
    assert graph_raw_downsample_factor(network.graph) == 4

    # A raw file of another shape is refused where the build reads it.
    tifffile.imwrite(tmp_path / "raw.tif", raw[:, :, :-1])
    with pytest.raises(ValueError, match="not the image's"):
        stages.build_network(settings, volume, SCHEMA)


# --- volumes read later, placed by what the graph records --------------------------------


def _reduced_graph(raw_factor: int = 1) -> nx.MultiGraph:
    """A network built on a 20 x 40 x 64 image of 1 um voxels reduced 2x."""
    G = nx.MultiGraph()
    G.graph["image_voxel_size_zyx"] = (2.0, 2.0, 2.0)
    record_on_graph(G, downsample_for(2, (20, 40, 64), (1.0, 1.0, 1.0)), raw_factor=raw_factor)
    return G


def _add_vessel(G: nx.MultiGraph) -> nx.MultiGraph:
    """One vessel along x at (z, y) = (10, 20) um."""
    G.add_node(0, pos=np.array([10.0, 20.0, 4.0]))
    G.add_node(1, pos=np.array([10.0, 20.0, 60.0]))
    G.add_edge(0, 1, key=0, length=56.0, branch_order="B01",
               voxels=[[10.0, 20.0, float(x)] for x in range(4, 61, 2)])
    return G


def test_a_mask_read_later_is_reduced_by_the_graphs_record():
    whole = np.zeros((20, 40, 64), dtype=bool)
    whole[3, 5, 7] = True

    placed = mask_volume_like_graph(whole, _reduced_graph(), description="pericyte mask")

    assert placed.shape == (10, 20, 32)
    assert placed[1, 2, 3] and placed.sum() == 1
    assert mask_volume_like_graph(whole, nx.MultiGraph(), description="pericyte mask") is whole


def test_a_raw_image_read_later_is_reduced_by_the_graphs_record_and_its_own_factor():
    whole = np.arange(20 * 40 * 64, dtype=np.float32).reshape(20, 40, 64)

    same = raw_volume_like_graph(whole, _reduced_graph(), description="raw image")
    coarser = raw_volume_like_graph(whole, _reduced_graph(raw_factor=4), description="raw image")

    assert np.array_equal(same, block_reduce_volume(whole, (2, 2, 2), "mean"))
    assert coarser.shape == (10, 20, 32)
    assert np.array_equal(coarser[0, 0, :2], block_reduce_volume(whole, (4, 4, 4), "mean")[0, 0, :1].repeat(2))
    assert graph_raw_downsample_factor(_reduced_graph(raw_factor=4)) == 4


def test_a_volume_that_is_not_the_whole_image_cannot_be_placed_on_a_reduced_network():
    with pytest.raises(ValueError, match="not on one grid"):
        mask_volume_like_graph(np.zeros((20, 40, 65), dtype=bool), _reduced_graph(), description="mask")


def test_cropped_then_reduced_a_volume_read_later_goes_through_both():
    G = _reduced_graph()
    # Built in the box (z, y, x) 4:16, 10:30, 16:48 of a 20 x 40 x 64 image,
    # reduced 2x: 6 x 10 x 16 voxels.
    G.graph[CROP_GRAPH_ATTRIBUTE] = CropBox((4, 10, 16), (16, 30, 48), (20, 40, 64)).to_record((1.0, 1.0, 1.0))
    record_on_graph(G, downsample_for(2, (12, 20, 32), (1.0, 1.0, 1.0)))
    whole = np.zeros((20, 40, 64), dtype=bool)
    whole[5, 11, 17] = True

    placed = mask_volume_like_graph(whole, G, description="cell mask")

    assert placed.shape == (6, 10, 16)
    assert placed[0, 0, 0] and placed.sum() == 1


def test_recording_nothing_drops_an_old_record_and_a_raw_factor_alone_keeps_an_identity():
    G = nx.MultiGraph()
    record_on_graph(G, downsample_for(2, (4, 4, 4), (1, 1, 1)))
    record_on_graph(G, None)
    assert GRAPH_ATTRIBUTE not in G.graph
    record_on_graph(G, None, raw_factor=2, image_shape_zyx=(4, 4, 4), voxel_size_zyx=(1, 1, 1))
    assert G.graph[GRAPH_ATTRIBUTE]["factor"] == 1
    assert G.graph[GRAPH_ATTRIBUTE]["raw_factor"] == 2
    assert graph_downsample(G) is None  # an identity is no reduction
    assert graph_raw_downsample_factor(G) == 2


def test_the_pericyte_mask_is_reduced_onto_the_networks_grid(tmp_path):
    from haemolynx.haemodynamics.pericyte_mask import set_poiseuille_resistances_with_pericyte_mask

    z, y, x = np.indices((20, 40, 64))
    blob = (z - 10) ** 2 + (y - 20) ** 2 + (x - 32) ** 2 <= 3.5**2
    tifffile.imwrite(tmp_path / "pericytes.tif", blob.astype(np.uint8) * 255)
    _G, results = set_poiseuille_resistances_with_pericyte_mask(
        _add_vessel(_reduced_graph()),
        diameter_by_branch_order={"B01": 5.0},
        constriction_factor_by_branch_order={"B01": 0.8},
        pericyte_mask_path=str(tmp_path / "pericytes.tif"),
        constriction_length=8.0,
    )

    # One pericyte at (10, 20, 32) um, on the vessel: a mask of 1 um voxels
    # without a voxel size of its own took the image's as read, reduced to
    # the run's 2 um, and was not refused against it.
    assert results["eligible_pericyte_count"] == 1
    assert results["assignment_distance_um_max"] == pytest.approx(0.0, abs=2.0)


def test_the_fwhm_raw_image_is_reduced_onto_the_networks_grid(monkeypatch):
    from haemolynx.haemodynamics import apply
    from haemolynx.haemodynamics.apply import HaemodynamicsApplyConfig, assign_edge_diameters

    whole = np.arange(20 * 40 * 64, dtype=np.float32).reshape(20, 40, 64)
    seen = []

    def fake_fwhm(G, _config, raw_volume=None, **_kwargs):
        seen.append(raw_volume)
        return {}

    monkeypatch.setattr(apply, "_measure_fwhm_diameters", fake_fwhm)
    monkeypatch.setattr(apply, "load_fwhm_raw_volume", lambda _config: whole)
    config = HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": {"B01": 5.0}},
        fwhm={"use_fwhm_edge_diameters": True, "do_fwhm_measurement": True,
              "fwhm_fix_blur_to_image_psf": False},
    )

    _G, _summary, raw = assign_edge_diameters(_add_vessel(_reduced_graph()), config)

    expected = block_reduce_volume(whole, (2, 2, 2), "mean")
    assert np.array_equal(seen[0], expected)
    assert np.array_equal(raw, expected)


# --- settings, preflight and the panel ---------------------------------------------------


def test_the_two_settings_offer_none_and_the_four_factors_and_default_to_none():
    for name in (SEGMENTED_SETTING, RAW_SETTING):
        assert SCHEMA[name].kind == "choice"
        assert SCHEMA[name].default == "none"
        assert SCHEMA[name].choices == DOWNSAMPLE_CHOICES
    assert "network_source=segmented_image" in SCHEMA[SEGMENTED_SETTING].requires
    assert "raw_image_path" in SCHEMA[RAW_SETTING].requires
    # A bare number in a config reads as the choice.
    assert SCHEMA[SEGMENTED_SETTING].coerce("4") == "4x"


def test_the_rows_sit_under_segmentation_source_and_raw_image_stain_on_the_input_tab():
    from haemolynx.gui.layout import layout_for
    from haemolynx.gui.tabs import tabs_for

    (input_tab,) = [tab for tab in tabs_for(SCHEMA) if tab.stage.call == "segment"]
    layout = layout_for("segment", [field.name for field in input_tab.fields], SCHEMA)
    segmented = list(layout.box("segmented_input").rows)
    assert segmented[segmented.index("segmentation_source") + 1] == SEGMENTED_SETTING
    raw = list(layout.box("raw_input").rows)
    assert raw[raw.index("raw_image_stain") + 1] == RAW_SETTING


def test_preflight_warns_when_the_raw_choice_differs_and_says_what_happens():
    from haemolynx.pipeline.checks import check_downsample_on_import

    base = {"raw_image_path": "raw.tif", SEGMENTED_SETTING: "4x", RAW_SETTING: "4x"}
    assert not check_downsample_on_import(base).warnings
    assert not check_downsample_on_import({SEGMENTED_SETTING: "none", RAW_SETTING: "none"}).passed
    finer = check_downsample_on_import({**base, RAW_SETTING: "2x"})
    assert finer.warnings and "read at 4x" in finer.warnings[0]
    coarser = check_downsample_on_import({**base, RAW_SETTING: "16x"})
    assert coarser.warnings and "spread over the 4x grid" in coarser.warnings[0]
    ilastik = check_downsample_on_import({**base, "use_ilastik_segmentation": True})
    assert any("ilastik" in w for w in ilastik.warnings)
    # A raw image left at none is read on the segmented image's grid: said in
    # the pass line, not a warning, since nothing was asked for it.
    left = check_downsample_on_import({**base, RAW_SETTING: "none"})
    assert not left.warnings and any("read at 4x" in str(p) for p in left.passed)
    # Without a raw image the raw choice is out of effect: no warning.
    assert not check_downsample_on_import({SEGMENTED_SETTING: "4x", RAW_SETTING: "16x"}).warnings
    assert not check_downsample_on_import(base).errors
