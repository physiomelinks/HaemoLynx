"""Crop on import: the box the settings describe, and every volume cut to it.

The box itself is pure arithmetic (``io.crop``), pinned here voxel for voxel
for each way of giving it. The stage tests then check what the crop is for:
the run analyses only the box, and every volume read alongside the main image
is cut to the same box, so the masks still land on the vessels they belong to.
"""
from __future__ import annotations

import logging
import pickle

import networkx as nx
import numpy as np
import pytest
import tifffile

from haemolynx.io import clear_vessel_mask_cache, load_and_validate_vessel_masks
from haemolynx.io.crop import (
    CROP_LOCATIONS,
    CROP_Z_LOCATIONS,
    GRAPH_ATTRIBUTE,
    CropBox,
    crop_box,
    crop_box_from_settings,
    crop_problems,
    crop_volume,
    crop_volume_and_box,
    crop_volume_from_settings,
    crop_volume_like_graph,
    graph_crop_box,
)
from haemolynx.pipeline import default_schema
from haemolynx.pipeline import stages

SCHEMA = default_schema()

#: (z, y, x): 10 slices of 20 rows by 30 columns, so every axis is distinct.
SHAPE = (10, 20, 30)


def _size_box(location="centre", z_location="centre", size_xyz=(10, 6, 4)):
    return crop_box(SHAPE, method="size", size_xyz=size_xyz, location=location, z_location=z_location)


# --- by size and location -----------------------------------------------------


@pytest.mark.parametrize(
    "location, y0, x0",
    [
        ("top_left", 0, 0),
        ("top_centre", 0, 10),
        ("top_right", 0, 20),
        ("centre_left", 7, 0),
        ("centre", 7, 10),
        ("centre_right", 7, 20),
        ("bottom_left", 14, 0),
        ("bottom_centre", 14, 10),
        ("bottom_right", 14, 20),
    ],
)
def test_each_location_places_the_box_in_the_xy_plane_as_napari_draws_it(location, y0, x0):
    # 10 voxels of x's 30 and 6 of y's 20: top is y = 0, left is x = 0.
    box = _size_box(location=location)
    assert box.start_zyx[1:] == (y0, x0)
    assert box.shape_zyx == (4, 6, 10)


def test_every_location_the_schema_offers_is_tested():
    assert set(CROP_LOCATIONS) == {
        "top_left", "top_centre", "top_right", "centre_left", "centre",
        "centre_right", "bottom_left", "bottom_centre", "bottom_right",
    }
    assert set(CROP_Z_LOCATIONS) == {"first", "centre", "last"}


@pytest.mark.parametrize("z_location, z0", [("first", 0), ("centre", 3), ("last", 6)])
def test_the_z_location_picks_the_first_middle_or_last_slices(z_location, z0):
    box = _size_box(z_location=z_location)
    assert box.start_zyx[0] == z0
    assert box.stop_zyx[0] == z0 + 4


def test_sizes_are_given_x_y_z_and_the_box_is_z_y_x():
    box = _size_box(location="top_left", z_location="first", size_xyz=(5, 3, 2))
    assert box.slices == (slice(0, 2), slice(0, 3), slice(0, 5))


def test_centred_with_an_odd_voxel_to_spare_the_box_sits_half_a_voxel_early():
    box = crop_box((1, 1, 5), method="size", size_xyz=(2, 1, 1))
    assert box.start_zyx[2] == 1  # (5 - 2) / 2 = 1.5 -> 1, never 2 one run and 1 the next


def test_a_size_larger_than_the_image_keeps_that_whole_axis_and_says_so(caplog):
    with caplog.at_level(logging.WARNING, logger="haemolynx.io.crop"):
        box = _size_box(size_xyz=(100, 6, 4))
    assert (box.start_zyx[2], box.stop_zyx[2]) == (0, 30)
    assert box.shape_zyx == (4, 6, 30)
    assert any("along x" in r.getMessage() and "30" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("size_xyz", [(0, 6, 4), (10, -1, 4)])
def test_a_size_of_zero_or_less_is_refused(size_xyz):
    with pytest.raises(ValueError, match="at least 1 voxel"):
        _size_box(size_xyz=size_xyz)


@pytest.mark.parametrize("size_xyz", [None, (10, 6)])
def test_a_size_without_three_values_is_refused(size_xyz):
    with pytest.raises(ValueError, match="crop_size_xyz"):
        _size_box(size_xyz=size_xyz)


def test_an_unknown_location_is_refused():
    with pytest.raises(ValueError, match="crop_location"):
        _size_box(location="middle")
    with pytest.raises(ValueError, match="crop_z_location"):
        _size_box(z_location="top")


# --- by coordinates -----------------------------------------------------------


def test_coordinates_keep_both_corners():
    box = crop_box(SHAPE, method="coordinates", start_xyz=(2, 3, 4), end_xyz=(11, 8, 6))
    assert box.slices == (slice(4, 7), slice(3, 9), slice(2, 12))
    assert box.shape_zyx == (3, 6, 10)
    assert box.describe().startswith("x 2-11, y 3-8, z 4-6")


def test_a_one_voxel_box_is_allowed():
    box = crop_box(SHAPE, method="coordinates", start_xyz=(5, 5, 5), end_xyz=(5, 5, 5))
    assert box.shape_zyx == (1, 1, 1)


def test_an_end_past_the_image_stops_at_its_edge_and_says_so(caplog):
    with caplog.at_level(logging.WARNING, logger="haemolynx.io.crop"):
        box = crop_box(SHAPE, method="coordinates", start_xyz=(25, 0, 0), end_xyz=(99, 19, 9))
    assert box.stop_zyx == (10, 20, 30)
    assert box.start_zyx == (0, 0, 25)
    assert len([r for r in caplog.records if "crop_end_xyz" in r.getMessage()]) == 1


def test_a_start_past_its_end_is_refused():
    with pytest.raises(ValueError, match="crop_start_xyz's y"):
        crop_box(SHAPE, method="coordinates", start_xyz=(0, 9, 0), end_xyz=(5, 8, 5))


def test_a_start_outside_the_image_is_refused():
    with pytest.raises(ValueError, match="outside the image"):
        crop_box(SHAPE, method="coordinates", start_xyz=(0, 0, 10), end_xyz=(5, 5, 12))


def test_a_negative_corner_is_refused():
    with pytest.raises(ValueError, match="0 or more"):
        crop_box(SHAPE, method="coordinates", start_xyz=(-1, 0, 0), end_xyz=(5, 5, 5))


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="crop_method"):
        crop_box(SHAPE, method="corners")


# --- the box, the settings and the volume --------------------------------------


def _settings(**overrides):
    settings = {
        "crop_on_import": True,
        "crop_method": "coordinates",
        "crop_size_xyz": None,
        "crop_location": "centre",
        "crop_z_location": "centre",
        "crop_start_xyz": (2, 3, 4),
        "crop_end_xyz": (11, 8, 6),
        "use_memmap_loading": False,
        "memmap_directory": None,
    }
    settings.update(overrides)
    return settings


def test_offset_um_places_a_position_back_on_the_whole_image():
    box = crop_box(SHAPE, method="coordinates", start_xyz=(2, 3, 4), end_xyz=(11, 8, 6))
    # voxel_size_zyx: z 2 um, y 0.5 um, x 0.25 um.
    assert box.offset_um((2.0, 0.5, 0.25)) == (8.0, 1.5, 0.5)


def test_crop_off_leaves_the_volume_alone():
    volume = np.arange(np.prod(SHAPE)).reshape(SHAPE)
    assert crop_box_from_settings(_settings(crop_on_import=False), SHAPE) is None
    assert crop_volume_from_settings(volume, _settings(crop_on_import=False)) is volume


def test_cropping_keeps_exactly_the_boxed_voxels_as_an_array_of_their_own():
    volume = np.arange(np.prod(SHAPE)).reshape(SHAPE)
    cropped, box = crop_volume_and_box(volume, _settings())
    assert np.array_equal(cropped, volume[4:7, 3:9, 2:12])
    assert box == CropBox((4, 3, 2), (7, 9, 12), SHAPE)
    # A copy, so the whole volume is freed once the caller drops it.
    assert cropped.base is None
    cropped[0, 0, 0] = -1
    assert volume[4, 3, 2] != -1


def test_with_the_low_ram_option_the_crop_is_disk_backed_and_identical(tmp_path):
    volume = np.arange(np.prod(SHAPE), dtype=np.int32).reshape(SHAPE)
    cropped = crop_volume_from_settings(
        volume, _settings(use_memmap_loading=True, memmap_directory=str(tmp_path))
    )
    assert isinstance(cropped, np.memmap)
    assert np.array_equal(np.asarray(cropped), volume[4:7, 3:9, 2:12])
    assert str(tmp_path) in str(cropped.filename)


def test_a_box_covering_the_whole_volume_cuts_nothing():
    volume = np.zeros(SHAPE, dtype=bool)
    settings = _settings(crop_method="size", crop_size_xyz=(30, 20, 10))
    assert crop_volume_from_settings(volume, settings) is volume


def test_a_volume_of_another_shape_is_never_cut_to_a_box_made_for_this_one():
    box = crop_box(SHAPE, method="coordinates", start_xyz=(0, 0, 0), end_xyz=(1, 1, 1))
    with pytest.raises(ValueError, match="worked out for"):
        crop_volume(np.zeros((10, 20, 31)), box)


def test_crop_problems_reports_only_what_fails_whatever_the_image():
    assert crop_problems(_settings(crop_on_import=False, crop_start_xyz=None)) == []
    assert crop_problems(_settings()) == []
    # Big numbers are fine before the image is known: it may be that big.
    assert crop_problems(_settings(crop_method="size", crop_size_xyz=(10**6, 10**6, 10**6))) == []
    assert crop_problems(_settings(crop_start_xyz=None))[0].startswith("crop_start_xyz is not set")
    assert "past" in crop_problems(_settings(crop_start_xyz=(12, 3, 4)))[0]
    assert "crop_size_xyz" in crop_problems(_settings(crop_method="size"))[0]


# --- the run: the image and the vessel masks, cut to one box ---------------------

#: (z, y, x) of a synthetic stack holding two rods along x: one inside the box
#: the tests crop to, one outside it.
STACK = (12, 40, 70)
#: The box: x 30-59, y 20-34, z 0-11, i.e. voxels (z, y, x) 0:12, 20:35, 30:60.
BOX_START_XYZ, BOX_END_XYZ = (30, 20, 0), (59, 34, 11)
BOX = (slice(0, 12), slice(20, 35), slice(30, 60))


def _rods() -> tuple[np.ndarray, np.ndarray]:
    inside = np.zeros(STACK, dtype=bool)
    inside[5:8, 24:27, 33:57] = True
    outside = np.zeros(STACK, dtype=bool)
    outside[5:8, 5:8, 3:27] = True
    return inside, outside


def _write(path, mask) -> str:
    tifffile.imwrite(path, np.asarray(mask, dtype=np.uint8) * 255)
    return str(path)


def _run_settings(tmp_path, **overrides) -> dict:
    inside, outside = _rods()
    values = SCHEMA.defaults()
    values.update(
        {
            "input_path": _write(tmp_path / "mask.tif", inside | outside),
            "vtk_output_prefix": tmp_path / "run" / "network",
            "plot_dir": tmp_path / "plots",
            "visualize_results": False,
            "crop_on_import": True,
            "crop_method": "coordinates",
            "crop_start_xyz": BOX_START_XYZ,
            "crop_end_xyz": BOX_END_XYZ,
            **overrides,
        }
    )
    return values


def test_the_run_analyses_only_the_box(tmp_path):
    settings = _run_settings(tmp_path)
    inside, _outside = _rods()

    volume = stages.skeletonise(settings, stages.segment(settings))

    assert volume.image.shape == (12, 15, 30)
    assert np.array_equal(volume.image != 0, inside[BOX])
    assert volume.skeleton.shape == (12, 15, 30)
    assert volume.skeleton.any()
    assert volume.crop == CropBox((0, 20, 30), (12, 35, 60), STACK)


def test_without_the_crop_the_same_stack_is_analysed_whole(tmp_path):
    settings = _run_settings(tmp_path, crop_on_import=False)

    volume = stages.skeletonise(settings, stages.segment(settings))

    assert volume.image.shape == STACK
    assert volume.crop is None


def test_the_network_is_built_in_the_box_and_records_where_it_sits(tmp_path):
    settings = _run_settings(tmp_path)
    volume = stages.skeletonise(settings, stages.segment(settings))

    network = stages.build_network(settings, volume, SCHEMA)

    G = network.graph
    assert G.number_of_edges() >= 1
    positions = np.array([G.nodes[n]["pos"] for n in G.nodes], dtype=float)
    # Voxels are 1 um: every node lies in the 12 x 15 x 30 box, measured from
    # its corner, and on the rod inside it -- the rod outside left no trace.
    assert positions.min() >= 0
    assert (positions.max(axis=0) <= np.array([12, 15, 30])).all()
    assert (np.abs(positions[:, 1] - 5) <= 2).all()
    assert G.graph[GRAPH_ATTRIBUTE] == {
        "start_zyx": (0, 20, 30),
        "stop_zyx": (12, 35, 60),
        "source_shape_zyx": STACK,
        "offset_um_zyx": (0.0, 20.0, 30.0),
    }
    assert graph_crop_box(G) == volume.crop


def test_a_network_built_on_the_whole_image_records_no_crop(tmp_path):
    settings = _run_settings(tmp_path, crop_on_import=False)
    volume = stages.skeletonise(settings, stages.segment(settings))

    network = stages.build_network(settings, volume, SCHEMA)

    assert GRAPH_ATTRIBUTE not in network.graph.graph
    assert graph_crop_box(network.graph) is None


def test_the_vessel_masks_are_cut_to_the_same_box_as_the_image(tmp_path):
    clear_vessel_mask_cache()
    arteriole, _outside = _rods()
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

    # The arteriole mask lies exactly on the image's rod, as it did uncut.
    assert np.array_equal(network.large_arteriole_mask, arteriole[BOX])
    assert np.array_equal(network.large_arteriole_mask, volume.image != 0)
    assert np.array_equal(network.large_venule_mask, venule[BOX])


def _mask_arguments(tmp_path, arteriole, venule, crop) -> dict:
    return dict(
        mask_role="large",
        enabled=True,
        use_ilastik=False,
        arteriole_mask_path=_write(tmp_path / "a.tif", arteriole),
        venule_mask_path=_write(tmp_path / "v.tif", venule),
        image_shape=crop.shape_zyx,
        main_voxel_size_xyz=(1.0, 1.0, 1.0),
        crop=crop,
    )


def test_a_vessel_mask_of_another_whole_shape_is_refused_though_its_crop_would_fit(tmp_path):
    """Regression guard: cut to the box first, a mask one voxel wider than the
    image comes out the box's shape and passes the shape check -- every voxel
    of it lying on the wrong part of the image."""
    crop = crop_box(STACK, method="coordinates", start_xyz=BOX_START_XYZ, end_xyz=BOX_END_XYZ)
    mask = np.zeros((STACK[0], STACK[1], STACK[2] + 1), dtype=bool)
    mask[5:8, 24:27, 33:57] = True
    clear_vessel_mask_cache()

    try:
        with pytest.raises(ValueError, match="before crop_on_import"):
            load_and_validate_vessel_masks(**_mask_arguments(tmp_path, mask, mask, crop))
    finally:
        clear_vessel_mask_cache()


def test_masks_remembered_for_one_box_are_not_handed_out_for_another(tmp_path):
    arteriole, venule = _rods()
    first = crop_box(STACK, method="coordinates", start_xyz=(0, 0, 0), end_xyz=(29, 14, 11))
    second = crop_box(STACK, method="coordinates", start_xyz=BOX_START_XYZ, end_xyz=BOX_END_XYZ)
    assert first.shape_zyx == second.shape_zyx
    clear_vessel_mask_cache()

    try:
        a1, *_ = load_and_validate_vessel_masks(**_mask_arguments(tmp_path, arteriole, venule, first))
        a2, *_ = load_and_validate_vessel_masks(**_mask_arguments(tmp_path, arteriole, venule, second))
    finally:
        clear_vessel_mask_cache()

    assert np.array_equal(a1, arteriole[first.slices])
    assert np.array_equal(a2, arteriole[second.slices])


def test_a_saved_skeleton_from_another_box_is_not_reused(tmp_path):
    settings = _run_settings(tmp_path)
    stages.skeletonise(settings, stages.segment(settings))

    resumed = _run_settings(tmp_path, do_skeletonize=False, crop_end_xyz=(59, 30, 11))
    with pytest.raises(ValueError, match="different crop_on_import box"):
        stages.skeletonise(resumed, stages.segment(resumed))


# --- volumes read later, cut by the box the graph records ------------------------


def _cropped_graph() -> nx.MultiGraph:
    """A network built in the box (z, y, x) 5:15, 10:30, 16:48 of a 20 x 40 x 64
    image of 1 um voxels."""
    G = nx.MultiGraph()
    G.graph["image_voxel_size_zyx"] = (1.0, 1.0, 1.0)
    G.graph[GRAPH_ATTRIBUTE] = CropBox((5, 10, 16), (15, 30, 48), (20, 40, 64)).to_record(
        (1.0, 1.0, 1.0)
    )
    return G


def _add_vessel(G: nx.MultiGraph) -> nx.MultiGraph:
    """One vessel along x at (z, y) = (5, 10) um from the box's corner."""
    G.add_node(0, pos=np.array([5.0, 10.0, 2.0]))
    G.add_node(1, pos=np.array([5.0, 10.0, 30.0]))
    G.add_edge(0, 1, key=0, length=28.0, branch_order="B01",
               voxels=[[5.0, 10.0, float(x)] for x in range(2, 31)])
    return G


def test_a_volume_read_later_is_cut_to_the_box_the_graph_was_built_in():
    whole = np.arange(20 * 40 * 64).reshape(20, 40, 64)

    cut = crop_volume_like_graph(whole, _cropped_graph(), description="raw image")

    assert np.array_equal(cut, whole[5:15, 10:30, 16:48])
    assert crop_volume_like_graph(whole, nx.MultiGraph(), description="raw image") is whole


def test_a_volume_that_is_not_the_whole_image_cannot_be_placed_on_a_cropped_network():
    with pytest.raises(ValueError, match=r"raw image is \(20, 40, 65\)"):
        crop_volume_like_graph(np.zeros((20, 40, 65)), _cropped_graph(), description="raw image")


def _pericytes_found(tmp_path, G: nx.MultiGraph) -> dict:
    """The pericyte mask run on *G*, with one pericyte in the whole-image mask
    at voxel (10, 20, 32) -- (5, 10, 16) in the box, on the vessel."""
    from haemolynx.haemodynamics.pericyte_mask import set_poiseuille_resistances_with_pericyte_mask

    z, y, x = np.indices((20, 40, 64))
    blob = (z - 10) ** 2 + (y - 20) ** 2 + (x - 32) ** 2 <= 3.5**2
    _G, results = set_poiseuille_resistances_with_pericyte_mask(
        G,
        diameter_by_branch_order={"B01": 5.0},
        constriction_factor_by_branch_order={"B01": 0.8},
        pericyte_mask_path=_write(tmp_path / "pericytes.tif", blob),
        constriction_length=8.0,
    )
    return results


def test_the_pericyte_mask_is_cut_to_the_networks_box(tmp_path):
    results = _pericytes_found(tmp_path, _add_vessel(_cropped_graph()))

    assert results["eligible_pericyte_count"] == 1
    assert results["assignment_distance_um_max"] == pytest.approx(0.0, abs=0.5)


def test_read_whole_the_same_pericyte_lies_off_the_vessel(tmp_path):
    # What the crop prevents: against the network's positions, the uncut
    # mask's pericyte sits 11 um from its vessel and is dropped.
    G = nx.MultiGraph()
    G.graph["image_voxel_size_zyx"] = (1.0, 1.0, 1.0)

    results = _pericytes_found(tmp_path, _add_vessel(G))

    assert results["eligible_pericyte_count"] == 0


def test_the_fwhm_raw_image_is_cut_to_the_networks_box(monkeypatch):
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

    _G, _summary, raw = assign_edge_diameters(_add_vessel(_cropped_graph()), config)

    assert np.array_equal(seen[0], whole[5:15, 10:30, 16:48])
    assert np.array_equal(raw, whole[5:15, 10:30, 16:48])


# --- settings, preflight and the panel ----------------------------------------------


def test_the_crop_settings_are_off_by_default_and_nest_under_their_toggle():
    assert SCHEMA["crop_on_import"].default is False
    for name in ("crop_method", "crop_size_xyz", "crop_location", "crop_z_location",
                 "crop_start_xyz", "crop_end_xyz"):
        assert "crop_on_import" in SCHEMA[name].requires
    for name in ("crop_size_xyz", "crop_location", "crop_z_location"):
        assert "crop_method=size" in SCHEMA[name].requires
    for name in ("crop_start_xyz", "crop_end_xyz"):
        assert "crop_method=coordinates" in SCHEMA[name].requires
    assert set(SCHEMA["crop_location"].choices) == set(CROP_LOCATIONS)
    assert set(SCHEMA["crop_z_location"].choices) == set(CROP_Z_LOCATIONS)
    for name in ("crop_size_xyz", "crop_start_xyz", "crop_end_xyz"):
        assert SCHEMA[name].unit == "voxels"


def test_crop_on_import_is_its_own_box_on_the_input_tab_under_segmentation_cleanup():
    from haemolynx.gui.layout import layout_for
    from haemolynx.gui.tabs import assign_to_stages, tabs_for

    crop_names = {name for name in SCHEMA.names if name.startswith("crop_")}
    owner = assign_to_stages(SCHEMA)
    assert {owner[name] for name in crop_names} == {owner["segmentation_cleanup"]}

    (input_tab,) = [tab for tab in tabs_for(SCHEMA) if tab.stage.call == "segment"]
    layout = layout_for("segment", [field.name for field in input_tab.fields], SCHEMA)
    keys = [box.key for box in layout.boxes]
    assert keys[keys.index("segmentation_cleanup") + 1] == "crop_on_import"
    box = layout.box("crop_on_import")
    assert box.title == "Crop on import"
    assert set(box.names) == crop_names
    assert box.rows[0] == "crop_on_import"


def test_preflight_reports_a_crop_no_image_could_hold():
    from haemolynx.pipeline.checks import check_crop_on_import

    assert not check_crop_on_import(_settings()).errors
    bad = check_crop_on_import(_settings(crop_start_xyz=(12, 3, 4)))
    assert bad.errors and "Crop on import" in bad.errors[0]
    assert not check_crop_on_import(_settings(crop_on_import=False, crop_start_xyz=None)).errors


# --- an H5 stack read only within the box ---------------------------------------
#
# The AIA stacks are 9.4 GB H5 files cut to a 5% box: the box is read alone, and
# a mask is binarised by the rule its whole file gives. Every test compares with
# reading the file whole and cutting it, as the loaders did before.

h5py = pytest.importorskip("h5py")


def _write_h5(path, volume, *, chunks=(4, 8, 8, 1)) -> str:
    """*volume* as ilastik writes a Simple Segmentation: ``exported_data``,
    a trailing size-1 channel axis, chunked."""
    with h5py.File(path, "w") as f:
        f.create_dataset("exported_data", data=np.asarray(volume)[..., None], chunks=chunks)
    return str(path)


def _labels(seed=0) -> np.ndarray:
    """A 1/2 ilastik label volume: 1 (vessel) the minority, as in the AIA masks."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random(STACK) < 0.2, 1, 2).astype(np.uint8)


@pytest.mark.parametrize("use_memmap", [False, True])
@pytest.mark.parametrize("axis_order", ["zyx", "xzy"])
def test_an_h5_box_read_holds_the_voxels_cutting_the_whole_stack_keeps(tmp_path, use_memmap, axis_order):
    from haemolynx.io import h5_volume_shape, load_3d_h5_with_voxel_size

    path = _write_h5(tmp_path / "stack.h5", np.arange(np.prod(STACK), dtype=np.uint16).reshape(STACK))
    whole, *whole_rest = load_3d_h5_with_voxel_size(path, axis_order=axis_order)
    assert h5_volume_shape(path, axis_order=axis_order) == whole.shape
    box = crop_box(whole.shape, method="coordinates", start_xyz=(3, 5, 1), end_xyz=(9, 17, 6)).slices

    cut, *cut_rest = load_3d_h5_with_voxel_size(
        path, axis_order=axis_order, use_memmap=use_memmap, memmap_directory=str(tmp_path), box=box
    )

    assert isinstance(cut, np.memmap) is use_memmap
    assert np.asarray(cut).dtype == whole.dtype and cut.shape == whole[box].shape
    assert np.array_equal(np.asarray(cut), whole[box])
    assert cut_rest == whole_rest  # the voxel size and its metadata status


@pytest.mark.parametrize(
    "volume",
    [
        _labels(),
        np.where(_labels(1) == 1, 255, 0).astype(np.uint8),
        (_labels(2) + (_labels(3) == 1)).astype(np.uint8),
        np.random.default_rng(4).integers(0, 200, STACK, dtype=np.uint8),
        np.random.default_rng(5).integers(-300, 300, STACK).astype(np.int16),
        np.random.default_rng(6).random(STACK).astype(np.float32),
        (np.random.default_rng(7).random(STACK) * 9 - 3).astype(np.float64),
        _labels(8) == 1,
    ],
    ids=["1/2", "0/255", "three labels", "grey", "int16", "float 0-1", "float", "bool"],
)
def test_the_rule_read_from_an_h5_file_binarises_as_the_whole_volume_does(tmp_path, volume):
    from haemolynx.io.load import _to_binary_volume_for_skeletonization, h5_binarization_rule

    path = _write_h5(tmp_path / "mask.h5", volume)
    rule = h5_binarization_rule(path)

    expected = _to_binary_volume_for_skeletonization(volume)
    assert np.array_equal(volume if rule is None else rule(volume), expected)


def test_an_h5_rule_counts_the_whole_file_not_the_box(tmp_path):
    """Inside the box label 2 is the minority, over the whole file label 1:
    a rule read from the box alone would take the wrong label for vessel."""
    from haemolynx.io.load import h5_binarization_rule

    volume = np.full(STACK, 2, dtype=np.uint8)
    volume[BOX] = 1
    volume[BOX][::3] = 2
    inside = volume[BOX]
    assert np.count_nonzero(inside == 2) < np.count_nonzero(inside == 1)
    assert np.count_nonzero(volume == 1) < np.count_nonzero(volume == 2)
    rule = h5_binarization_rule(_write_h5(tmp_path / "mask.h5", volume))
    assert np.array_equal(rule(inside), inside == 1)


@pytest.mark.parametrize("use_memmap", [False, True])
def test_an_h5_mask_is_read_within_the_box_and_is_the_mask_cut_from_the_whole(tmp_path, use_memmap):
    from haemolynx.io.automated_vessel_assignment import _load_mask_image

    path = tmp_path / "mask.h5"
    _write_h5(path, _labels(9))
    crop = crop_box(STACK, method="coordinates", start_xyz=BOX_START_XYZ, end_xyz=BOX_END_XYZ)
    low_ram = dict(use_memmap=use_memmap, memmap_directory=str(tmp_path))

    whole, xyz, status, whole_cut = _load_mask_image(path, **low_ram)
    cut, cut_xyz, cut_status, was_cut = _load_mask_image(path, crop=crop, **low_ram)

    assert not whole_cut and was_cut
    expected = crop_volume(whole, crop)
    assert type(cut) is type(expected) and cut.dtype == expected.dtype
    assert np.array_equal(cut, expected)
    assert (cut_xyz, cut_status) == (xyz, status)


def test_a_mask_the_box_cannot_be_read_from_comes_back_whole(tmp_path):
    from haemolynx.io.automated_vessel_assignment import _load_mask_image

    crop = crop_box(STACK, method="coordinates", start_xyz=BOX_START_XYZ, end_xyz=BOX_END_XYZ)
    wider = np.zeros((STACK[0], STACK[1], STACK[2] + 1), dtype=np.uint8)
    _write(tmp_path / "mask.tif", _rods()[0])
    _write_h5(tmp_path / "wider.h5", wider)

    tiff = _load_mask_image(tmp_path / "mask.tif", crop=crop)
    h5 = _load_mask_image(tmp_path / "wider.h5", crop=crop)

    assert tiff[0].shape == STACK and not tiff[3]
    assert h5[0].shape == wider.shape and not h5[3]


def test_an_h5_vessel_mask_of_another_whole_shape_is_still_refused(tmp_path):
    crop = crop_box(STACK, method="coordinates", start_xyz=BOX_START_XYZ, end_xyz=BOX_END_XYZ)
    wider = np.zeros((STACK[0], STACK[1], STACK[2] + 1), dtype=bool)
    wider[5:8, 24:27, 33:57] = True
    arguments = _mask_arguments(tmp_path, wider, wider, crop)
    arguments["arteriole_mask_path"] = _write_h5(tmp_path / "a.h5", wider)
    arguments["venule_mask_path"] = _write_h5(tmp_path / "v.h5", wider)
    clear_vessel_mask_cache()

    try:
        with pytest.raises(ValueError, match="large_arteriole_mask shape does not match input image shape before"):
            load_and_validate_vessel_masks(**arguments)
        arguments["venule_mask_path"] = None
        with pytest.raises(ValueError, match="large_global_mask shape does not match input image shape before"):
            load_and_validate_vessel_masks(**arguments)
    finally:
        clear_vessel_mask_cache()


@pytest.mark.parametrize("use_memmap_loading", [False, True])
def test_an_h5_run_reads_the_box_and_analyses_what_reading_it_whole_did(tmp_path, monkeypatch, use_memmap_loading):
    from haemolynx.io import automated_vessel_assignment as masks

    inside, outside = _rods()
    arteriole = np.where(inside, 1, 2).astype(np.uint8)
    venule = np.full(STACK, 2, dtype=np.uint8)
    venule[1:3, 30:33, 50:55] = 1
    settings = _run_settings(
        tmp_path,
        input_path=_write_h5(tmp_path / "stack.h5", np.where(inside | outside, 1, 2).astype(np.uint8)),
        use_memmap_loading=use_memmap_loading,
        automated_vessel_assignment=True,
        use_large_vessel_masks=True,
        large_arteriole_mask_path=_write_h5(tmp_path / "arteriole.h5", arteriole),
        large_venule_mask_path=_write_h5(tmp_path / "venule.h5", venule),
        large_vessel_mask_dilation_microns=0.0,
        large_vessel_min_component_volume_um3=0.0,
        large_vessel_swap_minority_components=False,
        large_vessel_remove_small_opposite_attached_components=False,
        exclude_smaller_overlapping_volumes=False,
    )

    def run():
        clear_vessel_mask_cache()
        try:
            volume = stages.skeletonise(settings, stages.segment(settings))
            return volume, stages.build_network(settings, volume, SCHEMA)
        finally:
            clear_vessel_mask_cache()

    volume, network = run()
    # Read whole and cut, as the loaders did before: the input and each mask.
    monkeypatch.setattr(stages, "_h5_input_crop_box", lambda settings, input_format: None)
    monkeypatch.setattr(masks, "h5_volume_shape", lambda *args, **kwargs: None)
    whole_volume, whole_network = run()

    assert volume.crop == whole_volume.crop == CropBox((0, 20, 30), (12, 35, 60), STACK)
    assert np.array_equal(np.asarray(volume.image), np.asarray(whole_volume.image))
    assert np.array_equal(np.asarray(volume.skeleton), np.asarray(whole_volume.skeleton))
    assert np.array_equal(network.large_arteriole_mask, inside[BOX])
    assert np.array_equal(network.large_arteriole_mask, whole_network.large_arteriole_mask)
    assert np.array_equal(network.large_venule_mask, whole_network.large_venule_mask)
    # The same graph, every attribute bit for bit, nodes and edges in the same order.
    assert pickle.dumps(network.graph) == pickle.dumps(whole_network.graph)
