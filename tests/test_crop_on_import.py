"""Crop on import: the box the settings describe, and every volume cut to it.

The box itself is pure arithmetic (``io.crop``), pinned here voxel for voxel
for each way of giving it. The stage tests then check what the crop is for:
the run analyses only the box, and every volume read alongside the main image
is cut to the same box, so the masks still land on the vessels they belong to.
"""
from __future__ import annotations

import logging

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
from haemolynx.io.downsample import downsample_for
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


# --- a box binarised as the whole file reads ------------------------------------
#
# An ilastik "Simple Segmentation" export labels vessel 1 and background 2, and
# which is vessel is decided from the values a volume holds (the minority of
# two). Decided on the box alone, a box holding only background (one value) came
# back solid vessel -- Harvey's 28_3_d_ipsi box, every vessel then classed
# Unknown_Large -- and a box where vessel is the majority came back inside out.

#: Background, as an ilastik Simple Segmentation export writes it.
ILASTIK_BACKGROUND, ILASTIK_VESSEL = 2, 1


def _ilastik_labels(vessel: np.ndarray) -> np.ndarray:
    return np.where(vessel, ILASTIK_VESSEL, ILASTIK_BACKGROUND).astype(np.uint8)


def _write_labels(path, labels: np.ndarray) -> str:
    """*labels* as ilastik exports them: an H5 ``exported_data`` dataset with a
    trailing channel axis, or a plain TIFF."""
    if str(path).endswith(".h5"):
        import h5py

        with h5py.File(path, "w") as f:
            f.create_dataset("exported_data", data=labels[..., np.newaxis])
    else:
        tifffile.imwrite(path, labels)
    return str(path)


def _vessel_outside_box() -> np.ndarray:
    _inside, outside = _rods()
    return outside


def _vessel_filling_box() -> np.ndarray:
    vessel = np.zeros(STACK, dtype=bool)
    vessel[BOX] = True
    return vessel


def _vessel_majority_in_box() -> np.ndarray:
    """Vessel in most of the box (a minority of the whole stack), with a
    strip of background across it."""
    vessel = _vessel_filling_box()
    vessel[0:12, 20:23, 30:60] = False
    return vessel


BOX_CASES = {
    "only background in the box": _vessel_outside_box,
    "only vessel in the box": _vessel_filling_box,
    "vessel the majority in the box": _vessel_majority_in_box,
}


@pytest.mark.parametrize("suffix", [".h5", ".tif"])
@pytest.mark.parametrize("low_ram", [False, True], ids=["in_ram", "low_ram"])
@pytest.mark.parametrize("case", list(BOX_CASES))
def test_the_segmented_image_box_is_binarised_as_the_whole_file_reads(tmp_path, case, low_ram, suffix):
    vessel = BOX_CASES[case]()
    path = _write_labels(tmp_path / f"segmented{suffix}", _ilastik_labels(vessel))
    settings = _run_settings(tmp_path, input_path=path, use_memmap_loading=low_ram)

    image, *_ = stages.load_volume_for_skeletonise(settings, suffix.lstrip("."))

    assert image.dtype == bool
    assert np.array_equal(np.asarray(image), vessel[BOX])


@pytest.mark.parametrize("low_ram", [False, True], ids=["in_ram", "low_ram"])
def test_a_downsampled_box_holding_only_background_is_empty(tmp_path, low_ram):
    """Harvey's run: crop, then 2x, of an ilastik H5 whose box has no label 1."""
    path = _write_labels(tmp_path / "segmented.h5", _ilastik_labels(_vessel_outside_box()))
    settings = _run_settings(
        tmp_path, input_path=path, use_memmap_loading=low_ram, segmented_image_downsample="2x"
    )

    image, *_ = stages.load_volume_for_skeletonise(settings, "h5")

    assert image.shape == (6, 8, 15)
    assert not np.asarray(image).any()


def test_the_whole_files_rule_is_decided_once_for_repeated_loads(tmp_path, monkeypatch):
    path = _write_labels(tmp_path / "segmented.h5", _ilastik_labels(_vessel_majority_in_box()))
    settings = _run_settings(tmp_path, input_path=path)
    decided = []
    real = stages.mask_binarization_rule
    monkeypatch.setattr(
        stages, "mask_binarization_rule", lambda volume: decided.append(volume.shape) or real(volume)
    )
    stages._SEGMENTED_IMAGE_RULE.clear()

    first, *_ = stages.load_volume_for_skeletonise(settings, "h5")
    second, *_ = stages.load_volume_for_skeletonise(settings, "h5")

    assert decided == [STACK]
    assert np.array_equal(first, second)


def test_crop_volume_binarises_the_box_by_the_rule_it_is_given():
    from haemolynx.io.load import mask_binarization_rule

    labels = _ilastik_labels(_vessel_outside_box())
    box = crop_box(STACK, method="coordinates", start_xyz=BOX_START_XYZ, end_xyz=BOX_END_XYZ)

    cut = crop_volume(labels, box, binarise=mask_binarization_rule(labels))

    assert cut.dtype == bool and not cut.any()
    # Without a rule the box keeps its labels, as before.
    assert np.array_equal(crop_volume(labels, box), labels[BOX])


@pytest.mark.parametrize(
    "value, expected",
    [(0, False), (ILASTIK_BACKGROUND, False), (ILASTIK_VESSEL, True), (255, True)],
)
def test_a_file_holding_one_value_is_read_explicitly_and_says_so(value, expected, caplog):
    from haemolynx.io.load import _to_binary_volume_for_skeletonization

    uniform = np.full((4, 5, 6), value, dtype=np.uint8)
    for low_ram in (False, True):
        with caplog.at_level(logging.WARNING, logger="haemolynx.io.load"):
            mask = _to_binary_volume_for_skeletonization(uniform, use_memmap=low_ram)
        assert mask.dtype == bool
        assert np.array_equal(np.asarray(mask), np.full(uniform.shape, expected))
    assert f"holds the one value {value} throughout" in caplog.text


def _global_mask_arguments(path, *, low_ram, memmap_directory) -> dict:
    crop = crop_box(STACK, method="coordinates", start_xyz=BOX_START_XYZ, end_xyz=BOX_END_XYZ)
    down = downsample_for(2, crop.shape_zyx, (1.0, 1.0, 1.0))
    return dict(
        mask_role="large",
        enabled=True,
        use_ilastik=False,
        arteriole_mask_path=path,
        venule_mask_path=None,
        image_shape=down.shape_zyx,
        main_voxel_size_xyz=down.scale_voxel_size_xyz((1.0, 1.0, 1.0)),
        crop=crop,
        downsample=down,
        use_memmap=low_ram,
        memmap_directory=memmap_directory,
    )


def _no_vessel_anywhere() -> np.ndarray:
    return np.zeros(STACK, dtype=bool)


MASK_CASES = {
    # The ipsi large-vessel export: label 2 throughout, read as solid vessel.
    "only background in the file": _no_vessel_anywhere,
    **BOX_CASES,
}


@pytest.mark.parametrize("low_ram", [False, True], ids=["in_ram", "low_ram"])
@pytest.mark.parametrize("case", list(MASK_CASES))
def test_a_cropped_and_downsampled_vessel_mask_is_binarised_as_the_whole_file_reads(tmp_path, case, low_ram):
    from haemolynx.io.downsample import block_reduce_volume

    vessel = MASK_CASES[case]()
    path = _write_labels(tmp_path / "large.h5", _ilastik_labels(vessel))
    arguments = _global_mask_arguments(path, low_ram=low_ram, memmap_directory=tmp_path)
    clear_vessel_mask_cache()

    try:
        mask, venule, *_ = load_and_validate_vessel_masks(**arguments)
    finally:
        clear_vessel_mask_cache()

    expected = block_reduce_volume(
        vessel[BOX], arguments["downsample"].axis_factors_zyx, "majority"
    )
    assert venule is None
    assert np.array_equal(np.asarray(mask), np.asarray(expected))
