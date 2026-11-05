"""Large-vessel masks split by width into large and small ones.

With only a large arteriole and venule mask given, ``split_large_vessel_masks_by_diameter``
cuts each into the part at least ``large_vessel_mask_split_diameter_um`` wide,
kept as the large mask, and the rest, used as the small mask of the same side
(``io/vessel_mask_split.py``). The panel offers the split only while no small
masks are given; the run then counts the small masks in use but reads no file
of theirs (``graph.vessel_mask_files_in_use``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from haemolynx.graph import (
    large_vessel_mask_split_diameter_um,
    small_vessel_masks_split_from_large,
    vessel_mask_files_in_use,
    vessel_masks_in_use,
)
from haemolynx.gui.form import shared_ilastik_host, unmet_chain
from haemolynx.gui.vessel_mask_search import roles_searched
from haemolynx.io import (
    clear_vessel_mask_cache,
    split_large_vessel_masks_by_diameter,
    vessel_mask_role_is_global,
    vessel_masks_are_global,
    wide_part_of_mask,
)
from haemolynx.pipeline import default_schema, stages
from haemolynx.pipeline.batch import batch_mask_sizes
from haemolynx.pipeline.checks import (
    check_large_vessel_mask_split,
    check_thick_vessel_restriction_prerequisites,
    ilastik_classifiers_in_effect,
    preflight,
)

SCHEMA = default_schema()


@pytest.fixture(autouse=True)
def _forget_loaded_masks():
    clear_vessel_mask_cache()
    yield
    clear_vessel_mask_cache()


def _tube(shape, spacing, start, end, radius) -> np.ndarray:
    """A round tube from *start* to *end* (microns, (z, y, x)), *radius* um."""
    points = np.indices(shape).reshape(3, -1).T * np.asarray(spacing, dtype=float)
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = end - start
    along = np.clip((points - start) @ axis / (axis @ axis), 0.0, 1.0)
    nearest = start + along[:, None] * axis
    return (np.linalg.norm(points - nearest, axis=1) <= radius).reshape(shape)


def _wide_tube_with_branch(spacing):
    """A 16 um tube across y-x, and a 5 um branch leaving it obliquely in z."""
    shape = tuple(int(np.ceil(80 / s)) for s in spacing)
    wide = _tube(shape, spacing, (40, 10, 10), (40, 70, 70), 8.0)
    branch = _tube(shape, spacing, (40, 40, 40), (75, 40, 60), 2.5)
    return wide, branch


# --- the split ---------------------------------------------------------------


def test_a_wide_vessel_stays_large_and_a_narrow_branch_becomes_small():
    spacing = (1.0, 1.0, 1.0)
    wide, branch = _wide_tube_with_branch(spacing)
    mask = wide | branch

    large, _venule, small, _ = split_large_vessel_masks_by_diameter(
        mask, None, diameter_um=10.0, voxel_size_xyz=spacing
    )

    assert not (large & small).any()
    assert np.array_equal(large | small, mask)
    assert large[wide].all(), "every voxel of the 16 um tube stays large"
    # The branch beyond the tube's wall, past a ball's reach, is all small.
    beyond = branch & ~_tube(mask.shape, spacing, (40, 10, 10), (40, 70, 70), 8.0 + 5.5)
    assert beyond.sum() > 500
    assert small[beyond].all()


@pytest.mark.parametrize("diameter, large_share", [(9.0, 1.0), (11.0, 1.0), (14.0, 0.0), (20.0, 0.0)])
def test_a_tube_is_large_up_to_its_width_and_small_beyond(diameter, large_share):
    """A 12 um tube (its voxel centres within 6 um of the axis; the distance
    transform reads 6-7 um at its axis, to the nearest background centre)."""
    spacing = (1.0, 1.0, 1.0)
    tube = _tube((20, 20, 60), spacing, (10, 10, 0), (10, 10, 59), 6.0)

    wide = wide_part_of_mask(tube, diameter_um=diameter, voxel_size_zyx=spacing)

    assert not (wide & ~tube).any()
    assert np.count_nonzero(wide) == pytest.approx(large_share * tube.sum())


def test_no_shell_of_narrow_voxels_is_left_round_a_wide_vessel_on_an_anisotropic_grid():
    """Without half a voxel of slack the wall's staircase at 2 x 0.6 x 0.6 um
    left 3% of the 16 um tube narrow, in 25 pieces."""
    spacing = (2.0, 0.6, 0.6)
    wide, branch = _wide_tube_with_branch(spacing)

    large, _, small, _ = split_large_vessel_masks_by_diameter(
        wide | branch, None, diameter_um=10.0, voxel_size_xyz=(0.6, 0.6, 2.0)
    )

    assert large[wide].all()
    assert small[branch & ~wide].sum() > 0.9 * (branch & ~wide).sum()


def test_the_load_time_dilation_is_added_back_so_the_width_is_the_vessels_own():
    """A tube read 10 um wide after a 1.5 um load-time dilation is a 7 um
    vessel: split at 8 um with that dilation added back, it is small."""
    spacing = (1.0, 1.0, 1.0)
    # Voxel centres within 4.5 um of the axis: 5 um to the nearest background.
    dilated = _tube((20, 20, 40), spacing, (10, 10, 0), (10, 10, 39), 4.5)

    as_read, *_ = split_large_vessel_masks_by_diameter(
        dilated, None, diameter_um=8.0, voxel_size_xyz=spacing
    )
    own, _, small, _ = split_large_vessel_masks_by_diameter(
        dilated, None, diameter_um=8.0, voxel_size_xyz=spacing, dilation_um=1.5
    )

    assert as_read.sum() == dilated.sum()
    assert not own.any() and np.array_equal(small, dilated)


def test_each_side_is_split_on_its_own_and_a_global_mask_stays_global():
    spacing = (1.0, 1.0, 1.0)
    wide, branch = _wide_tube_with_branch(spacing)

    pair = split_large_vessel_masks_by_diameter(wide, branch, diameter_um=10.0, voxel_size_xyz=spacing)
    assert pair[0].sum() == wide.sum() and not pair[1].any()
    assert not pair[2].any() and np.array_equal(pair[3], branch)

    large, venule, small, small_venule = split_large_vessel_masks_by_diameter(
        wide | branch, None, diameter_um=10.0, voxel_size_xyz=spacing
    )
    assert venule is None and small_venule is None
    assert large.any() and small.any()


# --- widths read on the segmentation ----------------------------------------------


def _thin_masks_on_wide_vessels(spacing):
    """The segmentation (the 16 um tube and its 5 um branch) and a thin
    classifier mask along both: 4 um round each axis, as Stack A's are
    4-7 um thick along vessels about 20 um wide."""
    shape = tuple(int(np.ceil(80 / s)) for s in spacing)
    wide, branch = _wide_tube_with_branch(spacing)
    core = _tube(shape, spacing, (40, 10, 10), (40, 70, 70), 2.0)
    branch_core = _tube(shape, spacing, (40, 40, 40), (75, 40, 60), 1.0)
    return wide | branch, core | branch_core, branch


def test_a_thin_mask_along_a_wide_vessel_is_read_by_the_vessels_width():
    """Read on its own the 4 um mask holds no 10 um ball, so the whole 16 um
    vessel was small; read on the segmentation, the mask says only the side."""
    spacing = (1.0, 1.0, 1.0)
    segmentation, mask, branch = _thin_masks_on_wide_vessels(spacing)
    core_of_wide = mask & ~branch

    own, _, _, _ = split_large_vessel_masks_by_diameter(
        mask, None, diameter_um=10.0, voxel_size_xyz=spacing
    )
    large, _, small, _ = split_large_vessel_masks_by_diameter(
        mask, None, diameter_um=10.0, voxel_size_xyz=spacing, segmentation=segmentation
    )

    assert not own.any(), "the mask's own width: nothing is 10 um wide"
    assert not (large & small).any() and np.array_equal(large | small, mask)
    assert large[core_of_wide].all(), "the 16 um vessel's mask is all large"
    # The branch's mask past the wide vessel's ball reach is all small.
    beyond = mask & branch & ~_tube(mask.shape, spacing, (40, 10, 10), (40, 70, 70), 8.0 + 5.5)
    assert beyond.sum() > 50 and small[beyond].all() and not large[beyond].any()
    # A wider split than the vessel leaves it small whatever the mask.
    wider, *_ = split_large_vessel_masks_by_diameter(
        mask, None, diameter_um=18.0, voxel_size_xyz=spacing, segmentation=segmentation
    )
    assert not wider.any()


def test_a_mask_dilated_past_the_vessels_wall_stays_large_to_its_edge():
    """A 12 um vessel's mask dilated 1.5 um at load reaches 1.5 um past the
    segmentation: read there, the load-time dilation takes the wide part out
    to the mask's edge rather than leaving a small shell round it."""
    spacing = (1.0, 1.0, 1.0)
    vessel = _tube((24, 24, 40), spacing, (12, 12, 0), (12, 12, 39), 6.0)
    dilated = _tube((24, 24, 40), spacing, (12, 12, 0), (12, 12, 39), 7.5)

    with_dilation, _, shell, _ = split_large_vessel_masks_by_diameter(
        dilated, None, diameter_um=10.0, voxel_size_xyz=spacing, segmentation=vessel, dilation_um=1.5
    )
    without, _, left_small, _ = split_large_vessel_masks_by_diameter(
        dilated, None, diameter_um=10.0, voxel_size_xyz=spacing, segmentation=vessel
    )

    assert np.array_equal(with_dilation, dilated) and not shell.any()
    assert left_small.any() and not (left_small & vessel).any(), "only the dilated rim is left out"


def test_a_narrow_vessel_touching_a_wide_one_takes_no_width_from_it():
    """The segmentation joins a 5 um arteriole to the 16 um vein it touches;
    a ball fits in the vein, not in the arteriole, so no arteriole voxel is
    large, though some lie within a vein ball's reach."""
    spacing = (1.0, 1.0, 1.0)
    shape = (40, 40, 60)
    vein = _tube(shape, spacing, (20, 20, 0), (20, 20, 59), 8.0)
    arteriole = _tube(shape, spacing, (20, 29.5, 0), (20, 29.5, 59), 2.5)
    segmentation = vein | arteriole

    large_arteriole, large_vein, small_arteriole, _ = split_large_vessel_masks_by_diameter(
        arteriole, vein, diameter_um=10.0, voxel_size_xyz=spacing, segmentation=segmentation
    )

    vein_ball_reach = _tube(shape, spacing, (20, 20, 0), (20, 20, 59), 8.5)
    assert (arteriole & vein_ball_reach).any()
    assert not large_arteriole.any() and np.array_equal(small_arteriole, arteriole)
    assert large_vein[vein & ~arteriole].all()


def test_the_segmentation_must_lie_on_the_masks_grid():
    mask = np.ones((4, 5, 6), dtype=bool)
    with pytest.raises(ValueError, match="one grid"):
        split_large_vessel_masks_by_diameter(
            mask, None, diameter_um=2.0, voxel_size_xyz=(1.0, 1.0, 1.0), segmentation=mask[:, :, :5]
        )


def test_small_blocks_and_the_low_ram_option_give_the_whole_transforms_masks(monkeypatch):
    """Blocks smaller than the volume, each with a halo of the ball's reach,
    answer every comparison as the whole transform does; under the low-RAM
    option the masks are on disk."""
    import haemolynx.io.vessel_mask_split as vessel_mask_split

    spacing = (2.0, 0.6, 0.6)
    wide, branch = _wide_tube_with_branch(spacing)
    mask = wide | branch
    monkeypatch.setattr(vessel_mask_split, "SPLIT_BLOCK_VOXELS", mask.size)
    whole = split_large_vessel_masks_by_diameter(
        mask, branch, diameter_um=10.0, voxel_size_xyz=(0.6, 0.6, 2.0)
    )
    monkeypatch.setattr(vessel_mask_split, "SPLIT_BLOCK_VOXELS", 1)

    blocks = split_large_vessel_masks_by_diameter(
        mask, branch, diameter_um=10.0, voxel_size_xyz=(0.6, 0.6, 2.0)
    )
    on_disk = split_large_vessel_masks_by_diameter(
        mask, branch, diameter_um=10.0, voxel_size_xyz=(0.6, 0.6, 2.0), use_memmap=True
    )

    assert whole[0].any() and whole[3].any()
    for reference, in_ram, disk in zip(whole, blocks, on_disk):
        assert isinstance(disk, np.memmap)
        assert np.array_equal(in_ram, reference)
        assert np.array_equal(np.asarray(disk), reference)


def test_widths_read_on_the_segmentation_are_the_same_by_blocks_and_on_disk(monkeypatch):
    import haemolynx.io.vessel_mask_split as vessel_mask_split

    spacing = (2.0, 0.6, 0.6)
    segmentation, mask, branch = _thin_masks_on_wide_vessels(spacing)
    kwargs = dict(diameter_um=10.0, voxel_size_xyz=(0.6, 0.6, 2.0), segmentation=segmentation, dilation_um=0.5)
    monkeypatch.setattr(vessel_mask_split, "SPLIT_BLOCK_VOXELS", mask.size)
    whole = split_large_vessel_masks_by_diameter(mask, branch, **kwargs)
    monkeypatch.setattr(vessel_mask_split, "SPLIT_BLOCK_VOXELS", 1)

    blocks = split_large_vessel_masks_by_diameter(mask, branch, **kwargs)
    on_disk = split_large_vessel_masks_by_diameter(mask, branch, use_memmap=True, **kwargs)

    assert whole[0].any() and whole[2].any() and whole[3].any()
    for reference, in_ram, disk in zip(whole, blocks, on_disk):
        assert isinstance(disk, np.memmap)
        assert np.array_equal(in_ram, reference)
        assert np.array_equal(np.asarray(disk), reference)


def test_an_empty_mask_splits_into_nothing_and_a_width_must_be_positive():
    empty = np.zeros((5, 6, 7), dtype=bool)
    large, _, small, _ = split_large_vessel_masks_by_diameter(
        empty, None, diameter_um=5.0, voxel_size_xyz=(1.0, 1.0, 1.0)
    )
    assert not large.any() and not small.any()
    for bad in (0.0, -3.0, float("nan")):
        with pytest.raises(ValueError, match="positive width"):
            wide_part_of_mask(empty, diameter_um=bad, voxel_size_zyx=(1.0, 1.0, 1.0))


# --- when the run splits -------------------------------------------------------


def _split_settings(**overrides) -> dict:
    settings = SCHEMA.defaults()
    settings.update(
        {
            "automated_vessel_assignment": True,
            "use_large_vessel_masks": True,
            "large_arteriole_mask_path": "large_arteriole.tif",
            "large_venule_mask_path": "large_venule.tif",
            "split_large_vessel_masks_by_diameter": True,
            "large_vessel_mask_split_diameter_um": 12.0,
        }
    )
    settings.update(overrides)
    return settings


def test_the_split_puts_the_small_masks_in_use_without_reading_files_of_theirs():
    settings = _split_settings()

    assert small_vessel_masks_split_from_large(settings)
    assert vessel_masks_in_use(settings, "small")
    assert not vessel_mask_files_in_use(settings, "small")
    assert vessel_mask_files_in_use(settings, "large")
    assert large_vessel_mask_split_diameter_um(settings) == 12.0


@pytest.mark.parametrize(
    "change",
    [
        {"split_large_vessel_masks_by_diameter": False},
        {"use_small_vessel_masks_for_boundary_assignment": True},
        {"use_large_vessel_masks": False},
        {"automated_vessel_assignment": False},
        {"network_source": "synthetic"},
    ],
)
def test_the_split_is_out_of_effect_unless_only_large_masks_are_given(change):
    settings = _split_settings(**change)

    assert not small_vessel_masks_split_from_large(settings)
    assert large_vessel_mask_split_diameter_um(settings) is None
    # The small masks are in use only by their own switch, and then from files.
    assert vessel_masks_in_use(settings, "small") == vessel_mask_files_in_use(settings, "small")


@pytest.mark.parametrize("width", [None, 0.0, -1.0, "wide"])
def test_a_split_with_no_positive_width_is_refused(width):
    settings = _split_settings(large_vessel_mask_split_diameter_um=width)

    with pytest.raises(ValueError, match="large_vessel_mask_split_diameter_um"):
        large_vessel_mask_split_diameter_um(settings)
    report = check_large_vessel_mask_split(settings)
    assert not report.ok and "large_vessel_mask_split_diameter_um" in report.errors[0]


def test_preflight_says_what_the_split_does_and_is_silent_without_it():
    report = check_large_vessel_mask_split(_split_settings())
    assert report.ok and report.passed and "12 um wide" in report.passed[0][1]

    out_of_effect = check_large_vessel_mask_split(
        _split_settings(use_small_vessel_masks_for_boundary_assignment=True, large_vessel_mask_split_diameter_um=None)
    )
    assert out_of_effect.ok and not out_of_effect.passed


def test_the_small_masks_are_global_exactly_when_the_large_ones_are():
    pair = _split_settings()
    assert not vessel_mask_role_is_global(pair, "small") and not vessel_masks_are_global(pair)

    global_ = _split_settings(large_venule_mask_path=None)
    assert vessel_mask_role_is_global(global_, "large")
    assert vessel_mask_role_is_global(global_, "small")


def test_the_panel_shows_the_split_only_while_no_small_masks_are_given():
    names = ("split_large_vessel_masks_by_diameter", "large_vessel_mask_split_diameter_um")
    shown = _split_settings()
    for name in names:
        assert unmet_chain(SCHEMA, name, shown) == ()

    with_small_masks = _split_settings(use_small_vessel_masks_for_boundary_assignment=True)
    for name in names:
        assert "!use_small_vessel_masks_for_boundary_assignment" in unmet_chain(SCHEMA, name, with_small_masks)
    without_large_masks = _split_settings(use_large_vessel_masks=False)
    for name in names:
        assert "use_large_vessel_masks" in unmet_chain(SCHEMA, name, without_large_masks)
    # The width hangs under the split's own switch.
    off = _split_settings(split_large_vessel_masks_by_diameter=False)
    assert unmet_chain(SCHEMA, "large_vessel_mask_split_diameter_um", off) == (
        "split_large_vessel_masks_by_diameter",
    )


def test_nothing_looks_for_small_mask_files_under_the_split():
    """Small-mask switches left on under the hidden small-mask row (ilastik
    among them) ask for nothing: the small masks are cut from the large."""
    settings = _split_settings(
        use_ilastik_small_vessel_segmentation=True,
        ilastik_small_arteriole_classifier_path="small.ilp",
    )

    assert batch_mask_sizes(settings) == ("large",)
    assert roles_searched(settings) == ("large",)
    assert shared_ilastik_host(settings) is None
    assert ilastik_classifiers_in_effect(settings) == []


def test_the_batch_window_says_its_small_mask_folder_goes_unread_under_the_split():
    from haemolynx.gui.batch_run import mask_switch_hint

    assert "cut from each image's large masks" in mask_switch_hint(SCHEMA, _split_settings(), "small")
    assert mask_switch_hint(SCHEMA, _split_settings(), "large") == ""
    unsplit = _split_settings(split_large_vessel_masks_by_diameter=False)
    assert "'Use small vessel masks for boundary assignment'" in mask_switch_hint(SCHEMA, unsplit, "small")


def test_preflight_asks_for_no_small_mask_file_under_the_split(tmp_path):
    mask = np.zeros((4, 5, 6), dtype=np.uint8)
    mask[1:3, 1:4, 1:5] = 255
    image = _write(tmp_path / "segmented.tif", mask)
    settings = _split_settings(
        input_path=image,
        output_dir=tmp_path,
        large_arteriole_mask_path=image,
        large_venule_mask_path=image,
        small_arteriole_mask_path=tmp_path / "missing_small_arteriole.tif",
        small_venule_mask_path=tmp_path / "missing_small_venule.tif",
    )

    report = preflight(settings, SCHEMA, echo=False)

    assert not [error for error in report.errors if "small_" in error], report.errors
    assert any(label == "large vessel masks split by width" for label, _ in report.passed)


@pytest.mark.parametrize("choice", ["small", "both"])
def test_the_thick_vessel_restriction_may_name_split_small_masks(choice):
    settings = _split_settings(
        use_thick_vessel_skeletonisation=True, skeleton_thick_vessel_restrict_to_mask=choice
    )
    assert check_thick_vessel_restriction_prerequisites(settings).ok

    unsplit = _split_settings(
        use_thick_vessel_skeletonisation=True,
        skeleton_thick_vessel_restrict_to_mask=choice,
        split_large_vessel_masks_by_diameter=False,
    )
    report = check_thick_vessel_restriction_prerequisites(unsplit)
    assert not report.ok and "use_small_vessel_masks_for_boundary_assignment" in report.errors[0]


# --- the run -----------------------------------------------------------------

#: (first x, last x + 1, width in voxels) of a straight vessel along x: a wide
#: arteriole, a mid one, a narrow stretch of capillary, a mid venule, a wide one.
PIECES = ((0, 30, 15), (30, 55, 9), (55, 85, 5), (85, 110, 9), (110, 140, 15))
#: A 5-voxel side branch runs +y to the image's face from each change of width.
BRANCH_X = (30, 55, 85, 110)
CENTRE, EXTENT, SIDE = 10, 21, 30


def _square(width: int) -> slice:
    return slice(CENTRE - width // 2, CENTRE + width // 2 + 1)


def _vessel_and_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The segmentation, the arteriole mask (the first two pieces) and the
    venule mask (the last two), (z, y, x) at 1 um voxels."""
    shape = (EXTENT, EXTENT + SIDE, PIECES[-1][1])
    segmentation = np.zeros(shape, dtype=bool)
    arteriole = np.zeros(shape, dtype=bool)
    venule = np.zeros(shape, dtype=bool)
    for index, (start, stop, width) in enumerate(PIECES):
        across = _square(width)
        segmentation[across, across, start:stop] = True
        if index < 2:
            arteriole[across, across, start:stop] = True
        if index > 2:
            venule[across, across, start:stop] = True
    for x in BRANCH_X:
        segmentation[_square(5), CENTRE:, x - 2 : x + 3] = True
    return segmentation, arteriole, venule


def _write(path: Path, mask: np.ndarray) -> str:
    tifffile.imwrite(path, np.asarray(mask, dtype=np.uint8) * 255)
    return str(path)


def _run_settings(tmp_path, **overrides) -> dict:
    segmentation, arteriole, venule = _vessel_and_masks()
    return _split_settings(
        input_path=_write(tmp_path / "segmented.tif", segmentation),
        output_dir=tmp_path / "run",
        plot_dir=tmp_path / "plots",
        visualize_results=False,
        large_arteriole_mask_path=_write(tmp_path / "large_arteriole.tif", arteriole),
        large_venule_mask_path=_write(tmp_path / "large_venule.tif", venule),
        large_vessel_min_component_volume_um3=0.0,
        large_vessel_swap_minority_components=False,
        large_vessel_remove_small_opposite_attached_components=False,
        exclude_smaller_overlapping_volumes=False,
        cut_network_at_large_vessel_volumes=False,
        assign_large_vessel_branch_orders=True,
        run_haemodynamics=False,
        **overrides,
    )


def _x_extent(mask) -> tuple[int, int]:
    xs = np.flatnonzero(np.asarray(mask).any(axis=(0, 1)))
    return int(xs.min()), int(xs.max())


def _x_of(G, node) -> float:
    return float(G.nodes[node]["pos"][2])


def test_a_run_splits_the_large_masks_and_orders_the_vessels_by_them(tmp_path):
    """The wide pieces are the Large tier, the mid pieces Art and Ven, the
    narrow ones capillaries: what small masks of the mid pieces would give."""
    settings = _run_settings(tmp_path)
    _segmentation, arteriole, venule = _vessel_and_masks()
    volume = stages.skeletonise(settings, stages.segment(settings))

    network = stages.build_network(settings, volume, SCHEMA)

    assert network.mask_split_diameter_um == 12.0
    for large, small, whole in (
        (network.large_arteriole_mask, network.small_arteriole_mask, arteriole),
        (network.large_venule_mask, network.small_venule_mask, venule),
    ):
        assert not (np.asarray(large) & np.asarray(small)).any()
        assert np.array_equal(np.asarray(large) | np.asarray(small), whole)
    # The 15-voxel pieces stay large (to within a ball's reach of their end);
    # the 9-voxel pieces beyond it are small.
    assert _x_extent(network.large_arteriole_mask) == (0, 32)
    assert _x_extent(network.large_venule_mask) == (107, 139)
    assert np.asarray(network.small_arteriole_mask)[:, :, 36:55].sum() == arteriole[:, :, 36:55].sum()
    assert np.asarray(network.small_venule_mask)[:, :, 85:104].sum() == venule[:, :, 85:104].sum()

    boundaries = stages.assign_boundaries(settings, network)
    G = boundaries.graph

    assert [_x_of(G, n) for n in settings["inlet_nodes"]] == [pytest.approx(8.0, abs=4)]
    assert [_x_of(G, n) for n in settings["outlet_nodes"]] == [pytest.approx(131.0, abs=4)]
    assert sorted(round(_x_of(G, n)) for n in settings["arteriole_boundary_nodes"]) == [29, 55]
    assert sorted(round(_x_of(G, n)) for n in settings["venule_boundary_nodes"]) == [85, 111]

    stages.assign_diameters(settings, network, boundaries, SCHEMA)
    order_at = {
        round(float(np.mean([p[2] for p in data["voxels"]]))): data["branch_order"]
        for _u, _v, data in G.edges(data=True)
        if abs(float(np.mean([p[1] for p in data["voxels"]])) - CENTRE) < 3
    }
    trunk = [order_at[x] for x in sorted(order_at)]
    assert trunk[0] == "Large_Art1" and trunk[-1] == "Large_Ven1"
    assert trunk[1] == "Art1" and trunk[-2] == "Ven1"
    assert all(order.startswith("B") for order in trunk[2:-2]) and trunk[2:-2]


def test_a_run_reads_thin_masks_by_the_segmentations_width(tmp_path):
    """Masks 3 voxels thick down each piece's axis, as a classifier's may be:
    the split reads the 15-voxel pieces' width off the segmentation, so they
    are still the Large tier and the 9-voxel ones Art and Ven."""
    settings = _run_settings(tmp_path)
    _segmentation, arteriole, venule = _vessel_and_masks()
    core = np.zeros(arteriole.shape, dtype=bool)
    core[_square(3), _square(3), :] = True
    for name, mask in (("large_arteriole", arteriole), ("large_venule", venule)):
        settings[f"{name}_mask_path"] = _write(tmp_path / f"thin_{name}.tif", mask & core)
    volume = stages.skeletonise(settings, stages.segment(settings))

    network = stages.build_network(settings, volume, SCHEMA)

    # The 15-voxel piece's core, to within a ball's reach of its end.
    in_reach = np.arange(arteriole.shape[2]) <= 32
    assert np.array_equal(np.asarray(network.large_arteriole_mask), arteriole & core & in_reach)
    boundaries = stages.assign_boundaries(settings, network)
    G = boundaries.graph
    stages.assign_diameters(settings, network, boundaries, SCHEMA)
    order_at = {
        round(float(np.mean([p[2] for p in data["voxels"]]))): data["branch_order"]
        for _u, _v, data in G.edges(data=True)
        if abs(float(np.mean([p[1] for p in data["voxels"]])) - CENTRE) < 3
    }
    trunk = [order_at[x] for x in sorted(order_at)]
    assert trunk[0] == "Large_Art1" and trunk[-1] == "Large_Ven1"
    assert trunk[1] == "Art1" and trunk[-2] == "Ven1"


def test_the_thick_vessel_restriction_reads_the_split_masks(tmp_path):
    settings = _run_settings(tmp_path, skeleton_thick_vessel_restrict_to_mask="small")
    segmentation, arteriole, venule = _vessel_and_masks()

    restriction = stages.thick_vessel_restriction_mask(segmentation, settings, (1.0, 1.0, 1.0))

    _, _, small_arteriole, small_venule = split_large_vessel_masks_by_diameter(
        arteriole, venule, diameter_um=12.0, voxel_size_xyz=(1.0, 1.0, 1.0), segmentation=segmentation
    )
    assert np.array_equal(restriction, small_arteriole | small_venule)
    both = stages.thick_vessel_restriction_mask(
        segmentation, {**settings, "skeleton_thick_vessel_restrict_to_mask": "both"}, (1.0, 1.0, 1.0)
    )
    assert np.array_equal(both, arteriole | venule)


def test_a_resumed_network_is_cut_again_when_the_split_changes(tmp_path):
    """The split sits on the Boundaries tab, after the masks load: a run
    resumed there with another width, or none, loads and cuts them again."""
    settings = _run_settings(tmp_path)
    _segmentation, arteriole, venule = _vessel_and_masks()
    volume = stages.skeletonise(settings, stages.segment(settings))
    network = stages.build_network(settings, volume, SCHEMA)

    assert stages._vessel_masks_now_in_use(settings, network) is network

    # Every piece is under 20 um: all of each mask is small.
    wider = stages._vessel_masks_now_in_use(
        {**settings, "large_vessel_mask_split_diameter_um": 20.0}, network
    )
    assert wider.mask_split_diameter_um == 20.0
    assert not np.asarray(wider.large_arteriole_mask).any()
    assert np.array_equal(np.asarray(wider.small_arteriole_mask), arteriole)
    assert np.array_equal(np.asarray(wider.small_venule_mask), venule)

    unsplit = stages._vessel_masks_now_in_use(
        {**settings, "split_large_vessel_masks_by_diameter": False}, network
    )
    assert unsplit.mask_split_diameter_um is None
    assert np.array_equal(np.asarray(unsplit.large_arteriole_mask), arteriole)
    assert np.array_equal(np.asarray(unsplit.large_venule_mask), venule)
    assert unsplit.small_arteriole_mask is None and unsplit.small_venule_mask is None
