"""Tests for io/automated_vessel_assignment.py — vessel-mask loading and validation.

Despite the module name this is I/O: it resolves mask paths, optionally runs
ilastik, loads the masks and checks they line up with the main image. The checks
are the point. A mask that disagrees with the image in shape or physical voxel
size still indexes fine and silently labels the wrong vessels, so the loader has
to refuse it rather than let it through. Voxel sizes cross an order boundary
here — metadata is physical ``(x, y, z)``, dilation needs per-array-axis
``(z, y, x)`` — so the tests below use three distinct spacings throughout.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx.io import (
    load_and_validate_vessel_masks,
    load_large_vessel_masks,
    vessel_mask_arguments,
)

# Coarse z, fine x — the usual confocal case, and all three spacings differ.
VOXEL_SIZE_XYZ = (0.4, 0.5, 2.0)


def _write_mask_tif(path: Path, shape: tuple[int, int, int]) -> None:
    tifffile.imwrite(path, np.zeros(shape, dtype=np.uint8))


def _write_mask_with_voxel_size(
    path: Path,
    volume: np.ndarray,
    voxel_size_xyz: tuple[float, float, float] = VOXEL_SIZE_XYZ,
) -> None:
    """Write a mask whose ImageJ metadata reports a known physical voxel size."""
    tifffile.imwrite(
        path,
        volume.astype(np.uint8),
        imagej=True,
        resolution=(1.0 / voxel_size_xyz[0], 1.0 / voxel_size_xyz[1]),
        metadata={"spacing": voxel_size_xyz[2], "unit": "um"},
    )


def test_disabled_returns_none_without_paths() -> None:
    result = load_and_validate_vessel_masks(
        mask_role="large",
        enabled=False,
        use_ilastik=False,
        arteriole_mask_path=None,
        venule_mask_path=None,
        image_shape=(4, 5, 6),
        main_voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    assert result == (None, None, None, None)


def test_ilastik_requires_enabled_flag() -> None:
    with pytest.raises(ValueError, match="use_ilastik_large_vessel_segmentation=True requires"):
        load_and_validate_vessel_masks(
            mask_role="large",
            enabled=False,
            use_ilastik=True,
            arteriole_mask_path=None,
            venule_mask_path=None,
            image_shape=(4, 5, 6),
            main_voxel_size_xyz=(1.0, 1.0, 1.0),
        )


def test_loads_and_validates_matching_masks(tmp_path: Path) -> None:
    shape = (4, 5, 6)
    arteriole_path = tmp_path / "arteriole.tif"
    venule_path = tmp_path / "venule.tif"
    _write_mask_tif(arteriole_path, shape)
    _write_mask_tif(venule_path, shape)

    arteriole_mask, venule_mask, art_vox, ven_vox = load_and_validate_vessel_masks(
        mask_role="large",
        enabled=True,
        use_ilastik=False,
        arteriole_mask_path=arteriole_path,
        venule_mask_path=venule_path,
        image_shape=shape,
        main_voxel_size_xyz=(1.0, 1.0, 1.0),
    )

    assert arteriole_mask is not None and venule_mask is not None
    assert arteriole_mask.shape == shape
    assert venule_mask.shape == shape
    assert art_vox == (1.0, 1.0, 1.0)
    assert ven_vox == (1.0, 1.0, 1.0)


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    arteriole_path = tmp_path / "arteriole.tif"
    venule_path = tmp_path / "venule.tif"
    _write_mask_tif(arteriole_path, (4, 5, 6))
    _write_mask_tif(venule_path, (4, 5, 6))

    with pytest.raises(ValueError, match="large_arteriole_mask shape does not match"):
        load_and_validate_vessel_masks(
            mask_role="large",
            enabled=True,
            use_ilastik=False,
            arteriole_mask_path=arteriole_path,
            venule_mask_path=venule_path,
            image_shape=(3, 5, 6),
            main_voxel_size_xyz=(1.0, 1.0, 1.0),
        )


# --- load_large_vessel_masks: the two masks are paired, and never swapped ---


def test_a_mask_path_supplied_while_masks_are_disabled_is_an_error() -> None:
    """Loading it anyway, or ignoring it, both hide that the config contradicts itself."""
    with pytest.raises(ValueError, match="Large-vessel masks are disabled"):
        load_large_vessel_masks(False, "arteriole.tif", None)


def test_supplying_only_one_of_the_two_masks_is_an_error(tmp_path: Path) -> None:
    """Arteriole-only would leave every terminal node an inflow with nowhere to drain."""
    arteriole_path = tmp_path / "arteriole.tif"
    _write_mask_tif(arteriole_path, (4, 5, 6))

    with pytest.raises(ValueError, match="or provide neither"):
        load_large_vessel_masks(True, arteriole_path, None)


def test_enabling_masks_without_any_paths_is_an_error() -> None:
    with pytest.raises(ValueError, match="mask paths are missing"):
        load_large_vessel_masks(True, None, None)


def test_the_arteriole_mask_comes_back_first_and_the_venule_second(tmp_path: Path) -> None:
    """Swapping the two returns inverts every inflow/outflow assignment downstream."""
    arteriole_path = tmp_path / "arteriole.tif"
    venule_path = tmp_path / "venule.tif"
    arteriole_volume = np.zeros((4, 5, 6), dtype=np.uint8)
    arteriole_volume[1, 1, 1] = 1
    venule_volume = np.zeros((4, 5, 6), dtype=np.uint8)
    venule_volume[3, 4, 5] = 1
    _write_mask_with_voxel_size(arteriole_path, arteriole_volume)
    _write_mask_with_voxel_size(venule_path, venule_volume)

    arteriole, venule, arteriole_voxel, venule_voxel = load_large_vessel_masks(
        True, arteriole_path, venule_path
    )

    assert bool(arteriole[1, 1, 1]) and not bool(arteriole[3, 4, 5])
    assert bool(venule[3, 4, 5]) and not bool(venule[1, 1, 1])
    assert arteriole_voxel == VOXEL_SIZE_XYZ
    assert venule_voxel == VOXEL_SIZE_XYZ


def test_an_unsupported_mask_format_is_reported_rather_than_guessed(tmp_path: Path) -> None:
    mask_path = tmp_path / "arteriole.npy"
    np.save(mask_path, np.zeros((4, 5, 6), dtype=np.uint8))

    with pytest.raises(ValueError, match="Unsupported mask format"):
        load_large_vessel_masks(True, mask_path, mask_path)


def test_masks_are_transposed_into_canonical_z_y_x_by_axis_order(tmp_path: Path) -> None:
    """A mask left in (x, y, z) order would be indexed with z and x interchanged."""
    arteriole_path = tmp_path / "arteriole.tif"
    venule_path = tmp_path / "venule.tif"
    volume_xyz = np.zeros((6, 5, 4), dtype=np.uint8)
    volume_xyz[5, 0, 0] = 1  # x = 5, y = 0, z = 0
    _write_mask_with_voxel_size(arteriole_path, volume_xyz)
    _write_mask_with_voxel_size(venule_path, volume_xyz)

    arteriole, _venule, _av, _vv = load_large_vessel_masks(
        True, arteriole_path, venule_path, axis_order="xyz"
    )

    assert arteriole.shape == (4, 5, 6)
    assert bool(arteriole[0, 0, 5])


# --- voxel-size agreement between masks and the main image ------------------


def test_masks_matching_the_main_image_voxel_size_are_accepted(tmp_path: Path) -> None:
    arteriole_path = tmp_path / "arteriole.tif"
    venule_path = tmp_path / "venule.tif"
    _write_mask_with_voxel_size(arteriole_path, np.zeros((4, 5, 6), dtype=np.uint8))
    _write_mask_with_voxel_size(venule_path, np.zeros((4, 5, 6), dtype=np.uint8))

    _a, _v, arteriole_voxel, venule_voxel = load_and_validate_vessel_masks(
        mask_role="large",
        enabled=True,
        use_ilastik=False,
        arteriole_mask_path=arteriole_path,
        venule_mask_path=venule_path,
        image_shape=(4, 5, 6),
        main_voxel_size_xyz=VOXEL_SIZE_XYZ,
    )

    assert arteriole_voxel == VOXEL_SIZE_XYZ
    assert venule_voxel == VOXEL_SIZE_XYZ


def test_a_mask_whose_z_and_x_spacings_are_swapped_is_rejected(tmp_path: Path) -> None:
    """The classic silent failure: same numbers, wrong axes, identical for cubic voxels.

    A comparison that sorted or reordered the triple would accept this and every
    physical distance measured against the mask would be wrong.
    """
    arteriole_path = tmp_path / "arteriole.tif"
    venule_path = tmp_path / "venule.tif"
    swapped = (VOXEL_SIZE_XYZ[2], VOXEL_SIZE_XYZ[1], VOXEL_SIZE_XYZ[0])
    _write_mask_with_voxel_size(arteriole_path, np.zeros((4, 5, 6), dtype=np.uint8), swapped)
    _write_mask_with_voxel_size(venule_path, np.zeros((4, 5, 6), dtype=np.uint8), swapped)

    with pytest.raises(ValueError, match="Voxel-size mismatch"):
        load_and_validate_vessel_masks(
            mask_role="large",
            enabled=True,
            use_ilastik=False,
            arteriole_mask_path=arteriole_path,
            venule_mask_path=venule_path,
            image_shape=(4, 5, 6),
            main_voxel_size_xyz=VOXEL_SIZE_XYZ,
        )


def test_two_masks_that_disagree_with_each_other_are_rejected(tmp_path: Path) -> None:
    arteriole_path = tmp_path / "arteriole.tif"
    venule_path = tmp_path / "venule.tif"
    _write_mask_with_voxel_size(arteriole_path, np.zeros((4, 5, 6), dtype=np.uint8))
    _write_mask_with_voxel_size(
        venule_path, np.zeros((4, 5, 6), dtype=np.uint8), (0.4, 0.5, 2.5)
    )

    with pytest.raises(ValueError, match="Voxel-size mismatch"):
        load_and_validate_vessel_masks(
            mask_role="large",
            enabled=True,
            use_ilastik=False,
            arteriole_mask_path=arteriole_path,
            venule_mask_path=venule_path,
            image_shape=(4, 5, 6),
            main_voxel_size_xyz=VOXEL_SIZE_XYZ,
        )


# --- dilation converts to per-array-axis spacing ----------------------------


def _load_dilated(tmp_path: Path, dilation_microns: float) -> np.ndarray:
    volume = np.zeros((9, 9, 9), dtype=np.uint8)
    volume[4, 4, 4] = 1
    arteriole_path = tmp_path / "arteriole.tif"
    venule_path = tmp_path / "venule.tif"
    _write_mask_with_voxel_size(arteriole_path, volume)
    _write_mask_with_voxel_size(venule_path, volume)

    arteriole, _v, _av, _vv = load_and_validate_vessel_masks(
        mask_role="large",
        enabled=True,
        use_ilastik=False,
        arteriole_mask_path=arteriole_path,
        venule_mask_path=venule_path,
        image_shape=(9, 9, 9),
        main_voxel_size_xyz=VOXEL_SIZE_XYZ,
        dilation_microns=dilation_microns,
    )
    return np.asarray(arteriole)


def test_dilation_uses_array_axis_spacing_not_the_metadata_order(tmp_path: Path) -> None:
    """A 1 um dilation crosses two x voxels (0.4 um) but no z slice (2.0 um).

    ``main_voxel_size_xyz`` is physical (x, y, z) and has to be converted before
    it scales array indices. Passing it through unconverted swaps the z and x
    spacings, and the mask would grow along axis 0 instead of axis 2.
    """
    dilated = _load_dilated(tmp_path, 1.0)

    assert not bool(dilated[3, 4, 4]) and not bool(dilated[5, 4, 4])  # z: 2.0 um apart
    assert bool(dilated[4, 4, 6]) and not bool(dilated[4, 4, 7])  # x: 0.4 um apart
    assert bool(dilated[4, 6, 4]) and not bool(dilated[4, 7, 4])  # y: 0.5 um apart


def test_zero_dilation_leaves_the_mask_exactly_as_loaded(tmp_path: Path) -> None:
    dilated = _load_dilated(tmp_path, 0.0)
    assert int(dilated.sum()) == 1
    assert bool(dilated[4, 4, 4])


def test_loader_removes_subthreshold_mask_components(tmp_path: Path) -> None:
    """Volume filtering runs after load (and would run after dilation when set)."""
    shape = (16, 16, 16)
    arteriole = np.zeros(shape, dtype=np.uint8)
    venule = np.zeros(shape, dtype=np.uint8)
    arteriole[1, 1, 1] = 1
    arteriole[8:11, 8:11, 8:11] = 1  # 27 voxels
    venule[2, 2, 2] = 1
    venule[10:12, 10:12, 10:12] = 1  # 8 voxels

    arteriole_path = tmp_path / "arteriole.tif"
    venule_path = tmp_path / "venule.tif"
    _write_mask_with_voxel_size(arteriole_path, arteriole, voxel_size_xyz=(1.0, 1.0, 1.0))
    _write_mask_with_voxel_size(venule_path, venule, voxel_size_xyz=(1.0, 1.0, 1.0))

    cleaned_a, cleaned_v, _, _ = load_and_validate_vessel_masks(
        mask_role="large",
        enabled=True,
        use_ilastik=False,
        arteriole_mask_path=arteriole_path,
        venule_mask_path=venule_path,
        image_shape=shape,
        main_voxel_size_xyz=(1.0, 1.0, 1.0),
        min_component_volume_um3=5.0,
        remove_small_opposite_attached_components=False,
    )

    assert int(np.count_nonzero(cleaned_a)) == 27
    assert int(np.count_nonzero(cleaned_v)) == 8
    assert not cleaned_a[1, 1, 1]
    assert not cleaned_v[2, 2, 2]


def test_one_two_label_masks_stay_sparse_under_default_volume_filters(
    tmp_path: Path,
) -> None:
    """Regression: ``1/2`` masks must not become all-True then wipe the venule.

    Harvey-style vessel masks often use background=1 / foreground=2 with no
    zero. A raw ``astype(bool)`` / ``> 0`` cast fills the arteriole; with the
    default opposite-attached cleanup every venule component then sits at
    distance 0 from that solid opposite mask and is removed. Automated
    inlet/outlet assignment and the napari mask layers both read these arrays,
    so the failure shows up as a solid arteriole, a blank venule, and missing
    outlets -- not as a colormap issue.
    """
    from haemolynx.graph import select_terminal_nodes_from_large_vessel_masks
    from haemolynx.gui.results import MASK_LAYERS, vessel_mask_volume_layers
    import networkx as nx

    shape = (20, 20, 20)
    # Arteriole: 1/2 encoding (no zero). Correct binarisation keeps the 8^3 blob;
    # a raw bool cast fills the volume.
    arteriole = np.ones(shape, dtype=np.uint8)
    arteriole[2:10, 2:10, 2:10] = 2  # 512 voxels
    # Venule: ordinary 0/1, sized to survive the 200 um^3 min-volume filter but
    # still be eligible for opposite-attached cleanup (<= 250 um^3) if the
    # arteriole were wrongly solid.
    venule = np.zeros(shape, dtype=np.uint8)
    venule[12:18, 12:18, 12:18] = 1  # 216 voxels

    arteriole_path = tmp_path / "arteriole.tif"
    venule_path = tmp_path / "venule.tif"
    _write_mask_with_voxel_size(arteriole_path, arteriole, voxel_size_xyz=(1.0, 1.0, 1.0))
    _write_mask_with_voxel_size(venule_path, venule, voxel_size_xyz=(1.0, 1.0, 1.0))

    cleaned_a, cleaned_v, _, _ = load_and_validate_vessel_masks(
        mask_role="large",
        enabled=True,
        use_ilastik=False,
        arteriole_mask_path=arteriole_path,
        venule_mask_path=venule_path,
        image_shape=shape,
        main_voxel_size_xyz=(1.0, 1.0, 1.0),
        # Schema defaults after e344137: volume filter + opposite-attached ON.
        min_component_volume_um3=200.0,
        remove_small_opposite_attached_components=True,
        opposite_attached_max_component_volume_um3=250.0,
        opposite_attached_max_distance_microns=3.0,
    )

    assert cleaned_a is not None and cleaned_v is not None
    assert cleaned_a.dtype == bool and cleaned_v.dtype == bool
    assert not bool(np.all(cleaned_a)), "arteriole must not fill the volume"
    assert int(np.count_nonzero(cleaned_a)) == 512
    assert int(np.count_nonzero(cleaned_v)) == 216
    assert bool(cleaned_a[5, 5, 5]) and not bool(cleaned_a[0, 0, 0])
    assert bool(cleaned_v[15, 15, 15]) and not bool(cleaned_v[0, 0, 0])

    graph = nx.MultiGraph()
    graph.add_node(0, pos=(5.0, 5.0, 5.0))  # inside arteriole
    graph.add_node(1, pos=(15.0, 15.0, 15.0))  # inside venule
    graph.add_node(2, pos=(10.0, 10.0, 10.0))  # junction
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    inlets, outlets = select_terminal_nodes_from_large_vessel_masks(
        graph,
        cleaned_a,
        cleaned_v,
        voxel_size_zyx=(1.0, 1.0, 1.0),
    )
    assert inlets == [0]
    assert outlets == [1]

    layers = vessel_mask_volume_layers(
        {
            "large_arteriole_mask": cleaned_a,
            "large_venule_mask": cleaned_v,
        },
        voxel_size_zyx=(1.0, 1.0, 1.0),
    )
    by_name = {spec.name: spec for spec in layers}
    art_layer = by_name[MASK_LAYERS["large_arteriole_mask"]]
    ven_layer = by_name[MASK_LAYERS["large_venule_mask"]]
    assert not bool(np.all(art_layer.data > 0.5))
    assert int(np.count_nonzero(art_layer.data > 0.5)) == 512
    assert int(np.count_nonzero(ven_layer.data > 0.5)) == 216
    assert np.array_equal(art_layer.data > 0.5, cleaned_a)
    assert np.array_equal(ven_layer.data > 0.5, cleaned_v)


# --- small-role wiring is not just the large role with a prefix -------------


def test_the_small_role_reports_its_own_setting_names(tmp_path: Path) -> None:
    """Copy-pasting the large-role config would tell users to flip the wrong flag."""
    with pytest.raises(
        ValueError,
        match="use_ilastik_small_vessel_segmentation=True requires "
              "use_small_vessel_masks_for_boundary_assignment=True",
    ):
        load_and_validate_vessel_masks(
            mask_role="small",
            enabled=False,
            use_ilastik=True,
            arteriole_mask_path=None,
            venule_mask_path=None,
            image_shape=(4, 5, 6),
            main_voxel_size_xyz=(1.0, 1.0, 1.0),
        )


def test_a_small_mask_shape_mismatch_names_the_small_mask(tmp_path: Path) -> None:
    arteriole_path = tmp_path / "small_arteriole.tif"
    venule_path = tmp_path / "small_venule.tif"
    _write_mask_tif(arteriole_path, (4, 5, 6))
    _write_mask_tif(venule_path, (4, 5, 6))

    with pytest.raises(ValueError, match="small_arteriole_mask shape does not match"):
        load_and_validate_vessel_masks(
            mask_role="small",
            enabled=True,
            use_ilastik=False,
            arteriole_mask_path=arteriole_path,
            venule_mask_path=venule_path,
            image_shape=(4, 5, 7),
            main_voxel_size_xyz=(1.0, 1.0, 1.0),
        )


# --- vessel_mask_arguments: the config-to-parameter mapping -----------------


LARGE_SETTINGS = {
    "use_large_vessel_masks": True,
    "use_ilastik_large_vessel_segmentation": False,
    "large_arteriole_mask_path": "large_a.tif",
    "large_venule_mask_path": "large_v.tif",
    "large_vessel_mask_dilation_microns": 2.5,
    "large_vessel_min_component_volume_um3": 200.0,
    "large_vessel_remove_small_opposite_attached_components": True,
    "large_vessel_opposite_attached_max_component_volume_um3": 250.0,
    "large_vessel_opposite_attached_max_distance_microns": 3.0,
    "exclude_smaller_overlapping_volumes": False,
    "use_small_vessel_masks_for_boundary_assignment": False,
    "small_arteriole_mask_path": "small_a.tif",
    "small_venule_mask_path": "small_v.tif",
    "small_vessel_min_component_volume_um3": 50.0,
    "ilastik_output_dir": "out",
    "ilastik_output_suffix": ".tif",
    "ilastik_executable": "/usr/bin/ilastik",
    "image_axis_order": "xyz",
}


def test_every_mapped_argument_is_a_real_parameter_of_the_loader() -> None:
    """A typo in the mapping table would only surface as a TypeError mid-pipeline."""
    accepted = set(inspect.signature(load_and_validate_vessel_masks).parameters)
    for role in ("large", "small"):
        assert set(vessel_mask_arguments(LARGE_SETTINGS, role)) <= accepted


def test_the_large_role_picks_the_large_paths_and_the_dilation() -> None:
    arguments = vessel_mask_arguments(LARGE_SETTINGS, "large")

    assert arguments["mask_role"] == "large"
    assert arguments["enabled"] is True
    assert arguments["arteriole_mask_path"] == "large_a.tif"
    assert arguments["venule_mask_path"] == "large_v.tif"
    assert arguments["dilation_microns"] == 2.5
    assert arguments["min_component_volume_um3"] == 200.0
    assert arguments["remove_small_opposite_attached_components"] is True
    assert arguments["opposite_attached_max_component_volume_um3"] == 250.0
    assert arguments["opposite_attached_max_distance_microns"] == 3.0
    assert arguments["exclude_smaller_overlapping_volumes"] is False
    assert arguments["axis_order"] == "xyz"


def test_the_small_role_picks_the_small_paths_and_has_no_dilation() -> None:
    """The small settings are not the large names with a prefix; mixing them up
    would dilate the boundary masks and load the wrong files."""
    arguments = vessel_mask_arguments(LARGE_SETTINGS, "small")

    assert arguments["mask_role"] == "small"
    assert arguments["enabled"] is False
    assert arguments["arteriole_mask_path"] == "small_a.tif"
    assert arguments["venule_mask_path"] == "small_v.tif"
    assert arguments["min_component_volume_um3"] == 50.0
    assert "dilation_microns" not in arguments
    assert "remove_small_opposite_attached_components" not in arguments
    assert "exclude_smaller_overlapping_volumes" not in arguments


def test_settings_absent_from_the_config_are_left_to_their_defaults() -> None:
    """Passing them as None would override the loader's own defaults with nothing."""
    arguments = vessel_mask_arguments({"use_large_vessel_masks": True}, "large")

    assert arguments == {"mask_role": "large", "enabled": True}


def test_an_unknown_mask_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="mask_role must be 'large' or 'small'"):
        vessel_mask_arguments(LARGE_SETTINGS, "medium")
