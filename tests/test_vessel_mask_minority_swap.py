"""Tests for relabelling a minority arteriole/venule mask component.

A classifier that exports one label per voxel can split one vessel between the
two classes: part of it arteriole, the rest venule. The pieces touch but never
share a voxel, so the overlap cleanup (``exclude_smaller_overlapping_volumes``)
has nothing to remove and the vessel stays two colours.
``swap_minority_touching_vessel_components`` moves the smaller piece into the
larger one's class, and must leave alone a genuine vessel that only runs beside
another -- telling those apart is what its contact fraction is for.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx.graph.large_vessels import (
    exclude_smaller_overlapping_large_vessel_components,
    swap_minority_touching_vessel_components,
)
from haemolynx.io import load_and_validate_vessel_masks, vessel_mask_arguments
from haemolynx.pipeline import default_schema

SHAPE = (21, 32, 64)


def _cylinder(
    radius: float, centre_zy: tuple[float, float], x_range: tuple[int, int]
) -> np.ndarray:
    """A solid tube running along x (array axis 2)."""
    z, y, x = np.indices(SHAPE)
    inside = (z - centre_zy[0]) ** 2 + (y - centre_zy[1]) ** 2 <= radius**2
    return inside & (x >= x_range[0]) & (x < x_range[1])


def _split_tube(arteriole_length: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One tube: arteriole over the top half of its first stretch, venule elsewhere.

    The labels split across the vessel's width and never overlap -- the way a
    one-label-per-voxel export cuts a vessel it could not classify.
    """
    tube = _cylinder(6.0, (10.0, 12.0), (2, 62))
    z = np.indices(SHAPE)[0]
    arteriole = tube & (z < 10) & (np.indices(SHAPE)[2] < 2 + arteriole_length)
    venule = tube & ~arteriole
    return tube, arteriole, venule


def test_a_vessel_split_across_its_width_becomes_one_venule() -> None:
    tube, arteriole, venule = _split_tube()
    assert not np.any(arteriole & venule)

    swapped_arteriole, swapped_venule, stats = swap_minority_touching_vessel_components(
        arteriole, venule
    )

    assert not np.any(swapped_arteriole)
    assert np.array_equal(swapped_venule, tube)
    assert stats["swapped_to_venule_component_count"] == 1
    assert stats["swapped_to_venule_voxel_count"] == int(arteriole.sum())
    assert stats["swapped_to_arteriole_component_count"] == 0
    (component,) = stats["components"]
    assert component["from"] == "arteriole" and component["to"] == "venule"
    assert component["largest_partner_voxel_count"] == int(venule.sum())
    assert component["contact_fraction"] > 0.3


def test_the_overlap_cleanup_cannot_fix_a_split_that_does_not_overlap() -> None:
    """Why the swap exists: the reported split vessel went through the overlap
    cleanup unchanged, because touching pieces share no voxel to remove."""
    _tube, arteriole, venule = _split_tube()

    cleaned_arteriole, cleaned_venule = exclude_smaller_overlapping_large_vessel_components(
        arteriole, venule
    )

    assert np.array_equal(cleaned_arteriole, arteriole)
    assert np.array_equal(cleaned_venule, venule)


def test_a_minority_venule_is_swapped_into_the_arteriole() -> None:
    tube, arteriole, venule = _split_tube()

    swapped_arteriole, swapped_venule, stats = swap_minority_touching_vessel_components(
        venule_mask=arteriole, arteriole_mask=venule
    )

    assert np.array_equal(swapped_arteriole, tube)
    assert not np.any(swapped_venule)
    assert stats["swapped_to_arteriole_component_count"] == 1
    assert stats["swapped_to_venule_component_count"] == 0


def test_an_arteriole_running_beside_a_larger_venule_is_kept() -> None:
    """Two separate tubes touching along one side: a real arteriole/venule pair."""
    arteriole = _cylinder(4.0, (10.0, 7.0), (2, 62))
    venule = _cylinder(6.0, (10.0, 18.0), (2, 62))
    assert not np.any(arteriole & venule)

    swapped_arteriole, swapped_venule, stats = swap_minority_touching_vessel_components(
        arteriole, venule
    )

    assert np.array_equal(swapped_arteriole, arteriole)
    assert np.array_equal(swapped_venule, venule)
    (component,) = stats["components"]
    assert not component["swapped"]
    assert 0.0 < component["contact_fraction"] < 0.3
    assert stats["kept_touching_component_count"] == 1
    assert stats["max_kept_contact_fraction"] == component["contact_fraction"]


def test_a_piece_joined_end_to_end_needs_a_lower_contact_fraction() -> None:
    """A short piece touching only at its end has a small contact fraction: the
    default keeps it, and lowering the fraction is how to catch it."""
    arteriole = _cylinder(5.0, (10.0, 12.0), (2, 12))
    venule = _cylinder(5.0, (10.0, 12.0), (12, 62))

    kept_arteriole, kept_venule, kept_stats = swap_minority_touching_vessel_components(
        arteriole, venule
    )
    swapped_arteriole, swapped_venule, _ = swap_minority_touching_vessel_components(
        arteriole, venule, min_contact_fraction=0.1
    )

    assert np.array_equal(kept_arteriole, arteriole)
    assert np.array_equal(kept_venule, venule)
    assert 0.1 < kept_stats["components"][0]["contact_fraction"] < 0.3
    assert not np.any(swapped_arteriole)
    assert np.array_equal(swapped_venule, arteriole | venule)


def test_the_size_ratio_limits_how_large_a_minority_may_be() -> None:
    _tube, arteriole, venule = _split_tube()
    ratio = arteriole.sum() / venule.sum()

    kept_arteriole, kept_venule, stats = swap_minority_touching_vessel_components(
        arteriole, venule, max_size_ratio=0.5 * ratio
    )
    swapped_arteriole, _swapped_venule, _ = swap_minority_touching_vessel_components(
        arteriole, venule, max_size_ratio=1.5 * ratio
    )

    assert np.array_equal(kept_arteriole, arteriole)
    assert np.array_equal(kept_venule, venule)
    assert stats["components"] == []
    assert not np.any(swapped_arteriole)


def test_equal_sized_touching_components_are_both_kept() -> None:
    """Neither is the minority, so neither is relabelled."""
    arteriole = np.zeros(SHAPE, dtype=bool)
    venule = np.zeros(SHAPE, dtype=bool)
    arteriole[2:6, 2:6, 2:12] = True
    venule[6:10, 2:6, 2:12] = True

    swapped_arteriole, swapped_venule, stats = swap_minority_touching_vessel_components(
        arteriole, venule
    )

    assert np.array_equal(swapped_arteriole, arteriole)
    assert np.array_equal(swapped_venule, venule)
    assert stats["components"] == []


def test_a_chain_of_pieces_ends_up_one_class() -> None:
    """small arteriole | venule | large arteriole, in a row.

    Largest first, the venule joins the large arteriole; the small arteriole
    then touches no venule and stays. Deciding the smallest first would flip it
    to venule just before its neighbour became arteriole, leaving one vessel
    in two classes again.
    """
    small_arteriole = np.zeros(SHAPE, dtype=bool)
    venule = np.zeros(SHAPE, dtype=bool)
    large_arteriole = np.zeros(SHAPE, dtype=bool)
    small_arteriole[4:8, 4:8, 2:4] = True
    venule[4:8, 4:8, 4:8] = True
    large_arteriole[4:8, 4:8, 8:22] = True

    # The cubes meet end to end: one face in six faces the next (1/6 of the area).
    swapped_arteriole, swapped_venule, stats = swap_minority_touching_vessel_components(
        small_arteriole | large_arteriole, venule, min_contact_fraction=0.15
    )

    assert np.array_equal(swapped_arteriole, small_arteriole | venule | large_arteriole)
    assert not np.any(swapped_venule)
    assert stats["swapped_to_arteriole_component_count"] == 1
    assert stats["swapped_to_venule_component_count"] == 0


def test_masks_that_do_not_touch_are_returned_unchanged() -> None:
    arteriole = _cylinder(4.0, (10.0, 6.0), (2, 62))
    venule = _cylinder(4.0, (10.0, 24.0), (2, 62))
    arteriole_before = arteriole.copy()
    venule_before = venule.copy()

    swapped_arteriole, swapped_venule, stats = swap_minority_touching_vessel_components(
        arteriole, venule
    )

    assert np.array_equal(swapped_arteriole, arteriole_before)
    assert np.array_equal(swapped_venule, venule_before)
    assert stats["components"] == []
    assert stats["swapped_to_venule_component_count"] == 0


def test_the_inputs_are_not_modified_and_the_outputs_are_boolean() -> None:
    _tube, arteriole, venule = _split_tube()
    arteriole_uint8 = arteriole.astype(np.uint8)
    venule_uint8 = venule.astype(np.uint8)

    swapped_arteriole, swapped_venule, _ = swap_minority_touching_vessel_components(
        arteriole_uint8, venule_uint8
    )

    assert np.array_equal(arteriole_uint8, arteriole.astype(np.uint8))
    assert np.array_equal(venule_uint8, venule.astype(np.uint8))
    assert swapped_arteriole.dtype == bool and swapped_venule.dtype == bool


def test_missing_masks_pass_through() -> None:
    venule = np.ones((2, 2, 2), dtype=bool)
    assert swap_minority_touching_vessel_components(None, venule)[:2] == (None, venule)


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("max_size_ratio", 0.0),
        ("max_size_ratio", 1.5),
        ("min_contact_fraction", -0.1),
        ("min_contact_fraction", 1.1),
    ],
)
def test_out_of_range_settings_are_refused(keyword: str, value: float) -> None:
    _tube, arteriole, venule = _split_tube()
    with pytest.raises(ValueError, match=keyword):
        swap_minority_touching_vessel_components(arteriole, venule, **{keyword: value})


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        swap_minority_touching_vessel_components(
            np.zeros((2, 2, 2), dtype=bool), np.zeros((2, 2, 3), dtype=bool)
        )


# --- the loader ------------------------------------------------------------


def _write_masks(tmp_path: Path, arteriole: np.ndarray, venule: np.ndarray):
    arteriole_path = tmp_path / "arteriole.tif"
    venule_path = tmp_path / "venule.tif"
    for path, volume in ((arteriole_path, arteriole), (venule_path, venule)):
        tifffile.imwrite(
            path,
            volume.astype(np.uint8),
            imagej=True,
            resolution=(1.0, 1.0),
            metadata={"spacing": 1.0, "unit": "um"},
        )
    return arteriole_path, venule_path


def _load(tmp_path: Path, mask_role: str, arteriole_length: int = 20, **options):
    _tube, arteriole, venule = _split_tube(arteriole_length)
    arteriole_path, venule_path = _write_masks(tmp_path, arteriole, venule)
    loaded_arteriole, loaded_venule, _, _ = load_and_validate_vessel_masks(
        mask_role=mask_role,
        enabled=True,
        use_ilastik=False,
        arteriole_mask_path=arteriole_path,
        venule_mask_path=venule_path,
        image_shape=SHAPE,
        main_voxel_size_xyz=(1.0, 1.0, 1.0),
        **options,
    )
    return loaded_arteriole, loaded_venule


@pytest.mark.parametrize("mask_role", ["large", "small"])
def test_the_loader_swaps_when_the_setting_is_on(tmp_path: Path, mask_role: str) -> None:
    tube, _arteriole, _venule = _split_tube()

    loaded_arteriole, loaded_venule = _load(
        tmp_path, mask_role, swap_minority_components=True
    )

    assert not np.any(loaded_arteriole)
    assert np.array_equal(loaded_venule, tube)


@pytest.mark.parametrize("mask_role", ["large", "small"])
def test_the_loader_leaves_the_split_alone_when_the_setting_is_off(
    tmp_path: Path, mask_role: str
) -> None:
    _tube, arteriole, venule = _split_tube()

    loaded_arteriole, loaded_venule = _load(tmp_path, mask_role)

    assert np.array_equal(loaded_arteriole, arteriole)
    assert np.array_equal(loaded_venule, venule)


def test_the_loader_passes_its_thresholds_through(tmp_path: Path) -> None:
    _tube, arteriole, venule = _split_tube()

    loaded_arteriole, loaded_venule = _load(
        tmp_path,
        "large",
        swap_minority_components=True,
        swap_max_size_ratio=0.05,
        swap_min_contact_fraction=0.3,
    )

    assert np.array_equal(loaded_arteriole, arteriole)
    assert np.array_equal(loaded_venule, venule)


def test_a_small_minority_is_swapped_before_opposite_attached_cleanup_deletes_it(
    tmp_path: Path,
) -> None:
    """The opposite-attached cleanup deletes tiny pieces touching the other
    class. Run first, it would leave a hole in the vessel; the swap runs before
    it and gives the piece back to the vessel instead."""
    tube, arteriole, venule = _split_tube(arteriole_length=4)
    assert arteriole.sum() <= 250  # small enough for the opposite-attached cleanup
    cleanup = {
        "arteriole_length": 4,
        "remove_small_opposite_attached_components": True,
        "opposite_attached_max_component_volume_um3": 250.0,
        "opposite_attached_max_distance_microns": 3.0,
    }

    deleted_arteriole, deleted_venule = _load(tmp_path, "large", **cleanup)
    swapped_arteriole, swapped_venule = _load(
        tmp_path, "large", swap_minority_components=True, **cleanup
    )

    assert not np.any(deleted_arteriole)
    assert np.array_equal(deleted_venule, venule)  # the hole
    assert not np.any(swapped_arteriole)
    assert np.array_equal(swapped_venule, tube)


def test_the_run_log_names_each_swap(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="haemolynx.io.automated_vessel_assignment"):
        _load(tmp_path, "large", swap_minority_components=True)

    assert "arteriole component of 1000 voxels -> venule" in caplog.text
    assert "swapped 1 arteriole component(s) to venule (1000 voxels)" in caplog.text


def test_the_run_log_counts_rather_than_lists_a_crowd_of_swaps(
    tmp_path: Path, caplog
) -> None:
    """Twenty-five mislabelled specks on one venule: twenty named, five counted."""
    shape = (6, 8, 100)
    venule = np.zeros(shape, dtype=bool)
    arteriole = np.zeros(shape, dtype=bool)
    venule[0:2, :, :] = True
    for speck in range(25):
        # Sunk halfway into the venule: it faces the venule over half its surface.
        arteriole[1:3, 2:4, 4 * speck : 4 * speck + 2] = True
    venule &= ~arteriole
    arteriole_path, venule_path = _write_masks(tmp_path, arteriole, venule)

    with caplog.at_level(logging.INFO, logger="haemolynx.io.automated_vessel_assignment"):
        loaded_arteriole, loaded_venule, _, _ = load_and_validate_vessel_masks(
            mask_role="small",
            enabled=True,
            use_ilastik=False,
            arteriole_mask_path=arteriole_path,
            venule_mask_path=venule_path,
            image_shape=shape,
            main_voxel_size_xyz=(1.0, 1.0, 1.0),
            swap_minority_components=True,
        )

    assert not np.any(loaded_arteriole)
    assert np.array_equal(loaded_venule, arteriole | venule)
    assert caplog.text.count("arteriole component of 8 voxels -> venule") == 20
    assert "and 5 smaller component(s)" in caplog.text


# --- the settings ----------------------------------------------------------


@pytest.mark.parametrize(
    "prefix, gate",
    [
        ("large_vessel", "use_large_vessel_masks"),
        ("small_vessel", "use_small_vessel_masks_for_boundary_assignment"),
    ],
)
def test_each_mask_size_has_its_own_opt_in_swap_settings(prefix: str, gate: str) -> None:
    schema = default_schema()
    toggle = schema[f"{prefix}_swap_minority_components"]
    ratio = schema[f"{prefix}_swap_max_size_ratio"]
    contact = schema[f"{prefix}_swap_min_contact_fraction"]

    assert toggle.default is False
    assert toggle.requires == (gate, "automated_vessel_assignment")
    assert ratio.default == 1.0 and contact.default == 0.3
    for setting in (ratio, contact):
        assert setting.requires == (
            gate,
            "automated_vessel_assignment",
            f"{prefix}_swap_minority_components",
        )
        assert setting.unit == "fraction" and setting.maximum == 1.0
    for setting in (toggle, ratio, contact):
        assert setting.section == "Vessel masks"


@pytest.mark.parametrize("role", ["large", "small"])
def test_the_config_reaches_the_loader_for_each_role(role: str) -> None:
    settings = {
        "use_large_vessel_masks": True,
        "use_small_vessel_masks_for_boundary_assignment": True,
        "large_vessel_swap_minority_components": True,
        "large_vessel_swap_max_size_ratio": 0.8,
        "large_vessel_swap_min_contact_fraction": 0.25,
        "small_vessel_swap_minority_components": False,
        "small_vessel_swap_max_size_ratio": 0.6,
        "small_vessel_swap_min_contact_fraction": 0.4,
    }

    arguments = vessel_mask_arguments(settings, role)

    expected = {
        "large": (True, 0.8, 0.25),
        "small": (False, 0.6, 0.4),
    }[role]
    assert (
        arguments["swap_minority_components"],
        arguments["swap_max_size_ratio"],
        arguments["swap_min_contact_fraction"],
    ) == expected


def _round_tube_split(plane: str):
    """A tube 12 um across on 2 x 0.5 x 0.5 um voxels, its first stretch
    labelled arteriole on one side of a lengthwise split: *plane* "z" splits
    it top from bottom, "y" side from side."""
    spacing = (2.0, 0.5, 0.5)
    z, y, x = np.indices((15, 40, 64))
    tube = ((z - 7) * spacing[0]) ** 2 + ((y - 20) * spacing[1]) ** 2 <= 6.0**2
    tube &= (x >= 2) & (x < 62)
    side = (z < 7) if plane == "z" else (y < 20)
    arteriole = tube & side & (x < 22)
    return arteriole, tube & ~arteriole, spacing


def test_the_contact_fraction_does_not_depend_on_which_way_the_split_runs() -> None:
    """Regression (audit): counted as surface voxels within 26-adjacency, a
    tube split lengthwise on 2 x 0.5 x 0.5 um voxels read about 0.54 split top
    from bottom and 0.21 split side from side -- swapped or kept against the
    0.3 threshold by the split's orientation alone. As facing surface area the
    two agree, and both are swapped."""
    fractions = {}
    for plane in ("z", "y"):
        arteriole, venule, spacing = _round_tube_split(plane)
        swapped_arteriole, _v, stats = swap_minority_touching_vessel_components(
            arteriole, venule, voxel_size_zyx=spacing
        )
        (component,) = stats["components"]
        fractions[plane] = component["contact_fraction"]
        assert component["swapped"] and not np.any(swapped_arteriole)

    assert abs(fractions["z"] - fractions["y"]) < 0.02
