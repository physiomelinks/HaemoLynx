"""Tests for vessel-mask connected-component volume and overlap cleanup."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx.graph import (
    exclude_smaller_overlapping_large_vessel_components,
    remove_small_opposite_attached_large_vessel_components,
    remove_small_vessel_components_by_volume,
)
from haemolynx.pipeline import default_schema


def test_remove_small_vessel_components_by_volume_filters_small_components():
    arteriole_mask = np.zeros((20, 20, 20), dtype=bool)
    venule_mask = np.zeros((20, 20, 20), dtype=bool)

    # Small components (size=1 and size=2 voxels) should be removed.
    arteriole_mask[1, 1, 1] = True
    venule_mask[2, 2, 2] = True
    venule_mask[2, 2, 3] = True

    # Larger components should be retained.
    arteriole_mask[10:12, 10:12, 10:12] = True  # 8 voxels
    venule_mask[14:16, 14:16, 14:16] = True  # 8 voxels

    cleaned_arteriole, cleaned_venule, stats = remove_small_vessel_components_by_volume(
        arteriole_mask,
        venule_mask,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        min_component_volume_um3=3.0,
    )

    assert int(np.count_nonzero(cleaned_arteriole)) == 8
    assert int(np.count_nonzero(cleaned_venule)) == 8
    assert int(stats["arteriole"]["removed_component_count"]) == 1
    assert int(stats["venule"]["removed_component_count"]) == 1


def test_remove_small_vessel_components_by_volume_disabled_at_zero_threshold():
    arteriole_mask = np.zeros((6, 6, 6), dtype=bool)
    venule_mask = np.zeros((6, 6, 6), dtype=bool)
    arteriole_mask[1, 1, 1] = True
    venule_mask[4, 4, 4] = True

    cleaned_arteriole, cleaned_venule, stats = remove_small_vessel_components_by_volume(
        arteriole_mask,
        venule_mask,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        min_component_volume_um3=0.0,
    )

    assert np.array_equal(cleaned_arteriole, arteriole_mask)
    assert np.array_equal(cleaned_venule, venule_mask)
    assert float(stats["threshold_um3"]) == 0.0


def test_remove_small_vessel_components_uses_physical_voxel_volume():
    """Anisotropic voxels: one voxel can exceed a threshold that would keep it in um^3."""
    arteriole_mask = np.zeros((8, 8, 8), dtype=bool)
    venule_mask = np.zeros((8, 8, 8), dtype=bool)
    arteriole_mask[1, 1, 1] = True  # 1 voxel * 0.4*0.5*2.0 = 0.4 um^3
    arteriole_mask[4:6, 4:6, 4:6] = True  # 8 voxels = 3.2 um^3
    venule_mask[2, 2, 2] = True

    cleaned_arteriole, cleaned_venule, stats = remove_small_vessel_components_by_volume(
        arteriole_mask,
        venule_mask,
        voxel_size_xyz=(0.4, 0.5, 2.0),
        min_component_volume_um3=1.0,
    )

    assert int(np.count_nonzero(cleaned_arteriole)) == 8
    assert not np.any(cleaned_venule)
    assert int(stats["arteriole"]["removed_component_count"]) == 1
    assert int(stats["venule"]["removed_component_count"]) == 1


def test_exclude_smaller_overlapping_large_vessel_components_venule_inside_arteriole():
    arteriole_mask = np.zeros((12, 12, 12), dtype=bool)
    venule_mask = np.zeros((12, 12, 12), dtype=bool)
    arteriole_mask[2:10, 2:10, 2:10] = True
    venule_mask[5:7, 5:7, 5:7] = True

    cleaned_arteriole, cleaned_venule = (
        exclude_smaller_overlapping_large_vessel_components(
            arteriole_mask, venule_mask
        )
    )

    assert np.array_equal(cleaned_arteriole, arteriole_mask)
    assert not np.any(cleaned_venule)


def test_exclude_smaller_overlapping_large_vessel_components_arteriole_inside_venule():
    arteriole_mask = np.zeros((12, 12, 12), dtype=bool)
    venule_mask = np.zeros((12, 12, 12), dtype=bool)
    venule_mask[2:10, 2:10, 2:10] = True
    arteriole_mask[5:7, 5:7, 5:7] = True

    cleaned_arteriole, cleaned_venule = (
        exclude_smaller_overlapping_large_vessel_components(
            arteriole_mask, venule_mask
        )
    )

    assert not np.any(cleaned_arteriole)
    assert np.array_equal(cleaned_venule, venule_mask)


def test_remove_small_opposite_attached_large_vessel_components():
    arteriole_mask = np.zeros((20, 20, 20), dtype=bool)
    venule_mask = np.zeros((20, 20, 20), dtype=bool)

    # Large arteriole body.
    arteriole_mask[5:15, 5:15, 5:15] = True
    # Tiny venule blob sitting next to the arteriole surface (should be removed).
    venule_mask[5:7, 5:7, 3:5] = True  # 8 voxels, adjacent in z
    # Distant tiny venule (should be kept: far from opposite mask).
    venule_mask[18, 18, 18] = True

    cleaned_arteriole, cleaned_venule, stats = (
        remove_small_opposite_attached_large_vessel_components(
            arteriole_mask,
            venule_mask,
            voxel_size_xyz=(1.0, 1.0, 1.0),
            max_component_volume_um3=20.0,
            max_attach_distance_microns=2.0,
        )
    )

    assert np.array_equal(cleaned_arteriole, arteriole_mask)
    assert not np.any(cleaned_venule[5:7, 5:7, 3:5])
    assert cleaned_venule[18, 18, 18]
    assert int(stats["venule"]["removed_component_count"]) == 1


def test_vessel_mask_volume_settings_exist_with_historical_defaults():
    schema = default_schema()
    assert schema["large_vessel_min_component_volume_um3"].default == 200.0
    assert schema["large_vessel_min_component_volume_um3"].unit == "um3"
    assert schema["small_vessel_min_component_volume_um3"].default == 50.0
    assert schema["small_vessel_min_component_volume_um3"].unit == "um3"
    assert schema["large_vessel_remove_small_opposite_attached_components"].default is True
    assert (
        schema["large_vessel_opposite_attached_max_component_volume_um3"].default
        == 250.0
    )
    assert schema["large_vessel_opposite_attached_max_distance_microns"].default == 3.0
    assert schema["exclude_smaller_overlapping_volumes"].default is False

    assert schema["large_vessel_min_component_volume_um3"].requires == (
        "use_large_vessel_masks",
    )
    assert schema["small_vessel_min_component_volume_um3"].requires == (
        "use_small_vessel_masks_for_boundary_assignment",
    )
    assert schema["large_vessel_opposite_attached_max_component_volume_um3"].requires == (
        "use_large_vessel_masks",
        "large_vessel_remove_small_opposite_attached_components",
    )
    assert schema["use_large_vessel_masks"].requires == ("automated_vessel_assignment",)
    assert schema["automated_vessel_assignment"].section == "Vessel masks"
    assert "override" in schema["automated_vessel_assignment"].help.lower()


def test_vessel_mask_volume_settings_are_not_skeleton_min_settings():
    schema = default_schema()
    assert "skeleton_min_component_percent" in schema
    assert schema["skeleton_min_component_percent"].section != "Vessel masks"
    assert schema["large_vessel_min_component_volume_um3"].section == "Vessel masks"
    assert schema["small_vessel_min_component_volume_um3"].section == "Vessel masks"
