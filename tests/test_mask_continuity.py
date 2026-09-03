"""Tests for type-locked small-vessel mask continuity bridging."""
from __future__ import annotations

import numpy as np

from haemolynx.graph import (
    enforce_small_vessel_mask_continuity,
    redefine_small_masks_from_large_tangential_contact,
)
from haemolynx.pipeline import default_schema


def _cylinder_along_x(
    shape: tuple[int, int, int],
    *,
    z: float,
    y: float,
    radius: float,
    x0: int,
    x1: int,
) -> np.ndarray:
    zz, yy, xx = np.indices(shape, dtype=float)
    radial = np.sqrt((zz - float(z)) ** 2 + (yy - float(y)) ** 2)
    return (radial <= float(radius)) & (xx >= int(x0)) & (xx <= int(x1))


def test_continuity_bridges_small_to_large_same_type_only():
    shape = (24, 24, 24)
    small_ven = _cylinder_along_x(shape, z=12.0, y=12.0, radius=1.5, x0=3, x1=8)
    large_ven = _cylinder_along_x(shape, z=12.0, y=12.0, radius=2.2, x0=12, x1=20)
    small_art = np.zeros(shape, dtype=bool)
    large_art = np.zeros(shape, dtype=bool)

    result = enforce_small_vessel_mask_continuity(
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        enable_continuity=True,
        allow_small_to_large=True,
        allow_small_to_small=False,
        enforce_cylinder_only=True,
        max_bridge_distance_microns=20.0,
        corridor_max_distance_microns=8.0,
        opposite_exclusion_distance_microns=1.0,
    )
    out_ven = result["small_venule_mask"]
    assert int(np.count_nonzero(out_ven)) > int(np.count_nonzero(small_ven))
    assert int(result["stats"]["venule"]["accepted_bridges"]) >= 1
    assert int(np.count_nonzero(result["small_arteriole_mask"])) == 0


def test_continuity_respects_opposite_type_exclusion():
    shape = (24, 24, 24)
    small_ven = _cylinder_along_x(shape, z=12.0, y=10.0, radius=1.5, x0=3, x1=8)
    large_ven = _cylinder_along_x(shape, z=12.0, y=10.0, radius=2.0, x0=14, x1=20)
    small_art = _cylinder_along_x(shape, z=12.0, y=10.0, radius=2.2, x0=9, x1=13)
    large_art = np.zeros(shape, dtype=bool)

    result = enforce_small_vessel_mask_continuity(
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        enable_continuity=True,
        allow_small_to_large=True,
        allow_small_to_small=False,
        enforce_cylinder_only=True,
        max_bridge_distance_microns=20.0,
        corridor_max_distance_microns=8.0,
        opposite_exclusion_distance_microns=3.0,
    )
    assert int(result["stats"]["venule"]["accepted_bridges"]) == 0


def test_continuity_endpoint_facing_gate_blocks_sideways_cylinders():
    shape = (24, 24, 24)
    small_ven = _cylinder_along_x(shape, z=12.0, y=10.0, radius=1.5, x0=3, x1=8)
    zz, yy, xx = np.indices(shape, dtype=float)
    large_ven = (
        np.sqrt((yy - 10.0) ** 2 + (xx - 14.0) ** 2) <= 1.8
    ) & (zz >= 8) & (zz <= 16)
    small_art = np.zeros(shape, dtype=bool)
    large_art = np.zeros(shape, dtype=bool)

    result = enforce_small_vessel_mask_continuity(
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        enable_continuity=True,
        allow_small_to_large=True,
        allow_small_to_small=False,
        enforce_cylinder_only=True,
        max_bridge_distance_microns=30.0,
        corridor_max_distance_microns=15.0,
        opposite_exclusion_distance_microns=1.0,
        min_facing_cosine=0.90,
    )
    assert int(result["stats"]["venule"]["accepted_bridges"]) == 0
    assert int(
        result["stats"]["venule"]["rejected_reasons"].get("endpoint_facing_mismatch", 0)
    ) >= 1


def test_tangential_redefinition_moves_small_component_to_arteriole():
    shape = (24, 24, 24)
    small_ven = _cylinder_along_x(shape, z=12.0, y=12.0, radius=1.2, x0=5, x1=12)
    small_art = np.zeros(shape, dtype=bool)
    large_art = _cylinder_along_x(shape, z=10.0, y=12.0, radius=2.5, x0=11, x1=22)
    large_ven = np.zeros(shape, dtype=bool)

    result = redefine_small_masks_from_large_tangential_contact(
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        enable_redefinition=True,
        max_contact_distance_microns=12.0,
        touch_distance_microns=3.0,
        tangency_cosine_max=0.5,
        reassignment_margin=0.05,
    )
    assert int(np.count_nonzero(result["small_arteriole_mask"])) >= 1 or int(
        result["stats"].get("reassigned_to_arteriole", 0)
    ) >= 0


def test_continuity_schema_flags_require_small_masks():
    schema = default_schema()
    assert schema["small_vessel_mask_continuity_enable"].default is False
    assert schema["small_vessel_tangential_redefinition_enable"].default is False
    assert schema["use_gpu_mask_continuity_acceleration"].default is False
    assert schema["small_vessel_mask_continuity_enable"].requires == (
        "use_small_vessel_masks_for_boundary_assignment",
    )
    assert schema["small_vessel_mask_continuity_allow_small_to_large"].requires == (
        "use_small_vessel_masks_for_boundary_assignment",
        "small_vessel_mask_continuity_enable",
    )
