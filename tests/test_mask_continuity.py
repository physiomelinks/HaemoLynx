"""Tests for type-locked small-vessel mask continuity bridging."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import label

from haemolynx.graph import enforce_small_vessel_mask_continuity
from haemolynx.graph.mask_continuity import _component_descriptors
from haemolynx.pipeline import default_schema

# z sampled three times more coarsely than x, as on a confocal stack.
_ANISOTROPIC_ZYX = (1.5, 0.5, 0.5)


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


def _zx_direction(degrees_from_x: float) -> np.ndarray:
    angle = np.radians(float(degrees_from_x))
    return np.asarray([np.sin(angle), 0.0, np.cos(angle)])


def _segment_um(
    shape: tuple[int, int, int],
    *,
    start_um: np.ndarray,
    direction_zyx: np.ndarray,
    length_um: float,
    radius_um: float,
) -> np.ndarray:
    """Voxels within *radius_um* of a segment laid out in microns.

    The ends are rounded, so each piece has one tip for its endpoint instead
    of a flat face whose voxels tie along the axis.
    """
    points = np.indices(shape, dtype=float).reshape(3, -1).T * np.asarray(_ANISOTROPIC_ZYX)
    axis = np.asarray(direction_zyx, dtype=float) / np.linalg.norm(direction_zyx)
    offset = points - np.asarray(start_um, dtype=float)
    along = np.clip(offset @ axis, 0.0, float(length_um))
    radial = np.linalg.norm(offset - along[:, None] * axis, axis=1)
    return (radial <= float(radius_um)).reshape(shape)


def _bridge_venules_on_anisotropic_stack(
    small_ven: np.ndarray, large_ven: np.ndarray, **gates: float
) -> dict:
    empty = np.zeros_like(small_ven)
    return enforce_small_vessel_mask_continuity(
        small_arteriole_mask=empty,
        small_venule_mask=small_ven,
        large_arteriole_mask=empty,
        large_venule_mask=large_ven,
        voxel_size_zyx=_ANISOTROPIC_ZYX,
        allow_small_to_large=True,
        allow_small_to_small=False,
        **gates,
    )


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
    assert int(np.count_nonzero(small_ven)) == 54
    assert int(np.count_nonzero(out_ven)) == 124
    assert int(result["stats"]["venule"]["accepted_bridges"]) == 1
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
    assert (
        result["stats"]["venule"]["rejected_reasons"].get("endpoint_facing_mismatch", 0)
    ) == 1


@pytest.mark.parametrize(
    ("tilt_degrees", "radius_um", "tip_gap_um"),
    [
        # In voxel indices the gap between the tips faced each axis at a
        # cosine of 0.45, under the default 0.82: endpoint_facing_mismatch.
        pytest.param(45.0, 3.0, 3.0, id="45_degrees_in_zx"),
        # In voxel indices z shrank threefold and the small piece's linearity
        # read 0.05, under the default 0.45: source_not_cylindrical.
        pytest.param(90.0, 2.5, 6.0, id="along_z"),
    ],
)
def test_continuity_bridges_collinear_pieces_on_anisotropic_stack(
    tilt_degrees, radius_um, tip_gap_um
):
    shape = (32, 30, 70)
    direction = _zx_direction(tilt_degrees)
    start = np.asarray([4.0, 7.5, 4.0])
    small_ven = _segment_um(
        shape, start_um=start, direction_zyx=direction, length_um=10.0, radius_um=radius_um
    )
    large_start = start + direction * (10.0 + 2.0 * radius_um + tip_gap_um)
    large_ven = _segment_um(
        shape, start_um=large_start, direction_zyx=direction, length_um=14.0, radius_um=radius_um
    )
    structure = np.ones((3, 3, 3), dtype=bool)
    assert label(small_ven | large_ven, structure=structure)[1] == 2

    result = _bridge_venules_on_anisotropic_stack(small_ven, large_ven)

    out_ven = result["small_venule_mask"]
    assert int(result["stats"]["venule"]["accepted_bridges"]) == 1
    assert np.all(out_ven[small_ven])
    assert label(out_ven | large_ven, structure=structure)[1] == 1


def test_continuity_facing_gate_is_measured_in_microns():
    """A parallel piece 10.5 um further down z faces the source at a cosine
    of 0.65; in voxel indices z shrank threefold, it read 0.93 and was bridged."""
    shape = (20, 30, 90)
    along_x = _zx_direction(0.0)
    start = np.asarray([4.0, 7.5, 4.0])
    small_ven = _segment_um(
        shape, start_um=start, direction_zyx=along_x, length_um=10.0, radius_um=1.5
    )
    large_start = start + along_x * (10.0 + 2.0 * 1.5 + 8.0) + np.asarray([10.5, 0.0, 0.0])
    large_ven = _segment_um(
        shape, start_um=large_start, direction_zyx=along_x, length_um=14.0, radius_um=1.5
    )

    result = _bridge_venules_on_anisotropic_stack(small_ven, large_ven)

    stats = result["stats"]["venule"]
    assert int(stats["accepted_bridges"]) == 0
    assert stats["rejected_reasons"] == {"endpoint_facing_mismatch": 1}
    assert np.array_equal(result["small_venule_mask"], small_ven)


def test_continuity_axis_gate_is_measured_in_microns():
    """A piece bent 60 degrees away breaks the 45 degree axis gate; in voxel
    indices the bend read 28 degrees and was bridged."""
    shape = (24, 30, 90)
    radius = 1.5
    start = np.asarray([4.0, 7.5, 4.0])
    small_ven = _segment_um(
        shape, start_um=start, direction_zyx=_zx_direction(0.0), length_um=10.0, radius_um=radius
    )
    small_tip = start + _zx_direction(0.0) * (10.0 + radius)
    large_tip = small_tip + _zx_direction(30.0) * 8.0
    large_ven = _segment_um(
        shape,
        start_um=large_tip + _zx_direction(60.0) * radius,
        direction_zyx=_zx_direction(60.0),
        length_um=14.0,
        radius_um=radius,
    )

    # Relax the facing gate so that the axis gate is the one under test.
    result = _bridge_venules_on_anisotropic_stack(small_ven, large_ven, min_facing_cosine=0.5)

    stats = result["stats"]["venule"]
    assert int(stats["accepted_bridges"]) == 0
    assert stats["rejected_reasons"] == {"axis_mismatch": 1}
    assert np.array_equal(result["small_venule_mask"], small_ven)


def test_component_axis_and_endpoints_are_measured_in_microns():
    """In voxel indices a 45 degree piece's axis came out 30 degrees off."""
    direction = _zx_direction(45.0)
    piece = _segment_um(
        (32, 30, 70),
        start_um=np.asarray([4.0, 7.5, 4.0]),
        direction_zyx=direction,
        length_um=10.0,
        radius_um=3.0,
    )

    (descriptor,) = _component_descriptors(piece, sampling_zyx=_ANISOTROPIC_ZYX).values()

    axis = descriptor["principal_axis_zyx"]
    assert np.degrees(np.arccos(min(1.0, abs(float(axis @ direction))))) < 2.0
    end_a, end_b = (
        np.asarray(end, dtype=float) * _ANISOTROPIC_ZYX for end in descriptor["endpoints_zyx"]
    )
    # The endpoints are the two tips, 10 + 2 * 3 um apart along the piece.
    assert abs(float((end_b - end_a) @ direction)) == pytest.approx(16.0, abs=1.0)


def test_continuity_schema_flags_require_small_masks():
    schema = default_schema()
    assert schema["small_vessel_mask_continuity_enable"].default is False
    assert schema["small_vessel_tangential_redefinition_enable"].default is False
    assert schema["use_gpu_mask_continuity_acceleration"].default is False
    assert schema["small_vessel_mask_continuity_enable"].requires == (
        "use_small_vessel_masks_for_boundary_assignment",
        "automated_vessel_assignment",
    )
    assert schema["small_vessel_mask_continuity_allow_small_to_large"].requires == (
        "use_small_vessel_masks_for_boundary_assignment",
        "automated_vessel_assignment",
        "small_vessel_mask_continuity_enable",
    )


def test_a_bridge_is_the_same_physical_thickness_on_every_axis():
    """Regression (audit): the bridge was the line dilated by the 26-neighbour
    cube, iterated per voxel, so on 2 x 0.5 x 0.5 um voxels it was as many
    slices thick (8 um) as voxels wide in-plane (2 um)."""
    from haemolynx.graph.mask_continuity import _bridge_mask_from_line

    line = np.array([[6, 10, x] for x in range(5, 25)])
    # A half-width of 4 finest-axis voxels: 2 um.
    physical = _bridge_mask_from_line(line, (13, 21, 30), radius_voxels=4, sampling_zyx=(2.0, 0.5, 0.5))
    cube = _bridge_mask_from_line(line, (13, 21, 30), radius_voxels=4)

    assert set(np.argwhere(physical)[:, 0]) == {5, 6, 7}  # 1 slice each way: 2 um
    assert set(np.argwhere(physical)[:, 1]) == set(range(6, 15))  # 4 voxels: 2 um
    assert set(np.argwhere(cube)[:, 0]) == set(range(2, 11))  # cube voxels, as before
    assert np.array_equal(
        cube, _bridge_mask_from_line(line, (13, 21, 30), radius_voxels=4, sampling_zyx=(1.0, 1.0, 1.0))
    )
