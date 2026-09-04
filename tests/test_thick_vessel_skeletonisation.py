"""Characterise when Lee thinning of a plasma-labelled object becomes a sheet.

The main input is a solid plasma-column mask, typically **one connected
object**: a fat region fused to capillaries. Isolated tubes are the wrong
fixture. These tests lock the radius gate from that fused object, show that
whole-object volume is the wrong discriminator, and check that thickness-gated
skeletonisation replaces the fat sheet without changing the attached
capillaries.

The pipeline exposes this as ``use_thick_vessel_skeletonisation`` on the
Skeletonise tab (off by default).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import label

from haemolynx.preprocessing import (
    BRAID_FACTOR_LIMIT,
    THICK_VESSEL_MIN_RADIUS_UM,
    braid_factor,
    foreground_volume_um3,
    lee_braid_factor,
    max_inscribed_radius_um,
    needs_thick_vessel_treatment,
    skeletonize_edt_ridge,
    skeletonize_thickness_gated,
    skeletonize_volume,
    thick_vessel_object_mask,
)
from haemolynx.preprocessing.thick_vessels import characterisation_rows

SPACING_ZYX = (1.0, 1.0, 1.0)
CAPILLARY_RADIUS_UM = 2.5
FAT_MAJOR_SCALE = 5.0
RADIUS_SWEEP = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)


def _disk_tube_along_axis(shape, origin, radius, length, axis: int) -> np.ndarray:
    """Solid circular tube starting at *origin* and running *length* voxels along *axis*."""
    coords = np.indices(shape)
    other = [0, 1, 2]
    other.remove(axis)
    a, b = other
    radial = (coords[a] - origin[a]) ** 2 + (coords[b] - origin[b]) ** 2 <= radius**2
    along = coords[axis] - origin[axis]
    return radial & (along >= 0) & (along < int(length))


def plasma_labelled_object(
    fat_minor_radius: float,
    *,
    fat_major_scale: float = FAT_MAJOR_SCALE,
    trunk_length: int = 40,
    capillary_radius: float = CAPILLARY_RADIUS_UM,
    capillary_length: int = 22,
    pad: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """One connected solid mask: elongated plasma trunk + attached capillaries.

    The fat part is a filled plasma-column segment (not a hollow wall, not a
    separate component). Flattened elliptical cross-section along x, capillaries
    fused on so the foreground is a single object.
    """
    minor = float(fat_minor_radius)
    major = minor * float(fat_major_scale)
    length = int(trunk_length)
    z = int(np.ceil(2.0 * minor + 2.0 * pad))
    y = int(np.ceil(2.0 * major + 2.0 * pad + capillary_length))
    x = int(length + 2 * pad + 2 * capillary_length)
    shape = (z, y, x)
    cz, cy, cx = z // 2, y // 2, x // 2
    zz, yy, xx = np.indices(shape)
    # Elliptical cross-section (z, y), elongated along x — a plasma column.
    cross = ((zz - cz) / max(minor, 0.5)) ** 2 + (
        (yy - cy) / max(major, 0.5)
    ) ** 2 <= 1.0
    along = np.abs(xx - cx) <= (length / 2.0)
    fat_roi = cross & along
    mask = fat_roi.copy()
    # Capillaries fused at the +x end and +y side so there is one CC.
    mask |= _disk_tube_along_axis(
        shape,
        (cz, cy, cx + length // 2 - 1),
        capillary_radius,
        capillary_length,
        axis=2,
    )
    mask |= _disk_tube_along_axis(
        shape,
        (cz, cy + int(major) - 1, cx),
        capillary_radius,
        capillary_length,
        axis=1,
    )
    labeled, n_cc = label(mask)
    assert n_cc == 1, f"plasma-labelled fixture split into {n_cc} objects"
    return mask.astype(bool), fat_roi.astype(bool)


def plasma_labelled_y(
    radius: float = 8.0,
    *,
    arm_length: int = 28,
    pad: int = 8,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One solid Y: three fat arms of equal radius fused at a junction.

    Returns ``(mask, (end_trunk, end_a, end_b))`` as voxel coordinates of the
    three arm tips, which the centreline tree must all reach.
    """
    r = float(radius)
    length = int(arm_length)
    z = int(np.ceil(2.0 * r + 2.0 * pad))
    y = int(np.ceil(2.0 * r + 2.0 * pad + length))
    x = int(np.ceil(2.0 * length + 2.0 * pad + 2.0 * r))
    shape = (z, y, x)
    cz = z // 2
    cy = pad + int(r) + 2
    cx = pad + int(r) + 2
    mask = np.zeros(shape, dtype=bool)
    # Trunk along +x, then a continuing +x arm and a +y arm.
    mask |= _disk_tube_along_axis(shape, (cz, cy, cx), r, length, axis=2)
    junction = (cz, cy, cx + length - 1)
    mask |= _disk_tube_along_axis(shape, junction, r, length, axis=2)
    mask |= _disk_tube_along_axis(shape, junction, r, length, axis=1)
    labeled, n_cc = label(mask)
    assert n_cc == 1, f"Y fixture split into {n_cc} objects"
    end_trunk = np.array([cz, cy, cx], dtype=int)
    end_a = np.array([cz, cy, cx + 2 * length - 2], dtype=int)
    end_b = np.array([cz, cy + length - 1, cx + length - 1], dtype=int)
    return mask.astype(bool), (end_trunk, end_a, end_b)


def plasma_labelled_cross(
    radius: float = 8.0,
    *,
    arm_length: int = 24,
    pad: int = 8,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Solid plus: four fat arms of equal radius from one junction."""
    r = float(radius)
    length = int(arm_length)
    z = int(np.ceil(2.0 * r + 2.0 * pad))
    y = int(np.ceil(2.0 * length + 2.0 * pad + 2.0 * r))
    x = int(np.ceil(2.0 * length + 2.0 * pad + 2.0 * r))
    shape = (z, y, x)
    cz = z // 2
    cy = y // 2
    cx = x // 2
    mask = np.zeros(shape, dtype=bool)
    # Arms in +x, -x, +y, -y, each starting at the junction.
    mask |= _disk_tube_along_axis(shape, (cz, cy, cx), r, length, axis=2)
    west = (cz, cy, cx - length + 1)
    mask |= _disk_tube_along_axis(shape, west, r, length, axis=2)
    mask |= _disk_tube_along_axis(shape, (cz, cy, cx), r, length, axis=1)
    south = (cz, cy - length + 1, cx)
    mask |= _disk_tube_along_axis(shape, south, r, length, axis=1)
    labeled, n_cc = label(mask)
    assert n_cc == 1, f"cross fixture split into {n_cc} objects"
    ends = (
        np.array([cz, cy, cx + length - 1], dtype=int),
        np.array([cz, cy, cx - length + 1], dtype=int),
        np.array([cz, cy + length - 1, cx], dtype=int),
        np.array([cz, cy - length + 1, cx], dtype=int),
    )
    return mask.astype(bool), ends


def _ridge_reaches(ridge: np.ndarray, point: np.ndarray, radius: float) -> bool:
    if not ridge.any():
        return False
    delta = np.argwhere(ridge) - point.reshape(1, 3)
    dist = np.sqrt(np.einsum("ij,ij->i", delta.astype(float), delta.astype(float)))
    return float(dist.min()) <= float(radius) + 2.0


def capillary_only_object() -> np.ndarray:
    """A fused Y of capillary-scale vessels, no fat trunk."""
    mask, _fat = plasma_labelled_object(
        CAPILLARY_RADIUS_UM, fat_major_scale=1.0, capillary_length=18
    )
    return mask


def _onset_radius(rows: list[dict]) -> float | None:
    sheets = [row for row in rows if row["lee_sheets"]]
    if not sheets:
        return None
    return min(float(row["max_radius_um"]) for row in sheets)


def test_plasma_labelled_fixture_is_one_solid_object():
    mask, fat_roi = plasma_labelled_object(6.0)
    assert int(mask.sum()) > int(fat_roi.sum())
    labeled, n_cc = label(mask)
    assert n_cc == 1
    # Plasma labelling fills the lumen: the fat ROI is not a hollow shell.
    from scipy.ndimage import binary_erosion

    eroded = binary_erosion(fat_roi)
    assert int(eroded.sum()) > 0


def test_capillary_only_object_stays_below_the_gate():
    mask = capillary_only_object()
    assert max_inscribed_radius_um(mask, SPACING_ZYX) < THICK_VESSEL_MIN_RADIUS_UM
    assert needs_thick_vessel_treatment(mask, voxel_size_zyx=SPACING_ZYX) is False


def test_volume_of_the_whole_object_is_not_the_gate():
    """A long capillary mesh can out-volume a compact fat blob fused to stubs."""
    skinny, _ = plasma_labelled_object(
        CAPILLARY_RADIUS_UM, fat_major_scale=1.0, trunk_length=20, capillary_length=200
    )
    chubby, fat_roi = plasma_labelled_object(
        6.0, fat_major_scale=1.2, trunk_length=16, capillary_length=8
    )
    assert foreground_volume_um3(skinny, SPACING_ZYX) > foreground_volume_um3(
        chubby, SPACING_ZYX
    )
    assert max_inscribed_radius_um(skinny, SPACING_ZYX) < 4.0
    assert max_inscribed_radius_um(chubby, SPACING_ZYX) >= 5.5
    assert foreground_volume_um3(fat_roi, SPACING_ZYX) > 0.0


def test_characterisation_sweep_records_fat_volume_and_braid():
    cases = []
    for radius in RADIUS_SWEEP:
        mask, fat_roi = plasma_labelled_object(radius)
        cases.append((f"plasma_r={radius:g}", mask, fat_roi, SPACING_ZYX))
    rows = characterisation_rows(cases)
    assert len(rows) == len(RADIUS_SWEEP)
    for row in rows:
        assert row["volume_um3"] > row["fat_volume_um3"] > 0.0
        assert row["max_radius_um"] > 0.0
        assert "lee_braid_factor" in row


def test_the_locked_radius_threshold_matches_the_measured_onset():
    """Lock THICK_VESSEL_MIN_RADIUS_UM to the first fat blob Lee sheets.

    The fixture is one plasma-labelled object. Sheet excess is Lee skeleton
    voxels in the fat ellipsoid over a single EDT-ridge polyline, so fused
    capillaries do not set the gate. A round blob stays a short tree; a
    flattened plasma region becomes a mid-plane sheet.
    """
    rows = characterisation_rows(
        [
            (f"plasma_r={radius:g}", *plasma_labelled_object(radius), SPACING_ZYX)
            for radius in RADIUS_SWEEP
        ]
    )
    onset = _onset_radius(rows)
    assert onset is not None, (
        "Lee never sheeted the fat plasma column; rows: "
        + ", ".join(
            f"{r['name']}: braid={r['lee_braid_factor']:.2f} r={r['max_radius_um']:.2f}"
            for r in rows
        )
    )
    measured_gate = float(np.floor(onset * 2.0) / 2.0)
    assert THICK_VESSEL_MIN_RADIUS_UM == pytest.approx(measured_gate, abs=0.51)
    assert THICK_VESSEL_MIN_RADIUS_UM > CAPILLARY_RADIUS_UM
    onset_row = min(
        (row for row in rows if row["lee_sheets"]),
        key=lambda row: float(row["max_radius_um"]),
    )
    assert onset_row["fat_volume_um3"] > 0.0


def test_catchment_keeps_fused_capillaries_out_of_the_fat_region():
    """A T-ball around the inscribed core, not a flood fill through the object."""
    mask, fat_roi = plasma_labelled_object(8.0)
    thick = thick_vessel_object_mask(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    assert thick.any()
    capillaries = mask & ~fat_roi
    swallowed = int((capillaries & thick).sum())
    assert swallowed / max(1, int(capillaries.sum())) < 0.35
    covered_fat = int((fat_roi & thick).sum()) / int(fat_roi.sum())
    assert covered_fat > 0.7


def test_gated_skeletonisation_matches_lee_when_nothing_is_fat():
    mask = capillary_only_object()
    lee = skeletonize_volume(mask)
    gated = skeletonize_thickness_gated(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    assert np.array_equal(gated, lee)


def test_gated_skeletonisation_collapses_the_fat_sheet_and_keeps_capillaries():
    mask, fat_roi = plasma_labelled_object(8.0)
    lee_braid = lee_braid_factor(fat_roi, axis=2)
    assert lee_braid > BRAID_FACTOR_LIMIT
    gated = skeletonize_thickness_gated(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    thick = thick_vessel_object_mask(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    gated_braid = braid_factor(gated & thick, axis=2)
    assert gated_braid < lee_braid
    capillaries = mask & ~fat_roi
    assert int((gated & capillaries).sum()) > 0


def test_min_radius_zero_is_plain_lee():
    mask, _ = plasma_labelled_object(6.0)
    assert np.array_equal(
        skeletonize_thickness_gated(mask, min_radius_um=0.0),
        skeletonize_volume(mask),
    )


def test_edt_ridge_on_the_fat_catchment_is_not_a_sheet():
    mask, fat_roi = plasma_labelled_object(8.0)
    thick = thick_vessel_object_mask(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    ridge = skeletonize_edt_ridge(thick)
    assert ridge.any()
    # A polyline tree, not a filled mid-plane.
    assert int(ridge.sum()) < 0.05 * int(thick.sum())


def test_edt_ridge_reaches_every_arm_of_a_fat_y():
    mask, (end_trunk, end_a, end_b) = plasma_labelled_y(8.0)
    assert needs_thick_vessel_treatment(mask, voxel_size_zyx=SPACING_ZYX) is True
    ridge = skeletonize_edt_ridge(mask)
    assert _ridge_reaches(ridge, end_trunk, 8.0)
    assert _ridge_reaches(ridge, end_a, 8.0)
    assert _ridge_reaches(ridge, end_b, 8.0)
    # A tree, not a filled sheet: still a small fraction of the fat volume.
    assert int(ridge.sum()) < 0.08 * int(mask.sum())


def test_edt_ridge_reaches_every_arm_of_a_fat_cross():
    mask, ends = plasma_labelled_cross(8.0)
    ridge = skeletonize_edt_ridge(mask)
    for tip in ends:
        assert _ridge_reaches(ridge, tip, 8.0)
    assert int(ridge.sum()) < 0.08 * int(mask.sum())


def test_gated_y_keeps_all_fat_arms_and_does_not_sheet():
    mask, (end_trunk, end_a, end_b) = plasma_labelled_y(8.0)
    gated = skeletonize_thickness_gated(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    assert _ridge_reaches(gated, end_trunk, 8.0)
    assert _ridge_reaches(gated, end_a, 8.0)
    assert _ridge_reaches(gated, end_b, 8.0)
    # A tree, not a filled mid-plane. Do not use braid along the trunk
    # axis: a real side arm occupies one x-slice and looks like a sheet
    # on that metric.
    assert int(gated.sum()) < 0.08 * int(mask.sum())
