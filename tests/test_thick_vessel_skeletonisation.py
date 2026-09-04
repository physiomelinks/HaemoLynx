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
from scipy.ndimage import generate_binary_structure, label

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
from haemolynx.preprocessing.thick_vessels import (
    _cover_around_path,
    _dijkstra_parents,
    _path_through_mask,
    _skeletonize_foreground,
    _traceback,
    characterisation_rows,
)

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


def test_gated_fat_ridge_connects_to_each_fused_thin_vessel():
    """Lee on a fused capillary must meet the fat centreline, not sit as a gap.

    The fixture is one plasma-labelled object with two capillaries. Surface
    flakes of Lee on the fat wall are not thin vessels and are not required
    to join the ridge.
    """
    from scipy.ndimage import generate_binary_structure

    mask, fat_roi = plasma_labelled_object(8.0)
    gated = skeletonize_thickness_gated(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    thick = thick_vessel_object_mask(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    fat_skel = gated & thick
    capillaries = mask & ~fat_roi
    assert fat_skel.any()
    assert int((gated & capillaries).sum()) > 0
    struct26 = generate_binary_structure(3, 3)
    gated_lab, _ = label(gated, structure=struct26)
    fat_labels = set(int(v) for v in gated_lab[fat_skel]) - {0}
    assert fat_labels
    cap_lab, n_cap = label(capillaries, structure=struct26)
    joined = 0
    for component_id in range(1, int(n_cap) + 1):
        cap_skel = gated & (cap_lab == component_id)
        if not cap_skel.any():
            continue
        cap_labels = set(int(v) for v in gated_lab[cap_skel]) - {0}
        assert cap_labels & fat_labels, (
            f"capillary {component_id} skeleton is not 26-connected to the fat ridge"
        )
        joined += 1
    assert joined == int(n_cap)
    # Joining arms must not re-sheet the fat catchment.
    assert int(gated.sum()) < 0.08 * int(mask.sum())


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


def test_edt_ridge_on_a_small_tube_in_a_large_volume_stays_local():
    """Geodesics are cropped to the fat component, not allocated for the stack."""
    volume = np.zeros((80, 80, 80), dtype=bool)
    volume[10:18, 10:18, 8:48] = True
    ridge = skeletonize_edt_ridge(volume)
    assert ridge[10:18, 10:18, 8:48].any()
    assert not ridge[40:, :, :].any()
    assert int(ridge.sum()) < int(volume[10:18, 10:18, 8:48].sum())


def test_dijkstra_traces_a_straight_tube_end_to_end():
    tube = np.zeros((5, 5, 20), dtype=bool)
    tube[2, 2, 2:18] = True
    cost = np.where(tube, 1.0, np.inf)
    root = (2, 2, 2)
    far = (2, 2, 17)
    walked = _dijkstra_parents(tube, cost, root)
    assert walked is not None
    parent, fg_coords, index_of = walked
    path = _traceback(parent, fg_coords, index_of, root, far)
    assert path[0] == root
    assert path[-1] == far
    assert all(voxel[0] == 2 and voxel[1] == 2 for voxel in path)


def test_cropped_lee_matches_lee_on_the_full_volume():
    volume = np.zeros((40, 40, 40), dtype=bool)
    volume[18:23, 18:23, 5:35] = True
    assert np.array_equal(_skeletonize_foreground(volume), skeletonize_volume(volume))

    two = np.zeros((40, 40, 40), dtype=bool)
    two[8:13, 8:13, 5:30] = True
    two[25:30, 25:30, 8:35] = True
    assert np.array_equal(_skeletonize_foreground(two), skeletonize_volume(two))


def test_cover_around_a_path_matches_a_full_volume_ball():
    from scipy.ndimage import distance_transform_edt

    shape = (30, 30, 30)
    path = [(12, 15, z) for z in range(5, 25)]
    fast = _cover_around_path(path, 4, shape)
    painted = np.zeros(shape, dtype=bool)
    for voxel in path:
        painted[voxel] = True
    naive = distance_transform_edt(~painted) <= 4
    assert np.array_equal(fast, naive)
    assert not fast[0, 0, 0]


def test_fat_catchment_on_a_padded_volume_matches_the_unpadded_object():
    mask, _ = plasma_labelled_object(8.0)
    thick_small = thick_vessel_object_mask(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    pad = 12
    padded = np.zeros(tuple(int(s) + 2 * pad for s in mask.shape), dtype=bool)
    padded[
        pad : pad + mask.shape[0],
        pad : pad + mask.shape[1],
        pad : pad + mask.shape[2],
    ] = mask
    thick_pad = thick_vessel_object_mask(
        padded, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    assert np.array_equal(
        thick_pad[
            pad : pad + mask.shape[0],
            pad : pad + mask.shape[1],
            pad : pad + mask.shape[2],
        ],
        thick_small,
    )
    assert not thick_pad[: pad // 2].any()


def _cycle_excess_26(skel: np.ndarray) -> int:
    """Extra 26-neighbour edges vs a voxel tree: n_edges - (n_voxels - n_cc).

    A 26-connected tree of voxels has ``#voxels = #undirected_edges + n_cc``.
    Local 26-triangles on a one-voxel centreline add a few extras; a Lee mesh
    wrapped around a fat trunk adds extras on the order of the voxel count.
    """
    from scipy.ndimage import generate_binary_structure
    from scipy.spatial import cKDTree

    coords = np.argwhere(np.asarray(skel, dtype=bool))
    n = int(coords.shape[0])
    if n == 0:
        return 0
    n_edges = len(cKDTree(coords.astype(float)).query_pairs(r=np.sqrt(3.0) + 1e-6))
    _, n_cc = label(skel, structure=generate_binary_structure(3, 3))
    return int(n_edges) - (n - int(n_cc))


def test_gated_fat_catchment_skeleton_is_a_tree_not_a_looped_lee_mesh():
    """Lee of the leftover fat-wall shell must not remain beside the ridge.

    The catchment leaves a thin shell of the fat lumen in ``mask & ~thick``.
    Lee of that wrap is a flake/loop mesh; those CCs do not extend away from
    thick, so they are not capillaries and must be dropped.
    """
    mask, fat_roi = plasma_labelled_object(8.0)
    thick = thick_vessel_object_mask(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    thin = mask & ~thick
    shell = fat_roi & ~thick
    lee_shell = skeletonize_volume(thin) & shell
    assert int(lee_shell.sum()) > 0, "fixture must leave a Lee-able fat-wall shell"

    gated = skeletonize_thickness_gated(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    # Wall flakes/loops gone; join paths may clip the shell as a tree edge.
    assert int((gated & shell).sum()) < 0.35 * int(lee_shell.sum())

    fat_skel = gated & thick
    assert fat_skel.any()
    n_fat = int(fat_skel.sum())
    excess = _cycle_excess_26(fat_skel)
    # A wrapping Lee mesh has excess ~ voxel count; a 26-tree has a few triangles.
    assert excess <= max(12, n_fat // 5), (
        f"fat-catchment skeleton has {excess} extra 26-edges on {n_fat} voxels"
    )
    n_voxels = n_fat
    from scipy.ndimage import generate_binary_structure
    from scipy.spatial import cKDTree

    coords = np.argwhere(fat_skel)
    n_edges = len(cKDTree(coords.astype(float)).query_pairs(r=np.sqrt(3.0) + 1e-6))
    _, n_cc = label(fat_skel, structure=generate_binary_structure(3, 3))
    assert n_voxels <= n_edges + int(n_cc) + max(12, n_voxels // 5)

    capillaries = mask & ~fat_roi
    assert int((gated & capillaries).sum()) > 0
    assert int((gated & shell).sum()) < int(lee_shell.sum())


def test_diagonally_pinched_fat_object_stays_one_tree_not_two():
    """A solid object joined only corner-to-corner must not split into two trees.

    ``skeletonize_edt_ridge`` used to label its cropped component mask with
    the default (6-connected, face-only) structure, while every other
    ``label()`` call in this module is 26-connected. A fat object whose mask
    boundary is only diagonally connected at a pinch -- routine on a
    thresholded, jagged microscopy surface -- was split into two
    "components", each given its own independent centreline tree with no
    check for tree-adjacency across the split. Locks the fix: one 26-connected
    object gets one connected, tree-shaped skeleton.
    """
    L = 14
    shape = (2 * L, 2 * L, 2 * L)
    mask = np.zeros(shape, dtype=bool)
    mask[0:L, 0:L, 0:L] = True
    mask[L : 2 * L, L : 2 * L, L : 2 * L] = True  # touches only at one corner

    _, n_6conn = label(mask)
    _, n_26conn = label(mask, structure=generate_binary_structure(3, 3))
    assert n_6conn == 2, "fixture must be a genuine 6-vs-26-connectivity pinch"
    assert n_26conn == 1, "fixture must be one object under 26-connectivity"

    ridge = skeletonize_edt_ridge(mask)
    assert ridge.any()
    _, n_cc = label(ridge, structure=generate_binary_structure(3, 3))
    assert n_cc == 1, "ridge split into multiple components across the diagonal pinch"
    excess = _cycle_excess_26(ridge)
    assert excess <= 2, f"ridge has {excess} extra 26-edges -- looks like a closed loop"


def test_join_falls_back_to_a_full_mask_geodesic_when_the_tight_box_has_no_path():
    """A join whose only real path leaves the tight box around start/end must not fail silently.

    ``_path_through_mask`` tried a straight line, then two corridor radii,
    then one more Dijkstra -- all cropped to a box tight around (start, end).
    On a horseshoe/ring shape, the two nearest-in-Euclidean-distance points
    across the opening are connected only by going the long way around,
    entirely outside that box. The old code gave up and returned
    ``[start, end]``: the caller then marks only those two (already-True)
    voxels and calls the arm "joined" because ``end`` sits on the ridge,
    leaving the arm's own voxels 26-disconnected from everything else -- an
    isolated fragment invisible to anything downstream that drops small
    disconnected components.
    """
    shape = (3, 21, 21)
    allowed = np.zeros(shape, dtype=bool)
    z = 1
    cy, cx = 10, 10
    outer_r, inner_r = 9, 8
    yy, xx = np.indices((21, 21))
    ring = ((yy - cy) ** 2 + (xx - cx) ** 2 <= outer_r**2) & (
        (yy - cy) ** 2 + (xx - cx) ** 2 >= inner_r**2
    )
    allowed[z] = ring

    def _snap(point):
        coords = np.argwhere(allowed)
        d = np.sum((coords - np.array(point)) ** 2, axis=1)
        return tuple(int(v) for v in coords[np.argmin(d)])

    start = _snap((z, cy, cx + outer_r - 1))
    end = _snap((z, cy, cx - (outer_r - 1)))
    # A tight box between these two points is empty (it spans the ring's
    # open middle); a straight line and both corridor radii must fail.
    assert not allowed[z, cy, cx], "fixture's middle must be empty, not part of the ring"

    path = _path_through_mask(start, end, allowed)
    assert len(path) > 2, "join must not silently give up as a bare [start, end]"
    assert all(allowed[p] for p in path), "every joined voxel must be foreground"
    assert all(
        max(abs(a - b) for a, b in zip(path[i], path[i + 1])) <= 1
        for i in range(len(path) - 1)
    ), "consecutive joined voxels must be 26-adjacent (a real connected path)"
    assert path[0] == start and path[-1] == end
