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
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label, maximum_filter

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
    _build_dijkstra_graph,
    _cover_around_path,
    _dijkstra_parents,
    _geodesic_on_crop,
    _join_thin_arms_to_fat_ridge,
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


def test_return_thick_mask_defaults_to_a_bare_array():
    """Every existing caller must keep getting a bare array back."""
    mask, _fat_roi = plasma_labelled_object(8.0)
    result = skeletonize_thickness_gated(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    assert isinstance(result, np.ndarray)


def test_return_thick_mask_reports_the_fat_catchment_when_requested():
    mask, fat_roi = plasma_labelled_object(8.0)
    skeleton, thick = skeletonize_thickness_gated(
        mask,
        min_radius_um=THICK_VESSEL_MIN_RADIUS_UM,
        voxel_size_zyx=SPACING_ZYX,
        return_thick_mask=True,
    )
    assert isinstance(skeleton, np.ndarray) and skeleton.dtype == bool
    assert isinstance(thick, np.ndarray) and thick.dtype == bool
    assert thick.any()
    covered_fat = int((fat_roi & thick).sum()) / int(fat_roi.sum())
    assert covered_fat > 0.7


def test_return_thick_mask_is_none_when_nothing_is_fat():
    mask = capillary_only_object()
    skeleton, thick = skeletonize_thickness_gated(
        mask,
        min_radius_um=THICK_VESSEL_MIN_RADIUS_UM,
        voxel_size_zyx=SPACING_ZYX,
        return_thick_mask=True,
    )
    assert isinstance(skeleton, np.ndarray)
    assert thick is None


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


def _fat_trunk_with_short_fused_branch(
    *, branch_len: int, r_fat: float = 8.0, branch_radius: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """A fat trunk with one thin branch fused to its side, close to the wall."""
    trunk_len = 90
    pad = 12
    shape = (
        int(2 * r_fat + 2 * pad),
        int(2 * r_fat + 2 * pad + branch_len),
        int(trunk_len + 2 * pad),
    )
    cz, cy, cx = shape[0] // 2, pad + int(r_fat), shape[2] // 2
    mask = _disk_tube_along_axis(
        shape, (cz, cy, cx - trunk_len // 2), r_fat, trunk_len, axis=2
    )
    branch_origin = (cz, cy + int(r_fat) - 2, cx)
    branch = _disk_tube_along_axis(shape, branch_origin, branch_radius, branch_len, axis=1)
    mask = mask | branch
    return mask, branch


def test_wall_absorption_um_override_shrinks_the_catchment_monotonically():
    """Lowering wall_absorption_um must strictly shrink the fat catchment.

    thick_vessel_object_mask's geodesic ``allowed`` gate (step 2, tracing the
    fat trunk's own possibly-irregular shape) always stays at half
    min_radius_um; only the wall dilation (step 3, the part that eats into a
    fused vessel's base) follows the override. Coupling both to the same
    value made a lower override non-monotonic: it could grow the geodesic
    body faster than it shrank the wall, absorbing *more* of a fused vessel,
    the opposite of what a user lowering it would expect.
    """
    mask, _branch = _fat_trunk_with_short_fused_branch(branch_len=6)
    default = thick_vessel_object_mask(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    lowered = thick_vessel_object_mask(
        mask,
        min_radius_um=THICK_VESSEL_MIN_RADIUS_UM,
        voxel_size_zyx=SPACING_ZYX,
        wall_absorption_um=1.0,
    )
    zeroed = thick_vessel_object_mask(
        mask,
        min_radius_um=THICK_VESSEL_MIN_RADIUS_UM,
        voxel_size_zyx=SPACING_ZYX,
        wall_absorption_um=0.0,
    )
    assert int(zeroed.sum()) < int(lowered.sum()) < int(default.sum())


def test_lowering_wall_absorption_and_flake_filter_recovers_a_short_fused_vessel():
    """A short vessel dropped at the default thresholds survives once both are lowered.

    At the default ~7.5 um combined reach (half min_radius_um wall absorption
    plus the flake filter), a 6-voxel branch fused to an 8 um-radius trunk is
    deleted entirely -- not a bug, the design's intended trade-off, but one a
    user needs to be able to relax for real data with shorter fused vessels.
    """
    mask, branch = _fat_trunk_with_short_fused_branch(branch_len=6)
    default = skeletonize_thickness_gated(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )
    assert int((default & branch).sum()) == 0, "fixture must reproduce the default loss"

    lowered = skeletonize_thickness_gated(
        mask,
        min_radius_um=THICK_VESSEL_MIN_RADIUS_UM,
        voxel_size_zyx=SPACING_ZYX,
        wall_absorption_um=1.0,
        flake_filter_um=1.0,
    )
    assert int((lowered & branch).sum()) > 0, "lowered thresholds must recover the branch"


def test_many_fused_fat_branches_all_get_a_skeleton_arm():
    """A trunk with more than a dozen fused fat branches must not lose most of them.

    The per-component arm search used to stop after 12 iterations regardless
    of whether real candidates remained -- a fixed safety bound, not a
    correctness threshold. A real fused network can have far more than a
    dozen genuine arms; on production data this was observed dropping over a
    hundred thousand high-EDT voxels' worth of arms in one component. Locks
    the fix: every one of 40 fused branches gets at least one skeleton voxel.
    """
    r_fat = 7.0
    trunk_len = 300
    n_branches = 40
    branch_radius = 6.5  # itself crosses THICK_VESSEL_MIN_RADIUS_UM
    branch_len = 20
    pad = 12
    shape = (
        int(2 * max(r_fat, branch_radius) + 2 * pad),
        int(2 * max(r_fat, branch_radius) + 2 * pad + branch_len),
        int(trunk_len + 2 * pad),
    )
    cz, cy, cx = shape[0] // 2, pad + int(max(r_fat, branch_radius)), shape[2] // 2
    mask = _disk_tube_along_axis(
        shape, (cz, cy, cx - trunk_len // 2), r_fat, trunk_len, axis=2
    )
    branch_masks = []
    for i in range(n_branches):
        t = cx - trunk_len // 2 + int((i + 0.5) * trunk_len / n_branches)
        origin = (cz, cy + int(r_fat) - 2, t)
        bm = _disk_tube_along_axis(shape, origin, branch_radius, branch_len, axis=1)
        mask = mask | bm
        branch_masks.append(bm)

    ridge = skeletonize_edt_ridge(mask)
    assert ridge.any()
    hits = sum(1 for bm in branch_masks if (ridge & bm).any())
    assert hits == n_branches, f"only {hits}/{n_branches} fused branches got a skeleton arm"


def test_disconnected_network_is_not_wrongly_joined_to_a_nearby_fat_trunk():
    """A self-contained thin-only network must not be bridged to an unrelated trunk.

    The nearest-fat-voxel search used to run over the whole image, so a thin
    vessel's Euclidean-nearest fat voxel could belong to a completely
    different, physically disconnected vessel network -- no path could ever
    exist between them (confirmed on production data: the full-mask Dijkstra
    fallback still failed and logged a warning). Scoping the search to fat
    voxels sharing the arm's own physically connected structure means: no
    such join is even attempted, no warning fires, and the self-contained
    network is left exactly as Lee thinned it.
    """
    shape = (40, 200, 200)
    r_fat = 8.0
    trunk_len = 60
    mask = _disk_tube_along_axis(shape, (10, 20, 20), r_fat, trunk_len, axis=2)
    branch = _disk_tube_along_axis(
        shape, (10, 20 + int(r_fat) - 2, 40), 1.5, 14, axis=1
    )
    mask = mask | branch
    # A second, physically separate structure: not touching the trunk at all.
    isolated = _disk_tube_along_axis(shape, (10, 20, 120), 2.0, 60, axis=2)
    mask = mask | isolated

    _, n_cc_mask = label(mask, structure=generate_binary_structure(3, 3))
    assert n_cc_mask == 2, "fixture must have two physically separate structures"

    gated = skeletonize_thickness_gated(
        mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
    )

    assert int((gated & branch).sum()) > 0, "the genuinely fused branch must still be joined"
    assert int((gated & isolated).sum()) > 0, "the isolated network must be preserved"
    _, n_cc_gated = label(gated, structure=generate_binary_structure(3, 3))
    assert n_cc_gated == 2, (
        "gated skeleton must keep the same two components as the input mask -- "
        "no bogus bridge drawn between physically unconnected structures"
    )


def test_join_skips_silently_when_the_arms_own_component_has_no_fat(caplog):
    """No fat anywhere in an arm's own component must skip the join, not warn.

    A self-contained small network with no fat trunk of its own is not an
    error case -- it has nothing to join to because it does not need
    joining. Only a genuine, still-unexplained failure to connect within a
    shared component should be loud.
    """
    shape = (40, 200, 200)
    r_fat = 8.0
    trunk_len = 60
    mask = _disk_tube_along_axis(shape, (10, 20, 20), r_fat, trunk_len, axis=2)
    isolated = _disk_tube_along_axis(shape, (10, 20, 120), 2.0, 60, axis=2)
    mask = mask | isolated

    with caplog.at_level("WARNING", logger="haemolynx.preprocessing.thick_vessels"):
        skeletonize_thickness_gated(
            mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, voxel_size_zyx=SPACING_ZYX
        )
    assert not any("Could not join" in record.message for record in caplog.records)


def _distant_arm_fixture(shape=(1, 1, 30), *, distance: int = 20):
    """A single fat voxel, a single thin-arm voxel *distance* apart, and a
    straight corridor between them so a join is possible if not capped."""
    thick = np.zeros(shape, dtype=bool)
    thick[0, 0, 0] = True
    thin_skel = np.zeros(shape, dtype=bool)
    thin_skel[0, 0, distance] = True
    allowed = np.zeros(shape, dtype=bool)
    allowed[0, 0, 0 : distance + 1] = True
    skeleton = thick | thin_skel
    return skeleton, thick, allowed


def test_a_bridge_past_the_distance_cap_is_left_disconnected():
    """A thin arm should merge into a fat vessel near where it touches it,
    not bridge to an arbitrarily distant point on the fat ridge."""
    skeleton, thick, allowed = _distant_arm_fixture(distance=20)

    joined = _join_thin_arms_to_fat_ridge(
        skeleton,
        thick,
        allowed,
        min_arm_extent_voxels=0.0,
        max_bridge_distance_um=10.0,
    )

    _, n_cc = label(joined, structure=generate_binary_structure(3, 3))
    assert n_cc == 2, "arm must stay a separate component, not bridged"
    assert int(joined.sum()) == 2, "no corridor voxels should have been drawn"


def test_a_bridge_within_the_distance_cap_still_joins():
    skeleton, thick, allowed = _distant_arm_fixture(distance=20)

    joined = _join_thin_arms_to_fat_ridge(
        skeleton,
        thick,
        allowed,
        min_arm_extent_voxels=0.0,
        max_bridge_distance_um=30.0,
    )

    _, n_cc = label(joined, structure=generate_binary_structure(3, 3))
    assert n_cc == 1, "arm must be bridged when within the cap"


def test_no_cap_leaves_the_join_search_unbounded_as_before():
    skeleton, thick, allowed = _distant_arm_fixture(distance=20)

    joined = _join_thin_arms_to_fat_ridge(
        skeleton, thick, allowed, min_arm_extent_voxels=0.0
    )

    _, n_cc = label(joined, structure=generate_binary_structure(3, 3))
    assert n_cc == 1


def _wide_trunk_arm_fixture(*, radius: int, gap: int, margin: int | None = None):
    """A fat trunk whose own local half-width (Y axis) is `radius` voxels,
    with a single already-drawn ridge point at its centre, and a thin arm
    `gap` voxels beyond the trunk's surface on the same column -- so the
    true surface-to-ridge distance is `radius + 1` voxels (EDT counts the
    first background voxel beyond the trunk), and the total straight-line
    arm-to-ridge distance is `radius + 1 + gap`."""
    if margin is None:
        margin = gap + 5
    width = margin + (2 * radius + 1) + margin
    length = 5
    shape = (1, width, length)
    thick = np.zeros(shape, dtype=bool)
    y0 = margin
    y1 = y0 + 2 * radius + 1
    thick[0, y0:y1, :] = True
    ridge_y = y0 + radius
    x = length // 2
    skeleton = np.zeros(shape, dtype=bool)
    skeleton[0, ridge_y, x] = True
    arm_y = y0 - 1 - gap
    skeleton[0, arm_y, x] = True
    allowed = thick.copy()
    allowed[0, arm_y:y1, x] = True
    return skeleton, thick, allowed


def test_bridge_radius_multiple_scales_with_the_local_fat_radius_not_a_fixed_constant():
    """A wide trunk's own local radius, not any fixed constant, sets the
    bridge cap -- an arm attaching near a major trunk has to cross roughly
    the trunk's own half-width to reach its centreline, and a cap sized for
    a thin vessel's radius would reject every such legitimate attachment
    (the reported bug this regresses: real trunks run tens of microns wide,
    far past any small fixed classification threshold)."""
    radius, gap, multiplier = 20, 15, 2.0
    expected_local_radius = radius + 1
    expected_distance = expected_local_radius + gap
    assert expected_distance < multiplier * expected_local_radius  # sanity: should join
    skeleton, thick, allowed = _wide_trunk_arm_fixture(radius=radius, gap=gap)

    joined = _join_thin_arms_to_fat_ridge(
        skeleton,
        thick,
        allowed,
        min_arm_extent_voxels=0.0,
        max_bridge_radius_multiple=multiplier,
    )

    _, n_cc = label(joined, structure=generate_binary_structure(3, 3))
    assert n_cc == 1, (
        "a cap based on the trunk's own local radius must accept an arm "
        "that a fixed small constant would have wrongly rejected"
    )


def test_bridge_radius_multiple_still_rejects_a_genuinely_far_arm():
    radius, gap, multiplier = 20, 60, 2.0
    expected_local_radius = radius + 1
    expected_distance = expected_local_radius + gap
    assert expected_distance > multiplier * expected_local_radius  # sanity: should reject
    skeleton, thick, allowed = _wide_trunk_arm_fixture(radius=radius, gap=gap)

    joined = _join_thin_arms_to_fat_ridge(
        skeleton,
        thick,
        allowed,
        min_arm_extent_voxels=0.0,
        max_bridge_radius_multiple=multiplier,
    )

    _, n_cc = label(joined, structure=generate_binary_structure(3, 3))
    assert n_cc == 2, "still capped -- just scaled to the trunk's own radius, not unbounded"


def _anisotropic_bridge_fixture(*, radius: int, gap_x: int, block_len: int, margin: int = 5):
    """A trunk whose half-width (Y axis, 1 micron/voxel in the caller's
    spacing) is `radius` voxels, with a thin arm on the same row, offset
    along X -- a much coarser-spaced axis in the caller's spacing -- by
    `gap_x` voxels past the trunk's own end."""
    width = margin + (2 * radius + 1) + margin
    length = block_len + gap_x + 5
    shape = (1, width, length)
    thick = np.zeros(shape, dtype=bool)
    y0 = margin
    y1 = y0 + 2 * radius + 1
    thick[0, y0:y1, 0:block_len] = True
    ridge_y = y0 + radius
    ridge_x = block_len // 2
    skeleton = np.zeros(shape, dtype=bool)
    skeleton[0, ridge_y, ridge_x] = True
    arm_x = block_len + gap_x
    skeleton[0, ridge_y, arm_x] = True
    allowed = thick.copy()
    allowed[0, ridge_y, ridge_x : arm_x + 1] = True
    return skeleton, thick, allowed


def test_bridge_cap_is_evaluated_in_physical_microns_not_raw_voxel_distance():
    """The trunk's half-width sits on the Y axis (1 micron/voxel); the arm
    is offset from the trunk purely along X, a much coarser axis (4
    microns/voxel). In raw voxel counts the arm looks close enough to pass
    the cap; converted to real microns via voxel_size_zyx it is not -- if
    the cap were computed on unscaled voxel distance, this arm would
    wrongly bridge across 28 real microns of tissue."""
    radius, gap_x, block_len = 5, 3, 8
    voxel_size_zyx = (1.0, 1.0, 4.0)
    multiplier = 2.0
    expected_local_radius_um = (radius + 1) * voxel_size_zyx[1]  # 6.0
    ridge_x = block_len // 2
    arm_x = block_len + gap_x
    expected_distance_um = (arm_x - ridge_x) * voxel_size_zyx[2]  # 28.0
    assert expected_distance_um > multiplier * expected_local_radius_um  # 28 > 12: reject
    raw_voxel_distance = arm_x - ridge_x  # 7
    assert raw_voxel_distance < multiplier * (radius + 1)  # 7 < 12: unscaled would wrongly accept
    skeleton, thick, allowed = _anisotropic_bridge_fixture(
        radius=radius, gap_x=gap_x, block_len=block_len
    )

    joined = _join_thin_arms_to_fat_ridge(
        skeleton,
        thick,
        allowed,
        voxel_size_zyx=voxel_size_zyx,
        min_arm_extent_voxels=0.0,
        max_bridge_radius_multiple=multiplier,
    )

    _, n_cc = label(joined, structure=generate_binary_structure(3, 3))
    assert n_cc == 2, (
        "distance must be measured in physical microns -- in raw voxel "
        "units this arm looks close enough to wrongly bridge"
    )


def _narrow_neck_then_wide_trunk_fixture(
    *, r_narrow: int, r_wide: int, neck_len: int, wide_len: int, gap: int, margin: int = 5
):
    """A trunk that starts narrow (half-width `r_narrow`, length `neck_len`)
    then widens (half-width `r_wide`), sharing one straight ridge row so the
    whole thing is a single connected component. The only marked ridge
    point sits mid-neck; the arm hangs `gap` voxels above it. The *raw*
    local radius there is small (r_narrow + 1) even though the trunk is
    genuinely wide again a short distance further along the very same
    connected mask."""
    width = margin + (2 * r_wide + 1) + margin
    length = neck_len + wide_len
    shape = (1, width, length)
    thick = np.zeros(shape, dtype=bool)
    y0 = margin
    ridge_y = y0 + r_wide
    thick[0, ridge_y - r_narrow : ridge_y + r_narrow + 1, 0:neck_len] = True
    thick[0, y0 : y0 + 2 * r_wide + 1, neck_len : neck_len + wide_len] = True
    point_a = (0, ridge_y, neck_len // 2)
    skeleton = np.zeros(shape, dtype=bool)
    skeleton[point_a] = True
    arm_y = ridge_y - r_narrow - 1 - gap
    skeleton[0, arm_y, neck_len // 2] = True
    allowed = thick.copy()
    allowed[0, arm_y:ridge_y, neck_len // 2] = True
    return skeleton, thick, allowed, point_a


def test_local_fat_radius_is_smoothed_over_a_transient_narrow_waist():
    """The only ridge point in reach of the arm sits in a narrow neck: read
    raw, its radius rejects a bridge that is otherwise clearly plausible,
    because the very same connected trunk is genuinely wide again a few
    microns further along. Smoothing the radius reading over a small
    physical neighbourhood (not searching other ridge points -- there are
    none here) must recover that and accept the join."""
    r_narrow, r_wide, gap, multiplier, radius_smoothing_um = 2, 15, 8, 2.0, 10.0
    raw_radius_a = r_narrow + 1
    distance_a = raw_radius_a + gap
    assert distance_a > multiplier * raw_radius_a  # unsmoothed: would reject
    skeleton, thick, allowed, point_a = _narrow_neck_then_wide_trunk_fixture(
        r_narrow=r_narrow, r_wide=r_wide, neck_len=10, wide_len=60, gap=gap
    )
    # Ground truth for the smoothed radius at the ridge point, computed
    # directly with scipy's own primitives rather than duplicating the
    # implementation's crop/padding logic.
    edt = distance_transform_edt(thick[0])
    footprint = 2 * int(np.ceil(radius_smoothing_um)) + 1
    smoothed = maximum_filter(edt, size=footprint)
    expected_smoothed_radius = float(smoothed[point_a[1], point_a[2]])
    assert distance_a <= multiplier * expected_smoothed_radius  # smoothed: should accept

    joined = _join_thin_arms_to_fat_ridge(
        skeleton,
        thick,
        allowed,
        min_arm_extent_voxels=0.0,
        max_bridge_radius_multiple=multiplier,
        radius_smoothing_um=radius_smoothing_um,
    )

    _, n_cc = label(joined, structure=generate_binary_structure(3, 3))
    assert n_cc == 1, (
        "smoothing the local radius over a small neighbourhood must "
        "recover a join that a single raw sample at a transient narrow "
        "waist would wrongly reject"
    )


def test_radius_smoothing_disabled_falls_back_to_the_raw_single_point_sample():
    """radius_smoothing_um=0 must reproduce the pre-smoothing behaviour
    exactly: the same narrow-neck arm, still rejected."""
    r_narrow, r_wide, gap, multiplier = 2, 15, 8, 2.0
    skeleton, thick, allowed, _point_a = _narrow_neck_then_wide_trunk_fixture(
        r_narrow=r_narrow, r_wide=r_wide, neck_len=10, wide_len=60, gap=gap
    )

    joined = _join_thin_arms_to_fat_ridge(
        skeleton,
        thick,
        allowed,
        min_arm_extent_voxels=0.0,
        max_bridge_radius_multiple=multiplier,
        radius_smoothing_um=0.0,
    )

    _, n_cc = label(joined, structure=generate_binary_structure(3, 3))
    assert n_cc == 2, "zero smoothing radius must behave like a raw single-point sample"


def test_join_fallback_stays_fast_in_a_large_image_with_unrelated_content():
    """A join needing the full-mask fallback must not scan the whole image.

    _path_through_mask's own local searches are cheap regardless of image
    size, but its last-resort fallback runs a full EDT + Dijkstra over
    whatever array it is given. Passing it the whole image (this session's
    first fix for silent join failures) froze a real run: a real stack is
    orders of magnitude larger than any one vessel's own connected
    structure, and that fallback could fire more than once. The caller must
    crop to the arm's own physically connected component first.

    A generous elapsed-time bound (not a tight one) catches a regression to
    whole-image scanning without making CI flaky on a slower runner: cropped
    to one ring's own component this takes milliseconds; scanning the whole
    5.4M-voxel image directly would not finish in the bound below.
    """
    import time

    big_shape = (60, 300, 300)
    allowed = np.zeros(big_shape, dtype=bool)
    rng = np.random.default_rng(0)
    for _ in range(60):
        z0 = rng.integers(0, big_shape[0] - 3)
        y0 = rng.integers(0, big_shape[1] - 40)
        x0 = rng.integers(200, big_shape[2] - 40)
        allowed[z0 : z0 + 3, y0 : y0 + 30, x0 : x0 + 30] = True

    z = 30
    cy, cx = 10, 10
    outer_r, inner_r = 9, 8
    yy, xx = np.indices((21, 21))
    ring = ((yy - cy) ** 2 + (xx - cx) ** 2 <= outer_r**2) & (
        (yy - cy) ** 2 + (xx - cx) ** 2 >= inner_r**2
    )
    allowed[z, 0:21, 0:21] |= ring

    thick = np.zeros(big_shape, dtype=bool)
    thick[z, cy, cx + outer_r - 1] = True
    allowed[z, cy, cx + outer_r - 1] = True

    thin_skel = np.zeros(big_shape, dtype=bool)
    thin_skel[z, 0:21, 0:21] = ring
    thin_skel[z, cy, cx + outer_r - 1] = False
    skeleton = thin_skel | thick

    _, n_cc = label(allowed, structure=generate_binary_structure(3, 3))
    assert n_cc > 1, "fixture must have unrelated content outside the ring's component"

    start = time.perf_counter()
    joined = _join_thin_arms_to_fat_ridge(
        skeleton, thick, allowed, min_arm_extent_voxels=0.0
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 10.0, f"join took {elapsed:.2f}s -- looks like it scanned the whole image"
    _, n_cc_ring = label(joined[z : z + 1], structure=generate_binary_structure(3, 3))
    assert n_cc_ring == 1, "ring must actually be bridged to the fat wall, not just fast"


def test_length_filter_stays_scoped_when_thick_and_thin_span_most_of_the_image():
    """The pre-join length filter must not run one EDT over the whole stack.

    Production data crashed here: "Unable to allocate 1.35 GiB for an array
    with shape (180948686,)". thick_vessel_object_mask legitimately produces
    a fat catchment spanning most of a real stack, so thin | thick's own
    bounding box is close to the whole image -- a dense EDT over that box
    is what ran out of memory. The fix scopes this per physically connected
    structure instead; this fixture is deliberately shaped to fail the old
    approach (one huge fat blob plus a thin arm, filling most of a
    moderately large volume) while staying small enough to run in a unit
    test.
    """
    import time

    shape = (60, 400, 400)
    r_fat = 25.0
    trunk_len = 380
    mask = _disk_tube_along_axis(
        shape, (30, 200, 10), r_fat, trunk_len, axis=2
    )
    thick = mask.copy()
    branch = _disk_tube_along_axis(
        shape, (30, 200 + int(r_fat) - 2, 200), 1.5, 30, axis=1
    )
    mask = mask | branch
    thin_skel = branch.copy()
    skeleton = thin_skel | thick

    assert mask.sum() > shape[0] * shape[1] * shape[2] * 0.05, (
        "fixture must occupy a real fraction of the volume, not a sliver"
    )

    start = time.perf_counter()
    joined = _join_thin_arms_to_fat_ridge(
        skeleton, thick, mask, min_arm_extent_voxels=4.0
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 15.0, f"length filter took {elapsed:.2f}s -- looks unscoped"
    assert int((joined & branch).sum()) > 0, "the fused branch must still be kept"


def test_joining_many_arms_does_not_rescan_the_whole_image_per_arm():
    """The per-arm join loop must not re-derive its fat-voxel set from scratch.

    Production data stalled here for 30+ minutes with no further log output.
    The loop rebuilt `fat_coords` via `result & thick_b` followed by
    `np.argwhere` -- both full-image scans -- after *every* accepted arm, so
    a component with many genuinely fused arms paid that cost once per arm
    instead of once total. Locks the fix (grow the fat-voxel set by
    appending only the arm's own newly-fat voxels) with enough arms that the
    old approach would be far slower than the bound below, on an image large
    enough for a per-arm full scan to actually be expensive.
    """
    import time

    r_fat = 15.0
    trunk_len = 1000
    n_arms = 300
    arm_radius = 1.5
    arm_len = 12
    pad = 12
    shape = (
        int(2 * r_fat + 2 * pad),
        int(2 * r_fat + 2 * pad + arm_len),
        int(trunk_len + 2 * pad),
    )
    cz, cy, cx = shape[0] // 2, pad + int(r_fat), shape[2] // 2

    thick = _disk_tube_along_axis(
        shape, (cz, cy, cx - trunk_len // 2), r_fat, trunk_len, axis=2
    )
    mask = thick.copy()
    thin_skel = np.zeros(shape, dtype=bool)
    for i in range(n_arms):
        t = cx - trunk_len // 2 + int((i + 0.5) * trunk_len / n_arms)
        y0 = cy + int(r_fat) - 2  # overlaps the trunk: genuinely fused
        arm = _disk_tube_along_axis(shape, (cz, y0, t), arm_radius, arm_len, axis=1)
        mask = mask | arm
        thin_skel = thin_skel | (arm & ~thick)
    skeleton = thin_skel | thick

    _, n_cc_mask = label(mask, structure=generate_binary_structure(3, 3))
    assert n_cc_mask == 1, "fixture must be one fused structure"

    start = time.perf_counter()
    joined = _join_thin_arms_to_fat_ridge(
        skeleton, thick, mask, min_arm_extent_voxels=0.0
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 20.0, (
        f"joining {n_arms} arms took {elapsed:.2f}s -- looks like a per-arm "
        "full-image rescan regressed"
    )
    _, n_cc_joined = label(joined, structure=generate_binary_structure(3, 3))
    assert n_cc_joined == 1, "every arm must actually be bridged into one tree"


def test_geodesic_on_crop_precomputed_cost_matches_computing_it_fresh():
    """A cached cost array must give the identical path to computing it fresh.

    _join_thin_arms_to_fat_ridge memoizes the fallback's Dijkstra cost per
    physically connected structure so a component with many arms needing
    the fallback pays for its EDT once instead of once per arm. This locks
    that the memoized path is correct, not just fast: passing a precomputed
    cost must not change the result.
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

    from scipy.ndimage import distance_transform_edt

    fresh_path = _geodesic_on_crop(allowed, start, end)

    edt = distance_transform_edt(allowed)
    cost = np.where(allowed, 1.0 / (np.square(edt) + 1e-6), np.inf)
    cached_path = _geodesic_on_crop(allowed, start, end, precomputed_cost=cost)

    assert fresh_path == cached_path
    assert len(fresh_path) > 2


def test_geodesic_on_crop_precomputed_graph_matches_computing_it_fresh():
    """A cached Dijkstra graph must give the identical path to computing it fresh.

    _build_dijkstra_graph lets _join_thin_arms_to_fat_ridge's fallback skip
    rebuilding the sparse 26-neighbour graph (the expensive part of a
    Dijkstra call, not the walk itself) for every arm sharing one physically
    connected structure. Passing a precomputed graph must not change the
    result -- only whether it gets rebuilt.
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

    from scipy.ndimage import distance_transform_edt

    fresh_path = _geodesic_on_crop(allowed, start, end)

    edt = distance_transform_edt(allowed)
    cost = np.where(allowed, 1.0 / (np.square(edt) + 1e-6), np.inf)
    graph = _build_dijkstra_graph(allowed, cost)
    cached_path = _geodesic_on_crop(allowed, start, end, precomputed_graph=graph)

    assert fresh_path == cached_path


def test_path_through_mask_reuses_one_fallback_graph_across_several_arms():
    """The fallback's cached graph must stay correct for more than one pair.

    A horseshoe (a ring with a small gap) forces the true full-mask fallback
    for two different (start, end) pairs on either side of the gap -- corridor
    dilation and the tight padded box around each pair both fail, since the
    only real path goes the long way around, well outside either. Both pairs
    share the same physically connected structure, the way two different
    arms of one fat vessel would, so a caller (_join_thin_arms_to_fat_ridge)
    builds the Dijkstra graph once and reuses it. This checks two things at
    once: the graph is actually built only once, and reusing it still gives
    a correct, connected path for the second pair, not a stale one.
    """
    shape = (3, 200, 200)
    allowed = np.zeros(shape, dtype=bool)
    z = 1
    cy, cx = 100, 100
    outer_r, inner_r = 80, 70
    yy, xx = np.indices((200, 200))
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    ring = (dist2 <= outer_r**2) & (dist2 >= inner_r**2)
    angle = np.degrees(np.arctan2(yy - cy, xx - cx))
    gap = (angle > -6) & (angle < 6)
    allowed[z] = ring & ~gap

    def _snap(point):
        coords = np.argwhere(allowed)
        d = np.sum((coords - np.array(point)) ** 2, axis=1)
        return tuple(int(v) for v in coords[np.argmin(d)])

    from scipy.ndimage import distance_transform_edt

    built: dict = {}
    build_calls = {"n": 0}

    def fallback_graph_fn():
        if "graph" not in built:
            build_calls["n"] += 1
            edt = distance_transform_edt(allowed)
            cost = np.where(allowed, 1.0 / (np.square(edt) + 1e-6), np.inf)
            built["graph"] = _build_dijkstra_graph(allowed, cost)
        return built["graph"]

    import math

    r_mid = (outer_r + inner_r) / 2

    def point_at(degrees):
        rad = math.radians(degrees)
        return (
            z,
            int(round(cy + r_mid * math.sin(rad))),
            int(round(cx + r_mid * math.cos(rad))),
        )

    for degrees_start, degrees_end in ((8, -8), (10, -10)):
        start = _snap(point_at(degrees_start))
        end = _snap(point_at(degrees_end))
        path = _path_through_mask(start, end, allowed, fallback_graph_fn=fallback_graph_fn)
        assert len(path) > 2, "join must not silently give up as a bare [start, end]"
        assert all(allowed[p] for p in path), "every joined voxel must be foreground"
        assert all(
            max(abs(a - b) for a, b in zip(path[i], path[i + 1])) <= 1
            for i in range(len(path) - 1)
        ), "path must be a real 26-connected walk, not a straight-line jump"

    assert build_calls["n"] == 1, "the graph must be built once, not once per arm"
