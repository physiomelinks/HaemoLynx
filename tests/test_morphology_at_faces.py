"""Morphology reads past an image face as the vessel running on, not as
background (PLAN_dale_review_fixes.md item 2; PLAN_dead_ends.md Phase 3): no
closing or opening trims a vessel the image cut."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import binary_closing

from haemolynx.preprocessing import segmentation_cleanup as sc
from haemolynx.preprocessing.memmap_support import at_faces_as_beyond, map_blockwise
from haemolynx.preprocessing.skeleton import close_binary_mask


def _tube_through_z(shape=(20, 21, 21), radius=3.0) -> np.ndarray:
    z, y, x = np.indices(shape)
    cy, cx = (shape[1] - 1) / 2, (shape[2] - 1) / 2
    return (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2


def _oblique_tube(shape=(16, 40, 40), radius=2.5) -> np.ndarray:
    """A tube running out through the x and y faces, at 45 degrees in-plane."""
    z, y, x = np.indices(shape).astype(float)
    return (z - 8) ** 2 + ((x - y) / np.sqrt(2)) ** 2 <= radius ** 2


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_closing_loses_no_voxel_of_a_tube_through_the_faces(radius):
    tube = _tube_through_z()
    closed = close_binary_mask(tube, radius=radius)
    assert not np.any(tube & ~closed)
    blockwise = close_binary_mask(tube, radius=radius, use_memmap=True)
    assert np.array_equal(np.asarray(blockwise), closed)


def test_closing_loses_nothing_at_x_and_y_faces_with_anisotropic_voxels():
    tube = _oblique_tube()
    closed = close_binary_mask(tube, radius=2, voxel_size_zyx=(2.0, 0.5, 0.5))
    assert not np.any(tube & ~closed)


def test_unpadded_closing_did_eat_the_face_slices():
    """The bug this fixes, pinned so it cannot quietly come back."""
    tube = _tube_through_z()
    plain = binary_closing(tube, iterations=2)
    assert np.count_nonzero(tube & ~plain) > 0
    padded = at_faces_as_beyond(lambda b: binary_closing(b, iterations=2), 2)(tube)
    assert np.count_nonzero(tube & ~padded) == 0


def test_closing_still_bridges_a_one_voxel_gap_and_keeps_far_vessels_apart():
    tube = _tube_through_z()
    gapped = tube.copy()
    gapped[10] = False
    from scipy.ndimage import label

    bridged = close_binary_mask(gapped, radius=1)
    assert label(gapped)[1] == 2 and label(bridged)[1] == 1
    two = np.zeros((10, 30, 30), dtype=bool)
    two[:, 5:8, 5:8] = True
    two[:, 20:23, 20:23] = True
    closed = close_binary_mask(two, radius=2)
    assert not closed[:, 12:16, 12:16].any()


def test_blockwise_is_exact_with_the_faces_padded():
    tube = _oblique_tube()
    op = at_faces_as_beyond(lambda b: binary_closing(b, iterations=2), 4)
    whole = op(tube)
    out = np.zeros_like(tube)
    map_blockwise(tube, op, out, halo=4, block_voxels=2000)
    assert np.array_equal(out, whole)


def test_an_opening_trims_a_face_slice_no_more_than_an_inner_one():
    tube = _tube_through_z(shape=(24, 25, 25), radius=4.0)
    opened = sc.remove_surface_whiskers(tube, voxel_size_zyx=(1.0, 1.0, 1.0), whisker_radius_um=1.0)
    lost = (tube & ~np.asarray(opened)).sum(axis=(1, 2))
    assert lost[0] == lost[12] and lost[-1] == lost[12]


@pytest.mark.parametrize("cleanup", ["close_small_gaps", "smooth"])
def test_gap_closing_and_smoothing_trim_nothing_at_the_faces(cleanup):
    tube = _tube_through_z(shape=(24, 25, 25), radius=4.0)
    if cleanup == "close_small_gaps":
        result = sc.close_small_gaps(tube, voxel_size_zyx=(1.0, 1.0, 1.0), closing_radius_um=2.0)
    else:
        result = sc._smooth_vessel_surfaces_morphological(tube, voxel_size_zyx=(1.0, 1.0, 1.0), radius_um=1.0)
    result = np.asarray(result)
    per_slice = result.sum(axis=(1, 2))
    assert per_slice[0] == per_slice[12] and per_slice[-1] == per_slice[12]
    blockwise = (
        sc.close_small_gaps(tube, voxel_size_zyx=(1.0, 1.0, 1.0), closing_radius_um=2.0, use_memmap=True)
        if cleanup == "close_small_gaps"
        else sc._smooth_vessel_surfaces_morphological(tube, voxel_size_zyx=(1.0, 1.0, 1.0), radius_um=1.0,
                                                      use_memmap=True)
    )
    assert np.array_equal(np.asarray(blockwise), result)
