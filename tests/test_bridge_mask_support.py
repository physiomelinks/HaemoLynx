"""``preprocessing.bridge_mask_support``: is a bridge a vessel, or background?

Nothing that joined two pieces of vessel used to ask the segmentation whether
there was vessel between them. These pin the measurement every bridge site now
shares: how much of a path lies in the mask, its longest run through
background, and whether a path runs beside an existing one in the same lumen.
"""
from __future__ import annotations

import numpy as np
import pytest

import importlib

from haemolynx.preprocessing.bridge_mask_support import (
    DEFAULT_MIN_MASK_FRACTION,
    BridgeMaskSupport,
    MaskSupport,
    PackedMask,
    _points_in_mask,
    bridge_is_supported,
    bridge_mask_support,
    path_shadows_existing_vessel,
)

VOXEL = (0.98, 0.98, 0.98)

# The package re-exports a function of the same name, so the module comes from importlib.
module = importlib.import_module("haemolynx.preprocessing.bridge_mask_support")


def _tube_along_x(shape, centre_zy, radius_vox):
    zz, yy, _xx = np.indices(shape)
    return (zz - centre_zy[0]) ** 2 + (yy - centre_zy[1]) ** 2 <= radius_vox**2


def _path_along_x(z, y, x0, x1, voxel=VOXEL):
    return [np.array([z, y, x0]) * voxel, np.array([z, y, x1]) * voxel]


def test_a_path_inside_the_mask_is_all_mask():
    mask = _tube_along_x((7, 7, 30), (3, 3), 2)

    support = bridge_mask_support(_path_along_x(3, 3, 2, 27), mask, VOXEL)

    assert support.inside_fraction == pytest.approx(1.0)
    assert support.longest_background_um == 0.0
    assert support.length_um == pytest.approx(25 * 0.98)
    assert bridge_is_supported(support)


@pytest.mark.parametrize("dropout_voxels, accepted", [(1, True), (2, True), (3, False)])
def test_a_dropout_of_up_to_two_voxels_is_bridged_and_three_is_not(dropout_voxels, accepted):
    """On a 0.98 um stack a one- or two-voxel hole in a vessel's segmentation
    reads as that much background, within the 2 um allowed; three voxels
    (2.94 um) is a gap, not a dropout."""
    mask = _tube_along_x((7, 7, 30), (3, 3), 2)
    mask[:, :, 15 : 15 + dropout_voxels] = False

    support = bridge_mask_support(_path_along_x(3, 3, 5, 25), mask, VOXEL)

    assert support.longest_background_um == pytest.approx(dropout_voxels * 0.98, abs=0.05)
    assert bridge_is_supported(support) is accepted


def test_a_three_micron_gap_is_refused_on_unit_voxels():
    mask = np.zeros((5, 5, 30), dtype=bool)
    mask[1:4, 1:4, :12] = mask[1:4, 1:4, 15:] = True

    support = bridge_mask_support([(2, 2, 5), (2, 2, 25)], mask, (1.0, 1.0, 1.0))

    assert support.longest_background_um == pytest.approx(3.0, abs=0.05)
    assert not bridge_is_supported(support)


def test_a_bent_path_measures_its_background_along_the_bend():
    """An L-shaped corridor with a one-voxel hole at the corner: the path's
    own length through it, not the straight line, is what is measured."""
    mask = np.zeros((3, 20, 20), dtype=bool)
    mask[1, 2:18, 2] = True
    mask[1, 17, 2:18] = True
    mask[1, 17, 2] = False  # the corner
    path = [np.array(p) * 0.98 for p in ((1, 3, 2), (1, 17, 2), (1, 17, 16))]

    support = bridge_mask_support(path, mask, VOXEL)

    assert support.length_um == pytest.approx(28 * 0.98)
    assert support.longest_background_um == pytest.approx(0.98, abs=0.05)
    assert bridge_is_supported(support)


@pytest.mark.parametrize(
    "axis, holes, expected_um, accepted",
    [(0, 1, 2.0, True), (0, 2, 4.0, False), (2, 4, 2.0, True), (2, 5, 2.5, False)],
)
def test_background_is_measured_in_microns_on_anisotropic_voxels(axis, holes, expected_um, accepted):
    """A 2 um z slice missing is 2 um of background; four 0.5 um x voxels are
    the same 2 um. Counted in voxels the z slice would read as a quarter."""
    spacing = (2.0, 0.5, 0.5)
    shape = [5, 5, 5]
    shape[axis] = 40
    mask = np.zeros(shape, dtype=bool)
    full = [slice(1, 4)] * 3
    full[axis] = slice(None)
    mask[tuple(full)] = True
    hole = [slice(None)] * 3
    hole[axis] = slice(20, 20 + holes)
    mask[tuple(hole)] = False
    start, end = np.array([2.0, 2.0, 2.0]), np.array([2.0, 2.0, 2.0])
    start[axis], end[axis] = 5, 35

    support = bridge_mask_support([start * spacing, end * spacing], mask, spacing)

    assert support.longest_background_um == pytest.approx(expected_um, abs=0.1)
    assert bridge_is_supported(support) is accepted


@pytest.mark.parametrize("hole_at", [20, 21])
@pytest.mark.parametrize("path_z", [(5, 35), tuple(range(5, 36)), (18, 23)])
def test_a_one_voxel_z_dropout_reads_one_micron_on_a_coarse_z(hole_at, path_z):
    """Regression (PLAN_bridging.md F10): pieces of 0.15 um, half the finest
    0.3 um voxel, straddle the boundaries of a 1 um z voxel, and one missing
    z slice read 0.86-1.14 um depending on the path's vertices. The step now
    divides every axis's half spacing (0.05 um), so it reads 1.0 um."""
    spacing = (1.0, 0.3, 0.3)
    mask = np.zeros((40, 5, 5), dtype=bool)
    mask[:, 1:4, 1:4] = True
    mask[hole_at] = False
    path = np.array([(z, 2, 2) for z in path_z], dtype=float) * np.array(spacing)

    support = bridge_mask_support(path, mask, spacing)

    assert support.longest_background_um == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "spacing, step",
    [((1.0, 0.3, 0.3), 0.05), ((2.0, 0.5, 0.5), 0.25), ((1.0, 1.0, 1.0), 0.5), ((1.0, 0.98, 0.98), 0.49)],
)
def test_the_sample_step_divides_every_half_spacing_where_a_small_divisor_does(spacing, step):
    """0.5 x 0.5 x 2 um and cube voxels keep their step; on 0.98 x 0.98 x 1 um
    (E14.5) no step within four times the lookups divides both half spacings,
    so the finest axis's step stays and nothing measured there moves."""
    from haemolynx.preprocessing.bridge_mask_support import _sample_step

    assert _sample_step(spacing) == pytest.approx(step)


def test_a_path_mostly_outside_the_mask_is_refused_however_short_its_gaps():
    """Every gap is a single voxel, but half the path is background: a
    straight line through tissue that happens to cross specks of mask."""
    mask = np.zeros((3, 3, 40), dtype=bool)
    mask[1, 1, ::2] = True

    support = bridge_mask_support([(1, 1, 2), (1, 1, 36)], mask, (1.0, 1.0, 1.0))

    assert support.longest_background_um <= 2.0
    assert support.inside_fraction == pytest.approx(0.5, abs=0.05)
    assert not bridge_is_supported(support)


def test_the_thresholds_are_the_callers():
    support = BridgeMaskSupport(inside_fraction=0.9, longest_background_um=2.5, length_um=40.0)

    assert not bridge_is_supported(support)
    assert bridge_is_supported(support, max_background_gap_um=3.0)
    assert not bridge_is_supported(support, max_background_gap_um=3.0, min_mask_fraction=0.95)


@pytest.mark.parametrize("dropout_voxels, accepted", [(1, True), (2, True), (3, False)])
def test_a_short_bridge_across_one_dropout_is_not_held_to_the_fraction(dropout_voxels, accepted):
    """Regression: a 4 um bridge between the two ends of a vessel broken by a
    one-voxel dropout is a quarter background, so the fraction alone refused
    every bridge the dropout tolerance exists for."""
    mask = _tube_along_x((7, 7, 30), (3, 3), 2)
    mask[:, :, 12 : 12 + dropout_voxels] = False

    support = bridge_mask_support(_path_along_x(3, 3, 11, 12 + dropout_voxels), mask, VOXEL)

    assert support.inside_fraction < DEFAULT_MIN_MASK_FRACTION
    assert bridge_is_supported(support) is accepted


def test_an_inside_test_can_stand_in_for_the_mask():
    def inside(points):
        return np.asarray(points)[:, 2] < 10.0

    support = bridge_mask_support([(0, 0, 0), (0, 0, 20)], inside=inside)

    assert support.inside_fraction == pytest.approx(0.5, abs=0.02)
    assert support.longest_background_um == pytest.approx(10.0, abs=0.5)


def test_the_lumen_radius_is_the_distance_to_background():
    mask = np.zeros((9, 21, 20), dtype=bool)
    mask[:, 4:17, :] = True  # a slab 13 voxels thick in y
    support = MaskSupport(mask, (1.0, 0.5, 1.0))

    radii = support.radius(np.array([[4, 5.0, 10], [4, 7.0, 10], [4, 0.0, 10]]))

    assert radii[0] == pytest.approx(3.5)  # voxel y=10: background at y=17, 7 x 0.5 um
    assert radii[1] == pytest.approx(1.5)  # voxel y=14
    assert radii[2] == 0.0
    assert support.inside(np.array([[4, 5.0, 10], [4, 0.0, 10]])).tolist() == [True, False]


# --- the same-lumen duplicate guard -----------------------------------------


def _shadows(path, existing, mask, spacing=(1.0, 1.0, 1.0), **options):
    support = MaskSupport(mask, spacing)
    return path_shadows_existing_vessel(
        np.asarray(path, dtype=float), None, np.asarray(existing, dtype=float),
        support.inside, support.radius, **options,
    )


def _line(z, y, x0, x1):
    return np.array([[z, y, x] for x in range(x0, x1 + 1)], dtype=float)


def test_a_second_strand_inside_one_wide_lumen_is_a_duplicate():
    """Lee thinning of a 12 um vessel leaves strands 5 um apart: one lumen,
    no background between, so a path along the second is the same vessel."""
    mask = _tube_along_x((15, 15, 60), (7, 7), 6)
    existing = _line(7, 4.5, 5, 55)

    assert _shadows(_line(7, 9.5, 5, 55), existing, mask)


def test_two_real_vessels_with_background_between_are_not_duplicates():
    """Two vessels 7 um wide, 8 um apart, with background between their walls
    -- and the same pair with that background filled in, which is one lumen."""
    mask = np.zeros((5, 20, 60), dtype=bool)
    mask[:, 2:9] = mask[:, 11:18] = True
    fused = mask.copy()
    fused[:, 9:11] = True

    assert not _shadows(_line(2, 14, 5, 55), _line(2, 6, 5, 55), mask)
    assert _shadows(_line(2, 14, 5, 55), _line(2, 6, 5, 55), fused)


def test_a_branch_leaving_a_vessel_is_not_a_duplicate_of_it():
    """A T-junction: the branch's attachment end sits on the trunk, and past
    it the trunk's nearest point lies straight behind the branch."""
    mask = _tube_along_x((9, 40, 60), (4, 5), 3)
    mask[1:8, 5:35, 28:33] = True
    trunk = _line(4, 5, 0, 59)
    branch = np.array([[4, y, 30] for y in range(5, 34)], dtype=float)

    assert not _shadows(branch, trunk, mask)


def test_a_bridge_continuing_a_vessel_is_not_a_duplicate_of_it():
    mask = _tube_along_x((9, 9, 60), (4, 4), 3)
    pieces = np.vstack([_line(4, 4, 0, 20), _line(4, 4, 30, 59)])

    assert not _shadows(_line(4, 4, 20, 30), pieces, mask)


def test_nothing_to_shadow_is_no_duplicate():
    mask = _tube_along_x((9, 9, 60), (4, 4), 3)

    assert not _shadows(_line(4, 4, 0, 30), np.empty((0, 3)), mask)
    # A path no longer than its own end margins has nothing left to judge.
    assert not _shadows(_line(4, 4, 10, 12), _line(4, 4, 0, 59), mask)


def test_a_bridge_shorter_than_its_end_margins_is_judged_at_its_middle():
    """Regression (PLAN_bridging.md F4): every sample within a lumen radius
    and a micron of either end is left out, so in a lumen 6 um round a path
    under 14 um had nothing judged and was never a duplicate -- a gap or
    orphan bridge could draw a second strand there unchallenged. Its middle
    is judged instead, when the caller asks (the reconnects drawing a
    bridge): beside a strand in the same lumen it is one; carrying a vessel
    on across a gap it is not."""
    mask = _tube_along_x((15, 15, 60), (7, 7), 6)
    strand = _line(7, 4, 5, 55)

    assert _shadows(_line(7, 8, 28, 32), strand, mask, judge_short_at_middle=True)
    assert not _shadows(_line(7, 8, 28, 32), strand, mask)
    assert not _shadows(_line(7, 4, 55, 59), _line(7, 4, 5, 55), mask, judge_short_at_middle=True)


# --- the mask as bits -------------------------------------------------------


def _all_indices(shape):
    return np.indices(shape).reshape(len(shape), -1).T


@pytest.mark.parametrize(
    "shape",
    [
        (20, 7, 9),   # 63 voxels a slice: eight slices make whole bytes
        (10, 3, 4),   # 12 a slice: two slices do
        (9, 2, 4),    # 8 a slice: every slice does, and the last slab is short
        (3, 16, 16),  # fewer slices than one slab needs
    ],
)
def test_a_packed_mask_holds_every_voxel_however_it_is_sliced_into_slabs(shape):
    rng = np.random.default_rng(int(np.prod(shape)))
    mask = rng.random(shape) > 0.5
    whole = PackedMask.from_mask(mask)
    by_slab = PackedMask.from_mask(mask, block_voxels=100)
    assert np.array_equal(whole.bits, by_slab.bits)
    assert whole.shape == shape and whole.bits.nbytes == (mask.size + 7) // 8
    assert np.array_equal(by_slab.contains(_all_indices(shape)), mask.ravel())


def test_a_packed_mask_answers_point_lookups_as_the_mask_does_inside_and_outside_the_volume():
    rng = np.random.default_rng(3)
    mask = rng.random((12, 20, 30)) > 0.4
    packed = PackedMask.from_mask(mask)
    voxel = (2.0, 0.5, 0.5)
    points = rng.uniform(-4.0, np.asarray(mask.shape) * voxel + 4.0, size=(5000, 3))
    assert np.array_equal(_points_in_mask(points, packed, voxel), _points_in_mask(points, mask, voxel))
    assert len(packed.contains(np.empty((0, 3), dtype=int))) == 0


@pytest.mark.parametrize("shape", [(12, 20, 30), (5, 7, 9), (3, 1, 13), (40, 33)])
def test_a_packed_masks_box_is_the_masks_own_box(shape):
    """Rows of every bit offset (a row of 30, 9 or 13 voxels starts anywhere
    in a byte), boxes touching the last voxel, one voxel wide or tall."""
    rng = np.random.default_rng(sum(shape))
    mask = rng.random(shape) > 0.5
    packed = PackedMask.from_mask(mask)
    boxes = [tuple(slice(0, n) for n in shape), tuple(slice(n - 1, n) for n in shape)]
    for _ in range(200):
        start = [int(rng.integers(0, n)) for n in shape]
        boxes.append(tuple(slice(a, int(rng.integers(a + 1, n + 1))) for a, n in zip(start, shape)))
    for box in boxes:
        got = packed.box(box)
        assert got.dtype == bool and np.array_equal(got, mask[box]), box
    empty = tuple(slice(2, 2) if axis == 0 else slice(0, n) for axis, n in enumerate(shape))
    assert packed.box(empty).shape == mask[empty].shape


def test_a_memmapped_mask_packs_to_the_same_bits(tmp_path):
    rng = np.random.default_rng(5)
    mask = rng.random((17, 7, 9)) > 0.5
    on_disk = np.memmap(tmp_path / "mask.bin", dtype=bool, mode="w+", shape=mask.shape)
    on_disk[...] = mask
    assert np.array_equal(PackedMask.from_mask(on_disk, block_voxels=200).bits, PackedMask.from_mask(mask).bits)
    on_disk._mmap.close()


def test_a_mask_support_packs_its_mask_once_enough_points_have_been_asked_about(monkeypatch):
    monkeypatch.setattr(module, "PACK_AFTER_POINTS", 50)
    rng = np.random.default_rng(8)
    mask = rng.random((6, 10, 10)) > 0.5
    support = MaskSupport(mask, (1.0, 1.0, 1.0))
    points = rng.uniform(-1.0, 10.0, size=(30, 3))
    before = support.inside(points)
    assert support._packed is None
    assert np.array_equal(before, _points_in_mask(points, mask, (1.0, 1.0, 1.0)))
    after = support.inside(points)  # the 60th point crosses the line
    assert isinstance(support._packed, PackedMask)
    assert np.array_equal(after, before)
    assert support.pack() is support._packed
    shared = MaskSupport(mask, (1.0, 1.0, 1.0), packed=support.pack())
    assert shared.pack() is support.pack()
    assert np.array_equal(shared.inside(points), before)


# --- one support for every reader of a mask (2026-10-18) ---------------------


def test_a_support_under_its_own_limits_is_itself_and_under_others_shares_its_bits_and_radii():
    rng = np.random.default_rng(4)
    mask = rng.random((8, 12, 12)) > 0.4
    support = MaskSupport(mask, VOXEL, max_background_gap_um=2.0, min_mask_fraction=0.8)
    support.pack()
    support.radius(np.array([[3.0, 3.0, 3.0]]))

    assert support.limited_to(2.0, 0.8) is support
    other = support.limited_to(1.0, 0.9)
    assert (other.max_background_gap_um, other.min_mask_fraction) == (1.0, 0.9)
    assert other.pack() is support.pack()
    assert other._distance is support._distance
    assert other.mask is support.mask


def test_the_lumen_radius_reads_the_supports_bits_once_it_has_packed_the_mask(monkeypatch):
    """The same radii either way; once packed, the mask itself is not read
    to ask whether a voxel is background."""
    monkeypatch.setattr(module, "PACK_AFTER_POINTS", 10)
    rng = np.random.default_rng(6)
    mask = np.zeros((10, 20, 20), dtype=bool)
    mask[2:8, 3:17, 3:17] = rng.random((6, 14, 14)) > 0.2
    points = rng.uniform(0.0, 9.0, size=(300, 3)) * np.array([1.0, 2.0, 2.0])
    plain = MaskSupport(mask.copy(), VOXEL).radius(points)

    support = MaskSupport(mask, VOXEL)
    before = support.radius(points[:5])
    support.inside(points[:20])  # past the line: packed
    assert support._distance.packed is support.pack()

    class Unread(np.ndarray):
        def __getitem__(self, key):
            raise AssertionError("the mask was read voxel by voxel")

    support._distance.features = support.mask = mask.view(Unread)
    assert np.array_equal(before, plain[:5])
    assert np.array_equal(support.radius(points), plain)


def test_a_radius_lookup_shared_from_elsewhere_is_given_the_bits_only_for_the_same_mask():
    from haemolynx.preprocessing.pointwise_distance import FeatureDistance

    mask = np.zeros((6, 8, 8), dtype=bool)
    mask[1:5, 2:6, 2:6] = True
    same = FeatureDistance(mask, feature_value=False, sampling=VOXEL)
    copy = FeatureDistance(mask.copy(), feature_value=False, sampling=VOXEL)

    MaskSupport(mask, VOXEL, feature_distance=same).pack()
    MaskSupport(mask, VOXEL, feature_distance=copy).pack()

    assert isinstance(same.packed, PackedMask)
    assert copy.packed is None, "bits of another array are not known to be its bits"


@pytest.mark.parametrize("pack_after", [1, 10**9])
def test_the_lumen_test_read_through_a_support_gives_the_same_answers(monkeypatch, pack_after):
    """``graph.assemble.mask_lumen_test`` with the run's support: the same
    answers, the points counted towards packing the support's mask."""
    from haemolynx.graph.assemble import mask_lumen_test

    monkeypatch.setattr(module, "PACK_AFTER_POINTS", pack_after)
    rng = np.random.default_rng(9)
    mask = rng.random((9, 11, 13)) > 0.5
    points = rng.uniform(-2.0, 14.0, size=(500, 3))
    support = MaskSupport(mask, VOXEL)

    through = mask_lumen_test(mask, VOXEL, support=support)(points)

    assert np.array_equal(through, mask_lumen_test(mask, VOXEL)(points))
    assert (support._packed is not None) is (pack_after == 1)
    with pytest.raises(ValueError):
        mask_lumen_test(mask, VOXEL, support=support)(points[:, :2])


def test_points_in_a_packed_mask_are_the_points_in_the_mask():
    from haemolynx.graph._helpers import points_inside_mask

    rng = np.random.default_rng(10)
    mask = rng.random((7, 9, 11)) > 0.5
    points = rng.uniform(-3.0, 13.0, size=(400, 3))

    assert np.array_equal(
        points_inside_mask(points, PackedMask.from_mask(mask), voxel_size_zyx=VOXEL),
        points_inside_mask(points, mask, voxel_size_zyx=VOXEL),
    )
