"""Skeleton bridging joins branch ends to what lies ahead of them, in microns.

``connect_skeleton_components`` used to join the nearest pair of voxels
between any two components, wherever on them those were, measured in voxels
with a hand-set z weight. Side-by-side capillaries a few voxels apart were
cross-linked, and on a coarse-z stack a gap along z counted as a quarter of
its real length. A bridge now starts at a branch end (a tip), goes only to
something within the tip's own heading, and reaches ``max_bridge_distance``
voxels of the *finest* axis, measured physically.

``inter_component_gap_distances`` -- the distribution of gaps the settings
optimiser picks candidate distances from -- is tested at the end.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label
from scipy.spatial import cKDTree

import haemolynx.preprocessing.skeleton as skeleton_mod
from haemolynx.preprocessing.skeleton import (
    connect_skeleton_components,
    inter_component_gap_distances,
    preprocess_skeleton_for_graph,
)

STRUCTURE = generate_binary_structure(3, 3)


def _count(volume) -> int:
    return int(label(volume, structure=STRUCTURE)[1])


def _components(skeleton, z_weight):
    labeled, n = label(skeleton, structure=generate_binary_structure(3, 3))
    weights = np.array([z_weight, 1.0, 1.0])
    coords = np.argwhere(labeled)
    labels = labeled[tuple(coords.T)]
    order = np.argsort(labels, kind="stable")
    coords, labels = coords[order], labels[order]
    starts = np.searchsorted(labels, np.arange(1, n + 2))
    comp = {c: coords[starts[c - 1]:starts[c]] for c in range(1, n + 1)}
    return comp, weights


def _fragments(seed):
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(6, 14)), int(rng.integers(16, 32)), int(rng.integers(16, 32)))
    skeleton = rng.random(shape) > rng.uniform(0.96, 0.99)
    for _ in range(int(rng.integers(1, 4))):  # a few long components
        skeleton[rng.integers(0, shape[0]), rng.integers(0, shape[1]), :] = True
    return skeleton


def test_two_parallel_vessels_are_not_cross_linked():
    """Regression: the nearest voxels of two side-by-side lines were bridged,
    though neither line ends there."""
    volume = np.zeros((5, 12, 30), dtype=bool)
    volume[2, 3, 2:28] = True
    volume[2, 6, 2:28] = True  # 3 voxels away along its whole length

    result = connect_skeleton_components(volume, max_bridge_distance=5)

    assert _count(result) == 2
    assert np.array_equal(result, volume)


def test_a_vessel_broken_in_two_is_joined():
    volume = np.zeros((5, 5, 30), dtype=bool)
    volume[2, 2, 2:12] = True
    volume[2, 2, 15:28] = True

    result = connect_skeleton_components(volume, max_bridge_distance=5)

    assert _count(result) == 1
    assert result[2, 2, 12:15].all()


def test_a_branch_ending_just_short_of_another_vessel_joins_its_side():
    """The target need not be a tip: a capillary cut off just before its
    junction reaches the side of the vessel it was joining."""
    volume = np.zeros((5, 20, 30), dtype=bool)
    volume[2, 15, 2:28] = True  # the trunk, along x
    volume[2, 2:12, 14] = True  # a branch along y, ending 3 voxels short of it

    result = connect_skeleton_components(volume, max_bridge_distance=5)

    assert _count(result) == 1


def _line_and_a_piece_beside_its_end():
    """A line ending at x=14 (heading +x), and a short piece running along z
    4 voxels to one side of it, behind its end -- within reach of the end,
    but not ahead of it, and with its own ends pointing along z, at nothing."""
    volume = np.zeros((5, 20, 30), dtype=bool)
    volume[2, 5, 2:15] = True
    volume[0:5, 9, 12] = True
    return volume


def test_a_fragment_beside_a_branch_end_is_not_reached_for():
    """Regression: its nearest voxel was 4 away, and that was enough."""
    volume = _line_and_a_piece_beside_its_end()

    result = connect_skeleton_components(volume, max_bridge_distance=5)

    assert _count(result) == 2


def test_facing_can_be_switched_off():
    volume = _line_and_a_piece_beside_its_end()

    result = connect_skeleton_components(volume, max_bridge_distance=5, min_facing_cosine=-1.0)

    assert _count(result) == 1


def test_a_lone_voxel_is_never_bridged_from():
    """A speck has no direction to be ahead of."""
    volume = np.zeros((5, 5, 20), dtype=bool)
    volume[2, 2, 2] = True
    volume[2, 2, 5] = True

    result = connect_skeleton_components(volume, max_bridge_distance=5)

    assert _count(result) == 2


def test_the_reach_is_physical_counted_in_voxels_of_the_finest_axis():
    """On 2 x 0.5 x 0.5 um voxels a reach of 4 is 2 um: ends 3 z steps apart
    (6 um) are too far, while ends 4 x steps apart (2 um) are not. In voxels
    the z gap counted as 3 -- within reach."""
    spacing = (2.0, 0.5, 0.5)
    along_z = np.zeros((20, 5, 5), dtype=bool)
    along_z[0:6, 2, 2] = True
    along_z[8:14, 2, 2] = True  # ends 3 z steps = 6 um apart
    along_x = np.zeros((5, 5, 30), dtype=bool)
    along_x[2, 2, 0:10] = True
    along_x[2, 2, 13:23] = True  # ends 4 x steps = 2 um apart

    assert _count(connect_skeleton_components(along_z, max_bridge_distance=4, voxel_size_zyx=spacing)) == 2
    assert _count(connect_skeleton_components(along_x, max_bridge_distance=4, voxel_size_zyx=spacing)) == 1
    # With no voxel size given every voxel is a unit cube, as before.
    assert _count(connect_skeleton_components(along_z, max_bridge_distance=4)) == 1


def test_a_short_piece_of_a_broken_vessel_is_joined_before_it_is_filtered():
    """Regression: min_branch_length deleted small components before
    bridging, so a larger value removed exactly the pieces bridging exists to
    reconnect."""
    volume = np.zeros((5, 5, 60), dtype=bool)
    volume[2, 2, 0:30] = True
    volume[2, 2, 33:39] = True  # a 6-voxel piece past a 2-voxel gap
    volume[2, 2, 42:60] = True

    result = preprocess_skeleton_for_graph(
        volume, min_branch_length=10, max_bridge_distance=4, bundle_scan_size=3,
        bundle_density_fraction=1.0,
    )

    assert _count(result) == 1
    assert result[2, 2, 35]


def test_what_bridging_could_not_join_is_still_filtered():
    volume = np.zeros((5, 20, 60), dtype=bool)
    volume[2, 2, 0:40] = True
    volume[2, 15, 20:25] = True  # a 5-voxel piece, nothing ahead of it

    result = preprocess_skeleton_for_graph(
        volume, min_branch_length=10, max_bridge_distance=4, bundle_scan_size=3,
        bundle_density_fraction=1.0,
    )

    assert _count(result) == 1
    assert not result[2, 15, 20:25].any()


#: The pipeline's default skeleton settings, closing and bridge_gaps included.
_PIPELINE_DEFAULTS = dict(
    min_branch_length=3, max_bridge_distance=4, closing_radius=2, bridge_gap_size=3,
    bundle_scan_size=9, bundle_density_fraction=0.35, bundle_max_connections_per_hub=8,
    bundle_hub_min_spacing=4,
)


def _vessel_with_a_row_of_specks(specks_y: int) -> np.ndarray:
    volume = np.zeros((24, 40, 60), dtype=bool)
    volume[12, 20, 5:55] = True
    volume[12, specks_y, 20:30:3] = True  # four lone noise voxels, 3 apart
    return volume


def test_a_row_of_noise_specks_is_dropped_before_bridge_gaps_can_grow_it():
    """Regression: with the size filter moved after bridging, bridge_gaps
    merged the specks into one blob whose skeleton was long enough to pass it,
    so the noise stayed as a component of its own (a noisy image's graph fell
    into pieces and could not be solved)."""
    result = preprocess_skeleton_for_graph(_vessel_with_a_row_of_specks(6), **_PIPELINE_DEFAULTS)

    assert _count(result) == 1
    assert not result[:, :14].any()


def test_noise_specks_beside_a_vessel_leave_it_as_it_was():
    """The same specks 6 voxels off the vessel's side were grown into it as a
    lump; a speck no bridge would reach for is not a piece of the vessel."""
    vessel_alone = np.zeros((24, 40, 60), dtype=bool)
    vessel_alone[12, 20, 5:55] = True
    expected = preprocess_skeleton_for_graph(vessel_alone, **_PIPELINE_DEFAULTS)

    result = preprocess_skeleton_for_graph(_vessel_with_a_row_of_specks(26), **_PIPELINE_DEFAULTS)

    assert np.array_equal(result, expected)


def test_a_short_piece_a_vessel_end_points_at_survives_the_early_filter():
    """The early filter keeps what bridging could join: here a 2-voxel piece
    the vessel's end points straight at, 3 voxels past it."""
    volume = np.zeros((5, 5, 40), dtype=bool)
    volume[2, 2, 0:30] = True
    volume[2, 2, 33:35] = True

    result = preprocess_skeleton_for_graph(
        volume, min_branch_length=5, max_bridge_distance=4, bundle_scan_size=3,
        bundle_density_fraction=1.0,
    )

    assert _count(result) == 1
    assert result[2, 2, 34]


def test_the_early_filter_gives_the_same_result_on_the_low_ram_path(tmp_path):
    volume = _vessel_with_a_row_of_specks(6)
    volume[2, 2, 10:13] = True  # a short piece in line with nothing: dropped as well

    in_ram = preprocess_skeleton_for_graph(volume, **_PIPELINE_DEFAULTS)
    on_disk = preprocess_skeleton_for_graph(
        volume, use_memmap=True, memmap_directory=tmp_path, **_PIPELINE_DEFAULTS
    )

    assert np.array_equal(np.asarray(on_disk), in_ram)
    assert _count(in_ram) == 1


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("voxel_size_zyx, z_weight", [((1.0, 1.0, 1.0), 1.0), ((1.3, 0.4, 0.4), 1.5)])
def test_gap_distances_are_the_all_pairs_minimum(seed, voxel_size_zyx, z_weight):
    """The loop this replaced, exactly: every pair's minimum voxel gap."""
    skeleton = _fragments(seed)
    comp, _ = _components(skeleton, 1.0)
    spacing = np.asarray(voxel_size_zyx) * np.array([z_weight, 1.0, 1.0])
    comp = {c: v * spacing for c, v in comp.items()}
    trees = {c: cKDTree(v) for c, v in comp.items()}
    ids = sorted(comp)
    expected = sorted(
        float(trees[b].query(comp[a])[0].min())
        for i, a in enumerate(ids)
        for b in ids[i + 1:]
    )
    gaps = inter_component_gap_distances(
        skeleton, voxel_size_zyx=voxel_size_zyx, z_distance_weight=z_weight
    )

    assert np.array_equal(gaps, np.asarray(expected))


def test_gap_distances_query_each_tree_once(monkeypatch):
    """One batched query per component tree, not one per pair."""
    from scipy import spatial

    skeleton = _fragments(1)
    n = label(skeleton, structure=generate_binary_structure(3, 3))[1]
    calls = []
    real = spatial.cKDTree.query

    class Counting(spatial.cKDTree):
        def query(self, *args, **kwargs):
            calls.append(1)
            return real(self, *args, **kwargs)

    monkeypatch.setattr(spatial, "cKDTree", Counting)
    gaps = inter_component_gap_distances(skeleton)

    assert n > 20
    assert len(gaps) == n * (n - 1) // 2
    assert len(calls) == n - 1


def test_gap_distances_are_the_same_in_parallel(monkeypatch):
    skeleton = _fragments(2)
    serial = inter_component_gap_distances(skeleton, voxel_size_zyx=(1.3, 0.4, 0.4))
    monkeypatch.setattr(skeleton_mod, "_PARALLEL_QUERY_POINTS", 1)

    assert np.array_equal(
        inter_component_gap_distances(skeleton, voxel_size_zyx=(1.3, 0.4, 0.4)), serial
    )


# --- closing and gap bridging reach physical distances -----------------------

_ANISOTROPIC_ZYX = (2.0, 0.5, 0.5)


def _two_lines(offset_axis: int, offset_voxels: int) -> np.ndarray:
    volume = np.zeros((20, 60, 60), dtype=bool)
    volume[5, 30, 5:55] = True
    index = [5, 30]
    index[offset_axis] += offset_voxels
    volume[index[0], index[1], 5:55] = True
    return volume


@pytest.mark.parametrize("axis, voxels", [(0, 5), (1, 20)])
def test_two_vessels_10_um_apart_stay_apart_along_any_axis(axis, voxels):
    """Regression (audit): closing and gap bridging counted voxels on every
    axis alike, so two vessels 10 um apart in z -- five 2 um slices -- were
    fused into one, while the same 10 um in-plane (twenty 0.5 um voxels)
    kept them apart."""
    result = preprocess_skeleton_for_graph(
        _two_lines(axis, voxels), **{**_PIPELINE_DEFAULTS, "max_bridge_distance": 0},
        voxel_size_zyx=_ANISOTROPIC_ZYX,
    )

    assert _count(result) == 2


def test_closing_reaches_a_physical_radius_on_every_axis():
    from haemolynx.preprocessing.skeleton import close_binary_mask

    # Solid slabs: closing fills a gap between thick parts, never in a line
    # one voxel thin (its erosion needs the whole footprint inside).
    in_plane = np.zeros((9, 9, 21), dtype=bool)
    in_plane[:, :, :10] = in_plane[:, :, 11:] = True  # a 0.5 um gap along x
    along_z = np.zeros((21, 9, 9), dtype=bool)
    along_z[:10] = along_z[11:] = True  # a 2 um gap along z

    closed_x = close_binary_mask(in_plane, 2, voxel_size_zyx=_ANISOTROPIC_ZYX)
    closed_z = close_binary_mask(along_z, 2, voxel_size_zyx=_ANISOTROPIC_ZYX)

    assert closed_x[4, 4, 10]  # 0.5 um is within the 1 um radius
    assert not closed_z[10, 4, 4]  # 2 um is not
    # Without a voxel size every voxel is a cube, as before.
    assert close_binary_mask(along_z, 2)[10, 4, 4]


@pytest.mark.parametrize("gap", [2, 5])
def test_gap_bridging_reaches_a_physical_distance_on_the_low_ram_path_too(tmp_path, gap):
    from haemolynx.preprocessing.skeleton import bridge_gaps

    volume = np.zeros((12, 30, 30), dtype=bool)
    volume[6, 15, 3:27] = True

    in_ram = bridge_gaps(volume, gap, voxel_size_zyx=_ANISOTROPIC_ZYX)
    on_disk = bridge_gaps(
        volume, gap, voxel_size_zyx=_ANISOTROPIC_ZYX, use_memmap=True, memmap_directory=tmp_path
    )

    assert np.array_equal(np.asarray(on_disk), in_ram)
    # gap x 0.5 um reaches floor(gap / 4) slices in z and gap voxels in y.
    assert in_ram[6, 15 + gap, 15] and not in_ram[6, 15 + gap + 1, 15]
    assert in_ram[6 + gap // 4, 15, 15] and not in_ram[6 + gap // 4 + 1, 15, 15]


def test_a_short_piece_is_judged_by_its_length_in_microns():
    """Regression (audit): the size filter counted voxels, so three voxels
    along a 2 um z (6 um) went the same way as three along a 0.5 um x
    (1.5 um). At a minimum of 5 finest-axis voxels (2.5 um), the z piece is a
    vessel and the x piece is not."""
    from haemolynx.preprocessing.skeleton import drop_small_components

    volume = np.zeros((20, 20, 40), dtype=bool)
    volume[2:5, 5, 5] = True  # 3 voxels along z: 6 um
    volume[10, 10, 5:8] = True  # 3 voxels along x: 1.5 um
    volume[10, 15, 10:30] = True  # a long piece, kept either way

    kept = drop_small_components(volume, min_size=5, connectivity=3, voxel_size_zyx=(2.0, 0.5, 0.5))

    assert kept[2:5, 5, 5].all()
    assert not kept[10, 10, 5:8].any()
    assert kept[10, 15, 10:30].all()


def test_on_cube_voxels_the_size_filter_is_the_voxel_count(tmp_path):
    from haemolynx.preprocessing.skeleton import drop_small_components
    from skimage.morphology import remove_small_objects

    rng = np.random.default_rng(3)
    volume = rng.random((16, 16, 16)) > 0.9
    expected = remove_small_objects(volume, min_size=4, connectivity=3)

    for kwargs in ({}, {"voxel_size_zyx": (1.3, 1.3, 1.3)}):
        assert np.array_equal(drop_small_components(volume, min_size=4, connectivity=3, **kwargs), expected)
    on_disk = drop_small_components(
        volume, min_size=4, connectivity=3, voxel_size_zyx=(2.0, 0.5, 0.5),
        use_memmap=True, memmap_directory=tmp_path,
    )
    in_ram = drop_small_components(volume, min_size=4, connectivity=3, voxel_size_zyx=(2.0, 0.5, 0.5))
    assert np.array_equal(np.asarray(on_disk), in_ram)


def test_a_bridge_routed_through_the_mask_measures_its_steps_in_microns():
    """Regression (audit): routed per voxel, a step along a 2 um z cost the
    same as one along a 0.5 um x. Here an obstacle sits between the bridge's
    ends in their own slice: around it in-plane is 10.5 voxel steps (5.2 um),
    over it through the next slice 8.8 steps -- but 7.1 um. The router took
    the slice above."""
    from haemolynx.preprocessing.skeleton import _bridge_path_through_mask

    mask = np.zeros((3, 9, 13), dtype=bool)
    mask[0, 1:8, :] = True
    mask[0, 2:7, 6] = False  # the obstacle, in the ends' own slice
    mask[1, 1:8, :] = True  # the slice above, open
    start, end = np.array([0, 4, 2]), np.array([0, 4, 10])

    physical = _bridge_path_through_mask(mask, start, end, np.array([4.0, 1.0, 1.0]))
    per_voxel = _bridge_path_through_mask(mask, start, end)

    assert physical is not None and set(physical[:, 0].tolist()) == {0}
    assert per_voxel is not None and 1 in set(per_voxel[:, 0].tolist())


# --- bridges held to the segmentation -----------------------------------------


def _support(mask, voxel_size_zyx=(1.0, 1.0, 1.0)):
    from haemolynx.preprocessing import MaskSupport

    return MaskSupport(mask, voxel_size_zyx)


def _broken_vessel_and_its_mask(background_voxels: int):
    """A vessel's centreline broken at x=12..14, inside a tube of mask with
    *background_voxels* of background in the middle of that break."""
    skeleton = np.zeros((5, 5, 30), dtype=bool)
    skeleton[2, 2, 2:12] = True
    skeleton[2, 2, 15:28] = True
    mask = np.zeros_like(skeleton)
    mask[1:4, 1:4, :] = True
    first = 12 + (3 - background_voxels) // 2
    mask[:, :, first:first + background_voxels] = False
    return skeleton, mask


@pytest.mark.parametrize("background_voxels, joined", [(1, True), (2, True), (3, False)])
def test_a_skeleton_bridge_may_cross_two_microns_of_background_but_not_three(
    background_voxels, joined
):
    """Regression: every tip within reach was bridged, the segmentation
    unasked. A one- or two-voxel dropout is a vessel the mask lost; three
    microns of background between two ends is two vessels."""
    skeleton, mask = _broken_vessel_and_its_mask(background_voxels)

    gated = connect_skeleton_components(
        skeleton, max_bridge_distance=5, mask_support=_support(mask)
    )

    assert (_count(gated) == 1) is joined
    assert bool(gated[2, 2, 12:15].all()) is joined
    assert _count(connect_skeleton_components(skeleton, max_bridge_distance=5)) == 1
    added = np.argwhere(np.asarray(gated) & ~skeleton)
    assert distance_transform_edt(~mask)[tuple(added.T)].max(initial=0.0) <= 1.0


def _a_tip_with_a_neighbour_across_background_and_its_own_vessel_ahead():
    """Vessel A ends at x=10 inside a mask tube running on to vessel B,
    which starts at x=18; C, a vessel along z in its own tube three voxels
    of background away, is nearer the tip and also ahead of it."""
    skeleton = np.zeros((9, 15, 30), dtype=bool)
    skeleton[4, 5, 0:11] = True  # A
    skeleton[4, 5, 18:30] = True  # B
    skeleton[0:9, 11, 14] = True  # C
    mask = np.zeros_like(skeleton)
    mask[3:6, 4:7, :] = True  # A and B's tube, y 4..6
    mask[:, 10:13, 13:16] = True  # C's tube, y 10..12: background at y 7..9
    return skeleton, mask


def test_a_tip_refused_its_nearest_target_bridges_to_the_next_one():
    skeleton, mask = _a_tip_with_a_neighbour_across_background_and_its_own_vessel_ahead()

    ungated = connect_skeleton_components(skeleton, max_bridge_distance=8)
    gated = connect_skeleton_components(
        skeleton, max_bridge_distance=8, mask_support=_support(mask)
    )

    assert ungated[4, 7:10, :].any()  # A (or B) reached across to C
    assert not gated[4, 7:10, :].any()
    assert gated[4, 5, 11:18].all()  # A joined to B along their own tube
    assert _count(gated) == 2


def _two_strands_and_a_broken_one(separated: bool):
    """Strand S1 along x at y=4 and strand S2 at y=10 or 11, broken at
    x=16..31. In one wide tube (*separated* False) bridging S2's break would
    draw a second centreline beside S1 in the same lumen; in two tubes with
    background between them it is S2's own vessel."""
    skeleton = np.zeros((13, 16, 50), dtype=bool)
    skeleton[6, 4, 2:48] = True
    y2 = 11 if separated else 10
    skeleton[6, y2, 2:16] = True
    skeleton[6, y2, 32:48] = True
    mask = np.zeros_like(skeleton)
    if separated:
        mask[1:12, 1:7, :] = True
        mask[1:12, 9:15, :] = True  # background at y 7..8
    else:
        mask[1:12, 1:15, :] = True
    return skeleton, mask


@pytest.mark.parametrize("separated", [False, True])
def test_a_skeleton_bridge_beside_a_strand_in_the_same_lumen_is_refused(separated):
    """No second, parallel centreline inside one segmented vessel -- and a
    real neighbour behind background does not stop a vessel being joined."""
    skeleton, mask = _two_strands_and_a_broken_one(separated)
    y2 = 11 if separated else 10
    settings = dict(max_bridge_distance=18, min_facing_cosine=0.95)

    ungated = connect_skeleton_components(skeleton, **settings)
    gated = connect_skeleton_components(skeleton, mask_support=_support(mask), **settings)

    assert ungated[6, y2, 16:32].all()
    assert bool(gated[6, y2, 16:32].all()) is separated


def _two_lines_in_their_own_tubes():
    """Two centrelines 6 voxels apart in y, each in its own tube, with three
    voxels of background between the tubes."""
    skeleton = np.zeros((9, 45, 40), dtype=bool)
    skeleton[4, 20, 5:35] = True
    skeleton[4, 26, 5:35] = True
    mask = np.zeros_like(skeleton)
    mask[3:6, 19:22, :] = True
    mask[3:6, 25:28, :] = True
    return skeleton, mask


@pytest.mark.parametrize("use_memmap", [False, True])
def test_closing_and_gap_filling_add_nothing_across_background(tmp_path, use_memmap):
    """Regression: closing and bridge_gaps fused two vessels 6 um apart into
    one lump, though the segmentation has three microns of background
    between them. Held to the mask (dilated by one voxel) they stay two."""
    skeleton, mask = _two_lines_in_their_own_tubes()
    settings = {
        **_PIPELINE_DEFAULTS, "max_bridge_distance": 0, "closing_radius": 2,
        "bridge_gap_size": 3, "use_memmap": use_memmap, "memmap_directory": tmp_path,
    }

    gated = preprocess_skeleton_for_graph(skeleton, segmentation_mask=mask, **settings)
    unasked = preprocess_skeleton_for_graph(
        skeleton, segmentation_mask=mask, bridge_require_mask_support=False, **settings
    )
    no_mask = preprocess_skeleton_for_graph(skeleton, **settings)

    assert _count(gated) == 2
    assert _count(unasked) == 1
    assert np.array_equal(np.asarray(unasked), np.asarray(no_mask))


@pytest.mark.parametrize("use_memmap", [False, True])
def test_a_vessel_its_mask_held_growth_thins_away_is_kept(tmp_path, use_memmap):
    """Regression: held to a 2 x 2 voxel vessel's mask (dilated by one voxel),
    closing and gap filling grew its centreline into a rod Lee thinning
    erases outright, and the vessel was gone from the skeleton."""
    skeleton = np.zeros((30, 30, 30), dtype=bool)
    skeleton[10:20, 15, 15] = True
    mask = np.zeros_like(skeleton)
    mask[10:20, 14:16, 14:16] = True
    settings = {
        **_PIPELINE_DEFAULTS, "min_branch_length": 0, "closing_radius": 2,
        "bridge_gap_size": 3, "use_memmap": use_memmap, "memmap_directory": tmp_path,
    }

    cleaned = np.asarray(preprocess_skeleton_for_graph(skeleton, segmentation_mask=mask, **settings))

    assert np.array_equal(cleaned, skeleton)


def test_bridge_gaps_adds_voxels_only_within_the_allowed_region():
    from haemolynx.preprocessing.skeleton import bridge_gaps, close_binary_mask

    volume = np.zeros((9, 20, 20), dtype=bool)
    volume[4, 10, 2:18] = True
    within = np.zeros_like(volume)
    within[:, :11] = True  # y <= 10 only

    grown = bridge_gaps(volume, 2)
    held = bridge_gaps(volume, 2, within=within)

    assert grown[4, 12, 10] and not held[4, 12, 10]
    assert held[4, 8, 10]
    assert np.array_equal(held, grown & (volume | within))
    slabs = np.zeros((9, 9, 21), dtype=bool)
    slabs[:, :, :10] = slabs[:, :, 11:] = True
    assert close_binary_mask(slabs, 2)[4, 4, 10]
    assert not close_binary_mask(slabs, 2, within=slabs)[4, 4, 10]


def _a_hub_with_two_branches():
    result = np.zeros((11, 11, 11), dtype=bool)
    result[5, 5, :] = True
    result[4:7, 4:7, 4:7] = True
    dense = np.zeros_like(result)
    dense[3:8, 3:8, 3:8] = True
    return result, dense


@pytest.mark.parametrize("accepts", [lambda c, t: True, lambda c, t: t[2] > 5, lambda c, t: False])
def test_a_hub_is_collapsed_only_when_two_of_its_links_are_accepted(accepts):
    """A link a hub could not draw would leave a branch cut off once the
    hub's window is erased, so a hub with fewer than two is left alone."""
    result, dense = _a_hub_with_two_branches()
    expected, _ = _a_hub_with_two_branches()
    skeleton_mod._collapse_hubs(expected, expected.copy(), dense, [np.array([5, 5, 5])], (5, 5, 5), 8)
    original = result.copy()

    skeleton_mod._collapse_hubs(
        result, result.copy(), dense, [np.array([5, 5, 5])], (5, 5, 5), 8, accepts
    )

    collapsed = bool(accepts(None, np.array([5, 5, 2]))) and bool(accepts(None, np.array([5, 5, 8])))
    assert np.array_equal(result, expected if collapsed else original)
    assert not np.array_equal(expected, original)


def test_a_hub_link_is_accepted_only_through_the_mask():
    mask = np.zeros((9, 12, 20), dtype=bool)
    mask[3:6, 3:6, :] = True
    mask[3:6, 9:12, :] = True  # a second tube, three voxels of background away
    mask[4, 4, 10] = False  # a one-voxel dropout
    accept = skeleton_mod._hub_link_gate(_support(mask))

    assert accept(np.array([4, 4, 2]), np.array([4, 4, 14]))
    assert not accept(np.array([4, 4, 5]), np.array([4, 10, 5]))
    assert skeleton_mod._hub_link_gate(None) is None
