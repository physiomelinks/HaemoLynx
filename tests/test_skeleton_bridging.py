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
from scipy.ndimage import generate_binary_structure, label
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
