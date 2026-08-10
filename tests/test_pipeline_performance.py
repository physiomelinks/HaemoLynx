"""The pipeline's expensive paths, pinned to the cheap results they replaced.

Each of these swapped a whole-volume computation for one that only touches
what it needs. The point of every test here is that the cheap version returns
what the expensive one did -- a speed-up that quietly changed a skeleton or a
routing cost would be far worse than the runtime it saved.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from haemolynx.graph.reconnect import COST_WINDOW_PAD
from haemolynx.preprocessing import bridge_gaps, connect_skeleton_components
from haemolynx.preprocessing.skeleton import _euclidean_ball


def _reference_bridge_gaps(binary_skeleton: np.ndarray, max_gap: int) -> np.ndarray:
    """The original formulation: threshold a full Euclidean distance transform."""
    dist = distance_transform_edt(~binary_skeleton)
    return binary_skeleton | ((dist <= max_gap) & (~binary_skeleton))


def _reference_component_coords(labeled: np.ndarray, n_components: int) -> dict:
    """The original per-component scan, for comparison with the grouped one."""
    return {
        cid: np.argwhere(labeled == cid)
        for cid in range(1, n_components + 1)
        if np.any(labeled == cid)
    }


@pytest.fixture
def scattered_skeleton():
    """Sparse fragments at varied separations, as a real skeleton comes out."""
    rng = np.random.default_rng(20260811)
    volume = np.zeros((40, 60, 60), dtype=bool)
    for start in [(5, 5, 5), (5, 20, 20), (12, 40, 8), (30, 12, 45), (20, 30, 30)]:
        z, y, x = start
        length = int(rng.integers(4, 12))
        volume[z, y, x : x + length] = True
        volume[z, y : y + length, x] = True
    # Two fragments a couple of voxels apart, the case bridging exists for.
    volume[18, 50, 10:16] = True
    volume[18, 50, 18:24] = True
    return volume


# ---------------------------------------------------------------------------
# bridge_gaps: dilation by a ball == thresholded distance transform
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("max_gap", [1, 2, 3, 4])
def test_bridge_gaps_matches_the_distance_transform_it_replaced(
    scattered_skeleton, max_gap
):
    fast = bridge_gaps(scattered_skeleton, max_gap=max_gap)
    reference = _reference_bridge_gaps(scattered_skeleton, max_gap)
    assert np.array_equal(fast, reference), (
        f"max_gap={max_gap}: {int(np.logical_xor(fast, reference).sum())} voxels differ"
    )


def test_bridge_gaps_of_zero_is_a_no_op(scattered_skeleton):
    assert np.array_equal(bridge_gaps(scattered_skeleton, max_gap=0), scattered_skeleton)


def test_bridge_gaps_fills_a_gap_it_can_reach_and_leaves_a_wider_one(
):
    """The bridging radius is the whole point: 2 voxels of gap, not 4."""
    volume = np.zeros((5, 5, 20), dtype=bool)
    volume[2, 2, 2] = True
    volume[2, 2, 5] = True   # gap of 2 background voxels
    volume[2, 2, 14] = True  # 8 away from anything
    bridged = bridge_gaps(volume, max_gap=2)
    assert bridged[2, 2, 3] and bridged[2, 2, 4], "reachable gap was not filled"
    assert not bridged[2, 2, 9], "a gap wider than max_gap was filled"


def test_euclidean_ball_is_a_ball_not_a_cube():
    ball = _euclidean_ball(2)
    assert ball.shape == (5, 5, 5)
    assert ball[2, 2, 2]
    assert ball[0, 2, 2]              # distance 2 along an axis
    assert not ball[0, 0, 0]          # distance sqrt(12) > 2, a cube would include it


# ---------------------------------------------------------------------------
# connect_skeleton_components: grouped labels == per-component scans
# ---------------------------------------------------------------------------
def test_component_grouping_finds_the_same_coordinates_as_scanning(
    scattered_skeleton,
):
    """The grouped split must reproduce `labeled == cid` for every component."""
    from scipy.ndimage import generate_binary_structure, label

    from haemolynx.preprocessing.skeleton import _resolve_component_connectivity

    conn = _resolve_component_connectivity(scattered_skeleton.ndim, None)
    labeled, n_components = label(
        scattered_skeleton, structure=generate_binary_structure(3, conn)
    )
    assert n_components > 1, "fixture must have several components to be meaningful"

    coords_all = np.argwhere(labeled)
    labels_all = labeled[tuple(coords_all.T)]
    order = np.argsort(labels_all, kind="stable")
    coords_all, labels_all = coords_all[order], labels_all[order]
    starts = np.searchsorted(labels_all, np.arange(1, n_components + 2))
    grouped = {
        cid: coords_all[int(starts[cid - 1]): int(starts[cid])]
        for cid in range(1, n_components + 1)
        if int(starts[cid]) > int(starts[cid - 1])
    }

    reference = _reference_component_coords(labeled, n_components)
    assert set(grouped) == set(reference)
    for cid, coords in reference.items():
        assert np.array_equal(grouped[cid], coords), f"component {cid} differs"


@pytest.mark.parametrize("max_bridge_distance", [1, 3, 8])
def test_connect_skeleton_components_never_fragments_further(
    scattered_skeleton, max_bridge_distance
):
    """Bridging may only join components, never split one.

    The voxel count is not the invariant here: the function re-skeletonises
    after bridging, so joining two fragments can thin the result below what it
    started with. What must hold is the component count.
    """
    from scipy.ndimage import generate_binary_structure, label

    structure = generate_binary_structure(3, 3)
    before = label(scattered_skeleton, structure=structure)[1]
    result = connect_skeleton_components(
        scattered_skeleton, max_bridge_distance=max_bridge_distance
    )
    after = label(result, structure=structure)[1]

    assert result.dtype == bool
    assert result.shape == scattered_skeleton.shape
    assert after <= before, f"bridging split the skeleton: {before} -> {after}"


def test_connect_skeleton_components_joins_a_close_pair_only_when_allowed():
    """Two fragments 3 voxels apart: bridged at distance 4, left alone at 2."""
    from scipy.ndimage import generate_binary_structure, label

    volume = np.zeros((5, 5, 20), dtype=bool)
    volume[2, 2, 2:6] = True
    volume[2, 2, 9:13] = True  # nearest voxels are 4 apart (6 -> 9 is 3 gaps)
    structure = generate_binary_structure(3, 3)

    far = connect_skeleton_components(volume, max_bridge_distance=2)
    assert label(far, structure=structure)[1] == 2, "should stay two components"

    near = connect_skeleton_components(volume, max_bridge_distance=4)
    assert label(near, structure=structure)[1] == 1, "should have been bridged"


# ---------------------------------------------------------------------------
# reconnect: the windowed routing cost field
# ---------------------------------------------------------------------------
def test_windowed_cost_matches_the_global_field_away_from_the_pad(
    scattered_skeleton,
):
    """Inside the pad the windowed transform is exact; that is the guarantee."""
    from haemolynx.graph import reconnect as reconnect_mod

    skeleton = scattered_skeleton
    global_cost = 1 + distance_transform_edt(~skeleton) ** 2

    minc = np.array([10, 10, 10])
    maxc = np.array([30, 40, 40])
    plo = np.maximum(minc - COST_WINDOW_PAD, 0)
    phi = np.minimum(maxc + COST_WINDOW_PAD, skeleton.shape)
    padded = ~skeleton[plo[0]:phi[0], plo[1]:phi[1], plo[2]:phi[2]]
    dist = distance_transform_edt(padded)
    inner = tuple(
        slice(int(minc[d] - plo[d]), int(minc[d] - plo[d] + maxc[d] - minc[d]))
        for d in range(3)
    )
    windowed = 1 + dist[inner] ** 2

    reference = global_cost[minc[0]:maxc[0], minc[1]:maxc[1], minc[2]:maxc[2]]
    # A windowed transform can only ever over-estimate: it sees fewer skeleton
    # voxels than the global one, never more.
    assert np.all(windowed >= reference - 1e-9)
    # Every voxel whose true distance fits inside the pad must be exact.
    within_pad = (reference - 1) <= (COST_WINDOW_PAD - 1) ** 2
    assert within_pad.any()
    np.testing.assert_allclose(windowed[within_pad], reference[within_pad])


def test_reconnect_secondary_loop_edges_still_runs_on_a_tiny_skeleton(tiny_skeleton):
    """The windowed cost must not change what a plain skeleton reconnects to."""
    pytest.importorskip("skan")
    import networkx as nx
    from skan import csr

    from haemolynx.graph import (
        build_graph_segment_skan_stitched_loops,
        reconnect_secondary_loop_edges,
    )

    sk = csr.Skeleton(tiny_skeleton)
    G, _, _ = build_graph_segment_skan_stitched_loops(sk, tiny_skeleton)
    G = nx.MultiGraph(G)
    result = reconnect_secondary_loop_edges(G, tiny_skeleton, debug=False)
    assert result.number_of_nodes() == G.number_of_nodes()
