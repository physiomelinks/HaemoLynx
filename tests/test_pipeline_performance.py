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
from haemolynx.preprocessing import (
    bridge_gaps,
    connect_skeleton_components,
    fill_binary_holes,
)
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


# ---------------------------------------------------------------------------
# fill_binary_holes: labelled background == scipy's inward flood fill
# ---------------------------------------------------------------------------
@pytest.fixture
def volume_with_a_hole():
    """A shell around a cavity, plus a dent that opens to the outside."""
    volume = np.zeros((12, 12, 12), dtype=bool)
    volume[3:8, 3:8, 3:8] = True
    volume[4:7, 4:7, 4:7] = False       # a sealed cavity: must be filled
    volume[2, 2, 2] = True              # an isolated speck
    volume[9:12, 9, 9] = True           # touches the border
    return volume


def test_fill_binary_holes_matches_scipy(volume_with_a_hole):
    from scipy.ndimage import binary_fill_holes

    ours = fill_binary_holes(volume_with_a_hole)
    theirs = binary_fill_holes(volume_with_a_hole)
    assert np.array_equal(ours, theirs), (
        f"{int(np.logical_xor(ours, theirs).sum())} voxels differ from scipy"
    )
    # And it did something, so the comparison is not vacuous.
    assert ours.sum() > volume_with_a_hole.sum()


def test_fill_binary_holes_fills_a_cavity_and_leaves_the_outside_alone(
    volume_with_a_hole,
):
    filled = fill_binary_holes(volume_with_a_hole)
    assert filled[5, 5, 5], "the sealed cavity was not filled"
    assert not filled[0, 0, 0], "background reaching the border was filled"
    assert not filled[10, 10, 10], "background reaching the border was filled"


def test_fill_binary_holes_leaves_a_cavity_that_leaks_to_the_border(
    volume_with_a_hole,
):
    """One voxel of the shell removed, and the cavity is no longer a hole."""
    from scipy.ndimage import binary_fill_holes

    leaky = volume_with_a_hole.copy()
    leaky[3, 5, 5] = False              # punch a face-connected channel out
    ours = fill_binary_holes(leaky)
    assert np.array_equal(ours, binary_fill_holes(leaky))
    assert not ours[5, 5, 5], "a cavity with a way out was filled anyway"


def test_fill_binary_holes_on_a_sparse_skeleton_changes_nothing(scattered_skeleton):
    """A thin 3D curve encloses no volume, so this is the pipeline's real case."""
    from scipy.ndimage import binary_fill_holes

    ours = fill_binary_holes(scattered_skeleton)
    assert np.array_equal(ours, binary_fill_holes(scattered_skeleton))


# ---------------------------------------------------------------------------
# log_skeleton_connectivity_stats: counting the foreground == counting all
# ---------------------------------------------------------------------------
def test_component_sizes_from_the_foreground_match_the_whole_volume(
    scattered_skeleton,
):
    from scipy.ndimage import generate_binary_structure, label

    from haemolynx.preprocessing.skeleton import _resolve_component_connectivity

    conn = _resolve_component_connectivity(scattered_skeleton.ndim, None)
    labeled, n_components = label(
        scattered_skeleton, structure=generate_binary_structure(3, conn)
    )
    whole_volume = np.bincount(labeled.ravel())
    foreground = np.bincount(
        labeled[scattered_skeleton], minlength=n_components + 1
    )
    whole_volume[0] = foreground[0] = 0  # background is not a component
    assert np.array_equal(whole_volume, foreground)


def test_connectivity_stats_are_skipped_when_nothing_logs_them(
    scattered_skeleton, monkeypatch, caplog
):
    """Diagnostics must not label a whole stack for a message nobody reads."""
    import logging

    from haemolynx.preprocessing import skeleton as skeleton_mod

    labels_computed = []
    real_label = skeleton_mod.label

    def counting_label(*args, **kwargs):
        labels_computed.append(1)
        return real_label(*args, **kwargs)

    monkeypatch.setattr(skeleton_mod, "label", counting_label)

    logger = logging.getLogger("haemolynx.preprocessing.skeleton")
    previous = logger.level
    try:
        logger.setLevel(logging.WARNING)
        skeleton_mod.log_skeleton_connectivity_stats("quiet", scattered_skeleton)
        assert labels_computed == [], "the volume was labelled with INFO off"

        logger.setLevel(logging.INFO)
        with caplog.at_level(logging.INFO, logger="haemolynx.preprocessing.skeleton"):
            skeleton_mod.log_skeleton_connectivity_stats("loud", scattered_skeleton)
        assert labels_computed == [1], "the stats were not computed with INFO on"
        assert any("[skeleton:loud]" in record.message for record in caplog.records)
    finally:
        logger.setLevel(previous)


# ---------------------------------------------------------------------------
# Guards: where each of these stops being the cheaper way round
# ---------------------------------------------------------------------------
def test_bridge_gaps_switches_to_the_transform_for_wide_gaps(
    scattered_skeleton, monkeypatch
):
    """A ball footprint grows as the cube of the radius, so past a point
    dilating by it is slower than measuring every distance. Measured on a
    329-million-voxel stack: radius 3 is 1.7x faster than the transform,
    radius 4 is 0.8x, radius 8 is nine times slower."""
    from haemolynx.preprocessing import skeleton as skeleton_mod

    dilations, transforms = [], []
    real_dilation = skeleton_mod.binary_dilation
    real_transform = skeleton_mod.distance_transform_edt

    monkeypatch.setattr(
        skeleton_mod,
        "binary_dilation",
        lambda *a, **kw: (dilations.append(1), real_dilation(*a, **kw))[1],
    )
    monkeypatch.setattr(
        skeleton_mod,
        "distance_transform_edt",
        lambda *a, **kw: (transforms.append(1), real_transform(*a, **kw))[1],
    )

    skeleton_mod.bridge_gaps(scattered_skeleton, max_gap=skeleton_mod.MAX_BALL_DILATION_RADIUS)
    assert (len(dilations), len(transforms)) == (1, 0), "narrow gap should dilate"

    dilations.clear(), transforms.clear()
    skeleton_mod.bridge_gaps(
        scattered_skeleton, max_gap=skeleton_mod.MAX_BALL_DILATION_RADIUS + 1
    )
    assert (len(dilations), len(transforms)) == (0, 1), "wide gap should measure"


@pytest.mark.parametrize("max_gap", [3, 4, 5, 7])
def test_bridge_gaps_agrees_with_itself_on_both_sides_of_the_guard(
    scattered_skeleton, max_gap
):
    """Which branch runs must not change the answer, only the runtime."""
    assert np.array_equal(
        bridge_gaps(scattered_skeleton, max_gap=max_gap),
        _reference_bridge_gaps(scattered_skeleton, max_gap),
    )


def test_reconnect_falls_back_to_one_global_field_when_windows_get_large(
    scattered_skeleton, monkeypatch, caplog
):
    """Windowing is only cheaper while the windows are small.

    Once the windows transformed so far cover more than the volume, doing them
    one at a time has already cost what the whole-volume field costs, so the
    whole-volume field is built and sliced from then on.
    """
    import logging

    import networkx as nx

    from haemolynx.graph import reconnect as reconnect_mod

    transform_shapes = []
    real_transform = reconnect_mod.distance_transform_edt

    def counting_transform(array, *a, **kw):
        transform_shapes.append(array.shape)
        return real_transform(array, *a, **kw)

    monkeypatch.setattr(reconnect_mod, "distance_transform_edt", counting_transform)
    # A pad wider than the volume makes every window the whole volume, which is
    # the regime the guard exists for.
    monkeypatch.setattr(reconnect_mod, "COST_WINDOW_PAD", max(scattered_skeleton.shape))

    G = nx.MultiGraph()
    shape = np.array(scattered_skeleton.shape)
    for node, position in enumerate([(5, 5, 5), (5, 20, 20), (12, 40, 8), (30, 12, 45)]):
        G.add_node(node, pos=np.array(position, dtype=float))
    for u, v in [(0, 1), (1, 2), (2, 3)]:
        G.add_edge(
            u,
            v,
            voxels=[G.nodes[u]["pos"].tolist(), G.nodes[v]["pos"].tolist()],
            length=10.0,
        )

    with caplog.at_level(logging.INFO, logger="haemolynx.graph.reconnect"):
        reconnect_mod.reconnect_secondary_loop_edges(
            G, scattered_skeleton, min_length_voxels=1, debug=False, max_workers=1
        )

    if len(transform_shapes) > 1:
        # Whenever it needed more than one window, the guard must have fired and
        # every later window must come from the one whole-volume field.
        assert any(
            "transforming it once" in record.message for record in caplog.records
        ), "windows exceeded the volume but no global field was built"
        assert transform_shapes[-1] == scattered_skeleton.shape


def test_reconnect_does_not_build_a_global_field_for_small_windows(
    scattered_skeleton, monkeypatch, caplog
):
    """The common case must stay windowed -- that is the whole saving."""
    import logging

    import networkx as nx

    from haemolynx.graph import reconnect as reconnect_mod

    transform_shapes = []
    real_transform = reconnect_mod.distance_transform_edt

    def counting_transform(array, *a, **kw):
        transform_shapes.append(array.shape)
        return real_transform(array, *a, **kw)

    monkeypatch.setattr(reconnect_mod, "distance_transform_edt", counting_transform)

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([18.0, 50.0, 10.0]))
    G.add_node(1, pos=np.array([18.0, 50.0, 15.0]))
    G.add_edge(0, 1, voxels=[[18.0, 50.0, 10.0], [18.0, 50.0, 15.0]], length=5.0)

    with caplog.at_level(logging.INFO, logger="haemolynx.graph.reconnect"):
        reconnect_mod.reconnect_secondary_loop_edges(
            G, scattered_skeleton, min_length_voxels=1, debug=False, max_workers=1
        )

    assert not any("transforming it once" in r.message for r in caplog.records)
    assert all(
        shape != scattered_skeleton.shape for shape in transform_shapes
    ), "a small window should never transform the whole volume"


def test_the_window_budget_is_not_one_whole_volume(scattered_skeleton):
    """A budget of 1.0 costs more than it saves, and did.

    Windows are transformed on worker threads and the transform releases the
    GIL, so several run at once while one whole-volume transform runs alone.
    Counting raw voxels against the volume therefore calls the windows too
    expensive while they are still the cheaper option -- on the nerve stack
    they reach 1.29x the volume at about a third of the whole-volume cost, and
    a budget of 1.0 tripped there and tripled the step's runtime.
    """
    from haemolynx.graph import reconnect as reconnect_mod

    assert reconnect_mod.GLOBAL_FIELD_BUDGET_MULTIPLE > 1.29, (
        "the budget must clear the window volume a real run is measured to use"
    )
