"""Click -> graph element resolution, without a napari viewer.

`hit_test_vessels`/`hit_test_nodes` are pure functions over plain arrays and
dicts, standing in for a Vectors layer's `.data`/`.features` and a Points
layer's native `get_value(...)` result -- so the "Edit" window's click
handling is testable without `make_napari_viewer` (which crashes in this
sandbox, see project memory).
"""
from __future__ import annotations

import numpy as np
import pytest

from haemolynx.gui.graph_click import EdgeHit, NodeHit, hit_test_nodes, hit_test_vessels


def _one_segment_vectors(origin, direction):
    """A single-segment (1, 2, 3) Vectors array, like napari's own layer.data."""
    return np.array([[list(origin), list(direction)]], dtype=float)


def test_hit_test_vessels_resolves_the_edge_and_the_closest_point():
    data = _one_segment_vectors((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    features = {"u": [1], "v": [2], "key": [0]}

    hit = hit_test_vessels(data, features, position=(3.0, 1.0, 0.0))

    assert hit == EdgeHit(u=1, v=2, key=0, point_um=(3.0, 0.0, 0.0))


def test_hit_test_vessels_clamps_to_the_segment_ends():
    data = _one_segment_vectors((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    features = {"u": [1], "v": [2], "key": [0]}

    # Past the far end: closest point is the end, not an extrapolation.
    # max_distance raised so this isolates clamping from the pickup radius,
    # which test_hit_test_vessels_returns_none_outside_the_pickup_radius
    # already covers.
    hit = hit_test_vessels(data, features, position=(50.0, 0.0, 0.0), max_distance=50.0)
    assert hit is not None
    assert hit.point_um == (10.0, 0.0, 0.0)

    # Before the start.
    hit = hit_test_vessels(data, features, position=(-50.0, 0.0, 0.0), max_distance=50.0)
    assert hit is not None
    assert hit.point_um == (0.0, 0.0, 0.0)


def test_hit_test_vessels_returns_none_outside_the_pickup_radius():
    data = _one_segment_vectors((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    features = {"u": [1], "v": [2], "key": [0]}

    hit = hit_test_vessels(data, features, position=(3.0, 100.0, 0.0), max_distance=2.0)
    assert hit is None


def test_hit_test_vessels_picks_the_nearer_of_two_segments():
    data = np.array(
        [
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[0.0, 5.0, 0.0], [10.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    features = {"u": ["a", "b"], "v": ["c", "d"], "key": [0, 1]}

    hit = hit_test_vessels(data, features, position=(5.0, 4.5, 0.0))
    assert hit is not None
    assert (hit.u, hit.v, hit.key) == ("b", "d", 1)


def test_hit_test_vessels_returns_none_with_no_vectors():
    hit = hit_test_vessels(np.zeros((0, 2, 3)), {"u": [], "v": [], "key": []}, position=(0, 0, 0))
    assert hit is None


def test_hit_test_nodes_resolves_the_node_id():
    features = {"node_id": [10, 20, 30]}
    assert hit_test_nodes(1, features) == NodeHit(node_id=20)


def test_hit_test_nodes_unpacks_a_tuple_result():
    # Some napari layer.get_value(...) calls return (index, ...) tuples.
    features = {"node_id": [10, 20, 30]}
    assert hit_test_nodes((2, "extra"), features) == NodeHit(node_id=30)


@pytest.mark.parametrize("result", [None, "not-an-index", -1, 5])
def test_hit_test_nodes_returns_none_for_a_miss_or_bad_index(result):
    features = {"node_id": [10, 20, 30]}
    assert hit_test_nodes(result, features) is None
