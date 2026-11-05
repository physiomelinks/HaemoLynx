"""A network cut to a box (``graph/crop.py``): vessels clipped at the faces,
each new tip leaving the image, each piece knowing where it came from."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.geometry import cumulative_lengths
from haemolynx.graph import crop_graph_to_box
from haemolynx.graph.crop import CROP_BOX, CROP_FROM, CROP_SOURCE_EDGE, CROP_SPAN
from haemolynx.graph.dead_ends import (
    DEAD_END_KIND,
    DEAD_END_KIND_SOURCE,
    LEAVES_IMAGE,
    SOURCE_CROP,
    node_leaves_image,
    open_end_nodes,
)


def _line(a, b, step=1.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
    return [list(a + (b - a) * t) for t in np.linspace(0.0, 1.0, n + 1)]


def _graph(nodes: dict, edges: list) -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.graph.update({
        "image_extent_um": (100.0, 100.0, 100.0),
        "image_voxel_size_zyx": (1.0, 1.0, 1.0),
        "open_end_conditions": {"nodes": [0]},
        "statistics_report": {"basic": {}},
    })
    for node, pos in nodes.items():
        G.add_node(node, pos=np.asarray(pos, float), pressure=1000.0)
    for u, v, voxels in edges:
        G.add_edge(
            u, v, voxels=voxels, length=float(cumulative_lengths(np.asarray(voxels))[-1]),
            diameter_um=6.0, branch_order="Art2", resistance=2.0, conductance=0.5,
            flow_abs=3.0, flow_signed=-3.0, pressure_drop=1.5, flow_solved=True,
        )
    return G


def test_a_vessel_crossing_a_face_is_cut_there_and_its_tip_leaves_the_image():
    G = _graph({0: (5, 5, 5), 1: (5, 5, 25)}, [(0, 1, _line((5, 5, 5), (5, 5, 25)))])
    H = crop_graph_to_box(G, (0, 0, 0), (10, 10, 10))
    assert set(H.nodes) == {0, 2}
    assert np.allclose(H.nodes[2]["pos"], (5, 5, 10))
    assert H.nodes[2][DEAD_END_KIND] == LEAVES_IMAGE and H.nodes[2][DEAD_END_KIND_SOURCE] == SOURCE_CROP
    (_u, _v, data), = H.edges(data=True)
    assert data["length"] == pytest.approx(5.0)
    assert data[CROP_SOURCE_EDGE] == (0, 1, 0) and data[CROP_FROM] == 0
    assert data[CROP_SPAN] == pytest.approx((0.0, 0.25))
    assert np.allclose(data["voxels"][0], (5, 5, 5)) and np.allclose(data["voxels"][-1], (5, 5, 10))
    # What the uncropped solve left, and the cut vessel's resistance, are gone.
    for name in ("resistance", "conductance", "flow_abs", "flow_signed", "pressure_drop", "flow_solved"):
        assert name not in data, name
    assert data["diameter_um"] == 6.0 and data["branch_order"] == "Art2"
    assert "pressure" not in H.nodes[0]
    assert H.graph["image_extent_um"] == (10.0, 10.0, 10.0)
    assert H.graph[CROP_BOX] == {"lo": (0.0, 0.0, 0.0), "hi": (10.0, 10.0, 10.0)}
    assert "open_end_conditions" not in H.graph and "statistics_report" not in H.graph
    assert H.graph["image_voxel_size_zyx"] == (1.0, 1.0, 1.0)
    assert node_leaves_image(H, 2) and open_end_nodes(H) == [2]


def test_the_box_is_moved_to_start_at_zero_unless_told_not_to():
    G = _graph({0: (15, 15, 15), 1: (15, 15, 35)}, [(0, 1, _line((15, 15, 15), (15, 15, 35)))])
    H = crop_graph_to_box(G, (10, 10, 10), (20, 20, 20))
    assert np.allclose(H.nodes[0]["pos"], (5, 5, 5)) and np.allclose(H.nodes[2]["pos"], (5, 5, 10))
    assert np.allclose(next(iter(H.edges(data=True)))[2]["voxels"][-1], (5, 5, 10))
    kept = crop_graph_to_box(G, (10, 10, 10), (20, 20, 20), translate=False)
    assert np.allclose(kept.nodes[2]["pos"], (15, 15, 20))
    assert kept.graph["image_extent_um"] == (10.0, 10.0, 10.0)


def test_a_vessel_wholly_inside_is_kept_whole_with_its_resistance():
    G = _graph(
        {0: (2, 2, 2), 1: (2, 2, 8), 2: (2, 2, 30), 3: (2, 2, 40)},
        [(0, 1, _line((2, 2, 2), (2, 2, 8))), (1, 2, _line((2, 2, 8), (2, 2, 30))), (2, 3, _line((2, 2, 30), (2, 2, 40)))],
    )
    H = crop_graph_to_box(G, (0, 0, 0), (10, 10, 10))
    whole = H.edges[0, 1, 0]
    assert whole["resistance"] == 2.0 and whole["length"] == G.edges[0, 1, 0]["length"]
    assert whole[CROP_SPAN] == (0.0, 1.0) and whole[CROP_FROM] == 0
    assert "flow_abs" not in whole
    # The vessel lying wholly outside goes, with its nodes.
    assert 2 not in H and 3 not in H
    assert H.number_of_edges() == 2


def test_a_vessel_leaving_and_coming_back_is_two_pieces():
    out_and_back = _line((5, 5, 5), (5, 5, 15)) + _line((5, 5, 15), (5, 8, 15))[1:] + _line((5, 8, 15), (5, 8, 5))[1:]
    G = _graph({0: (5, 5, 5), 1: (5, 8, 5)}, [(0, 1, out_and_back)])
    H = crop_graph_to_box(G, (0, 0, 0), (10, 10, 10))
    pieces = sorted(H.edges(data=True), key=lambda e: e[2][CROP_SPAN])
    assert len(pieces) == 2
    total = G.edges[0, 1, 0]["length"]  # 10 + 3 + 10
    assert [p[2][CROP_SPAN] for p in pieces] == [pytest.approx((0.0, 5 / total)), pytest.approx((18 / total, 1.0))]
    assert [p[2]["length"] for p in pieces] == [pytest.approx(5.0), pytest.approx(5.0)]
    assert pieces[0][2][CROP_FROM] == 0
    assert pieces[1][2][CROP_FROM] not in (0, 1)  # the tip it comes back in by, toward the source's u
    tips = [n for n in H if H.nodes[n].get(DEAD_END_KIND) == LEAVES_IMAGE]
    assert len(tips) == 2 and all(H.degree(n) == 1 for n in tips)


def test_a_vessel_passing_through_the_box_is_a_piece_between_two_tips():
    G = _graph({0: (5, 5, -10), 1: (5, 5, 20)}, [(0, 1, _line((5, 5, -10), (5, 5, 20)))])
    H = crop_graph_to_box(G, (0, 0, 0), (10, 10, 10))
    assert 0 not in H and 1 not in H
    (u, v, data), = H.edges(data=True)
    assert data["length"] == pytest.approx(10.0) and data[CROP_SPAN] == pytest.approx((1 / 3, 2 / 3))
    assert np.allclose(H.nodes[data[CROP_FROM]]["pos"], (5, 5, 0))
    assert all(H.nodes[n][DEAD_END_KIND] == LEAVES_IMAGE for n in (u, v))


def test_a_junction_keeps_its_arms_cut_to_the_box():
    G = _graph(
        {0: (5, 5, 5), 1: (5, 5, 1), 2: (5, 5, 30), 3: (5, 30, 5)},
        [(0, 1, _line((5, 5, 5), (5, 5, 1))), (0, 2, _line((5, 5, 5), (5, 5, 30))), (0, 3, _line((5, 5, 5), (5, 30, 5)))],
    )
    G.nodes[1][DEAD_END_KIND] = "blind"  # the run's own call on a dead end inside the box
    H = crop_graph_to_box(G, (0, 0, 0), (10, 10, 10))
    assert H.degree(0) == 3
    assert sorted(H.edges[e]["length"] for e in H.edges(keys=True)) == pytest.approx([4.0, 5.0, 5.0])
    assert H.nodes[1][DEAD_END_KIND] == "blind"
    assert len(open_end_nodes(H)) == 2


def test_a_vessel_touching_the_box_at_one_point_is_dropped():
    G = _graph({0: (5, 20, 0), 1: (5, 0, 20)}, [(0, 1, [[5, 20, 0], [5, 0, 20]])])  # meets it at (5, 10, 10)
    H = crop_graph_to_box(G, (0, 0, 0), (10, 10, 10))
    assert H.number_of_edges() == 0 and H.number_of_nodes() == 0
    # One running along a face is inside it, and kept.
    along = _graph({0: (5, 5, 20), 1: (5, 20, 10)}, [(0, 1, [[5, 5, 20], [5, 5, 10], [5, 20, 10]])])
    (_u, _v, data), = crop_graph_to_box(along, (0, 0, 0), (10, 10, 10)).edges(data=True)
    assert data["length"] == pytest.approx(5.0)


def test_a_vessel_without_a_centreline_is_cut_along_its_chord():
    G = _graph({0: (5, 5, 5), 1: (5, 5, 25)}, [])
    G.add_edge(0, 1, length=20.0, diameter_um=4.0)
    H = crop_graph_to_box(G, (0, 0, 0), (10, 10, 10))
    (_u, _v, data), = H.edges(data=True)
    assert data["length"] == pytest.approx(5.0) and np.allclose(data["voxels"], [[5, 5, 5], [5, 5, 10]])


def test_a_box_with_no_extent_is_refused():
    with pytest.raises(ValueError, match="positive extent"):
        crop_graph_to_box(_graph({}, []), (0, 0, 0), (10, 0, 10))


def test_a_synthetic_network_cropped_keeps_every_centreline_inside_the_box():
    from synthetic_fixtures import fixture_template

    from haemolynx.synthetic import GenerationOptions, generate_network

    T = fixture_template()
    G = generate_network(T, seed=2, options=GenerationOptions(scale=2.0, scale_axes="xy"))
    lo = np.array([0.0, 100.0, 100.0])
    hi = lo + np.asarray(T.extent_um)
    H = crop_graph_to_box(G, lo, hi)
    extent = np.asarray(T.extent_um)
    for _n, data in H.nodes(data=True):
        assert np.all(data["pos"] >= -1e-9) and np.all(data["pos"] <= extent + 1e-9)
    for *_e, data in H.edges(data=True):
        voxels = np.asarray(data["voxels"])
        assert np.all(voxels >= -1e-9) and np.all(voxels <= extent + 1e-9)
    tips = [n for n in H if H.nodes[n].get(DEAD_END_KIND_SOURCE) == SOURCE_CROP]
    assert tips and all(H.degree(n) == 1 for n in tips)
    assert set(tips) <= set(open_end_nodes(H))
    # Each source vessel's pieces never cover more of it than it has.
    covered: dict = {}
    for *_e, data in H.edges(data=True):
        s0, s1 = data[CROP_SPAN]
        assert 0.0 <= s0 < s1 <= 1.0 + 1e-12
        covered[data[CROP_SOURCE_EDGE]] = covered.get(data[CROP_SOURCE_EDGE], 0.0) + (s1 - s0)
    assert max(covered.values()) <= 1.0 + 1e-9
    # The length inside, measured along the uncropped centrelines.
    inside = 0.0
    for u, v, key, data in G.edges(keys=True, data=True):
        points = np.asarray(data["voxels"], float)
        mids = 0.5 * (points[1:] + points[:-1])
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        inside += float(steps[np.all((mids >= lo) & (mids <= hi), axis=1)].sum())
    assert sum(d["length"] for *_e, d in H.edges(data=True)) == pytest.approx(inside, rel=0.01)
