"""Which step of graph building made a connection
(``scripts/thick_bridge_review/connection_steps.py``): a course found in a
graph, the step that made it, a bridge's course before the split, and the
skeleton layer that drew it. The E14.5 trace itself is run by hand."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from thick_bridge_review.connection_steps import (
    StepIndex,
    bridge_course,
    course_present,
    densify,
    extend_course,
    made_by,
    polyline_length,
    skeleton_graph,
    skeleton_maker,
    summary_markdown,
    trace_case,
)


def _edge(G, u, v, a, b, **attrs):
    count = int(np.ceil(np.linalg.norm(np.subtract(b, a)))) + 1
    path = [tuple(float(c) for c in q) for q in np.linspace(a, b, count)]
    G.add_node(u, pos=path[0])
    G.add_node(v, pos=path[-1])
    G.add_edge(u, v, voxels=path, length=float(np.linalg.norm(np.subtract(b, a))), **attrs)


def test_densify_keeps_the_ends_and_spaces_points_no_further_than_asked():
    points = densify([(0, 0, 0), (0, 0, 3.0), (0, 4.0, 3.0)], 0.5)
    assert np.allclose(points[0], (0, 0, 0)) and np.allclose(points[-1], (0, 4, 3))
    assert np.linalg.norm(np.diff(points, axis=0), axis=1).max() <= 0.5 + 1e-9
    assert polyline_length(points) == pytest.approx(7.0)


def test_a_course_is_there_only_where_a_centreline_runs_along_it_unbroken():
    """A vessel along the course has it; one 3 um beside it does not; two
    pieces with a 1 um break between them cover it but do not join its ends."""
    course = np.array([(10.0, 10.0, x) for x in np.arange(0.0, 10.5, 0.5)])
    along = nx.MultiGraph()
    _edge(along, 0, 1, (10, 10, -5), (10, 10, 15))
    beside = nx.MultiGraph()
    _edge(beside, 0, 1, (10, 13, -5), (10, 13, 15))
    broken = nx.MultiGraph()
    _edge(broken, 0, 1, (10, 10, -5), (10, 10, 4.5))
    _edge(broken, 2, 3, (10, 10, 5.5), (10, 10, 15))

    def present(G):
        return course_present(StepIndex(G).local(course[10]), course)

    assert present(along)
    assert not present(beside)
    assert not present(broken)
    # A detour much longer than the course does not count as joining it.
    detour = broken.copy()
    _edge(detour, 1, "far", (10, 10, 4.5), (10, 30, 4.5))
    _edge(detour, "far", 2, (10, 30, 4.5), (10, 10, 5.5))
    assert not present(detour)
    # A short join across the break does.
    joined = broken.copy()
    _edge(joined, 1, 2, (10, 10, 4.5), (10, 10, 5.5), bridge_kind="orphan")
    assert present(joined)


def test_the_step_that_made_a_course_is_the_start_of_its_last_unbroken_run():
    labels = ["skan", "reconnect", "collapse", "split"]
    assert made_by(labels, [True, True, True, True]) == "skan"
    assert made_by(labels, [False, True, True, True]) == "reconnect"
    assert made_by(labels, [True, False, True, True]) == "collapse"
    assert made_by(labels, [True, True, True, False]) is None
    assert made_by([], []) is None


def test_a_bridges_course_is_the_way_its_vessel_ran_from_the_wall_to_the_centreline():
    """A vessel crossing the wall at x = 20 and running on to the thick
    centreline at x = 10, through a node at x = 15: its course is the way
    from the wall to the centreline, over both edges. The graph's own
    centreline, 3 um off the tree, is where the bridge ends."""
    G = nx.MultiGraph()
    _edge(G, 0, 1, (10, 20, 40), (10, 20, 15))
    _edge(G, 1, 2, (10, 20, 15), (10, 20, 13))
    _edge(G, "T0", "T1", (10, 0, 13), (10, 40, 13))
    _edge(G, 2, "T0", (10, 20, 13), (10, 0, 13))
    local = StepIndex(G).local((10, 20, 20))
    course = bridge_course(local, (10, 20, 20), (10, 20, 13))
    assert np.linalg.norm(course[0] - (10, 20, 20)) <= 2.0
    assert np.linalg.norm(course[-1] - (10, 20, 13)) <= 2.0
    assert np.allclose(course[:, :2], (10, 20))
    # A way in longer than the limit is another vessel's.
    assert bridge_course(local, (10, 20, 20), (10, 20, 13), max_length_um=2.0) is None
    # No centreline through the wall point: the split drew the bridge.
    G = nx.MultiGraph()
    _edge(G, 0, 1, (10, 30, 40), (10, 30, 25))
    _edge(G, "T0", "T1", (10, 0, 13), (10, 40, 13))
    assert bridge_course(StepIndex(G).local((10, 20, 20)), (10, 20, 20), (10, 20, 13)) is None


def _line(z, y, xs):
    return np.array([(z, y, x) for x in xs])


def test_the_skeleton_step_that_drew_a_course_is_the_first_whose_skeleton_joins_it():
    """A course along x from 0 to 10: Lee's arm stops at x = 4, the tree
    starts at x = 9. The skeleton that first runs unbroken along it drew
    it; a strand 1 voxel beside it, moved by the clean-up's thinning, still
    counts, a strand 4 voxels off does not."""
    course = _line(5.0, 5.0, np.arange(0.0, 10.5, 0.5))
    lee = _line(5, 5, range(-5, 5))
    tree = _line(5, 5, range(9, 20))
    join = _line(5, 6, range(5, 9))                     # drawn a voxel off the line
    gap_filled = _line(5, 5, range(-5, 20))

    def maker(final, after, before, lee_only):
        skeletons = {name: skeleton_graph(np.vstack(v) if v else np.zeros((0, 3), int), (1.0, 1.0, 1.0))
                     for name, v in (("final", final), ("after_join", after), ("before_join", before),
                                     ("lee", lee_only))}
        return skeleton_maker(course, skeletons)

    assert maker([gap_filled], [lee, join, tree], [lee, tree], [lee]) == ("join", "final=1 after_join=1 before_join=0 lee=0")
    assert maker([gap_filled], [lee, tree], [lee, tree], [lee])[0] == "cleanup"
    assert maker([gap_filled], [gap_filled], [gap_filled], [gap_filled])[0] == "lee"
    assert maker([gap_filled], [gap_filled], [gap_filled], [lee])[0] == "tree"
    assert maker([lee, tree], [lee, tree], [lee, tree], [lee])[0] == "none"
    beside = _line(5, 9, range(-5, 20))
    assert maker([gap_filled], [beside], [beside], [beside])[0] == "cleanup"


def test_a_short_course_is_carried_on_along_its_own_vessel():
    G = nx.MultiGraph()
    _edge(G, 0, 1, (10, 10, 0), (10, 10, 40))
    _edge(G, 2, 3, (10, 0, 20), (10, 40, 20))
    local = StepIndex(G).local((10, 10, 20))
    course = extend_course(local, np.array([(10.0, 10.0, 19.0), (10.0, 10.0, 21.0)]), 3.0)
    assert polyline_length(course) == pytest.approx(8.0, abs=0.6)
    assert np.allclose(course[:, :2], (10, 10))


def test_every_bridge_becomes_a_case_with_the_lumen_radius_at_its_inner_end():
    """A thick region 5 um in radius along y and one bridge into its
    centreline: one case, read at the region's radius deep inside."""
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    from thick_bridge_review.connection_steps import bridge_cases

    p = np.stack(np.indices((30, 40, 30)), axis=-1).astype(float)
    thick = np.linalg.norm(p[..., [0, 2]] - (15.0, 15.0), axis=-1) <= 5.0
    G = nx.MultiGraph()
    _edge(G, "T0", "T1", (15, 0, 15), (15, 39, 15))
    _edge(G, "O", "C", (15, 20, 21), (15, 20, 15), **{IS_ZERO_RESISTANCE: True})
    (case,) = bridge_cases(G, thick, (1.0, 1.0, 1.0))
    assert case["kind"] == "bridge" and case["edge"][:2] in (("O", "C"), ("C", "O"))
    assert case["local_radius_um"] == pytest.approx(5.0, abs=0.5)
    assert len(case["piece_um"]) >= 13


def _case(id_, kind, piece, radius=6.0):
    return {"id": id_, "kind": kind, "vox": [0, 0, 0], "piece_um": [list(map(float, p)) for p in piece],
            "local_radius_um": radius}


def test_a_case_is_traced_to_the_step_that_joined_it_or_to_the_split():
    """A thick centreline along y at x = 10, a vessel coming in from x = 40.
    A near-miss piece the reconnect drew is put down to the reconnect; a
    bridge whose vessel ran to the centreline from the skeleton on is put
    down to the skeleton layer that drew that way in; a bridge the split
    drew where no vessel ran to the centreline is put down to the split."""
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    tree = np.array([(10.0, y, 10.0) for y in range(0, 61)])
    spine = nx.MultiGraph()
    _edge(spine, "T0", "T1", (10, 0, 10), (10, 60, 10))
    _edge(spine, "V0", "V1", (10, 40, 40), (10, 40, 22))

    skan = spine.copy()
    _edge(skan, "V1", "C", (10, 40, 22), (10, 40, 10))           # the skeleton's way in
    reconnect = skan.copy()
    _edge(reconnect, "V0", "W", (10, 40, 40), (10, 50, 40), bridge_kind="orphan")
    split_input = reconnect.copy()
    final = split_input.copy()
    final.remove_edge("V1", "C")
    _edge(final, "V1", "C", (10, 40, 16), (10, 40, 10), **{IS_ZERO_RESISTANCE: True})
    _edge(final, "V0", "V1b", (10, 20, 40), (10, 20, 16))
    _edge(final, "V1b", "C2", (10, 20, 16), (10, 20, 10), **{IS_ZERO_RESISTANCE: True})

    steps = [(label, StepIndex(G)) for label, G in (
        ("build_graph_segment_skan_stitched_loops", skan),
        ("reconnect_secondary_loop_edges", reconnect),
        ("finish_thick_split", final),
    )]
    lee = np.array([(10, 40, x) for x in range(22, 41)])
    before = np.vstack([tree.astype(int), lee])
    after = np.vstack([before, [(10, 40, x) for x in range(11, 22)]])
    layers = {"final": after, "after_join": after, "before_join": before, "lee": lee}
    args = (steps, StepIndex(split_input), StepIndex(final), tree, layers, (1.0, 1.0, 1.0))

    near_miss = trace_case(_case(1, "near_miss", [(10, 40, 40), (10, 50, 40)]), *args)
    assert near_miss["now"] == "present" and near_miss["made_by"] == "reconnect_secondary_loop_edges"
    assert near_miss["present"] == "011"

    bridge = trace_case(_case(2, "bridge", [(10, 40, 16), (10, 40, 10)]), *args)
    assert bridge["now"] == "bridge"
    assert bridge["made_by"] == "build_graph_segment_skan_stitched_loops"
    assert bridge["skeleton"] == "join"

    drawn = trace_case(_case(3, "bridge", [(10, 20, 16), (10, 20, 10)]), *args)
    assert drawn["now"] == "bridge" and drawn["made_by"] == "finish_thick_split"

    gone = trace_case(_case(4, "bridge", [(10, 5, 16), (10, 5, 30)]), *args)
    assert gone["now"] == "gone"

    table = summary_markdown([near_miss, bridge, drawn, gone])
    assert "| skeleton: join | 1 | 2 b (0, 0, 0) |" in table
    assert "| gone at this commit | 1 |" in table
