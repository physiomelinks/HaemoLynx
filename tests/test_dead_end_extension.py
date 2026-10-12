"""Finishing the vessels a dead end cut short (graph/dead_end_extension.py,
PLAN_dead_ends.md Phases 3, 5 and 6), the prune's face protection by the one
definition, and boundary selection reading it -- each off by default until
marked grids calibrate it."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from dead_end_fixtures import CASES, JUNCTION, RADIUS, SHAPE, VOXEL_SIZE, _trunk_mask
from haemolynx.graph import dead_end_extension as dx
from haemolynx.graph import dead_ends as de
from haemolynx.preprocessing import MaskSupport
from lumen_artefact_fixtures import capsule, polyline_graph

EXTENT = de.image_extent_um(SHAPE, VOXEL_SIZE)
by_name = {make.__name__: make for make in CASES}


def _support(mask):
    return MaskSupport(mask, VOXEL_SIZE)


def test_a_dead_end_short_of_the_face_is_drawn_on_to_it():
    case = by_name["stops_short_of_face"]()
    G = case.graph
    assert dx.extend_dead_ends_to_faces(G, _support(case.mask), extent_um=EXTENT) == 1
    assert G.degree(case.tip) == 2
    (new,) = [n for n in G.neighbors(case.tip) if n != JUNCTION]
    assert G.nodes[new]["pos"][1] == pytest.approx(59.0, abs=0.6)
    assert G.edges[case.tip, new, 0]["bridge_kind"] == dx.EXTENDED_TO_FACE
    ends = de.classify_dead_ends(G, extent_um=EXTENT, support=_support(case.mask))
    assert ends[new].kind == de.LEAVES_IMAGE
    # One already on the face, or not leaving the image, is left alone.
    for name in ("leaves_through_face", "interior_near_face", "blind_sprout"):
        other = by_name[name]()
        assert dx.extend_dead_ends_to_faces(other.graph, _support(other.mask), extent_um=EXTENT) == 0


def test_tracing_on_stops_where_the_mask_ends():
    case = by_name["mask_runs_on"]()  # the mask runs on 13 um past the tip, ending at y = 2.5
    G = case.graph
    counts = dx.trace_on_dead_ends(G, _support(case.mask), extent_um=EXTENT)
    assert counts["to_mask_end"] == 1 and counts["joined"] == 0
    (new,) = [n for n in G.neighbors(case.tip) if n != JUNCTION]
    assert G.nodes[new]["pos"][1] == pytest.approx(2.5, abs=0.6)
    assert G.edges[case.tip, new, 0]["bridge_kind"] == dx.TRACED_ON
    ends = de.classify_dead_ends(G, extent_um=EXTENT, support=_support(case.mask))
    assert ends[new].kind != de.STOPS_SHORT


def test_tracing_on_joins_the_vessel_the_mask_runs_into():
    mask = _trunk_mask()
    capsule(mask, JUNCTION, (10, 5, 40), RADIUS)
    capsule(mask, (10, 5, -5), (10, 5, 85), RADIUS)
    G = polyline_graph([[(10, 30, 0), JUNCTION], [JUNCTION, (10, 30, 79)], [JUNCTION, (10, 18, 40)],
                        [(10, 5, 0), (10, 5, 79)]])
    tip = (10.0, 18.0, 40.0)
    counts = dx.trace_on_dead_ends(G, _support(mask), extent_um=EXTENT)
    assert counts["joined"] == 1
    assert G.degree(tip) == 2
    (target,) = [n for n in G.neighbors(tip) if n != JUNCTION]
    assert G.degree(target) == 3
    assert G.nodes[target]["pos"][1] == pytest.approx(5.0, abs=0.6)
    assert nx.number_connected_components(G) == 1


def test_a_dead_end_is_joined_to_the_side_of_the_vessel_ahead():
    case = by_name["broken_towards_side"]()
    G = case.graph
    assert dx.join_dead_ends_to_vessel_sides(G, _support(case.mask), extent_um=EXTENT) == 1
    (target,) = [n for n in G.neighbors(case.tip) if n != JUNCTION]
    assert G.degree(target) == 3
    assert G.edges[case.tip, target, 0]["bridge_kind"] == dx.JOINED_TO_SIDE
    assert G.edges[case.tip, target, 0]["length"] == pytest.approx(9.0, abs=0.6)
    assert nx.number_connected_components(G) == 1


def test_a_neighbour_ahead_is_not_joined():
    case = by_name["broken_towards_side"]()
    G = case.graph
    far, near, mid = (10.0, 10.0, 0.0), (10.0, 10.0, 79.0), (10.0, 10.0, 60.0)
    voxels = np.asarray(G.edges[far, near, 0]["voxels"])
    G.remove_edge(far, near, 0)
    G.add_node(mid, pos=np.asarray(mid))
    i = int(np.argmin(np.abs(voxels[:, 2] - 60.0)))
    G.add_edge(far, mid, voxels=voxels[: i + 1].tolist(), length=60.0)
    G.add_edge(mid, near, voxels=voxels[i:].tolist(), length=19.0)
    G.add_edge(mid, JUNCTION, voxels=[list(mid), list(JUNCTION)], length=28.3)
    assert dx.join_dead_ends_to_vessel_sides(G, _support(case.mask), extent_um=EXTENT) == 0


def test_every_repair_is_off_in_a_default_run():
    from haemolynx.pipeline import default_schema
    from haemolynx.pipeline.stages import graph_build_arguments

    settings = default_schema().defaults()
    arguments = graph_build_arguments(settings)
    assert arguments["extend_dead_ends_to_faces"] is False
    assert arguments["trace_on_dead_ends"] is False
    assert arguments["join_dead_ends_to_vessel_sides"] is False
    assert arguments["stub_face_protection"] == "threshold"
    assert settings["boundary_open_end_test"] == "distance"


def _near_face_stub(mask_to_y: float):
    """A 4 um-radius trunk along x at y = 50 and a 6.5 um branch of radius 2
    towards the y = 59 face (under the prune's 3 radii, 12 um; protruding
    more than its own radius from the trunk), its mask ending at
    *mask_to_y*; its tip 2.5 um from the face, not touching it."""
    mask = capsule(np.zeros(SHAPE, dtype=bool), (10, 50, -5), (10, 50, 85), 4.0)
    capsule(mask, (10, 50, 40), (10, mask_to_y, 40), 2.0)
    G = polyline_graph([[(10, 50, 0), (10, 50, 40)], [(10, 50, 40), (10, 50, 79)], [(10, 50, 40), (10, 56.5, 40)]])
    return mask, G


@pytest.mark.parametrize(
    "protection, mask_to_y, kept",
    [
        ("threshold", 56.5, True),      # within its 12 um threshold of the face: kept, as before
        ("leaves_image", 56.5, False),  # but the vessel does not reach the face: a spur
        ("leaves_image", 70.0, True),   # the vessel runs on out through the face: kept
    ],
)
def test_the_prune_keeps_at_a_face_only_what_leaves_the_image_when_asked(protection, mask_to_y, kept):
    from haemolynx.graph.assemble import lumen_cleanup

    mask, G = _near_face_stub(mask_to_y)
    cleanup = lumen_cleanup(
        mask, VOXEL_SIZE, image_shape=SHAPE, min_stub_length=10.0, min_stub_length_radius_multiple=3.0,
        stub_face_protection=protection,
    )
    pruned = cleanup.prune(G)
    assert ((10.0, 56.5, 40.0) in pruned) is kept


def test_boundary_selection_can_read_the_dead_end_definition():
    from haemolynx.graph.boundaries import select_boundary_nodes_by_method

    G = nx.MultiGraph()
    for n, p in {0: (50, 50, 2), 1: (50, 50, 20), 2: (50, 50, 52), 3: (50, 50, 40)}.items():
        G.add_node(n, pos=np.asarray(p, dtype=float))
    G.add_edge(0, 1, length=18.0)
    G.add_edge(1, 2, length=32.0)
    G.add_edge(1, 3, length=20.0)
    G.graph["image_voxel_size_zyx"] = (1.0, 1.0, 1.0)
    G.graph["image_extent_um"] = (100.0, 100.0, 60.0)
    G.graph["voxel_size"] = (1.0, 1.0, 1.0)
    # Node 2 lies 7 um from the x face: within a 10 um distance, but the
    # graph's own verdict is that it ends inside the image.
    G.nodes[0][de.DEAD_END_KIND] = de.LEAVES_IMAGE
    G.nodes[2][de.DEAD_END_KIND] = de.UNKNOWN
    G.nodes[3][de.DEAD_END_KIND] = de.UNKNOWN
    by_distance = select_boundary_nodes_by_method(
        G, (101, 101, 61), method="edge_percent", node_role="outlet", axis=2, edge_percent=50, end_percent=50,
        open_end_max_distance_um=10.0,
    )
    by_kind = select_boundary_nodes_by_method(
        G, (101, 101, 61), method="edge_percent", node_role="inlet", axis=2, edge_percent=50, end_percent=50,
        open_end_test="leaves_image",
    )
    assert by_distance == [2]
    assert by_kind == [0]
    with pytest.raises(ValueError):
        select_boundary_nodes_by_method(G, (101, 101, 61), method="edge_percent", node_role="inlet",
                                        open_end_test="near")
