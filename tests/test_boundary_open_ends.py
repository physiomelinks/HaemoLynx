"""Position-only boundary methods choose open ends, not dead ends.

``edge_percent`` and ``degree_1_from_inlet`` pick terminals by where they sit,
and used to take any terminal there -- including a vessel's dead end inside
the tissue (a segmentation gap, most often), which then held a boundary
pressure and pushed flow in or out where none enters the tissue. They now take
only terminals near an image face, where the image cut a vessel.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph.boundaries import (
    open_terminal_nodes,
    select_boundary_nodes_by_method,
    select_boundary_nodes_for_role,
)

IMAGE_SHAPE = (51, 51, 101)  # 50 x 50 x 100 um at 1 um voxels


def _network():
    """A trunk along x from face to face, with a branch ending inside the
    tissue near the low-x end -- in the first 20% band, 15 um from any face."""
    G = nx.MultiGraph(voxel_size=(1.0, 1.0, 1.0))
    positions = {
        "low_face": (25, 25, 1),
        "junction_a": (25, 25, 30),
        "junction_b": (25, 25, 70),
        "high_face": (25, 25, 99),
        "dead_end": (25, 30, 15),
        "side_face": (49, 25, 70),
    }
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v in (("low_face", "junction_a"), ("junction_a", "junction_b"),
                 ("junction_b", "high_face"), ("junction_a", "dead_end"),
                 ("junction_b", "side_face")):
        G.add_edge(u, v)
    return G


def test_open_terminal_nodes_are_the_terminals_near_a_face():
    G = _network()

    assert set(open_terminal_nodes(G, IMAGE_SHAPE, max_distance_um=10.0)) == {
        "low_face", "high_face", "side_face"
    }


def test_open_terminal_nodes_say_nothing_without_a_voxel_size():
    G = _network()
    G.graph.pop("voxel_size")

    assert open_terminal_nodes(G, IMAGE_SHAPE, max_distance_um=10.0) is None


def test_edge_percent_leaves_out_a_dead_end_in_its_band():
    """Regression: the dead end sat in the first 20% of the terminals' span
    along x and became an inlet beside the real one."""
    G = _network()

    before = select_boundary_nodes_by_method(
        G, IMAGE_SHAPE, method="edge_percent", node_role="inlet",
        edge_percent=20.0, end_percent=20.0, axis=2, open_end_max_distance_um=None,
    )
    inlets = select_boundary_nodes_by_method(
        G, IMAGE_SHAPE, method="edge_percent", node_role="inlet",
        edge_percent=20.0, end_percent=20.0, axis=2,
    )

    assert set(before) == {"low_face", "dead_end"}
    assert inlets == ["low_face"]


def test_degree_1_from_inlet_leaves_out_dead_ends():
    """Regression: every terminal but the inlet became an outlet."""
    G = _network()

    outlets = select_boundary_nodes_by_method(
        G, IMAGE_SHAPE, method="degree_1_from_inlet", node_role="outlet",
        inlet_nodes_for_distance=["low_face"],
    )

    assert set(outlets) == {"high_face", "side_face"}


def test_a_band_with_no_open_end_keeps_its_terminals_and_warns(caplog):
    """No boundary at all would fail the run later, with less to go on."""
    G = _network()

    with caplog.at_level(logging.WARNING, logger="haemolynx.graph.boundaries"):
        inlets = select_boundary_nodes_by_method(
            G, IMAGE_SHAPE, method="edge_percent", node_role="inlet",
            edge_percent=20.0, end_percent=20.0, axis=2, open_end_max_distance_um=0.5,
        )

    assert set(inlets) == {"low_face", "dead_end"}
    assert any("interior dead ends" in record.getMessage() for record in caplog.records)


def test_coordinates_are_taken_as_placed_even_at_a_dead_end():
    """A point the user placed is a decision, not a guess to second-guess."""
    G = _network()

    chosen = select_boundary_nodes_by_method(
        G, IMAGE_SHAPE, method="coordinates", node_role="inlet", coordinates=[(25, 30, 15)],
    )

    assert chosen == ["dead_end"]


@pytest.mark.parametrize("value, expected", [(10.0, ["low_face"]), (None, ["dead_end", "low_face"])])
def test_the_setting_reaches_the_selector_and_empty_turns_it_off(value, expected):
    G = _network()
    settings = {
        "inlet_node_selection_method": "edge_percent",
        "boundary_axis": 2,
        "boundary_first_percent": 20.0,
        "boundary_last_percent": 20.0,
        "boundary_open_end_max_distance_um": value,
    }

    assert sorted(select_boundary_nodes_for_role(G, IMAGE_SHAPE, settings, "inlet")) == expected


def test_open_terminal_nodes_read_the_faces_on_three_distinct_spacings():
    """Every test above runs on 1 um voxels. On (2.0, 0.5, 0.4) um a 41-voxel
    stack is 80 um deep in z but 16 um across x: a terminal 2 um from the
    high-x face is an open end, one 2 um short of 16 um in z is not. Read
    x-first, the voxel size makes the stack 16 um deep in z and swaps the
    two."""
    G = nx.MultiGraph(voxel_size=(2.0, 0.5, 0.4))
    positions = {
        "hub": (8.0, 10.0, 8.0),
        "near_high_x": (8.0, 10.0, 14.0),  # 2 um from x = 16 um; 8 um from every other face
        "z_14": (14.0, 10.0, 8.0),  # 14 um deep in an 80 um stack; 8 um from the x faces
        "near_high_z": (79.0, 10.0, 8.0),  # 1 um from z = 80 um
    }
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for node in ("near_high_x", "z_14", "near_high_z"):
        G.add_edge("hub", node)

    open_ends = open_terminal_nodes(G, (41, 41, 41), max_distance_um=5.0)

    assert sorted(open_ends) == ["near_high_x", "near_high_z"]


def _kinds_marked(G):
    """The kinds graph building marks: both x tips leave the image, while the
    side tip 1 um from the high-z face and the branch inside the tissue stop
    short."""
    from haemolynx.graph.dead_ends import DEAD_END_KIND, LEAVES_IMAGE, STOPS_SHORT

    G.nodes["low_face"][DEAD_END_KIND] = LEAVES_IMAGE
    G.nodes["high_face"][DEAD_END_KIND] = LEAVES_IMAGE
    G.nodes["side_face"][DEAD_END_KIND] = STOPS_SHORT
    G.nodes["dead_end"][DEAD_END_KIND] = STOPS_SHORT
    return G


@pytest.mark.parametrize(
    "open_end_test, expected",
    [("leaves_image", ["low_face"]), (None, ["dead_end", "low_face"])],
    ids=["leaves_image", "distance"],
)
def test_the_leaves_image_test_reaches_the_selector_from_the_settings(open_end_test, expected):
    """boundary_open_end_test=leaves_image takes the dead ends' marked kinds,
    not the distance: within 20 um of a face the dead end 15 um from the
    low-x face is an open end by distance, but graph building marked it as
    stopping short, so only the low-x tip is an inlet."""
    G = _kinds_marked(_network())
    settings = {
        "inlet_node_selection_method": "edge_percent",
        "boundary_axis": 2,
        "boundary_first_percent": 20.0,
        "boundary_last_percent": 20.0,
        "boundary_open_end_max_distance_um": 20.0,
    }
    if open_end_test is not None:
        settings["boundary_open_end_test"] = open_end_test

    assert sorted(select_boundary_nodes_for_role(G, IMAGE_SHAPE, settings, "inlet")) == expected


def test_the_leaves_image_test_picks_outlets_by_kind_from_the_settings():
    """degree_1_from_inlet with leaves_image keeps the terminals marked as
    leaving the image: the high-x tip, not the side tip graph building marked
    as stopping short, though it lies 1 um from the high-z face (the distance
    test takes both)."""
    G = _kinds_marked(_network())
    settings = {
        "inlet_node_selection_method": "edge_percent",
        "outlet_node_selection_method": "degree_1_from_inlet",
        "boundary_axis": 2,
        "boundary_first_percent": 20.0,
        "boundary_last_percent": 20.0,
        "inlet_nodes": ["low_face"],
    }

    by_distance = select_boundary_nodes_for_role(
        G, IMAGE_SHAPE, settings, "outlet", exclude_nodes=["low_face"]
    )
    by_kind = select_boundary_nodes_for_role(
        G, IMAGE_SHAPE, {**settings, "boundary_open_end_test": "leaves_image"}, "outlet",
        exclude_nodes=["low_face"],
    )

    assert sorted(by_distance) == ["high_face", "side_face"]
    assert by_kind == ["high_face"]
