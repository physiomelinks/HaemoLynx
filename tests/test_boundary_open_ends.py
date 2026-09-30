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
