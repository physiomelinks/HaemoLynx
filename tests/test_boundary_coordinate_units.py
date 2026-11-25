"""A boundary coordinate is microns, and saying so is not enough.

``inlet_node_selection_method="coordinates"`` snaps every point to the
*nearest* terminal, so a point in the wrong units never fails: it selects
something, and the run solves a network whose inlets are somewhere else. The
shipped nerve config carried six such points -- voxel indices read off a
viewer, 117 to 185 um from the terminals they were meant to name, three of
them landing on the same node.

These tests pin the report that makes that visible, and the one signature that
identifies it: on an anisotropic stack the same numbers read as voxel indices
land on a terminal, and by a wide margin.
"""
from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph import BoundaryCoordinateWarning, select_boundary_nodes_for_role
from haemolynx.graph.boundaries import select_boundary_nodes_by_method

IMAGE_SHAPE = (60, 200, 200)
#: Anisotropic on purpose: z is what a voxel-index mistake gets wrong most.
VOXEL_SIZE_ZYX = (2.0, 1.0, 1.0)

#: The terminals, as voxel indices into the image above.
TERMINAL_VOXELS = {
    1: (40, 20, 20),
    2: (40, 180, 20),
    3: (10, 20, 180),
    4: (10, 180, 180),
}


def _network(voxel_size_zyx=VOXEL_SIZE_ZYX) -> nx.MultiGraph:
    """A hub with four terminals, positioned in physical microns."""
    G = nx.MultiGraph()
    scale = np.asarray(voxel_size_zyx, dtype=float) if voxel_size_zyx else np.ones(3)
    G.add_node(0, pos=np.asarray((25, 100, 100), dtype=float) * scale)
    for node_id, voxel in TERMINAL_VOXELS.items():
        G.add_node(node_id, pos=np.asarray(voxel, dtype=float) * scale)
        G.add_edge(0, node_id, length=1.0)
    if voxel_size_zyx is not None:
        G.graph["voxel_size"] = tuple(float(v) for v in voxel_size_zyx)
    return G


def _microns(voxel: tuple[float, float, float]) -> list[float]:
    return list(np.asarray(voxel, dtype=float) * np.asarray(VOXEL_SIZE_ZYX, dtype=float))


def _settings(coordinates) -> dict:
    return {
        "inlet_node_selection_method": "coordinates",
        "inlet_node_coordinates": coordinates,
        "inlet_node_volumes": [],
        "inlet_nodes": [],
    }


def test_coordinates_in_microns_select_their_terminals_without_a_warning():
    G = _network()
    coordinates = [_microns(TERMINAL_VOXELS[1]), _microns(TERMINAL_VOXELS[4])]

    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryCoordinateWarning)
        selected = select_boundary_nodes_for_role(
            G, IMAGE_SHAPE, _settings(coordinates), "inlet"
        )

    assert selected == [1, 4]


def test_voxel_indices_are_reported_as_voxel_indices():
    """The regression: these are the numbers the nerve config used to carry."""
    G = _network()
    coordinates = [list(map(float, TERMINAL_VOXELS[1])), list(map(float, TERMINAL_VOXELS[2]))]

    with pytest.warns(BoundaryCoordinateWarning) as caught:
        selected = select_boundary_nodes_for_role(
            G, IMAGE_SHAPE, _settings(coordinates), "inlet"
        )

    messages = [str(warning.message) for warning in caught]
    assert len(messages) == 2
    for message in messages:
        assert "inlet_node_coordinates[" in message
        assert "looks like a voxel index" in message
        assert "multiply it by the voxel size" in message
    # The first point is 40 voxels of z out, and still selects a terminal.
    assert selected and set(selected) <= set(TERMINAL_VOXELS)


def test_the_message_names_the_setting_of_the_role_that_was_asked_for():
    G = _network()
    settings = {
        "venule_boundary_selection_method": "coordinates",
        "venule_boundary_node_coordinates": [list(map(float, TERMINAL_VOXELS[1]))],
        "venule_boundary_node_volumes": [],
        "inlet_nodes": [],
    }

    with pytest.warns(BoundaryCoordinateWarning, match="venule_boundary_node_coordinates"):
        select_boundary_nodes_for_role(G, IMAGE_SHAPE, settings, "venule_boundary")


def test_a_point_near_its_terminal_is_not_reported():
    """Picking is accurate to a few voxels, and collapse moves a node further."""
    G = _network()
    near = np.asarray(_microns(TERMINAL_VOXELS[2]), dtype=float) + np.array([2.0, 3.0, 1.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryCoordinateWarning)
        selected = select_boundary_nodes_for_role(
            G, IMAGE_SHAPE, _settings([near.tolist()]), "inlet"
        )

    assert selected == [2]


def test_a_point_inside_the_network_but_on_no_terminal_is_reported():
    """No voxel size means no second reading to offer, but still a report."""
    G = _network(voxel_size_zyx=None)
    G.graph.pop("voxel_size", None)

    with pytest.warns(BoundaryCoordinateWarning) as caught:
        select_boundary_nodes_by_method(
            G,
            IMAGE_SHAPE,
            method="coordinates",
            node_role="inlet",
            coordinates=[[25.0, 100.0, 100.0]],
            coordinates_setting_name="inlet_node_coordinates",
        )

    message = str(caught[0].message)
    assert "from the nearest terminal node" in message
    assert "voxel index" not in message


def test_pointing_at_a_corner_of_the_image_is_left_alone():
    """"The terminal nearest this corner" is a deliberate way to use this."""
    G = _network()

    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryCoordinateWarning)
        selected = select_boundary_nodes_for_role(
            G, IMAGE_SHAPE, _settings([[0.0, 0.0, 0.0]]), "inlet"
        )

    assert selected == [1]


def test_methods_that_take_no_coordinates_report_nothing():
    G = _network()
    settings = {
        "inlet_node_selection_method": "all_degree_1",
        "inlet_node_coordinates": [],
        "inlet_node_volumes": [],
        "inlet_nodes": [],
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryCoordinateWarning)
        selected = select_boundary_nodes_for_role(G, IMAGE_SHAPE, settings, "inlet")

    assert set(selected) == set(TERMINAL_VOXELS)
