"""Face-restricted boundary selection.

The band method assigns arterial pressure to whichever degree-1 nodes fall in a positional
band. About 86% of degree-1 nodes in these graphs are interior skeletonisation spurs rather
than vessels crossing the region boundary, so most selected inlets are mask defects. A vessel
that enters the region must cross one of its faces; a dead end in the middle of the volume
cannot be a pressure inlet whatever its coordinate.
"""
import networkx as nx
import numpy as np
import pytest

from ImageLynx.graph.boundaries import (
    select_boundary_terminal_nodes,
    select_boundary_terminal_nodes_by_face,
)

VOX = (1.8639, 1.866, 1.866)
SHAPE = (160, 160, 160)
EXTENT = [(SHAPE[i] - 1) * VOX[i] for i in range(3)]


def _graph(positions):
    """A star: every listed position is a degree-1 terminal joined to one hub."""
    G = nx.MultiGraph()
    hub = "hub"
    G.add_node(hub, pos=np.array([EXTENT[0] / 2, EXTENT[1] / 2, EXTENT[2] / 2]))
    for name, p in positions.items():
        G.add_node(name, pos=np.asarray(p, dtype=float))
        G.add_edge(hub, name, length=1.0)
    return G


def test_terminals_on_the_low_and_high_faces_become_inlets_and_outlets():
    G = _graph({"lo": [0.0, 100.0, 100.0], "hi": [EXTENT[0], 100.0, 100.0]})
    inlets, outlets = select_boundary_terminal_nodes_by_face(
        G, SHAPE, axis=0, voxel_size=VOX)
    assert inlets == ["lo"] and outlets == ["hi"]


def test_an_interior_terminal_is_never_a_pressure_boundary():
    """The defect this exists to remove: 86% of terminals are interior spurs."""
    G = _graph({"lo": [0.0, 100.0, 100.0],
                "hi": [EXTENT[0], 100.0, 100.0],
                "spur": [EXTENT[0] * 0.05, 150.0, 150.0]})   # deep inside the band
    inlets, outlets = select_boundary_terminal_nodes_by_face(
        G, SHAPE, axis=0, voxel_size=VOX)
    assert "spur" not in inlets and "spur" not in outlets

    # The band method does select it, which is the contrast being drawn.
    band_in, _ = select_boundary_terminal_nodes(
        G, SHAPE, edge_percent=25.0, end_percent=25.0, axis=0, voxel_size=VOX)
    assert "spur" in band_in


def test_the_band_width_parameter_does_not_exist():
    """Removing it removes the 18.2% of ratio movement it accounted for."""
    import inspect

    params = inspect.signature(select_boundary_terminal_nodes_by_face).parameters
    assert "edge_percent" not in params and "end_percent" not in params


def test_tolerance_is_measured_in_voxels_and_scales_with_spacing():
    just_inside = 1.5 * VOX[0]
    G = _graph({"lo": [0.0, 100.0, 100.0],              # keeps the face non-empty
                "near": [just_inside, 100.0, 100.0],
                "hi": [EXTENT[0], 100.0, 100.0]})
    tight, _ = select_boundary_terminal_nodes_by_face(
        G, SHAPE, axis=0, voxel_size=VOX, face_tolerance_voxels=1.0)
    loose, _ = select_boundary_terminal_nodes_by_face(
        G, SHAPE, axis=0, voxel_size=VOX, face_tolerance_voxels=2.0)
    assert "near" not in tight
    assert "near" in loose


def test_a_face_node_that_is_not_a_terminal_is_not_selected():
    G = _graph({"lo": [0.0, 100.0, 100.0], "hi": [EXTENT[0], 100.0, 100.0]})
    # A vessel passing through the face rather than ending at it: degree 2, on the face.
    G.add_node("through", pos=np.array([0.0, 50.0, 50.0]))
    G.add_node("beyond", pos=np.array([20.0, 50.0, 50.0]))
    G.add_edge("hub", "beyond", length=1.0)
    G.add_edge("beyond", "through", length=1.0)
    G.add_edge("through", "hub", length=1.0)
    inlets, _ = select_boundary_terminal_nodes_by_face(
        G, SHAPE, axis=0, voxel_size=VOX)
    assert "through" not in inlets
    assert "lo" in inlets


def test_the_other_two_axes_are_available_and_give_different_sets():
    G = _graph({"z": [0.0, 100.0, 100.0], "y": [100.0, 0.0, 100.0],
                "x": [100.0, 100.0, 0.0],
                "zh": [EXTENT[0], 100.0, 100.0], "yh": [100.0, EXTENT[1], 100.0],
                "xh": [100.0, 100.0, EXTENT[2]]})
    for axis, lo, hi in ((0, "z", "zh"), (1, "y", "yh"), (2, "x", "xh")):
        inlets, outlets = select_boundary_terminal_nodes_by_face(
            G, SHAPE, axis=axis, voxel_size=VOX)
        assert inlets == [lo] and outlets == [hi], f"axis {axis}"


def test_an_empty_face_raises_rather_than_falling_back():
    """The band method silently falls back to the extreme 10% of *all* nodes.

    A fallback that quietly redefines what a boundary is turns an unsolvable region into a
    solved one with invented boundaries, which is worse than refusing.
    """
    G = _graph({"spur": [EXTENT[0] * 0.5, 100.0, 100.0]})
    with pytest.raises(ValueError, match="no terminal nodes on the"):
        select_boundary_terminal_nodes_by_face(G, SHAPE, axis=0, voxel_size=VOX)


def test_a_node_on_two_faces_is_assigned_once_and_not_to_both():
    corner = [0.0, 0.0, 100.0]
    G = _graph({"corner": corner, "hi": [EXTENT[0], 100.0, 100.0]})
    inlets, outlets = select_boundary_terminal_nodes_by_face(
        G, SHAPE, axis=0, voxel_size=VOX)
    assert inlets.count("corner") == 1
    assert "corner" not in outlets


def test_interior_terminals_can_be_tagged_for_a_robin_condition():
    G = _graph({"lo": [0.0, 100.0, 100.0], "hi": [EXTENT[0], 100.0, 100.0],
                "spur": [100.0, 100.0, 100.0]})
    select_boundary_terminal_nodes_by_face(
        G, SHAPE, axis=0, voxel_size=VOX, boundary_permeability_mode="robin_resistance")
    assert G.nodes["spur"].get("is_robin_boundary") is True
    assert "is_robin_boundary" not in G.nodes["lo"]
