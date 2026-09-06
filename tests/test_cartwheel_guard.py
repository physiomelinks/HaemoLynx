"""Cartwheel hub detection: hand-verified geometry, no pipeline involved."""
from __future__ import annotations

import math

import numpy as np
import networkx as nx
import pytest

from haemolynx.graph.cartwheel_guard import (
    CartwheelHub,
    detect_cartwheel_hubs,
    format_cartwheel_hub_report,
    hub_radial_dispersion,
    hub_spoke_directions,
)


# --- hub_radial_dispersion: pure math -----------------------------------


def test_dispersion_of_two_opposite_vectors_is_zero():
    assert hub_radial_dispersion([np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])]) == pytest.approx(0.0)


def test_dispersion_of_identical_vectors_is_one():
    v = np.array([0.6, 0.8, 0.0])
    assert hub_radial_dispersion([v, v, v]) == pytest.approx(1.0)


def test_fewer_than_two_directions_is_not_dispersed():
    assert hub_radial_dispersion([]) == 1.0
    assert hub_radial_dispersion([np.array([1.0, 0.0, 0.0])]) == 1.0


def test_eight_evenly_spaced_directions_cancel_exactly():
    angles = np.linspace(0.0, 2 * math.pi, 8, endpoint=False)
    directions = [np.array([0.0, math.sin(a), math.cos(a)]) for a in angles]
    assert hub_radial_dispersion(directions) == pytest.approx(0.0, abs=1e-9)


# --- hub_spoke_directions: node positions vs. a voxel centreline --------


def _straight_chain(*positions: tuple[float, float, float]) -> nx.MultiGraph:
    G = nx.MultiGraph()
    for i, pos in enumerate(positions):
        G.add_node(i, pos=pos)
    for i in range(len(positions) - 1):
        G.add_edge(i, i + 1, key=0)
    return G


def test_direction_falls_back_to_the_straight_line_to_the_neighbor():
    """No `voxels` on the edge: the direction is node position to node position."""
    G = _straight_chain((0.0, 0.0, 0.0), (0.0, 3.0, 4.0))

    directions = hub_spoke_directions(G, 0)

    assert set(directions) == {(1, 0)}
    np.testing.assert_allclose(directions[(1, 0)], [0.0, 0.6, 0.8])


def test_direction_uses_the_voxel_path_not_the_chord():
    """A curving path leaves in a different direction than the straight chord
    to its far end -- the whole reason to prefer voxels when they exist."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 5.0, 10.0))
    G.add_edge(
        0, 1, key=0,
        voxels=[(0.0, 0.0, 0.0), (0.0, 0.0, 5.0), (0.0, 5.0, 10.0)],
    )

    directions = hub_spoke_directions(G, 0, tangent_length_um=10.0)

    chord = np.array([0.0, 5.0, 10.0]) / math.hypot(5.0, 10.0)
    assert not np.allclose(directions[(1, 0)], chord)


def test_a_voxel_path_stored_neighbor_to_hub_is_reoriented():
    """`voxels` can run either way; the direction must always leave *node*."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 20.0))
    G.add_edge(
        0, 1, key=0,
        # Deliberately listed from neighbor to hub.
        voxels=[(0.0, 0.0, 20.0), (0.0, 0.0, 10.0), (0.0, 0.0, 0.0)],
    )

    directions = hub_spoke_directions(G, 0, tangent_length_um=10.0)

    np.testing.assert_allclose(directions[(1, 0)], [0.0, 0.0, 1.0], atol=1e-9)


def test_a_node_missing_pos_yields_no_directions():
    G = nx.MultiGraph()
    G.add_node(0)  # no "pos"
    G.add_node(1, pos=(0.0, 0.0, 1.0))
    G.add_edge(0, 1, key=0)

    assert hub_spoke_directions(G, 0) == {}


# --- detect_cartwheel_hubs: the full detector ----------------------------


def _cartwheel_hub(n_spokes: int, *, spoke_length: float = 10.0) -> nx.MultiGraph:
    """A hub with *n_spokes* neighbors evenly spaced around it in one plane."""
    G = nx.MultiGraph()
    G.add_node("hub", pos=(0.0, 0.0, 0.0))
    for i in range(n_spokes):
        angle = 2 * math.pi * i / n_spokes
        pos = (0.0, spoke_length * math.sin(angle), spoke_length * math.cos(angle))
        G.add_node(f"spoke{i}", pos=pos)
        G.add_edge("hub", f"spoke{i}", key=0, length=spoke_length)
    return G


def _coherent_bundle(n_spokes: int, *, half_arc_degrees: float = 30.0) -> nx.MultiGraph:
    """A hub whose neighbors all sit within one narrow arc -- a busy but
    directionally coherent junction, not a cartwheel."""
    G = nx.MultiGraph()
    G.add_node("hub", pos=(0.0, 0.0, 0.0))
    angles = np.linspace(-half_arc_degrees, half_arc_degrees, n_spokes)
    for i, degrees in enumerate(angles):
        angle = math.radians(degrees)
        pos = (0.0, 10.0 * math.sin(angle), 10.0 * math.cos(angle))
        G.add_node(f"spoke{i}", pos=pos)
        G.add_edge("hub", f"spoke{i}", key=0, length=10.0)
    return G


def test_an_eight_spoke_cartwheel_is_flagged():
    G = _cartwheel_hub(8)

    hubs = detect_cartwheel_hubs(G)

    assert [h.node for h in hubs] == ["hub"]
    hub = hubs[0]
    assert hub.degree == 8
    assert hub.spoke_count == 8
    assert hub.radial_dispersion == pytest.approx(0.0, abs=1e-9)
    assert hub.mean_spoke_length_um == pytest.approx(10.0)


def test_a_coherent_bundle_is_not_flagged_even_at_the_same_degree():
    """High degree alone is not enough -- the daughters must also disagree
    about direction, which a real busy junction's usually do not."""
    G = _coherent_bundle(6)

    hubs = detect_cartwheel_hubs(G)

    assert hubs == []


def test_below_min_degree_is_never_flagged_however_spread_out():
    G = _cartwheel_hub(5)  # below the default min_degree=6

    assert detect_cartwheel_hubs(G) == []
    # Lowering the bar to match reaches the same hub.
    hubs = detect_cartwheel_hubs(G, min_degree=5)
    assert [h.node for h in hubs] == ["hub"]


def test_dispersion_threshold_is_configurable():
    G = _coherent_bundle(6, half_arc_degrees=80.0)

    # Default threshold (0.5): a wide-but-not-full-circle bundle may or may
    # not clear it depending on the exact spread; assert both directions
    # explicitly against the dispersion this fixture actually produces.
    hubs_default = detect_cartwheel_hubs(G)
    dispersion = hub_radial_dispersion(
        list(hub_spoke_directions(G, "hub").values())
    )
    if dispersion <= 0.5:
        assert hubs_default and hubs_default[0].node == "hub"
    else:
        assert hubs_default == []
    # A permissive threshold above the measured dispersion always flags it.
    hubs_permissive = detect_cartwheel_hubs(G, max_radial_dispersion=dispersion + 0.01)
    assert hubs_permissive and hubs_permissive[0].node == "hub"
    # A strict threshold below it never does.
    hubs_strict = detect_cartwheel_hubs(G, max_radial_dispersion=max(dispersion - 0.01, 0.0))
    assert hubs_strict == []


def test_multiple_hubs_are_sorted_most_spread_out_first():
    """Two flagged hubs, worst (lowest radial dispersion) first.

    A full-circle wheel's directions cancel exactly (R=0, proven above); an
    arc-bounded bundle's do not fully cancel, so its R is strictly above 0
    yet still low enough to flag -- computed here from the same
    already-verified primitives rather than hand-derived trigonometry, since
    what this test checks is detect_cartwheel_hubs's sort, not the sum itself.
    """
    wheel = _cartwheel_hub(8)
    bundle = _coherent_bundle(6, half_arc_degrees=100.0)
    bundle_dispersion = hub_radial_dispersion(
        list(hub_spoke_directions(bundle, "hub").values())
    )
    assert 0.0 < bundle_dispersion <= 0.5, "fixture must be flagged but not perfectly cancel"

    G = nx.union(wheel, bundle, rename=("wheel-", "bundle-"))

    hubs = detect_cartwheel_hubs(G)

    assert [h.node for h in hubs] == ["wheel-hub", "bundle-hub"]
    assert hubs[0].radial_dispersion < hubs[1].radial_dispersion


def test_a_hub_missing_its_own_position_is_skipped_not_crashed():
    G = _cartwheel_hub(8)
    del G.nodes["hub"]["pos"]

    assert detect_cartwheel_hubs(G) == []


def test_min_degree_below_two_is_rejected():
    with pytest.raises(ValueError, match="min_degree"):
        detect_cartwheel_hubs(nx.MultiGraph(), min_degree=1)


def test_dispersion_threshold_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="max_radial_dispersion"):
        detect_cartwheel_hubs(nx.MultiGraph(), max_radial_dispersion=1.5)


# --- format_cartwheel_hub_report -----------------------------------------


def test_report_of_no_hubs_says_so():
    assert format_cartwheel_hub_report([]) == "Cartwheel hub guard: no hubs flagged."


def test_report_names_every_flagged_hub():
    hubs = detect_cartwheel_hubs(_cartwheel_hub(8))

    report = format_cartwheel_hub_report(hubs)

    assert "1 hub(s) flagged" in report
    assert "node=hub" in report
    assert "degree=8" in report
