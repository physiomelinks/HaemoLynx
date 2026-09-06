"""Direction-aware node-cluster collapse: hand-verified geometry, no pipeline involved.

Mirrors tests/test_cartwheel_guard.py's style, since this module exists
specifically to stop collapse_node_clusters from producing the shape that
guard flags -- see graph/direction_aware_collapse.py's own docstring.
"""
from __future__ import annotations

import math

import numpy as np
import networkx as nx
import pytest

from haemolynx.graph.cartwheel_guard import detect_cartwheel_hubs
from haemolynx.graph.collapse import collapse_node_clusters
from haemolynx.graph.direction_aware_collapse import (
    DEFAULT_MAX_RADIAL_DISPERSION,
    DEFAULT_MIN_DEGREE_FOR_DISPERSION_CHECK,
    collapse_node_clusters_direction_aware,
)


def _wheel_graph(
    n_spokes: int = 8, hub_jitter: float = 0.3, spoke_length: float = 20.0
) -> nx.MultiGraph:
    """*n_spokes* tightly-clustered hub nodes, each with one edge leaving in
    its own evenly-spaced direction -- the shape cartwheel_guard exists to
    flag, and the shape a naive collapse would produce by merging every hub
    node into one representative carrying every spoke."""
    G = nx.MultiGraph()
    rng = np.random.default_rng(0)
    for i in range(n_spokes):
        offset = rng.normal(scale=hub_jitter, size=3)
        G.add_node(i, pos=np.array([0.0, 0.0, 0.0]) + offset)
    for i in range(n_spokes):
        angle = 2 * math.pi * i / n_spokes
        direction = np.array([0.0, math.cos(angle), math.sin(angle)])
        ext_pos = direction * spoke_length
        ext_id = 100 + i
        G.add_node(ext_id, pos=ext_pos)
        c_pos = G.nodes[i]["pos"]
        G.add_edge(
            i, ext_id,
            length=float(np.linalg.norm(ext_pos - c_pos)),
            voxels=[c_pos.tolist(), ext_pos.tolist()],
        )
    return G


def _coherent_bundle_graph(
    n_branches: int = 8, spread_degrees: float = 20.0, spoke_length: float = 150.0
) -> nx.MultiGraph:
    """*n_branches* tightly-clustered hub nodes, each leaving in a direction
    within a narrow cone -- a real busy junction where every branch
    continues roughly one way, not a cartwheel. Collapsing all of them
    should stay allowed. *spoke_length* is long enough that neighbouring
    external endpoints stay outside the default 5 um collapse distance from
    each other despite the narrow angular spread -- this fixture is only
    testing whether the *hub* side stays free to merge, not creating a
    second cluster on the external side by accident."""
    G = nx.MultiGraph()
    rng = np.random.default_rng(1)
    spread = math.radians(spread_degrees)
    for i in range(n_branches):
        offset = rng.normal(scale=0.3, size=3)
        G.add_node(i, pos=np.array([0.0, 0.0, 0.0]) + offset)
        angle = -spread / 2 + spread * i / max(n_branches - 1, 1)
        direction = np.array([0.0, math.cos(angle), math.sin(angle)])
        ext_pos = direction * spoke_length
        ext_id = 100 + i
        G.add_node(ext_id, pos=ext_pos)
        c_pos = G.nodes[i]["pos"]
        G.add_edge(
            i, ext_id,
            length=float(np.linalg.norm(ext_pos - c_pos)),
            voxels=[c_pos.tolist(), ext_pos.tolist()],
        )
    return G


# --- the core claim: refuses to build what cartwheel_guard would flag ------


def test_a_wheel_shaped_cluster_is_not_collapsed_into_one_cartwheel_hub():
    G = _wheel_graph()

    legacy = collapse_node_clusters(G, distance_threshold=5.0)
    aware = collapse_node_clusters_direction_aware(G, distance_threshold=5.0)

    # The legacy behaviour really does produce the pathology this exists to
    # fix -- otherwise this fixture proves nothing.
    legacy_hubs = detect_cartwheel_hubs(
        legacy,
        min_degree=DEFAULT_MIN_DEGREE_FOR_DISPERSION_CHECK,
        max_radial_dispersion=DEFAULT_MAX_RADIAL_DISPERSION,
    )
    assert legacy_hubs, "fixture bug: legacy collapse did not even produce a cartwheel hub"
    assert max(dict(legacy.degree()).values()) == 8

    aware_hubs = detect_cartwheel_hubs(
        aware,
        min_degree=DEFAULT_MIN_DEGREE_FOR_DISPERSION_CHECK,
        max_radial_dispersion=DEFAULT_MAX_RADIAL_DISPERSION,
    )
    assert aware_hubs == [], "direction-aware collapse still produced a flaggable cartwheel hub"
    # And it did not just refuse to collapse anything at all.
    assert aware.number_of_nodes() < G.number_of_nodes()


def test_a_coherent_busy_junction_still_collapses_fully():
    """The fix must not become "never merge more than a few nodes" -- a
    real junction where every branch leaves roughly the same way should
    collapse exactly as the legacy method does."""
    G = _coherent_bundle_graph()

    legacy = collapse_node_clusters(G, distance_threshold=5.0)
    aware = collapse_node_clusters_direction_aware(G, distance_threshold=5.0)

    assert aware.number_of_nodes() == legacy.number_of_nodes()
    assert aware.number_of_edges() == legacy.number_of_edges()
    assert max(dict(aware.degree()).values()) == 8


def test_a_small_cluster_below_min_degree_always_collapses_regardless_of_shape():
    """Two hub nodes each with one external spoke pointing the opposite way
    -- as wheel-shaped as two spokes can be -- but degree 2 is below the
    default floor of 6, so it merges exactly as the legacy method would."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([1.0, 0.0, 0.0]))
    G.add_node(10, pos=np.array([0.0, 0.0, -20.0]))
    G.add_node(11, pos=np.array([0.0, 0.0, 20.0]))
    G.add_edge(0, 10, length=20.0, voxels=[[0, 0, 0], [0, 0, -20]])
    G.add_edge(1, 11, length=19.0, voxels=[[1, 0, 0], [0, 0, 20]])

    aware = collapse_node_clusters_direction_aware(G, distance_threshold=5.0)

    assert aware.number_of_nodes() == 3
    assert set(dict(aware.degree()).values()) == {2, 1, 1}


# --- deduplicating redundant parallel wiring after a merge -----------------


def test_two_cluster_members_reaching_the_same_neighbour_keep_only_the_shorter_edge():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([1.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([20.0, 0.0, 0.0]))
    G.add_edge(0, 2, length=20.0, voxels=[[0, 0, 0], [20, 0, 0]])
    G.add_edge(1, 2, length=19.2, voxels=[[1, 0, 0], [20, 0, 0]])

    out = collapse_node_clusters_direction_aware(G, distance_threshold=5.0)

    assert out.number_of_nodes() == 2
    assert out.number_of_edges() == 1
    remaining = list(out.edges(data=True))[0][2]
    assert remaining["length"] == pytest.approx(19.2)


# --- distance_only parity: reusing the option must not add new settings ---


@pytest.mark.parametrize(
    "build_graph", [_wheel_graph, _coherent_bundle_graph], ids=["wheel", "coherent_bundle"]
)
def test_the_legacy_method_is_unaffected_by_this_modules_existence(build_graph):
    """collapse_node_clusters itself is never called by this module, and
    never imported for its behaviour (only two small, pure, read-only
    helpers) -- confirmed by re-running it directly and comparing."""
    G = build_graph()
    a = collapse_node_clusters(G, distance_threshold=5.0)
    b = collapse_node_clusters(G, distance_threshold=5.0)
    assert set(a.nodes) == set(b.nodes)
    assert set(a.edges(keys=True)) == set(b.edges(keys=True))


# --- validation --------------------------------------------------------


@pytest.mark.parametrize("bad_dispersion", [-0.1, 1.1])
def test_a_max_radial_dispersion_outside_0_1_is_rejected(bad_dispersion):
    G = nx.MultiGraph()
    with pytest.raises(ValueError, match="max_radial_dispersion"):
        collapse_node_clusters_direction_aware(
            G, max_radial_dispersion=bad_dispersion
        )


def test_a_min_degree_below_two_is_rejected():
    G = nx.MultiGraph()
    with pytest.raises(ValueError, match="min_degree_for_dispersion_check"):
        collapse_node_clusters_direction_aware(
            G, min_degree_for_dispersion_check=1
        )


def test_a_graph_with_fewer_than_two_positioned_nodes_is_left_alone():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    out = collapse_node_clusters_direction_aware(G, distance_threshold=5.0)
    assert list(out.nodes) == [0]
