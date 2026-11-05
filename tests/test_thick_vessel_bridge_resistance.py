"""A thin-vessel-to-fat-vessel bridge edge must carry negligible resistance.

graph.thick_vessel_junctions.insert_thick_vessel_junction_nodes tags the
segment inside a fat vessel's mask boundary IS_ZERO_RESISTANCE (see that
module). This is the other half: haemodynamics.apply must actually turn that
tag into a resistance small enough to be negligible, without corrupting the
flow solve the way a literal zero (infinite conductance) would.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.haemodynamics import (
    apply_poiseuille_haemodynamics,
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
    solve_flow_from_conductance_matrix,
)
from haemolynx.haemodynamics.apply import ZERO_RESISTANCE_FRACTION

EDGE_LENGTH_UM = 1000.0


def _chain_graph(num_edges: int = 3) -> nx.MultiGraph:
    """A straight capillary chain; the middle edge is a tagged bridge."""
    graph = nx.MultiGraph()
    for node in range(num_edges + 1):
        graph.add_node(node, pos=np.asarray([0.0, 0.0, node * EDGE_LENGTH_UM]))
    for node in range(num_edges):
        attrs = dict(
            key=0,
            length=EDGE_LENGTH_UM,
            branch_order="B01",
            voxels=[
                [0.0, 0.0, node * EDGE_LENGTH_UM],
                [0.0, 0.0, (node + 1) * EDGE_LENGTH_UM],
            ],
        )
        if node == 1:
            attrs[IS_ZERO_RESISTANCE] = True
        graph.add_edge(node, node + 1, **attrs)
    return graph


def test_a_tagged_edge_gets_a_negligible_resistance_not_a_poiseuille_one():
    G, summary = apply_poiseuille_haemodynamics(
        _chain_graph(), diameter_by_branch_order={"B01": 5.0}
    )

    tagged = [
        data["resistance"] for _u, _v, data in G.edges(data=True) if data.get(IS_ZERO_RESISTANCE)
    ]
    untagged = [
        data["resistance"]
        for _u, _v, data in G.edges(data=True)
        if not data.get(IS_ZERO_RESISTANCE)
    ]

    assert len(tagged) == 1
    assert len(untagged) == 2
    # Scaled to the network's own smallest real resistance, not a fixed
    # absolute value (a fixed epsilon would be astronomically far from real
    # Poiseuille resistances -- see ZERO_RESISTANCE_FRACTION's docstring).
    assert tagged[0] == pytest.approx(min(untagged) * ZERO_RESISTANCE_FRACTION)
    assert all(r > tagged[0] * 1e3 for r in untagged)
    assert summary["resistances"]["thick_vessel_bridges"] == 1


def test_conductance_and_flow_solve_stay_finite_with_a_tagged_edge():
    """A literal zero resistance (infinite conductance) would make the
    Laplacian's diagonal inf - inf = nan and corrupt the whole solve."""
    G, _summary = apply_poiseuille_haemodynamics(
        _chain_graph(), diameter_by_branch_order={"B01": 5.0}
    )

    conductance, node_list = build_conductance_matrix_from_graph(G)
    assert np.all(np.isfinite(conductance))

    laplacian = calc_laplacian_from_conductance_matrix(conductance)
    assert np.all(np.isfinite(laplacian))

    result = solve_flow_from_conductance_matrix(
        conductance,
        node_list,
        inlet_p_bc=100.0,
        outlet_p_bc=0.0,
        inlet_nodes=[0],
        outlet_nodes=[3],
    )
    assert np.all(np.isfinite(result["pressure"]))


def test_no_tagged_edges_leaves_the_bridge_count_at_zero():
    G = nx.MultiGraph()
    for node in range(2):
        G.add_node(node, pos=np.asarray([0.0, 0.0, node * EDGE_LENGTH_UM]))
    G.add_edge(
        0,
        1,
        key=0,
        length=EDGE_LENGTH_UM,
        branch_order="B01",
        voxels=[[0.0, 0.0, 0.0], [0.0, 0.0, EDGE_LENGTH_UM]],
    )

    _G, summary = apply_poiseuille_haemodynamics(G, diameter_by_branch_order={"B01": 5.0})

    assert summary["resistances"]["thick_vessel_bridges"] == 0
