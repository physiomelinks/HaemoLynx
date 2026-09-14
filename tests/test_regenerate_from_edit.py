"""End-to-end: a hand-edited graph, regenerated through real pipeline stages.

Builds on test_diameter_assignment.py's own fixtures (a straight branch-order
chain, boundaries, settings) -- see that file's own docstring for why
`assign_diameters` and `build_haemodynamic_model` are the two stages this
exercises for real, not stubbed. What is new here is the *edited* graph in
between: a deleted edge that collapses a node, and a newly drawn branch with
neither `branch_order` nor `length` yet -- exactly the "Edit" window's Add
branch / Delete edge outputs, fed through haemolynx.graph.edit's own
primitives, then through the real stages Regenerate resumes at.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx.graph import EdgeDraft, commit_new_edge, delete_edge_and_collapse  # noqa: E402
from haemolynx.pipeline.stages import assign_diameters, build_haemodynamic_model  # noqa: E402

from test_diameter_assignment import (  # noqa: E402
    BRANCH_ORDERS,
    SCHEMA,
    _boundaries,
    _network,
    _settings,
    _vessel_network,
)


def _network_with_a_pendant_branch() -> nx.MultiGraph:
    """The BRANCH_ORDERS chain (0..4), plus a dead-end pendant off node 2."""
    graph = _network()
    graph.add_node(5, pos=np.asarray([100.0, 0.0, 2 * 400.0]))
    graph.add_edge(
        2,
        5,
        key=0,
        branch_order="B01",
        length=100.0,
        voxels=[[0.0, 0.0, 2 * 400.0], [100.0, 0.0, 2 * 400.0]],
    )
    return graph


def test_regenerate_recomputes_diameters_and_resistance_for_the_edited_topology(
    tmp_path,
):
    graph = _network_with_a_pendant_branch()
    last_node = len(BRANCH_ORDERS)  # 4, the chain's own outlet

    # --- Delete edge (2, 5): node 5 drops to degree 0 (removed); node 2
    # drops from degree 3 to degree 2 and collapses into a direct 1-3 edge.
    delete_edge_and_collapse(graph, 2, 5, 0)
    assert 5 not in graph
    assert 2 not in graph
    assert graph.has_edge(1, 3)

    # --- Add branch: a brand-new pendant off node 3, Finish with no target
    # hit (a dangling terminal) -- has no branch_order/length yet.
    draft = EdgeDraft(start_node=3, points_um=[tuple(graph.nodes[3]["pos"])])
    draft.extend([tuple(graph.nodes[3]["pos"]), (50.0, 0.0, 3 * 400.0 + 50.0)])
    _start, new_node, new_key = commit_new_edge(graph, draft, end_node=None)
    assert "branch_order" not in graph[3][new_node][new_key]

    network = _vessel_network(tmp_path, graph=graph)
    boundaries = _boundaries()
    settings = _settings(tmp_path)
    assert settings["inlet_nodes"] == [0]
    assert settings["outlet_nodes"] == [last_node]

    model = assign_diameters(settings, network, boundaries, SCHEMA)

    # The edit survived assign_diameters' own branch-order BFS: the collapsed
    # node stays gone, the merge stays a single edge, and the new pendant
    # picked up a real branch order (and therefore a table diameter) instead
    # of staying stranded with neither. (The old pendant node's id -- 5 -- is
    # legitimately recycled by next_node_id for the new one, so identity is
    # checked through the edge below, not through a node id.)
    assert 2 not in model.graph
    assert model.graph.has_edge(1, 3)
    new_edge_data = model.graph[3][new_node][new_key]
    assert new_edge_data.get("branch_order")
    assert new_edge_data.get("diameter_um", 0) > 0

    solved = build_haemodynamic_model(settings, model, SCHEMA)

    resistance = solved.graph[3][new_node][new_key].get("resistance")
    conductance = solved.graph[3][new_node][new_key].get("conductance")
    assert resistance is not None and np.isfinite(resistance) and resistance > 0
    assert conductance is not None and conductance > 0
