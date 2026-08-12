"""Tests for graph/automated_vessel_assignment.py — terminal-node assignment.

These functions decide where blood enters and leaves the network, so a mistake
does not crash anything: it produces a plausible model of the wrong vasculature.
The failure modes worth guarding are a node assigned to the wrong side, a node
assigned to both sides at once, an edge labelled from a mask lookup done with
the wrong voxel spacing, and stale labels left over from a previous run.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph import (
    infer_boundary_nodes_from_small_vessel_masks,
    select_terminal_nodes_from_large_vessel_masks,
)
from haemolynx.graph.automated_vessel_assignment import (
    compute_overlapping_terminal_assignment_metrics,
    resolve_overlapping_terminal_node_assignment,
)

MASK_SHAPE = (12, 8, 8)
ISOTROPIC = (1.0, 1.0, 1.0)
# Coarse z, fine x; all three distinct so a (z, y, x) / (x, y, z) swap shows up.
ANISOTROPIC = (2.0, 0.5, 0.4)


def _empty_mask(shape=MASK_SHAPE) -> np.ndarray:
    return np.zeros(shape, dtype=bool)


def _branching_graph() -> nx.MultiGraph:
    """Three degree-1 terminals around one degree-3 junction."""
    G = nx.MultiGraph()
    positions = {
        0: (1.0, 4.0, 4.0),   # terminal, sits in the arteriole mask
        1: (10.0, 4.0, 4.0),  # terminal, sits in the venule mask
        2: (5.0, 7.0, 4.0),   # terminal, in neither mask
        3: (5.0, 4.0, 4.0),   # junction, sits in the arteriole mask
    }
    for node, pos in positions.items():
        G.add_node(node, pos=np.array(pos))
    for terminal in (0, 1, 2):
        G.add_edge(
            terminal,
            3,
            voxels=[positions[terminal], positions[3]],
        )
    return G


def _chain_graph(z_positions=(0.0, 2.0, 4.0, 6.0, 8.0)) -> nx.MultiGraph:
    """A 5-node chain along z, each edge carrying its own voxel path."""
    G = nx.MultiGraph()
    for node, z in enumerate(z_positions):
        G.add_node(node, pos=np.array([z, 3.0, 3.0]))
    for node in range(len(z_positions) - 1):
        z_start, z_end = z_positions[node], z_positions[node + 1]
        G.add_edge(
            node,
            node + 1,
            voxels=[
                [float(z), 3.0, 3.0]
                for z in np.arange(z_start, z_end + 0.5, 1.0)
            ],
        )
    return G


# --- select_terminal_nodes_from_large_vessel_masks --------------------------


def test_terminals_are_split_by_which_mask_they_sit_in():
    arteriole, venule = _empty_mask(), _empty_mask()
    arteriole[1, 4, 4] = True
    venule[10, 4, 4] = True

    starting, output = select_terminal_nodes_from_large_vessel_masks(
        _branching_graph(), arteriole, venule, voxel_size_zyx=ISOTROPIC
    )

    assert starting == [0]
    assert output == [1]


def test_a_junction_inside_a_mask_is_not_treated_as_an_inlet():
    """Only free ends can be boundary conditions; an interior node would inject
    flow into the middle of the network."""
    arteriole, venule = _empty_mask(), _empty_mask()
    arteriole[5, 4, 4] = True  # the degree-3 junction
    venule[10, 4, 4] = True

    starting, output = select_terminal_nodes_from_large_vessel_masks(
        _branching_graph(), arteriole, venule, voxel_size_zyx=ISOTROPIC
    )

    assert starting == []
    assert output == [1]


def test_a_terminal_outside_the_mask_volume_is_skipped_not_clamped():
    """Clamping an out-of-bounds index would assign a node by the nearest face."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([100.0, 4.0, 4.0]))
    G.add_node(1, pos=np.array([-5.0, 4.0, 4.0]))
    G.add_node(2, pos=np.array([5.0, 4.0, 4.0]))
    G.add_edge(0, 2, voxels=[(100.0, 4.0, 4.0), (5.0, 4.0, 4.0)])
    G.add_edge(1, 2, voxels=[(-5.0, 4.0, 4.0), (5.0, 4.0, 4.0)])

    arteriole, venule = _empty_mask(), _empty_mask()
    arteriole[:] = True
    venule[:] = True

    starting, output = select_terminal_nodes_from_large_vessel_masks(
        G, arteriole, venule, voxel_size_zyx=ISOTROPIC
    )

    assert starting == []
    assert output == []


def test_a_terminal_in_both_masks_lands_on_exactly_one_side():
    """A node that is both an inlet and an outlet makes the flow solve ill-posed."""
    arteriole, venule = _empty_mask(), _empty_mask()
    arteriole[1, 4, 4] = True
    venule[1, 4, 4] = True
    arteriole[0:3, 4, 4] = True
    venule[0:3, 4, 4] = True

    starting, output = select_terminal_nodes_from_large_vessel_masks(
        _branching_graph(), arteriole, venule, voxel_size_zyx=ISOTROPIC
    )

    assert set(starting) & set(output) == set()
    assert len(set(starting) | set(output)) == 1


def test_allow_overlap_keeps_a_shared_terminal_in_both_groups():
    """The diagnostic path needs to see the ambiguity rather than a resolved guess."""
    arteriole, venule = _empty_mask(), _empty_mask()
    arteriole[1, 4, 4] = True
    venule[1, 4, 4] = True

    starting, output = select_terminal_nodes_from_large_vessel_masks(
        _branching_graph(), arteriole, venule, voxel_size_zyx=ISOTROPIC, allow_overlap=True
    )

    assert starting == [0]
    assert output == [0]


def test_masks_of_different_shapes_are_rejected():
    with pytest.raises(ValueError, match="must share a shape"):
        select_terminal_nodes_from_large_vessel_masks(
            _branching_graph(),
            _empty_mask(),
            _empty_mask((12, 8, 9)),
            voxel_size_zyx=ISOTROPIC,
        )


@pytest.mark.parametrize("bad", [(0.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (1.0, 1.0)])
def test_a_non_physical_voxel_size_is_rejected(bad):
    """Zero or missing spacing would divide by zero or broadcast into nonsense."""
    with pytest.raises(ValueError, match="voxel_size_zyx"):
        select_terminal_nodes_from_large_vessel_masks(
            _branching_graph(), _empty_mask(), _empty_mask(), voxel_size_zyx=bad
        )


def test_a_graph_with_no_terminals_yields_no_boundary_nodes():
    G = nx.MultiGraph()
    for node, pos in enumerate([(1.0, 4.0, 4.0), (5.0, 4.0, 4.0), (9.0, 4.0, 4.0)]):
        G.add_node(node, pos=np.array(pos))
    G.add_edge(0, 1, voxels=[(1.0, 4.0, 4.0), (5.0, 4.0, 4.0)])
    G.add_edge(1, 2, voxels=[(5.0, 4.0, 4.0), (9.0, 4.0, 4.0)])
    G.add_edge(2, 0, voxels=[(9.0, 4.0, 4.0), (1.0, 4.0, 4.0)])
    arteriole = _empty_mask()
    arteriole[:] = True

    assert select_terminal_nodes_from_large_vessel_masks(
        G, arteriole, _empty_mask(), voxel_size_zyx=ISOTROPIC
    ) == ([], [])


# --- overlap resolution metrics --------------------------------------------


def test_overlap_metrics_report_each_mask_separately():
    arteriole, venule = _empty_mask(), _empty_mask()
    arteriole[0:3, 4, 4] = True

    metrics = compute_overlapping_terminal_assignment_metrics(
        _branching_graph(),
        0,
        node_pos=np.array([1.0, 4.0, 4.0]),
        large_arteriole_mask=arteriole,
        large_venule_mask=venule,
        voxel_size_zyx=ISOTROPIC,
    )

    assert metrics["arteriole_overlap_fraction"] > 0.0
    assert metrics["venule_overlap_fraction"] == 0.0
    assert metrics["venule_intersection"] is None
    assert np.isinf(metrics["venule_midpoint_distance"])
    assert np.isfinite(metrics["arteriole_midpoint_distance"])


def test_resolution_is_deterministic_when_the_two_masks_are_identical():
    """Two runs of the same data must not disagree about which side a node is on."""
    mask = _empty_mask()
    mask[0:3, 4, 4] = True
    G = _branching_graph()

    assignments = {
        resolve_overlapping_terminal_node_assignment(
            G,
            0,
            node_pos=np.array([1.0, 4.0, 4.0]),
            large_arteriole_mask=mask.copy(),
            large_venule_mask=mask.copy(),
            voxel_size_zyx=ISOTROPIC,
        )
        for _ in range(3)
    }

    assert assignments == {"inlet"}


def test_the_node_closer_to_the_venule_midline_is_assigned_to_the_outputs():
    """The tie-break must follow the geometry, not the argument order."""
    arteriole, venule = _empty_mask(), _empty_mask()
    # Node at (1, 4, 4): dead centre of the venule, off to one side of the arteriole.
    venule[0:3, 3:6, 3:6] = True
    arteriole[0:3, 4:8, 4:8] = True

    assignment = resolve_overlapping_terminal_node_assignment(
        _branching_graph(),
        0,
        node_pos=np.array([1.0, 4.0, 4.0]),
        large_arteriole_mask=arteriole,
        large_venule_mask=venule,
        voxel_size_zyx=ISOTROPIC,
    )

    assert assignment == "outlet"


# --- infer_boundary_nodes_from_small_vessel_masks ---------------------------


def test_boundary_nodes_are_where_the_mask_region_meets_the_capillary_bed():
    """Every labelled node is not a boundary — only the ones with an unlabelled edge.

    Nodes 0 and 1 sit inside the arteriole mask with all their edges labelled;
    treating them as boundaries would impose a pressure inside the feeding
    vessel instead of at its outlet.
    """
    arteriole, venule = _empty_mask(), _empty_mask()
    arteriole[0:5, 3, 3] = True
    venule[6:12, 3, 3] = True

    result = infer_boundary_nodes_from_small_vessel_masks(
        _chain_graph(), arteriole, venule, voxel_size_zyx=ISOTROPIC
    )

    assert result["arteriole_nodes"] == [0, 1, 2]
    assert result["venule_nodes"] == [3, 4]
    assert result["arteriole_boundary_nodes"] == [2]
    assert result["venule_boundary_nodes"] == [3]
    assert result["arteriole_edge_count"] == 2
    assert result["venule_edge_count"] == 1
    assert result["overlap_edge_count"] == 0


def test_labels_are_written_onto_the_nodes_and_edges():
    arteriole, venule = _empty_mask(), _empty_mask()
    arteriole[0:5, 3, 3] = True
    venule[6:12, 3, 3] = True
    G = _chain_graph()

    infer_boundary_nodes_from_small_vessel_masks(
        G, arteriole, venule, voxel_size_zyx=ISOTROPIC
    )

    labels = {node: data.get("mask_vessel_type") for node, data in G.nodes(data=True)}
    assert labels == {
        0: "arteriole", 1: "arteriole", 2: "arteriole", 3: "venule", 4: "venule"
    }
    assert G.edges[0, 1, 0]["mask_vessel_type"] == "arteriole"
    assert G.edges[3, 4, 0]["mask_vessel_type"] == "venule"
    # The unlabelled middle edge is the capillary bed between the two trees.
    assert "mask_vessel_type" not in G.edges[2, 3, 0]


def test_labels_from_an_earlier_run_are_cleared_before_relabelling():
    """Reruns are how a user tunes the threshold; stale labels make the result
    depend on the order the runs happened in."""
    arteriole, venule = _empty_mask(), _empty_mask()
    arteriole[0:5, 3, 3] = True
    venule[6:12, 3, 3] = True
    G = _chain_graph()

    infer_boundary_nodes_from_small_vessel_masks(
        G, arteriole, venule, voxel_size_zyx=ISOTROPIC
    )
    result = infer_boundary_nodes_from_small_vessel_masks(
        G, _empty_mask(), _empty_mask(), voxel_size_zyx=ISOTROPIC
    )

    assert result["arteriole_nodes"] == []
    assert result["venule_nodes"] == []
    assert all("mask_vessel_type" not in data for _n, data in G.nodes(data=True))
    assert all(
        "mask_vessel_type" not in data for _u, _v, data in G.edges(data=True)
    )


def test_the_overlap_threshold_is_inclusive_at_the_configured_fraction():
    """An edge exactly at the threshold counts; drifting to a strict `>` would
    silently drop the edges that sit on the boundary of the criterion."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 3.0, 3.0]))
    G.add_node(1, pos=np.array([3.0, 3.0, 3.0]))
    G.add_edge(0, 1, voxels=[[float(z), 3.0, 3.0] for z in range(4)])
    arteriole = _empty_mask()
    arteriole[0:2, 3, 3] = True  # 2 of the 4 sampled points

    at_threshold = infer_boundary_nodes_from_small_vessel_masks(
        G, arteriole, _empty_mask(), voxel_size_zyx=ISOTROPIC,
        minimum_overlap_fraction=0.5,
    )
    above_threshold = infer_boundary_nodes_from_small_vessel_masks(
        G, arteriole, _empty_mask(), voxel_size_zyx=ISOTROPIC,
        minimum_overlap_fraction=0.51,
    )

    assert at_threshold["arteriole_edge_count"] == 1
    assert above_threshold["arteriole_edge_count"] == 0


def test_an_edge_in_both_masks_is_resolved_to_the_stronger_overlap():
    arteriole, venule = _empty_mask(), _empty_mask()
    arteriole[0:5, 3, 3] = True
    venule[0:3, 3, 3] = True
    G = _chain_graph()

    result = infer_boundary_nodes_from_small_vessel_masks(
        G, arteriole, venule, voxel_size_zyx=ISOTROPIC
    )

    # Edge 0-1 lies in both masks; edge 1-2 only in the arteriole one.
    assert result["overlap_edge_count"] == 1
    assert result["arteriole_edge_count"] == 2
    assert result["venule_edge_count"] == 0
    assert set(result["arteriole_nodes"]) & set(result["venule_nodes"]) == set()
    assert G.edges[0, 1, 0]["mask_vessel_type"] == "arteriole"


def test_allow_overlap_records_the_ambiguity_instead_of_resolving_it():
    arteriole, venule = _empty_mask(), _empty_mask()
    arteriole[0:5, 3, 3] = True
    venule[0:5, 3, 3] = True
    G = _chain_graph()

    result = infer_boundary_nodes_from_small_vessel_masks(
        G, arteriole, venule, voxel_size_zyx=ISOTROPIC, allow_overlap=True
    )

    assert G.edges[0, 1, 0]["mask_vessel_type"] == "overlap"
    assert result["arteriole_edge_count"] == result["venule_edge_count"] == 2
    assert set(result["arteriole_nodes"]) == set(result["venule_nodes"]) == {0, 1, 2}


def test_edge_mask_lookups_use_per_array_axis_spacing():
    """Edge positions are physical (z, y, x) microns; the lookup must divide by
    the matching per-axis spacing. Under a z/x swap this edge misses the mask
    entirely and the vessel is never labelled."""
    arteriole = _empty_mask((8, 8, 8))
    arteriole[1, 2, 3] = True
    arteriole[2, 2, 3] = True
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([1 * 2.0, 2 * 0.5, 3 * 0.4]))
    G.add_node(1, pos=np.array([2 * 2.0, 2 * 0.5, 3 * 0.4]))
    G.add_edge(0, 1, voxels=[G.nodes[0]["pos"].tolist(), G.nodes[1]["pos"].tolist()])

    result = infer_boundary_nodes_from_small_vessel_masks(
        G, arteriole, _empty_mask((8, 8, 8)), voxel_size_zyx=ANISOTROPIC
    )

    assert result["arteriole_edge_count"] == 1
    assert result["arteriole_nodes"] == [0, 1]


def test_small_masks_of_different_shapes_are_rejected():
    with pytest.raises(ValueError, match="must share a shape"):
        infer_boundary_nodes_from_small_vessel_masks(
            _chain_graph(),
            _empty_mask(),
            _empty_mask((12, 8, 9)),
            voxel_size_zyx=ISOTROPIC,
        )


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_an_overlap_fraction_outside_zero_to_one_is_rejected(fraction):
    """A fraction above 1 can never be met, so every edge would go unlabelled."""
    with pytest.raises(ValueError, match=r"must be in \[0.0, 1.0\]"):
        infer_boundary_nodes_from_small_vessel_masks(
            _chain_graph(),
            _empty_mask(),
            _empty_mask(),
            voxel_size_zyx=ISOTROPIC,
            minimum_overlap_fraction=fraction,
        )


def test_a_graph_without_node_positions_is_rejected():
    """Without positions there is nothing to look up in the mask, and silently
    returning empty results would read as 'no vessels found'."""
    G = nx.MultiGraph()
    G.add_edge(0, 1)

    with pytest.raises(ValueError, match="no node positions"):
        infer_boundary_nodes_from_small_vessel_masks(
            G, _empty_mask(), _empty_mask(), voxel_size_zyx=ISOTROPIC
        )


def test_a_plain_graph_is_labelled_the_same_way_as_a_multigraph():
    """The two branches are written out separately, so they can drift apart."""
    arteriole, venule = _empty_mask(), _empty_mask()
    arteriole[0:5, 3, 3] = True
    venule[6:12, 3, 3] = True

    multi = _chain_graph()
    plain = nx.Graph()
    plain.add_nodes_from(multi.nodes(data=True))
    for u, v, data in multi.edges(data=True):
        plain.add_edge(u, v, **data)

    multi_result = infer_boundary_nodes_from_small_vessel_masks(
        multi, arteriole, venule, voxel_size_zyx=ISOTROPIC
    )
    plain_result = infer_boundary_nodes_from_small_vessel_masks(
        plain, arteriole, venule, voxel_size_zyx=ISOTROPIC
    )

    for key in (
        "arteriole_boundary_nodes",
        "venule_boundary_nodes",
        "arteriole_nodes",
        "venule_nodes",
        "arteriole_edge_count",
        "venule_edge_count",
    ):
        assert multi_result[key] == plain_result[key]
