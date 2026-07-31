"""Geometry checks for degree-2 edge merging.

Edge ``voxels`` and node ``pos`` are physical microns. The three merge helpers
used to insert the junction as an *integer* tuple — two truncated, one rounded —
which displaced the junction and, because the de-duplication guard compared a
float against that integer, also duplicated it. A merge of two 1.5 um segments
about a junction at z = 1.6 um measured 4.200 um instead of 3.000 um.

The error vanished whenever the junction landed on a whole micron, which is why
the suite never caught it: almost every fixture uses ``voxel_size = 1.0``.
"""
import numpy as np
import pytest

from ImageLynx.graph._helpers import (
    JUNCTION_TOLERANCE_UM,
    calculate_path_length,
    create_merged_edge_attributes,
    merge_curved_edges,
    merge_edge_voxels_at_node,
    merge_voxel_paths_at_node,
)
from ImageLynx.graph.degree2 import create_trivial_merged_edge

TRUE_LENGTH_UM = 3.0


def _collinear_pair(junction_z: float):
    """Two collinear segments meeting at *junction_z*, spanning 0 -> 3 um."""
    first = [[0.0, 0.0, 0.0], [0.0, 0.0, junction_z]]
    second = [[0.0, 0.0, junction_z], [0.0, 0.0, TRUE_LENGTH_UM]]
    return first, second, [0.0, 0.0, junction_z]


# --- the regression ---------------------------------------------------------


@pytest.mark.parametrize("junction_z", [0.4, 1.0, 1.4, 1.5, 1.6, 2.5, 2.75])
def test_merge_preserves_length_for_sub_voxel_junctions(junction_z):
    """Length must be exact wherever the junction sits, not only on whole microns."""
    first, second, node = _collinear_pair(junction_z)
    merged = merge_voxel_paths_at_node(first, second, node)
    assert calculate_path_length(merged) == pytest.approx(TRUE_LENGTH_UM, abs=1e-9)


@pytest.mark.parametrize("junction_z", [1.4, 1.6, 2.5])
def test_all_three_entry_points_agree(junction_z):
    """The three public helpers are one implementation and must not diverge."""
    first, second, node = _collinear_pair(junction_z)

    by_curved = merge_curved_edges(first, second, np.asarray(node))
    by_node = merge_edge_voxels_at_node(first, second, node)
    by_attrs = create_merged_edge_attributes(
        {"voxels": first, "length": junction_z},
        {"voxels": second, "length": TRUE_LENGTH_UM - junction_z},
        node,
    )["voxels"]

    assert by_curved == by_node == by_attrs


def test_junction_is_not_duplicated():
    """The old float-vs-int guard appended the junction even when already present."""
    first, second, node = _collinear_pair(1.6)
    merged = merge_voxel_paths_at_node(first, second, node)
    assert merged == [(0.0, 0.0, 0.0), (0.0, 0.0, 1.6), (0.0, 0.0, 3.0)]


def test_junction_coordinates_stay_physical_not_quantised():
    first, second, node = _collinear_pair(1.6)
    merged = merge_voxel_paths_at_node(first, second, node)
    assert all(isinstance(c, float) for point in merged for c in point)
    assert not any(point == (0, 0, 1) for point in merged)


# --- "is it breaking things" — behaviour preserved where it was already right


@pytest.mark.parametrize("junction_z", [1.0, 2.0])
def test_integer_junctions_match_the_previous_behaviour(junction_z):
    """On whole-micron junctions the old code was correct; the new code must agree.

    These are the exact paths and lengths the pre-consolidation implementation
    produced, hard-coded so a future change cannot quietly drift.
    """
    first, second, node = _collinear_pair(junction_z)
    merged = merge_voxel_paths_at_node(first, second, node)
    assert merged == [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, junction_z),
        (0.0, 0.0, TRUE_LENGTH_UM),
    ]
    assert calculate_path_length(merged) == pytest.approx(TRUE_LENGTH_UM)


@pytest.mark.parametrize("reverse_first", [False, True])
@pytest.mark.parametrize("reverse_second", [False, True])
def test_merge_is_orientation_independent(reverse_first, reverse_second):
    """Either input may arrive reversed; the merged path must be the same."""
    first, second, node = _collinear_pair(1.6)
    if reverse_first:
        first = first[::-1]
    if reverse_second:
        second = second[::-1]
    merged = merge_voxel_paths_at_node(first, second, node)
    assert calculate_path_length(merged) == pytest.approx(TRUE_LENGTH_UM)
    assert merged[0] == (0.0, 0.0, 0.0)
    assert merged[-1] == (0.0, 0.0, TRUE_LENGTH_UM)


def test_curved_path_interior_points_are_preserved():
    """Consolidation must not drop or reorder interior geometry."""
    first = [[0.0, 0.0, 0.0], [0.0, 0.7, 0.8], [0.0, 0.0, 1.6]]
    second = [[0.0, 0.0, 1.6], [0.0, -0.7, 2.4], [0.0, 0.0, 3.0]]
    merged = merge_voxel_paths_at_node(first, second, [0.0, 0.0, 1.6])
    assert merged == [
        (0.0, 0.0, 0.0),
        (0.0, 0.7, 0.8),
        (0.0, 0.0, 1.6),
        (0.0, -0.7, 2.4),
        (0.0, 0.0, 3.0),
    ]


def test_junction_is_inserted_when_a_path_stops_short():
    """The insert still fires when re-routing left a path short of the node."""
    first = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]]
    second = [[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
    merged = merge_voxel_paths_at_node(first, second, [0.0, 0.0, 1.6])
    assert (0.0, 0.0, 1.6) in merged
    assert calculate_path_length(merged) == pytest.approx(TRUE_LENGTH_UM)


def test_points_within_tolerance_are_treated_as_the_junction():
    first, second, node = _collinear_pair(1.6)
    first[-1] = [0.0, 0.0, 1.6 + JUNCTION_TOLERANCE_UM / 2.0]
    merged = merge_voxel_paths_at_node(first, second, node)
    assert len(merged) == 3


# --- length is recomputed from the path, consistently --------------------


def test_trivial_merge_length_matches_its_own_voxel_path():
    """`create_trivial_merged_edge` used an additive sum that could go stale."""
    first, second, node = _collinear_pair(1.6)
    merged = create_trivial_merged_edge(
        {"voxels": first, "length": 1.6},
        {"voxels": second, "length": 1.4},
        node,
    )
    assert merged["length"] == pytest.approx(calculate_path_length(merged["voxels"]))
    assert merged["length"] == pytest.approx(TRUE_LENGTH_UM)


def test_merged_length_survives_inputs_with_no_length_attribute():
    """The old `< length_additive * 0.5` guard returned 0 when either input lacked length."""
    first, second, node = _collinear_pair(1.6)
    merged = create_merged_edge_attributes(
        {"voxels": first}, {"voxels": second}, node
    )
    assert merged["length"] == pytest.approx(TRUE_LENGTH_UM)
    assert merged["additive_length"] == 0
