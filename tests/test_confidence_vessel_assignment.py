"""Tests for confidence-based progressive large-vessel assignment."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph import (
    select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence,
)
from haemolynx.pipeline import default_schema


def test_confidence_mode_replaces_exact_tie_with_unresolved():
    """Equal arteriole/venule evidence should be flagged unresolved.

    Positions and masks use canonical physical (z, y, x).
    """
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([2.0, 2.0, 2.0]))  # terminal in overlap
    G.add_node(1, pos=np.array([2.0, 2.0, 3.0]))
    G.add_node(2, pos=np.array([2.0, 2.0, 4.0]))
    G.add_node(3, pos=np.array([2.0, 2.0, 5.0]))  # distal terminal: venule-only
    G.add_edge(0, 1, length=1.0, voxels=[(2.0, 2.0, 2.0), (2.0, 2.0, 3.0)])
    G.add_edge(1, 2, length=1.0, voxels=[(2.0, 2.0, 3.0), (2.0, 2.0, 4.0)])
    G.add_edge(2, 3, length=1.0, voxels=[(2.0, 2.0, 4.0), (2.0, 2.0, 5.0)])

    arteriole_mask = np.zeros((8, 8, 8), dtype=bool)
    venule_mask = np.zeros((8, 8, 8), dtype=bool)
    arteriole_mask[1:4, 1:4, 1:4] = True
    venule_mask[1:4, 1:4, 1:4] = True
    venule_mask[1:4, 1:4, 4:6] = True

    result = select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        max_dilation_microns=0.0,
        confidence_margin=0.05,
        minimum_confidence=0.05,
        topology_penalty=0.0,
    )
    assert result["input_nodes"] == []
    assert result["output_nodes"] == [3]
    assert result["unresolved_nodes"] == [0]
    assert result["node_confidence"][0]["reason"] == "low_score_gap"


def test_confidence_mode_topology_penalty_biases_label():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([2.0, 2.0, 2.0]))
    G.add_node(1, pos=np.array([2.0, 2.0, 3.0]))
    G.add_node(2, pos=np.array([2.0, 2.0, 4.0]))
    G.add_edge(
        0,
        1,
        length=1.0,
        branch_order="Ven1",
        vessel_type="venule",
        voxels=[(2.0, 2.0, 2.0), (2.0, 2.0, 3.0)],
    )
    G.add_edge(1, 2, length=1.0, branch_order="Ven2", vessel_type="venule")

    arteriole_mask = np.zeros((8, 8, 8), dtype=bool)
    venule_mask = np.zeros((8, 8, 8), dtype=bool)
    arteriole_mask[1:4, 1:4, 1:4] = True
    venule_mask[1:4, 1:4, 1:4] = True

    result = select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        max_dilation_microns=0.0,
        confidence_margin=0.0,
        minimum_confidence=0.01,
        topology_penalty=0.2,
    )
    assert result["output_nodes"] == [0]
    assert result["input_nodes"] == []
    assert result["unresolved_nodes"] == [2]
    assert result["node_confidence"][0]["decision"] == "output"


def test_quality_gate_can_trigger_conservative_mode():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([2.0, 2.0, 2.0]))
    G.add_node(1, pos=np.array([2.0, 2.0, 3.0]))
    G.add_edge(0, 1, length=1.0)

    arteriole_mask = np.zeros((10, 10, 10), dtype=bool)
    venule_mask = np.zeros((10, 10, 10), dtype=bool)
    # Heavy overlap plus many disconnected fragments.
    for offset in range(0, 8, 2):
        arteriole_mask[offset : offset + 1, 1:3, 1:3] = True
        venule_mask[offset : offset + 1, 1:3, 1:3] = True
        arteriole_mask[offset : offset + 1, 6:8, 6:8] = True
        venule_mask[offset : offset + 1, 6:8, 6:8] = True

    conservative_max = 5.0
    result = select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        max_dilation_microns=25.0,
        confidence_margin=0.08,
        minimum_confidence=0.12,
        quality_max_overlap_fraction=0.05,
        quality_min_terminal_coverage=0.99,
        quality_max_component_count=2,
        conservative_max_dilation_microns=conservative_max,
    )
    assert result["conservative_mode"] is True
    assert float(result["effective_max_dilation_microns"]) == conservative_max


@pytest.mark.parametrize("low_memory", [False, True])
def test_an_empty_venule_mask_claims_no_terminal_at_the_volumes_corner(tmp_path, low_memory):
    """Dilated, an empty mask used to read as a blob at voxel (0, 0, 0)
    (scipy's distance transform of a volume with no feature), so a terminal
    there became an outlet of a venule mask with no venule in it."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([1.0, 1.0, 1.0]))  # beside the corner, in no mask
    G.add_node(1, pos=np.array([5.0, 5.0, 5.0]))
    G.add_node(2, pos=np.array([10.0, 10.0, 10.0]))  # inside the arteriole
    G.add_edge(0, 1, length=6.9, voxels=[(1.0, 1.0, 1.0), (5.0, 5.0, 5.0)])
    G.add_edge(1, 2, length=8.7, voxels=[(5.0, 5.0, 5.0), (10.0, 10.0, 10.0)])

    arteriole_mask = np.zeros((14, 14, 14), dtype=bool)
    arteriole_mask[8:13, 8:13, 8:13] = True
    venule_mask = np.zeros((14, 14, 14), dtype=bool)

    result = select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        max_dilation_microns=4.0,
        confidence_margin=0.0,
        minimum_confidence=0.0,
        topology_penalty=0.0,
        low_memory=low_memory,
        memmap_directory=tmp_path,
    )
    assert result["input_nodes"] == [2]
    assert 0 not in result["output_nodes"]
    assert result["output_nodes"] == []


def test_confidence_schema_requires_non_legacy_mode():
    schema = default_schema()
    assert schema["automated_vessel_assignment_use_legacy_mode"].default is True
    assert "!automated_vessel_assignment_use_legacy_mode" in schema[
        "automated_vessel_confidence_margin"
    ].requires
    assert schema["automated_vessel_confidence_margin"].default == 0.08
    assert schema["automated_vessel_min_confidence"].default == 0.12
    assert schema["automated_vessel_topology_penalty"].default == 0.12
    assert schema["automated_vessel_conservative_max_dilation_microns"].default == 15.0
