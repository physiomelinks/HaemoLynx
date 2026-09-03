"""Tests for progressive vessel-mask assignment dilation."""
from __future__ import annotations

import networkx as nx
import numpy as np

from haemolynx.graph import (
    infer_boundary_nodes_from_small_vessel_masks_progressive_dilation,
    select_terminal_nodes_from_large_vessel_masks_progressive_dilation,
)
from haemolynx.graph.automated_vessel_assignment import _build_dilation_schedule_microns
from haemolynx.pipeline import default_schema


def test_build_dilation_schedule_includes_zero_and_exact_max():
    assert _build_dilation_schedule_microns(
        max_dilation_microns=0.0, dilation_step_microns=5.0
    ) == [0.0]
    assert _build_dilation_schedule_microns(
        max_dilation_microns=10.0, dilation_step_microns=5.0
    ) == [0.0, 5.0, 10.0]
    assert _build_dilation_schedule_microns(
        max_dilation_microns=12.0, dilation_step_microns=5.0
    ) == [0.0, 5.0, 10.0, 12.0]


def test_progressive_dilation_assignment_locks_earlier_nodes():
    """Nodes assigned early remain fixed across later dilation steps.

    Positions and masks use canonical physical (z, y, x).
    """
    G = nx.MultiGraph()
    # Terminal near arteriole and venule volumes with staged overlap behavior.
    G.add_node(0, pos=np.array([1.0, 1.0, 1.0]))  # arteriole at 0 µm
    G.add_node(1, pos=np.array([1.0, 1.0, 7.0]))  # venule at +5, arteriole at +10
    G.add_node(2, pos=np.array([1.0, 1.0, 9.0]))  # venule at 0 µm
    G.add_node(10, pos=np.array([1.0, 1.0, 2.0]))
    G.add_node(11, pos=np.array([1.0, 1.0, 7.0]))
    G.add_edge(0, 10, length=1.0)
    G.add_edge(10, 11, length=5.0)
    G.add_edge(11, 2, length=2.0)
    G.add_edge(1, 10, length=5.0)

    arteriole_mask = np.zeros((12, 12, 12), dtype=bool)
    venule_mask = np.zeros((12, 12, 12), dtype=bool)
    arteriole_mask[1, 1, 1] = True
    venule_mask[1, 1, 9] = True

    start_nodes, out_nodes = select_terminal_nodes_from_large_vessel_masks_progressive_dilation(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        max_dilation_microns=10.0,
        dilation_step_microns=5.0,
        allow_overlap=False,
    )
    assert start_nodes == [0]
    assert out_nodes == [1, 2]


def test_progressive_small_vessel_zero_max_matches_single_shot_shape():
    """A 0 µm schedule labels art/ven edges and finds capillary transitions."""
    G = nx.MultiGraph()
    # art -- art/cap -- capillary -- ven/cap -- ven -- capillary past venule
    for node, z in enumerate((0.0, 2.0, 4.0, 6.0, 8.0, 10.0)):
        G.add_node(node, pos=np.array([z, 3.0, 3.0]))
    for node in range(5):
        z0, z1 = float(node * 2), float(node * 2 + 2)
        G.add_edge(
            node,
            node + 1,
            voxels=[[z, 3.0, 3.0] for z in (z0, z1)],
        )

    art = np.zeros((12, 8, 8), dtype=bool)
    ven = np.zeros((12, 8, 8), dtype=bool)
    art[0:3, 3, 3] = True
    # Cover the venule segment (z=6..7) but leave the distal edge (z=8..10) unlabeled
    # so a capillary transition exists at the venule/capillary boundary.
    ven[6:8, 3, 3] = True

    result = infer_boundary_nodes_from_small_vessel_masks_progressive_dilation(
        G,
        small_arteriole_mask=art,
        small_venule_mask=ven,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        max_dilation_microns=0.0,
        minimum_overlap_fraction=0.5,
        allow_overlap=False,
    )
    assert result["arteriole_boundary_nodes"] == [2]
    assert result["venule_boundary_nodes"] == [4]
    assert result["arteriole_nodes"] == [0, 1, 2]
    assert result["venule_nodes"] == [3, 4]
    assert result["arteriole_edge_count"] == 2
    assert result["venule_edge_count"] == 2
    assert set(result["arteriole_nodes"]).isdisjoint(result["venule_nodes"])
    for node_id in result["arteriole_boundary_nodes"]:
        assert G.nodes[node_id].get("mask_vessel_type") == "arteriole"
    for node_id in result["venule_boundary_nodes"]:
        assert G.nodes[node_id].get("mask_vessel_type") == "venule"


def test_progressive_dilation_schema_settings_and_requires():
    schema = default_schema()
    load = schema["large_vessel_mask_dilation_microns"]
    assign = schema["large_vessel_assignment_max_dilation_microns"]
    small = schema["small_vessel_mask_dilation_microns"]

    load_help = load.help.lower()
    assert "one-shot" in load_help
    assert "load-time" in load_help
    assert "progressive" in assign.help.lower()
    assert "progressive" in small.help.lower()
    assert "load-time one-shot" in assign.help.lower()
    assert assign.requires == ("use_large_vessel_masks", "automated_vessel_assignment")
    assert small.requires == ("use_small_vessel_masks_for_boundary_assignment",)
    assert assign.default == 0.0
    assert small.default == 0.0
