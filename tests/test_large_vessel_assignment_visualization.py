"""Tests for large-vessel assignment Plotly visualizations."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ImageLynx.visualization.large_vessel_assignment import (
    visualize_3d_plotly_large_vessel_assignment_flow_direction,
)


def test_visualize_large_vessel_assignment_with_flow_direction_adds_arrows():
    G = nx.MultiGraph()
    G.add_node(1, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([5.0, 0.0, 0.0]))
    G.add_node(3, pos=np.array([10.0, 0.0, 0.0]))

    G.add_edge(1, 2, flow_signed=2.0, branch_order="Art1")
    G.add_edge(2, 3, flow_signed=-1.0, branch_order="Ven1")

    mask_shape = (12, 12, 12)
    large_art = np.zeros(mask_shape, dtype=bool)
    large_ven = np.zeros(mask_shape, dtype=bool)
    small_art = np.zeros(mask_shape, dtype=bool)
    small_ven = np.zeros(mask_shape, dtype=bool)
    large_art[0:3, 0:3, 0:3] = True
    large_ven[8:11, 0:3, 0:3] = True

    fig = visualize_3d_plotly_large_vessel_assignment_flow_direction(
        G,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        input_nodes=[1],
        output_nodes=[3],
        show=False,
    )

    cone_traces = [trace for trace in fig.data if getattr(trace, "type", "") == "cone"]
    assert len(cone_traces) == 1
    assert len(cone_traces[0]["x"]) == 2
