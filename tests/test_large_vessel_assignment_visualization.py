"""Tests for large-vessel assignment Plotly visualizations."""
from __future__ import annotations

import networkx as nx
import numpy as np

from haemolynx.pipeline import default_schema
from haemolynx.visualization import (
    visualize_3d_plotly_large_vessel_assignment_flow_direction,
)


def test_visualize_large_vessel_assignment_with_flow_direction_adds_arrows():
    G = nx.MultiGraph()
    G.add_node(1, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([0.0, 0.0, 5.0]))
    G.add_node(3, pos=np.array([0.0, 0.0, 10.0]))

    G.add_edge(1, 2, flow_signed=2.0, branch_order="Art1")
    G.add_edge(2, 3, flow_signed=-1.0, branch_order="Ven1")

    mask_shape = (12, 12, 12)
    large_art = np.zeros(mask_shape, dtype=bool)
    large_ven = np.zeros(mask_shape, dtype=bool)
    small_art = np.zeros(mask_shape, dtype=bool)
    small_ven = np.zeros(mask_shape, dtype=bool)
    large_art[0:3, 0:3, 0:3] = True
    large_ven[0:3, 0:3, 8:11] = True

    fig = visualize_3d_plotly_large_vessel_assignment_flow_direction(
        G,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        input_nodes=[1],
        output_nodes=[3],
        voxel_size_zyx=(1.0, 1.0, 1.0),
        show=False,
    )

    cone_traces = [trace for trace in fig.data if getattr(trace, "type", "") == "cone"]
    assert len(cone_traces) == 1
    assert len(cone_traces[0]["x"]) == 2


def test_large_vessel_assignment_viz_schema_default():
    schema = default_schema()
    assert schema["large_vessel_3d_volume_downsample_stride"].default == 1
    assert schema["large_vessel_3d_volume_downsample_stride"].requires == (
        "use_large_vessel_masks",
        "automated_vessel_assignment",
    )
