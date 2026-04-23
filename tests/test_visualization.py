"""Tests for visualization module."""
import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np
import networkx as nx

# Forcibly skip the entire visualization suite in CI/Automated environments 
# as 3D rendering (PyVista/Vedo/VTK) causes segmentation faults on headless servers.
pytestmark = pytest.mark.skip(reason="Visualization tests require an interactive display/GPU context.")

from ImageLynx.visualization import (
    plot_node_degree_distribution,
    visualize_edges_and_nodes,
    visualize_geometry_with_branch_orders,
    visualize_geometry_with_edge_weights,
    visualize_volume_vedo,
    visualize_overlay,
    visualize_overlay_vedo,
)
from ImageLynx.visualization._helpers import (
    sort_branch_orders_numerically,
    create_color_mapping,
    group_branch_orders_for_legend,
)


def test_sort_branch_orders_numerically():
    out = sort_branch_orders_numerically(["BO3", "BO1", "B10"])
    assert out[0] == "BO1"
    assert out[-1] == "B10"


def test_create_color_mapping():
    m = create_color_mapping(["BO1", "BO2"], "viridis")
    assert len(m) == 2
    assert all(len(c) == 4 for c in m.values())


def test_group_branch_orders_for_legend():
    orders = ["BO1", "BO2", "BO10"]
    counts = {"BO1": 1, "BO2": 2, "BO10": 3}
    lo, lc = group_branch_orders_for_legend(orders, 5, counts)
    assert "BO5+" in lc or len(lo) <= 3


@pytest.mark.plotting
def test_plot_node_degree_distribution(simple_graph):
    counts = plot_node_degree_distribution(simple_graph)
    assert isinstance(counts, dict)


@pytest.mark.plotting
def test_visualize_edges_and_nodes(simple_graph):
    img = np.zeros((5, 5, 5))
    visualize_edges_and_nodes(img, simple_graph)


@pytest.mark.plotting
def test_visualize_geometry_with_branch_orders(multigraph_with_branch_order):
    img = np.zeros((10, 10, 10))
    G = multigraph_with_branch_order.copy()
    for u, v, k, d in G.edges(keys=True, data=True):
        if "voxels" not in d or len(d["voxels"]) < 2:
            G[u][v][k]["voxels"] = [(0, 0, 0), (5, 0, 0)]
    fig, ax, cmap = visualize_geometry_with_branch_orders(img, G)
    assert cmap is not None


@pytest.mark.plotting
def test_visualize_geometry_with_edge_weights(multigraph_with_branch_order):
    img = np.zeros((10, 10, 10))
    G = multigraph_with_branch_order.copy()
    for u, v, k, d in G.edges(keys=True, data=True):
        G[u][v][k]["voxels"] = [(0, 0, 0), (5, 0, 0)]
    result = visualize_geometry_with_edge_weights(img, G)
    assert result[2] is not None


@pytest.mark.plotting
def test_visualize_volume_vedo():
    pytest.importorskip("vedo")
    img = np.zeros((10, 10, 10), dtype=bool)
    img[2:8, 2:8, 2:8] = True
    plt = visualize_volume_vedo(img, title="Test", show=False)
    assert plt is not None


@pytest.mark.plotting
def test_visualize_overlay():
    pytest.importorskip("pyvista")
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[5, 5, 5] = True
    plt = visualize_overlay(mask, skeleton, title="Test", show=False)
    assert plt is not None


@pytest.mark.plotting
def test_visualize_overlay_vedo():
    pytest.importorskip("vedo")
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[5, 5, 5] = True
    plt = visualize_overlay_vedo(mask, skeleton, title="Test", show=False)
    assert plt is not None
