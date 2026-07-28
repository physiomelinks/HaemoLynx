"""Pytest fixtures and configuration."""
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests

import pytest
import numpy as np
import networkx as nx


@pytest.fixture
def small_binary_3d():
    """Small 3D binary volume for testing."""
    arr = np.zeros((10, 10, 10), dtype=bool)
    arr[4:6, 4:6, :] = True  # small rod
    return arr


@pytest.fixture
def tiny_skeleton():
    """Tiny skeleton for graph tests (linear path)."""
    skel = np.zeros((8, 8, 8), dtype=bool)
    for i in range(2, 6):
        skel[i, 4, 4] = True
    return skel


@pytest.fixture
def simple_graph():
    """Simple graph with 3 nodes, 2 edges, positions."""
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([1.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([2.0, 0.0, 0.0]))
    G.add_edge(0, 1, length=1.0, voxels=[(0, 0, 0), (1, 0, 0)])
    G.add_edge(1, 2, length=1.0, voxels=[(1, 0, 0), (2, 0, 0)])
    return G


@pytest.fixture
def multigraph_with_branch_order():
    """MultiGraph with branch_order on edges."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([5.0, 0.0, 0.0]))
    G.add_edge(0, 1, length=5.0, branch_order="BO1", voxels=[(0, 0, 0), (5, 0, 0)])
    return G


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "plotting: marks tests that create matplotlib figures"
    )
