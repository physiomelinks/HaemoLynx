"""Pytest fixtures and configuration."""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests
# Keep PyVista from opening a render window locally; CI already sets this.
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
# The same for Qt: the gui-marked tests each build a napari viewer, and a
# plain `pytest` run opens one window per test across the desktop. `setdefault`
# so `QT_QPA_PLATFORM=xcb pytest -m gui` still shows them when that is what you
# want.
#
# Not on Windows, where the offscreen plugin comes with no OpenGL at all:
# napari's canvas asks it for GL_MAX_TEXTURE_SIZE before it draws anything and
# every gui test dies there, in glGetIntegerv rather than in anything of ours.
# CI borrows a software GL stack from xvfb; Windows has nothing to borrow, so
# the real platform -- windows across the desktop and all -- is what runs them.
os.environ.setdefault(
    "QT_QPA_PLATFORM", "windows" if sys.platform == "win32" else "offscreen"
)

import pytest
import numpy as np
import networkx as nx

#: Plots produced by tests are written here rather than displayed.
#: Covered by the `tests/plots/` .gitignore rule.
PLOT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


@pytest.fixture
def plot_output_dir() -> Path:
    """Directory for test-generated plot files."""
    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return PLOT_OUTPUT_DIR


@pytest.fixture
def plot_subdir(plot_output_dir, request) -> Path:
    """A per-module folder under `tests/plots/`, so figures do not collide."""
    directory = plot_output_dir / request.module.__name__.removeprefix("test_")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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


@pytest.fixture(autouse=True)
def _reset_vessel_draw_mode():
    """Tubes/lines is a session view pref; do not leak it between tests."""
    mod = sys.modules.get("haemolynx.gui._widget")
    if mod is not None and hasattr(mod, "DEFAULT_VESSEL_DRAW"):
        mod._vessel_draw_mode = mod.DEFAULT_VESSEL_DRAW
    yield
    mod = sys.modules.get("haemolynx.gui._widget")
    if mod is not None and hasattr(mod, "DEFAULT_VESSEL_DRAW"):
        mod._vessel_draw_mode = mod.DEFAULT_VESSEL_DRAW
