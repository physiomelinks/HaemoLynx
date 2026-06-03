"""Tests for tutorials.tutorial_plots helpers."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_DIR = REPO_ROOT / "tutorials"
if str(TUTORIALS_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIALS_DIR))

from tutorial_plots import GraphBuildPlotter, in_jupyter, show_saved_plot


def test_in_jupyter_false_outside_ipython() -> None:
    assert in_jupyter() is False


def test_show_saved_plot_missing_file(capsys) -> None:
    assert show_saved_plot("/nonexistent/plot.png") is False
    assert "plot not found" in capsys.readouterr().out


def test_graph_build_plotter_tracks_paths(tmp_path) -> None:
    import numpy as np
    import networkx as nx

    from ImageLynx import visualization

    image = np.zeros((4, 8, 8), dtype=np.uint8)
    G = nx.MultiGraph()
    G.add_node(0, pos=(1.0, 2.0, 3.0))
    G.add_node(1, pos=(1.0, 2.0, 5.0))
    G.add_edge(0, 1, voxels=[(1, 2, 3), (1, 2, 4), (1, 2, 5)])

    plotter = GraphBuildPlotter(
        image,
        tmp_path,
        show_inline=False,
        steps_to_display=frozenset({"test_step"}),
    )
    plotter(G, "test_step")
    assert len(plotter.saved) == 1
    assert plotter.plot_paths()[0].is_file()
