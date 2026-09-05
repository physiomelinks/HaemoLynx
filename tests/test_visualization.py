"""Tests for visualization module."""
import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np
import networkx as nx

from haemolynx.visualization import (
    plot_node_degree_distribution,
    visualize_3d_plotly_vessel_types,
    visualize_edges_and_nodes,
    visualize_geometry_with_branch_orders,
    visualize_geometry_with_edge_resistance,
)
from haemolynx.visualization._helpers import (
    sort_branch_orders_numerically,
    create_color_mapping,
    group_branch_orders_for_legend,
)


def test_visualize_3d_plotly_vessel_types_renders_large_vessel_traces():
    """Large_Art/Large_Ven edges must get their own coloured trace, not just
    a bucket in an internal dict nobody reads: the trace-building loop used
    to iterate a hardcoded 4-category tuple that silently dropped any
    vessel type added to type_to_color/type_to_label without also being
    added there."""
    G = nx.MultiGraph()
    for node, z in enumerate((0.0, 1.0, 2.0, 3.0, 4.0, 5.0)):
        G.add_node(node, pos=(z, 0.0, 0.0))
    G.add_edge(0, 1, branch_order="Large_Art1")
    G.add_edge(1, 2, branch_order="Art1")
    G.add_edge(2, 3, branch_order="B01")
    G.add_edge(3, 4, branch_order="Ven1")
    G.add_edge(4, 5, branch_order="Large_Ven1")

    fig = visualize_3d_plotly_vessel_types(G, show=False)

    trace_names = [trace.name for trace in fig.data]
    assert any(name.startswith("Large arterioles") for name in trace_names)
    assert any(name.startswith("Large venules") for name in trace_names)
    assert any(name.startswith("Arterioles") for name in trace_names)
    assert any(name.startswith("Venules") for name in trace_names)
    assert any(name.startswith("Capillaries") for name in trace_names)

    large_art_trace = next(t for t in fig.data if t.name.startswith("Large arterioles"))
    assert large_art_trace.line.color == "#8b0000"
    large_ven_trace = next(t for t in fig.data if t.name.startswith("Large venules"))
    assert large_ven_trace.line.color == "#08306b"


def test_sort_branch_orders_numerically():
    out = sort_branch_orders_numerically(["BO3", "BO1", "B10"])
    assert out[0] == "BO1"
    assert out[-1] == "B10"


def test_sort_branch_orders_numerically_places_large_vessel_tiers_outermost():
    """Large_Art sorts before Art, Large_Ven sorts after Ven -- not
    alphabetically, which would put Large_Art after BO and Large_Ven
    before Ven."""
    out = sort_branch_orders_numerically(
        ["Ven1", "BO1", "Large_Ven1", "Art1", "Large_Art1"]
    )
    assert out == ["Large_Art1", "Art1", "BO1", "Ven1", "Large_Ven1"]


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
def test_plot_node_degree_distribution(simple_graph, plot_output_dir):
    out = plot_output_dir / "node_degree_distribution.png"
    counts = plot_node_degree_distribution(simple_graph, save_path=out, show=False)
    assert isinstance(counts, dict)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.plotting
def test_visualize_edges_and_nodes(simple_graph, plot_output_dir):
    img = np.zeros((5, 5, 5))
    out = plot_output_dir / "edges_and_nodes.png"
    visualize_edges_and_nodes(img, simple_graph, save_path=out, show=False)
    # Previously this test had no assertion at all.
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.plotting
def test_visualize_geometry_with_branch_orders(multigraph_with_branch_order, plot_output_dir):
    img = np.zeros((10, 10, 10))
    G = multigraph_with_branch_order.copy()
    for u, v, k, d in G.edges(keys=True, data=True):
        if "voxels" not in d or len(d["voxels"]) < 2:
            G[u][v][k]["voxels"] = [(0, 0, 0), (5, 0, 0)]
    out = plot_output_dir / "geometry_branch_orders.png"
    fig, ax, cmap = visualize_geometry_with_branch_orders(
        img, G, save_path=out, show=False
    )
    assert cmap is not None
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.plotting
def test_visualize_geometry_with_edge_resistance(multigraph_with_branch_order, plot_output_dir):
    img = np.zeros((10, 10, 10))
    G = multigraph_with_branch_order.copy()
    for u, v, k, d in G.edges(keys=True, data=True):
        G[u][v][k]["voxels"] = [(0, 0, 0), (5, 0, 0)]
        # The plot colours by haemodynamic resistance, so it must be present.
        G[u][v][k]["resistance"] = 1.0e16
        G[u][v][k]["conductance"] = 1.0e-16
    out = plot_output_dir / "geometry_edge_resistance.png"
    result = visualize_geometry_with_edge_resistance(img, G, save_path=out, show=False)
    assert result[2] is not None
    assert out.exists() and out.stat().st_size > 0


def test_no_display_is_attempted_under_a_non_interactive_backend():
    """Guards the UserWarning storm: Agg cannot show, so nothing must try."""
    import warnings

    from haemolynx.visualization.plot import (
        _show_matplotlib_blocking,
        _show_matplotlib_non_blocking,
        backend_can_display,
    )

    assert not backend_can_display(), "tests must run on a non-interactive backend"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _show_matplotlib_non_blocking()
        _show_matplotlib_blocking()
