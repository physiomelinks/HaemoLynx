"""Tests for visualization module."""
import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np
import networkx as nx

from haemolynx.visualization import (
    plot_node_degree_distribution,
    visualize_3d_plotly,
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


def test_visualize_3d_plotly_snaps_edge_endpoints_to_node_positions():
    """Regression: plot.py used to draw an edge's raw ``voxels`` as stored,

    the same duplicated logic geometry.edge_polyline was written to replace
    (see its own module docstring: "two plotly writers" used to answer this
    separately). A voxels path left a little short of its node -- exactly
    what centreline smoothing/cluster collapse produces -- used to leave a
    visible gap; edge_polyline's snap closes it.
    """
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 10.0))
    # Falls 2 microns short of node 1's actual position.
    G.add_edge(0, 1, voxels=[(0.0, 0.0, 0.0), (0.0, 0.0, 8.0)])

    fig = visualize_3d_plotly(G, show=False)

    edge_trace = next(t for t in fig.data if t.name == "Edges")
    # Stored (z, y, x) -> plotted (x, y, z): the last real point before the
    # None separator must land exactly on node 1's x-coordinate (10.0), not
    # the raw voxels path's short 8.0.
    real_x = [x for x in edge_trace.x if x is not None]
    assert real_x[-1] == pytest.approx(10.0)


def test_visualize_3d_plotly_falls_back_to_a_node_to_node_segment():
    """An edge with no ``voxels`` at all must still be drawn, from pos alone."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 5.0))
    G.add_edge(0, 1)

    fig = visualize_3d_plotly(G, show=False)

    edge_trace = next(t for t in fig.data if t.name == "Edges")
    real_x = [x for x in edge_trace.x if x is not None]
    assert real_x == pytest.approx([0.0, 5.0])


def _two_vessels() -> nx.MultiGraph:
    G = nx.MultiGraph()
    for node, pos in {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 100.0), 2: (0.0, 50.0, 100.0)}.items():
        G.add_node(node, pos=pos)
    # A straight run stored one point per micron, as a centreline is.
    G.add_edge(0, 1, diameter_um=6.0, voxels=[(0.0, 0.0, float(x)) for x in range(101)])
    G.add_edge(1, 2, diameter_um=12.0, voxels=[(0.0, float(y), 100.0) for y in range(51)])
    return G


def test_the_3d_plot_colours_each_vessel_by_its_diameter():
    """Regression: every vessel was one plain colour."""
    fig = visualize_3d_plotly(_two_vessels(), show=False)

    edge_trace = fig.data[0]
    assert edge_trace.name == "Edges (colour: diameter_um)"
    assert edge_trace.line.colorbar.title.text == "diameter_um"
    colours = list(edge_trace.line.color)
    assert len(colours) == len(edge_trace.x)
    assert set(colours) == {6.0, 12.0}


def test_the_3d_plot_can_still_draw_every_vessel_one_colour():
    fig = visualize_3d_plotly(_two_vessels(), show=False, colour_by=None)

    assert fig.data[0].line.color == "cyan"


def test_the_3d_plot_draws_a_straight_run_from_its_two_ends():
    """Regression: every centreline point was written -- a hundred for a
    straight hundred-micron vessel that two draw exactly."""
    fig = visualize_3d_plotly(_two_vessels(), show=False)

    points = [x for x in fig.data[0].x if x is not None]
    assert len(points) == 4


def test_thinning_keeps_a_centreline_within_its_tolerance():
    from haemolynx.visualization import thin_polyline

    t = np.linspace(0.0, 4.0 * np.pi, 400)
    wavy = np.column_stack([np.zeros_like(t), 3.0 * np.sin(t), 10.0 * t])

    thinned = thin_polyline(wavy, 0.25)

    assert 4 < len(thinned) < 100
    assert np.array_equal(thinned[0], wavy[0]) and np.array_equal(thinned[-1], wavy[-1])
    # Every original point lies within the tolerance of the thinned polyline.
    segments = list(zip(thinned[:-1], thinned[1:]))

    def distance(point):
        best = np.inf
        for a, b in segments:
            ab = b - a
            s = np.clip(np.dot(point - a, ab) / np.dot(ab, ab), 0.0, 1.0)
            best = min(best, float(np.linalg.norm(point - (a + s * ab))))
        return best

    assert max(distance(point) for point in wavy) <= 0.25 + 1e-9


def test_a_saved_3d_plot_opens_without_the_internet(tmp_path):
    """Regression: plotly.js was loaded from its CDN, so a saved page stayed
    blank offline."""
    path = tmp_path / "network.html"
    visualize_3d_plotly(_two_vessels(), show=False, save_html_path=str(path))

    html = path.read_text(encoding="utf-8")
    assert 'src="https://cdn.plot.ly' not in html  # no script loaded from the CDN
    assert len(html) > 1_000_000  # plotly.js itself is inside the page


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


# --- off the main thread (a napari panel run) ----------------------------------


def _pyplot_must_not_be_used(monkeypatch):
    """Make pyplot's figure makers fail: under napari they build Qt windows."""
    import matplotlib.pyplot as plt

    def refuse(*_args, **_kwargs):
        raise AssertionError("a saved figure went through pyplot")

    monkeypatch.setattr(plt, "figure", refuse)
    monkeypatch.setattr(plt, "subplots", refuse)


def _two_vessel_graph() -> nx.MultiGraph:
    G = nx.MultiGraph()
    for node, pos in {0: (0.0, 1.0, 1.0), 1: (0.0, 4.0, 4.0), 2: (0.0, 1.0, 7.0)}.items():
        G.add_node(node, pos=np.asarray(pos))
    G.add_edge(0, 1, branch_order="B01", resistance=2.0, voxels=[(0.0, 1.0, 1.0), (0.0, 4.0, 4.0)])
    G.add_edge(1, 2, branch_order="B02", resistance=4.0, voxels=[(0.0, 4.0, 4.0), (0.0, 1.0, 7.0)])
    return G


def test_a_saved_figure_never_goes_through_pyplot(tmp_path, monkeypatch):
    """Every run's skeleton figures and an export's plots were pyplot figures,
    which under napari's Qt resolve to Qt windows built on the run's worker
    thread ("Starting a Matplotlib GUI outside of the main thread")."""
    from pathlib import Path

    from haemolynx.visualization import visualize_skeleton
    from haemolynx.visualization.perturbation_plots import (
        SWEEP_AXIS_BY_TYPE,
        SWEEP_Y_METRICS,
        plot_sweep_curves,
    )

    _pyplot_must_not_be_used(monkeypatch)
    G = _two_vessel_graph()
    image = np.zeros((2, 8, 9), dtype=np.uint8)
    image[:, 1:5, 1:8] = 1
    written = {
        "degree": tmp_path / "degree.png",
        "overlay": tmp_path / "overlay.png",
        "orders": tmp_path / "orders.png",
        "resistance": tmp_path / "resistance.png",
        "skeleton": tmp_path / "skeleton.png",
    }
    plot_node_degree_distribution(G, save_path=written["degree"], show=False)
    visualize_edges_and_nodes(image, G, save_path=written["overlay"], show=False)
    visualize_geometry_with_branch_orders(image, G, save_path=written["orders"], show=False)
    visualize_geometry_with_edge_resistance(image, G, save_path=written["resistance"], show=False)
    visualize_skeleton(image.astype(bool), save_path=written["skeleton"])
    for name, path in written.items():
        assert path.is_file() and path.stat().st_size > 0, name

    axis = SWEEP_AXIS_BY_TYPE["pressure_sweep"]
    rows = [
        {axis.x_key: pressure, **{column: float(i + 1) for i, column in enumerate(SWEEP_Y_METRICS)}}
        for pressure in (1000.0, 2000.0)
    ]
    curves = plot_sweep_curves(rows, tmp_path / "sweep", axis=axis)
    assert len(curves) == len(SWEEP_Y_METRICS)
    assert all(Path(path).is_file() for path in curves.values())


def test_no_window_is_opened_off_the_main_thread(tmp_path, monkeypatch):
    """A blocking show on the panel's worker held the run for ever."""
    import threading

    from haemolynx.visualization.plot import backend_can_display

    _pyplot_must_not_be_used(monkeypatch)
    seen: dict[str, object] = {}

    def run() -> None:
        try:
            seen["can_display"] = backend_can_display()
            plot_node_degree_distribution(_two_vessel_graph(), show=True, block=True)
            seen["shown"] = True
        except BaseException as error:  # noqa: BLE001 - reported below
            seen["error"] = error

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=60)

    assert not worker.is_alive(), "showing a figure off the main thread blocked"
    assert "error" not in seen, seen.get("error")
    assert seen == {"can_display": False, "shown": True}
