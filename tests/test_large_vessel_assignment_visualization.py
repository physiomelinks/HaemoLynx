"""Tests for large-vessel assignment Plotly visualizations."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np

from haemolynx.pipeline import default_schema
from haemolynx.pipeline.stages import (
    HaemodynamicModel,
    Solution,
    VesselNetwork,
    _write_run_final_graph_3d_html,
    export_results,
)
from haemolynx.visualization import (
    VESSEL_VOLUME_TRACE_STYLES,
    add_binary_mask_volume_trace,
    selected_vessel_masks_for_html,
    visualize_3d_plotly_large_vessel_assignment,
    visualize_3d_plotly_large_vessel_assignment_flow_direction,
    write_final_graph_3d_html,
)


def _tiny_graph() -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.add_node(1, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([0.0, 0.0, 5.0]))
    G.add_node(3, pos=np.array([0.0, 0.0, 10.0]))
    G.add_edge(1, 2, flow_signed=2.0, branch_order="Art1")
    G.add_edge(2, 3, flow_signed=-1.0, branch_order="Ven1")
    return G


def _tiny_masks(shape=(8, 6, 6)):
    large_art = np.zeros(shape, dtype=bool)
    large_ven = np.zeros(shape, dtype=bool)
    small_art = np.zeros(shape, dtype=bool)
    small_ven = np.zeros(shape, dtype=bool)
    large_art[0:2, 0:2, 0:2] = True
    large_ven[0:2, 0:2, 4:6] = True
    small_art[6:8, 4:6, 0:2] = True
    small_ven[6:8, 4:6, 4:6] = True
    return large_art, large_ven, small_art, small_ven


def _volume_traces(fig):
    return [trace for trace in fig.data if getattr(trace, "type", "") == "volume"]


def _volume_style_tuple(trace):
    colorscale = list(trace.colorscale)
    color = colorscale[-1][1]
    return (trace.name, float(trace.opacity), str(color))


def test_visualize_large_vessel_assignment_with_flow_direction_adds_arrows():
    G = _tiny_graph()
    large_art, large_ven, small_art, small_ven = _tiny_masks((12, 12, 12))

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


def test_assignment_view_volume_traces_use_shared_styles():
    G = _tiny_graph()
    large_art, large_ven, small_art, small_ven = _tiny_masks()
    fig = visualize_3d_plotly_large_vessel_assignment(
        G,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        input_nodes=[1],
        output_nodes=[3],
        show=False,
    )
    styles = {_volume_style_tuple(trace) for trace in _volume_traces(fig)}
    expected = {
        (
            str(item["name"]),
            float(item["opacity"]),
            str(item["color"]),
        )
        for item in VESSEL_VOLUME_TRACE_STYLES.values()
    }
    assert styles == expected


def test_assignment_view_omits_large_volumes_when_masks_are_none():
    G = _tiny_graph()
    _large_art, _large_ven, small_art, small_ven = _tiny_masks()
    fig = visualize_3d_plotly_large_vessel_assignment(
        G,
        large_arteriole_mask=None,
        large_venule_mask=None,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        input_nodes=[1],
        output_nodes=[3],
        show=False,
    )
    names = {trace.name for trace in _volume_traces(fig)}
    assert names == {
        VESSEL_VOLUME_TRACE_STYLES["small_arteriole"]["name"],
        VESSEL_VOLUME_TRACE_STYLES["small_venule"]["name"],
    }


def test_assignment_view_calls_shared_volume_helper(monkeypatch):
    G = _tiny_graph()
    large_art, large_ven, small_art, small_ven = _tiny_masks()
    calls: list[dict] = []
    real = add_binary_mask_volume_trace

    def spy(fig, mask, **kwargs):
        calls.append(kwargs)
        return real(fig, mask, **kwargs)

    monkeypatch.setattr(
        "haemolynx.visualization.large_vessel_assignment.add_binary_mask_volume_trace",
        spy,
    )
    visualize_3d_plotly_large_vessel_assignment(
        G,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        input_nodes=[1],
        output_nodes=[3],
        volume_downsample_stride=2,
        show=False,
    )
    by_name = {call["name"]: call for call in calls}
    for style in VESSEL_VOLUME_TRACE_STYLES.values():
        assert by_name[style["name"]]["color"] == style["color"]
        assert by_name[style["name"]]["opacity"] == style["opacity"]
        assert by_name[style["name"]]["volume_downsample_stride"] == 2


def test_selected_vessel_masks_follow_gui_flags():
    large_art, large_ven, small_art, small_ven = _tiny_masks()
    both = selected_vessel_masks_for_html(
        use_large_vessel_masks=True,
        use_small_vessel_masks=True,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
    )
    assert both["large_arteriole_mask"] is large_art
    assert both["small_venule_mask"] is small_ven

    large_only = selected_vessel_masks_for_html(
        use_large_vessel_masks=True,
        use_small_vessel_masks=False,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
    )
    assert large_only["large_venule_mask"] is large_ven
    assert large_only["small_arteriole_mask"] is None
    assert large_only["small_venule_mask"] is None

    none = selected_vessel_masks_for_html(
        use_large_vessel_masks=False,
        use_small_vessel_masks=False,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
    )
    assert all(value is None for value in none.values())


def test_final_graph_html_includes_pipeline_volume_traces_when_masks_enabled(tmp_path):
    G = _tiny_graph()
    large_art, large_ven, small_art, small_ven = _tiny_masks()
    html_path = tmp_path / "final_graph_3d.html"
    fig = write_final_graph_3d_html(
        G,
        save_html_path=html_path,
        use_large_vessel_masks=True,
        use_small_vessel_masks=True,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        input_nodes=[1],
        output_nodes=[3],
        volume_downsample_stride=1,
        show=False,
    )
    pipeline = visualize_3d_plotly_large_vessel_assignment(
        G,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        input_nodes=[1],
        output_nodes=[3],
        volume_downsample_stride=1,
        show=False,
    )
    assert {_volume_style_tuple(t) for t in _volume_traces(fig)} == {
        _volume_style_tuple(t) for t in _volume_traces(pipeline)
    }
    assert html_path.is_file() and html_path.stat().st_size > 0


def test_final_graph_html_omits_unselected_mask_volumes(tmp_path):
    G = _tiny_graph()
    large_art, large_ven, small_art, small_ven = _tiny_masks()

    large_only = write_final_graph_3d_html(
        G,
        save_html_path=tmp_path / "large.html",
        use_large_vessel_masks=True,
        use_small_vessel_masks=False,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        show=False,
    )
    large_names = {trace.name for trace in _volume_traces(large_only)}
    assert large_names == {
        VESSEL_VOLUME_TRACE_STYLES["large_arteriole"]["name"],
        VESSEL_VOLUME_TRACE_STYLES["large_venule"]["name"],
    }

    small_only = write_final_graph_3d_html(
        G,
        save_html_path=tmp_path / "small.html",
        use_large_vessel_masks=False,
        use_small_vessel_masks=True,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        show=False,
    )
    small_names = {trace.name for trace in _volume_traces(small_only)}
    assert small_names == {
        VESSEL_VOLUME_TRACE_STYLES["small_arteriole"]["name"],
        VESSEL_VOLUME_TRACE_STYLES["small_venule"]["name"],
    }

    neither = write_final_graph_3d_html(
        G,
        save_html_path=tmp_path / "none.html",
        use_large_vessel_masks=False,
        use_small_vessel_masks=False,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        show=False,
    )
    assert _volume_traces(neither) == []
    scatter_names = {trace.name for trace in neither.data}
    assert "Edges" in scatter_names
    assert "Nodes" in scatter_names


def test_final_graph_html_writer_delegates_to_pipeline_assignment_view(
    monkeypatch, tmp_path
):
    G = _tiny_graph()
    large_art, large_ven, small_art, small_ven = _tiny_masks()
    captured = {}

    def fake_assignment(*args, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(data=())

    monkeypatch.setattr(
        "haemolynx.visualization.pipeline_artifacts.visualize_3d_plotly_large_vessel_assignment",
        fake_assignment,
    )
    write_final_graph_3d_html(
        G,
        save_html_path=tmp_path / "final_graph_3d.html",
        use_large_vessel_masks=True,
        use_small_vessel_masks=True,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        volume_downsample_stride=3,
        show=False,
    )
    kwargs = captured["kwargs"]
    assert kwargs["large_arteriole_mask"] is large_art
    assert kwargs["small_venule_mask"] is small_ven
    assert kwargs["volume_downsample_stride"] == 3


def test_final_graph_html_uses_full_mask_extent_not_a_z_crop():
    G = _tiny_graph()
    shape = (10, 4, 4)
    large_art = np.zeros(shape, dtype=bool)
    large_ven = np.zeros(shape, dtype=bool)
    large_art[0, 1, 1] = True
    large_art[9, 2, 2] = True
    large_ven[0, 2, 1] = True
    voxel_size_zyx = (2.0, 1.0, 1.0)
    fig = write_final_graph_3d_html(
        G,
        save_html_path=None,
        use_large_vessel_masks=True,
        use_small_vessel_masks=False,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        voxel_size_zyx=voxel_size_zyx,
        show=False,
    )
    art = next(
        trace
        for trace in _volume_traces(fig)
        if trace.name == VESSEL_VOLUME_TRACE_STYLES["large_arteriole"]["name"]
    )
    z_coords = np.asarray(art.z, dtype=float)
    assert z_coords.min() == 0.0
    assert z_coords.max() == 9.0 * voxel_size_zyx[0]


def test_run_html_helper_passes_loaded_masks_and_flags(monkeypatch, tmp_path):
    G = _tiny_graph()
    large_art, large_ven, small_art, small_ven = _tiny_masks()
    captured = {}

    def fake_writer(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(
        "haemolynx.visualization.write_final_graph_3d_html",
        fake_writer,
    )
    settings = {
        "plot_dir": tmp_path,
        "use_large_vessel_masks": True,
        "use_small_vessel_masks_for_boundary_assignment": True,
        "show_plots_in_ide": False,
        "interactive_plots": False,
        "large_vessel_3d_volume_downsample_stride": 2,
        "inlet_nodes": [1],
        "outlet_nodes": [3],
        "arteriole_boundary_nodes": [],
        "venule_boundary_nodes": [],
    }
    _write_run_final_graph_3d_html(
        settings,
        G,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        show=False,
    )
    assert captured["use_large_vessel_masks"] is True
    assert captured["use_small_vessel_masks"] is True
    assert captured["large_arteriole_mask"] is large_art
    assert captured["small_venule_mask"] is small_ven
    assert captured["volume_downsample_stride"] == 2
    assert captured["save_html_path"] == tmp_path / "final_graph_3d.html"
    assert large_art.shape[0] == 8


def test_export_results_rewrites_final_graph_html_with_network_masks(
    monkeypatch, tmp_path
):
    G = _tiny_graph()
    large_art, large_ven, small_art, small_ven = _tiny_masks()
    captured = {}

    def fake_writer(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(
        "haemolynx.visualization.write_final_graph_3d_html",
        fake_writer,
    )
    monkeypatch.setattr(
        "haemolynx.visualization.plot_node_degree_distribution",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "haemolynx.visualization.visualize_3d_plotly",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "haemolynx.visualization.visualize_geometry_with_branch_orders",
        lambda *args, **kwargs: None,
    )
    volume = SimpleNamespace(
        image=np.zeros((8, 6, 6), dtype=np.uint8),
        output_dir=tmp_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    network = VesselNetwork(
        graph=G,
        volume=volume,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
    )
    model = HaemodynamicModel(graph=G)
    solution = Solution(graph=G)
    settings = {
        "statistics": False,
        "measurement_3d_to_cell_mask": False,
        "run_haemodynamics": False,
        "vtk_export": False,
        "visualize_vtk": False,
        "visualize_results": True,
        "ide_plot_mode": "final_only",
        "show_plots_in_ide": False,
        "interactive_plots": False,
        "hold_ide_plots_open": False,
        "final_render_mode": "3d",
        "plot_dir": tmp_path,
        "input_path": Path("stack.tif"),
        "use_large_vessel_masks": True,
        "use_small_vessel_masks_for_boundary_assignment": False,
        "large_vessel_3d_volume_downsample_stride": 1,
        "inlet_nodes": [1],
        "outlet_nodes": [3],
        "arteriole_boundary_nodes": [],
        "venule_boundary_nodes": [],
    }
    export_results(settings, network, model, solution)
    assert captured["use_large_vessel_masks"] is True
    assert captured["use_small_vessel_masks"] is False
    assert captured["large_arteriole_mask"] is large_art
    assert captured["small_arteriole_mask"] is small_art
    assert captured["save_html_path"] == tmp_path / "final_graph_3d.html"


def test_export_skips_final_graph_html_when_produce_ide_plots_is_off(
    monkeypatch, tmp_path
):
    def boom(*_args, **_kwargs):
        raise AssertionError(
            "final_graph_3d.html must not be written when Produce IDE plots is off"
        )

    monkeypatch.setattr(
        "haemolynx.pipeline.stages.visualization.write_final_graph_3d_html",
        boom,
    )
    G = _tiny_graph()
    large_art, large_ven, small_art, small_ven = _tiny_masks()
    volume = SimpleNamespace(
        image=np.zeros((8, 6, 6), dtype=np.uint8),
        output_dir=tmp_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    network = VesselNetwork(
        graph=G,
        volume=volume,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
    )
    export_results(
        {
            "statistics": False,
            "measurement_3d_to_cell_mask": False,
            "run_haemodynamics": False,
            "vtk_export": False,
            "visualize_vtk": False,
            "visualize_results": False,
            "plot_dir": tmp_path,
            "input_path": Path("stack.tif"),
        },
        network,
        HaemodynamicModel(graph=G),
        Solution(graph=G),
    )


def test_build_network_and_export_write_final_graph_html_via_shared_helper():
    import inspect

    from haemolynx.pipeline import stages

    build_src = inspect.getsource(stages.build_network)
    export_src = inspect.getsource(stages.export_results)
    assert "_write_run_final_graph_3d_html" in build_src
    assert "_write_run_final_graph_3d_html" in export_src
