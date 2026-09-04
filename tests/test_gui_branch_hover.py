"""Branch-hover tooltip helpers: availability, formatting, polyline hit-testing.

Pure logic for the napari Vectors hover path that shows branchID (always) plus
user-selected metrics. No napari / Qt imports -- those live in the widget tests.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.gui.branch_hover import (
    BRANCH_HOVER_LABELS,
    BRANCH_HOVER_MAX_DISTANCE,
    BRANCH_HOVER_METRICS,
    available_branch_hover_metrics,
    available_metrics_from_features,
    branch_hover_rows,
    default_selected_metrics,
    edge_tortuosity,
    filter_selected_metrics,
    format_branch_tooltip,
    format_metric_value,
    hover_features_for_segments,
    nearest_vector_index,
    panel_metric_options,
    tooltips_from_feature_table,
)
from test_gui_results import a_graph, built, network, spec_named
from haemolynx.gui.results import (
    BRANCH_HOVER,
    FLOW_DIRECTION,
    VESSEL_LABELS,
    VESSELS,
    edge_polylines,
    polylines_to_vectors,
)


def graph_with(**attrs) -> nx.MultiGraph:
    """``a_graph`` plus per-edge attributes (flow, order, resistance, ...)."""
    return a_graph(**attrs)


def test_branch_id_is_edge_index_even_when_segment_ids_collide():
    """Tooltip identity is the graph-edge enumeration, not ``segment_id``.

    Every simple MultiGraph edge has key 0, and later topology steps often
    leave several edges sharing ``segment_id=0``. Those used to make every
    tooltip read ``branchID: 0``.
    """
    graph = a_graph()
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["segment_id"] = 0
    ids, features = branch_hover_rows(graph, selected=())
    assert ids == ["0", "1", "2"]
    assert list(features["branch_id"]) == ["0", "1", "2"]
    assert list(features["tooltip"]) == ["branchID: 0", "branchID: 1", "branchID: 2"]


def test_branch_id_matches_identity_edge_index_when_an_edge_cannot_be_drawn():
    """Skipped polylines keep the same numbering as the layer ``edge_index``."""
    graph = nx.MultiGraph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, length=1.0, segment_id=99)
    for node_id, z in enumerate((0.0, 10.0, 20.0), start=2):
        graph.add_node(node_id, pos=np.array([z, 0.0, 0.0]))
    for u, v in ((2, 3), (3, 4)):
        graph.add_edge(
            u, v, key=0,
            voxels=[graph.nodes[u]["pos"].tolist(), graph.nodes[v]["pos"].tolist()],
            length=10.0, segment_id=0,
        )
    paths, identity = edge_polylines(graph)
    ids, features = branch_hover_rows(graph, selected=())
    assert len(paths) == 2
    assert list(identity["edge_index"]) == [1, 2]
    assert ids == ["1", "2"]
    assert list(features["branch_id"]) == ["1", "2"]
    assert list(features["tooltip"]) == ["branchID: 1", "branchID: 2"]


def test_available_metrics_on_geometry_only_graph():
    """After build_network: length and tortuosity, nothing haemodynamic."""
    graph = a_graph()
    assert available_branch_hover_metrics(graph) == ("tortuosity", "length")


def test_available_metrics_include_order_when_branch_order_present():
    graph = a_graph(branch_order="A1")
    assert available_branch_hover_metrics(graph) == (
        "order",
        "tortuosity",
        "length",
    )


def test_available_metrics_include_resistance_when_present():
    graph = a_graph(resistance=1e15)
    assert available_branch_hover_metrics(graph) == (
        "resistance",
        "tortuosity",
        "length",
    )


def test_available_metrics_include_flow_when_present():
    graph = a_graph(flow_abs=1.5e-12)
    assert available_branch_hover_metrics(graph) == (
        "flow",
        "tortuosity",
        "length",
    )


def test_available_metrics_full_solved_graph():
    graph = a_graph(
        branch_order="C0",
        resistance=2e14,
        flow_abs=3e-13,
        diameter_um=5.0,
        diameter_source="measured",
    )
    assert available_branch_hover_metrics(graph) == BRANCH_HOVER_METRICS


def test_empty_branch_order_does_not_count_as_available():
    graph = a_graph(branch_order="")
    assert "order" not in available_branch_hover_metrics(graph)


def test_nan_flow_does_not_count_as_available():
    graph = a_graph(flow_abs=float("nan"))
    assert "flow" not in available_branch_hover_metrics(graph)


def test_tortuosity_is_path_over_straight():
    graph = a_graph()
    u, v, key, data = next(iter(graph.edges(keys=True, data=True)))
    # Straight vessels of length 10 along z -> tortuosity 1.0
    assert edge_tortuosity(graph, u, v, data) == pytest.approx(1.0)
    data["length"] = 20.0
    assert edge_tortuosity(graph, u, v, data) == pytest.approx(2.0)


def test_tortuosity_none_without_length_or_positions():
    graph = nx.MultiGraph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, key=0, length=5.0)
    assert edge_tortuosity(graph, 0, 1, graph[0][1][0]) is None
    graph.nodes[0]["pos"] = np.array([0.0, 0.0, 0.0])
    graph.nodes[1]["pos"] = np.array([0.0, 0.0, 0.0])  # zero distance
    assert edge_tortuosity(graph, 0, 1, graph[0][1][0]) is None


def test_format_branch_tooltip_always_includes_branch_id():
    text = format_branch_tooltip("12", {}, selected=())
    assert text == "branchID: 12"


def test_format_branch_tooltip_includes_only_selected_available_metrics():
    values = {
        "flow": 1.5e-12,
        "order": "A1",
        "resistance": 1e15,
        "tortuosity": 1.25,
        "length": 10.0,
    }
    text = format_branch_tooltip(
        "3",
        values,
        selected=("flow", "length"),
    )
    assert text == (
        "branchID: 3\n"
        f"branch flow: {format_metric_value('flow', 1.5e-12)}\n"
        f"branch length: {format_metric_value('length', 10.0)}"
    )


def test_format_branch_tooltip_excludes_unselected_and_missing():
    values = {"flow": 1.0, "length": 10.0, "order": None}
    text = format_branch_tooltip(
        "0",
        values,
        selected=("flow", "order", "resistance"),
    )
    assert text == (
        "branchID: 0\n"
        f"branch flow: {format_metric_value('flow', 1.0)}"
    )
    assert "branch order" not in text
    assert "branch resistance" not in text
    assert "branch length" not in text


def test_format_branch_tooltip_metric_order_is_declared_order():
    values = {
        m: ("X" if m in {"order", "diameter_source"} else 1.0)
        for m in BRANCH_HOVER_METRICS
    }
    text = format_branch_tooltip("1", values, selected=BRANCH_HOVER_METRICS)
    lines = text.split("\n")
    assert lines[0] == "branchID: 1"
    assert [line.split(":")[0] for line in lines[1:]] == [
        BRANCH_HOVER_LABELS[m] for m in BRANCH_HOVER_METRICS
    ]


def test_panel_metric_options_hide_unavailable():
    assert panel_metric_options(("length", "flow")) == (
        ("flow", "branch flow"),
        ("length", "branch length"),
    )
    assert panel_metric_options(()) == ()


def test_filter_selected_metrics_drops_unavailable_and_unknown():
    assert filter_selected_metrics(
        ("flow", "length", "nope", "order"),
        available=("length", "order"),
    ) == ("order", "length")


def test_default_selected_metrics_is_all_available_in_declared_order():
    assert default_selected_metrics(("length", "flow")) == ("flow", "length")


def test_branch_hover_rows_tooltips_match_formatter():
    graph = a_graph(flow_abs=2e-12, resistance=5e14, branch_order="B2")
    selected = ("flow", "order", "resistance", "tortuosity", "length")
    ids, features = branch_hover_rows(graph, selected)
    assert ids == ["0", "1", "2"]
    assert list(features["branch_id"]) == ["0", "1", "2"]
    first = format_branch_tooltip(
        "0",
        {
            "flow": 2e-12,
            "order": "B2",
            "resistance": 5e14,
            "tortuosity": 1.0,
            "length": 10.0,
        },
        selected,
    )
    assert features["tooltip"][0] == first


def test_tooltips_from_feature_table_respects_selection():
    graph = a_graph(flow_abs=1e-12)
    _ids, features = branch_hover_rows(graph, selected=("flow", "length"))
    only_id = tooltips_from_feature_table(features, selected=())
    assert only_id[0] == "branchID: 0"
    length_only = tooltips_from_feature_table(features, selected=("length",))
    assert length_only[0] == (
        "branchID: 0\n"
        f"branch length: {format_metric_value('length', 10.0)}"
    )


def test_available_metrics_from_features_matches_graph_detection():
    graph = a_graph(branch_order="A0", flow_abs=1e-12)
    _ids, features = branch_hover_rows(graph, selected=BRANCH_HOVER_METRICS)
    assert available_metrics_from_features(features) == (
        "flow",
        "order",
        "tortuosity",
        "length",
    )


def test_result_layers_put_tooltip_features_on_the_vessel_vectors():
    """Hover lives on the drawn branch, not a midpoint circle beside it."""
    group = built().stage_finished("build_network", network(a_graph()))
    names = [spec.name for spec in group.layers]
    assert BRANCH_HOVER not in names
    vessels = spec_named(group, VESSELS)
    assert vessels.kind == "vectors"
    assert set(vessels.features) >= {"branch_id", "tooltip", "length", "tortuosity"}
    assert vessels.features["tooltip"][0].startswith("branchID: ")
    assert vessels.options["branch_hover_available"] == ("tortuosity", "length")
    assert vessels.options["branch_hover_selected"] == ("tortuosity", "length")
    assert "size" not in vessels.options
    assert "face_color" not in vessels.options


def test_branch_hover_hides_flow_until_attrs_exist_on_graph():
    """Flow is offered only once edges carry ``flow_abs`` (after a solve)."""
    from types import SimpleNamespace

    graph = a_graph()
    results = built(graph)
    before = spec_named(
        results.stage_finished("build_network", network(graph)),
        VESSELS,
    )
    assert before.options["branch_hover_available"] == ("tortuosity", "length")

    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["flow_abs"] = 1e-12
        data["resistance"] = 1e15
        data["branch_order"] = "C0"
        data["diameter_um"] = 5.0
        data["diameter_source"] = "measured"

    group = results.stage_finished(
        "solve",
        SimpleNamespace(
            pressure=np.zeros(4),
            node_list=[0, 1, 2, 3],
            equivalent_resistance=1.0,
        ),
    )
    vessels = spec_named(group, VESSELS)
    assert vessels.options["branch_hover_available"] == BRANCH_HOVER_METRICS


def test_vessel_labels_layer_still_emitted_without_a_hover_circle():
    group = built().stage_finished("build_network", network(a_graph()))
    names = [spec.name for spec in group.layers]
    assert names.count(VESSELS) == 1
    assert names.count(VESSEL_LABELS) == 1
    assert BRANCH_HOVER not in names


def test_hover_features_repeat_across_polyline_segments():
    graph = a_graph()
    paths, identity = edge_polylines(graph)
    _vectors, owner = polylines_to_vectors(paths)
    features, available, selected = hover_features_for_segments(graph, owner)
    assert available == ("tortuosity", "length")
    assert selected == ("tortuosity", "length")
    assert len(features["tooltip"]) == len(owner)
    expected = [str(int(i)) for i in np.asarray(identity["edge_index"])[owner]]
    assert list(features["branch_id"]) == expected


def test_vessel_tooltip_branch_id_matches_edge_index_column():
    """Hover ``branchID`` is the same number colour-by ``edge_index`` uses."""
    graph = a_graph()
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["segment_id"] = 0
    vessels = spec_named(
        built(graph).stage_finished("build_network", network(graph)),
        VESSELS,
    )
    branch_ids = [str(v) for v in vessels.features["branch_id"]]
    edge_ids = [str(int(v)) for v in vessels.features["edge_index"]]
    assert branch_ids == edge_ids
    assert set(branch_ids) == {"0", "1", "2"}


def test_nearest_vector_index_hits_along_the_polyline_not_only_the_midpoint():
    """A point on the centreline far from the old midpoint circle still hits."""
    paths, _ = edge_polylines(a_graph())
    vectors, owner = polylines_to_vectors(paths)
    # First vessel runs (0,0,0) -> (10,0,0); midpoint was (5,0,0).
    near_end = nearest_vector_index(
        (1.0, 0.0, 0.0), vectors, max_distance=BRANCH_HOVER_MAX_DISTANCE
    )
    assert near_end is not None
    assert int(owner[near_end]) == 0
    midpoint = nearest_vector_index(
        (5.0, 0.0, 0.0), vectors, max_distance=BRANCH_HOVER_MAX_DISTANCE
    )
    assert midpoint is not None
    assert int(owner[midpoint]) == 0
    second = nearest_vector_index(
        (15.0, 0.0, 0.0), vectors, max_distance=BRANCH_HOVER_MAX_DISTANCE
    )
    assert second is not None
    assert int(owner[second]) == 1


def test_nearest_vector_index_misses_off_the_branch():
    paths, _ = edge_polylines(a_graph())
    vectors, _owner = polylines_to_vectors(paths)
    assert nearest_vector_index(
        (5.0, 50.0, 0.0), vectors, max_distance=BRANCH_HOVER_MAX_DISTANCE
    ) is None
    assert nearest_vector_index(
        (5.0, 0.0, 0.0), vectors, max_distance=0.1
    ) is not None
    assert nearest_vector_index(
        (0.0, 0.0, 0.0), np.empty((0, 2, 3))
    ) is None


def test_nearest_vector_index_projects_to_the_view_plane():
    """A 3D camera offset along the view ray still hits the centreline."""
    vectors = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], dtype=float)
    # Cursor is 8 µm "in front" of the vessel along +y; same on-screen location.
    index = nearest_vector_index(
        (5.0, 8.0, 0.0),
        vectors,
        max_distance=BRANCH_HOVER_MAX_DISTANCE,
        view_direction=(0.0, 1.0, 0.0),
    )
    assert index == 0
    miss = nearest_vector_index(
        (5.0, 8.0, 20.0),
        vectors,
        max_distance=BRANCH_HOVER_MAX_DISTANCE,
        view_direction=(0.0, 1.0, 0.0),
    )
    assert miss is None


def test_flow_direction_layer_carries_the_same_tooltip_table():
    """Flow-direction arrows get the same hover strings as the vessel polylines."""
    from types import SimpleNamespace

    graph = a_graph(flow_signed=1.0, flow_abs=1.5e-12)
    results = built(graph)
    results.settings["show_flow_direction_layer"] = True
    group = results.stage_finished("export_results", SimpleNamespace())
    flow = spec_named(group, FLOW_DIRECTION)
    assert "tooltip" in flow.features
    assert "branch_id" in flow.features
    assert str(flow.features["tooltip"][0]).startswith("branchID: ")
    assert len(flow.features["tooltip"]) == len(flow.data)
    origin = np.asarray(flow.data[0, 0], dtype=float)
    hit = nearest_vector_index(
        origin, flow.data, max_distance=BRANCH_HOVER_MAX_DISTANCE
    )
    assert hit == 0
