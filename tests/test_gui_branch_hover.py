"""Branch-hover tooltip helpers: availability, formatting, panel options.

Pure logic for the napari midpoint layer that shows branchID (always) plus
user-selected metrics. No napari / Qt imports -- those live in the widget tests.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.gui.branch_hover import (
    BRANCH_HOVER_LABELS,
    BRANCH_HOVER_METRICS,
    available_branch_hover_metrics,
    available_metrics_from_features,
    branch_hover_rows,
    branch_id_for_edge,
    default_selected_metrics,
    edge_tortuosity,
    filter_selected_metrics,
    format_branch_tooltip,
    format_metric_value,
    panel_metric_options,
    tooltips_from_feature_table,
)
from test_gui_results import a_graph, built, network, spec_named
from haemolynx.gui.results import BRANCH_HOVER, VESSEL_LABELS, VESSELS


def graph_with(**attrs) -> nx.MultiGraph:
    """``a_graph`` plus per-edge attributes (flow, order, resistance, ...)."""
    return a_graph(**attrs)


def test_branch_id_prefers_segment_id_over_edge_key():
    data = {"segment_id": 7}
    assert branch_id_for_edge(0, 1, key=99, data=data) == "7"
    assert branch_id_for_edge(0, 1, key=99, data={}) == "99"


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
    values = {m: 1.0 if m != "order" else "X" for m in BRANCH_HOVER_METRICS}
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


def test_result_layers_emit_branch_hover_with_tooltip_features():
    group = built().stage_finished("build_network", network(a_graph()))
    hover = spec_named(group, BRANCH_HOVER)
    assert hover.kind == "points"
    assert hover.visible is True
    assert set(hover.features) >= {"branch_id", "tooltip", "length", "tortuosity"}
    assert hover.features["tooltip"][0].startswith("branchID: ")
    assert hover.options["branch_hover_available"] == ("tortuosity", "length")
    assert hover.options["branch_hover_selected"] == ("tortuosity", "length")
    # Must stay at the vessel-label midpoint size (not the brief 8.0 enlargement).
    from haemolynx.gui.results import BRANCH_HOVER_POINT_SIZE, VESSEL_LABEL_POINT_SIZE

    assert hover.options["size"] == BRANCH_HOVER_POINT_SIZE == VESSEL_LABEL_POINT_SIZE


def test_branch_hover_hides_flow_until_attrs_exist_on_graph():
    """Flow is offered only once edges carry ``flow_abs`` (after a solve)."""
    from types import SimpleNamespace

    graph = a_graph()
    results = built(graph)
    before = spec_named(
        results.stage_finished("build_network", network(graph)),
        BRANCH_HOVER,
    )
    assert before.options["branch_hover_available"] == ("tortuosity", "length")

    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["flow_abs"] = 1e-12
        data["resistance"] = 1e15
        data["branch_order"] = "C0"

    group = results.stage_finished(
        "solve",
        SimpleNamespace(
            pressure=np.zeros(4),
            node_list=[0, 1, 2, 3],
            equivalent_resistance=1.0,
        ),
    )
    hover = spec_named(group, BRANCH_HOVER)
    assert hover.options["branch_hover_available"] == BRANCH_HOVER_METRICS


def test_vessel_labels_layer_still_emitted_alongside_branch_hover():
    group = built().stage_finished("build_network", network(a_graph()))
    names = [spec.name for spec in group.layers]
    assert names.count(VESSELS) == 1
    assert names.count(VESSEL_LABELS) == 1
    assert names.count(BRANCH_HOVER) == 1
