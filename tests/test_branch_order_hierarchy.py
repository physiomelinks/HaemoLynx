"""Integration-style test and demo visualization for hierarchical branch orders."""
import re
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

# Ensure package import works when running this file directly.
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from haemolynx.graph import (
    MissingSmallVesselAssignmentWarning,
    assign_hierarchical_branch_orders,
    assign_vessel_branch_orders,
)


def _edge_id(u: int, v: int, key: int) -> tuple[int, int, int]:
    return (u, v, key) if u <= v else (v, u, key)


def _edge_ids_from_node_path(
    G: nx.MultiGraph, path: list[int]
) -> list[tuple[int, int, int]]:
    ids: list[tuple[int, int, int]] = []
    for u, v in zip(path[:-1], path[1:]):
        keys = list(G[u][v].keys())
        assert keys, f"No edge found between path nodes {u} and {v}"
        ids.append(_edge_id(u, v, keys[0]))
    return ids


def _extract_order_number(label: str) -> int:
    m = re.search(r"(\d+)$", label)
    assert m is not None, f"Missing numeric suffix in branch label: {label}"
    return int(m.group(1))


def _vessel_type_from_branch_label(label: str) -> str:
    if label.startswith("Art"):
        return "arteriole"
    if label.startswith("Ven"):
        return "venule"
    if label.startswith("B"):
        return "capillary"
    return "unknown"


def _build_demo_graph() -> tuple[nx.MultiGraph, list[int], list[int], list[int], list[int]]:
    """
    Build one arteriole-to-venule capillary tree:
    input -> arteriole trunk -> arteriole boundary -> branching capillaries -> venule boundary -> venule trunk -> output
    """
    G = nx.MultiGraph()
    node_positions = {
        0: (0.0, 0.0, 0.0),   # input
        1: (1.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),   # arteriole boundary
        3: (3.0, 1.0, 0.3),
        4: (3.0, -1.0, -0.3),
        5: (4.0, 1.2, 0.4),
        6: (4.0, -1.2, -0.4),  # venule boundary
        7: (5.0, 0.0, 0.0),
        8: (6.0, 0.0, 0.0),
        9: (7.0, 0.0, 0.0),   # output
        10: (1.0, 0.8, 0.2),  # arteriole side branch
        11: (6.0, -0.8, -0.2),  # venule side branch
    }
    for node, pos in node_positions.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))

    # Arteriole side.
    for u, v in [(0, 1), (1, 2), (1, 10)]:
        G.add_edge(u, v, length=1.0, weight=1.0)

    # Capillary bed branching from one arteriole boundary to one venule boundary.
    for u, v in [(2, 3), (2, 4), (3, 5), (4, 6), (5, 6)]:
        G.add_edge(u, v, length=1.0, weight=1.0)

    # Venule side.
    for u, v in [(6, 7), (7, 8), (8, 9), (8, 11)]:
        G.add_edge(u, v, length=1.0, weight=1.0)

    input_nodes = [0]
    outlet_nodes = [9]
    arteriole_boundary_nodes = [2]
    venule_boundary_nodes = [6]
    return G, input_nodes, outlet_nodes, arteriole_boundary_nodes, venule_boundary_nodes


def _render_rotatable_3d_html(
    G: nx.MultiGraph,
    output_html_path: Path,
    input_nodes: list[int],
    outlet_nodes: list[int],
    arteriole_boundary_nodes: list[int],
    venule_boundary_nodes: list[int],
) -> Path:
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise RuntimeError(
            "plotly is required for 3D visualization. Install with `pip install plotly`."
        ) from exc

    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "arteriole": "#d62728",  # red
        "capillary": "#2ca02c",  # green
        "venule": "#1f77b4",  # blue
        "unknown": "#888888",
    }
    display_names = {
        "arteriole": "Arterioles",
        "capillary": "Capillaries",
        "venule": "Venules",
        "unknown": "Unknown",
    }

    traces = []
    for vessel_type in ("arteriole", "capillary", "venule", "unknown"):
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        edge_count = 0
        for u, v, key, data in G.edges(keys=True, data=True):
            label = str(data.get("branch_order", "unknown"))
            if _vessel_type_from_branch_label(label) != vessel_type:
                continue
            pu = np.asarray(G.nodes[u]["pos"], dtype=float)
            pv = np.asarray(G.nodes[v]["pos"], dtype=float)
            xs.extend([float(pu[0]), float(pv[0]), None])
            ys.extend([float(pu[1]), float(pv[1]), None])
            zs.extend([float(pu[2]), float(pv[2]), None])
            edge_count += 1
        if edge_count == 0:
            continue
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=f"{display_names[vessel_type]} ({edge_count})",
                line={"color": colors[vessel_type], "width": 6},
            )
        )

    node_x = []
    node_y = []
    node_z = []
    node_labels = []
    node_text = []
    for n, attrs in G.nodes(data=True):
        p = np.asarray(attrs["pos"], dtype=float)
        node_x.append(float(p[0]))
        node_y.append(float(p[1]))
        node_z.append(float(p[2]))
        node_labels.append(str(n))
        node_text.append(f"Node {n}")

    traces.append(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers+text",
            text=node_labels,
            textposition="top center",
            marker={"size": 4, "color": "black"},
            name="Nodes",
            hovertext=node_text,
            hoverinfo="text",
        )
    )

    # Add branch-order labels at edge midpoints.
    edge_label_x = []
    edge_label_y = []
    edge_label_z = []
    edge_label_text = []
    for u, v, key, data in G.edges(keys=True, data=True):
        pu = np.asarray(G.nodes[u]["pos"], dtype=float)
        pv = np.asarray(G.nodes[v]["pos"], dtype=float)
        midpoint = (pu + pv) / 2.0
        edge_label_x.append(float(midpoint[0]))
        edge_label_y.append(float(midpoint[1]))
        edge_label_z.append(float(midpoint[2]))
        edge_label_text.append(str(data.get("branch_order", "No_BO")))
    traces.append(
        go.Scatter3d(
            x=edge_label_x,
            y=edge_label_y,
            z=edge_label_z,
            mode="text",
            text=edge_label_text,
            textposition="middle center",
            textfont={"size": 11, "color": "black"},
            name="Branch order labels",
            hoverinfo="skip",
        )
    )

    # Mark assigned special nodes with thin black arrows and labels above nodes.
    special_nodes = [
        ("Input", input_nodes),
        ("Output", outlet_nodes),
        ("Terminal arteriole", arteriole_boundary_nodes),
        ("Terminal venule", venule_boundary_nodes),
    ]
    label_x = []
    label_y = []
    label_z = []
    label_text = []
    for label, nodes in special_nodes:
        for node_id in nodes:
            p = np.asarray(G.nodes[node_id]["pos"], dtype=float)
            label_x.append(float(p[0]))
            label_y.append(float(p[1]))
            label_z.append(float(p[2] + 0.28))
            label_text.append(f"{label} assigned (node {node_id})")
    if label_x:
        traces.append(
            go.Scatter3d(
                x=label_x,
                y=label_y,
                z=label_z,
                mode="text",
                text=label_text,
                textfont={"size": 12, "color": "black"},
                name="Assignment labels",
                hoverinfo="text",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Hierarchical Vessel Types (Art=Red, Capillary=Green, Ven=Blue)",
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    fig.write_html(str(output_html_path), include_plotlyjs="cdn")
    return output_html_path


def test_hierarchical_branch_order_pipeline_flow():
    G, input_nodes, outlet_nodes, arteriole_boundary_nodes, venule_boundary_nodes = (
        _build_demo_graph()
    )

    assign_hierarchical_branch_orders(
        G,
        inlet_nodes=input_nodes,
        outlet_nodes=outlet_nodes,
        arteriole_boundary_nodes=arteriole_boundary_nodes,
        venule_boundary_nodes=venule_boundary_nodes,
    )

    input_to_art_path = nx.shortest_path(G, input_nodes[0], arteriole_boundary_nodes[0])
    output_to_ven_path = nx.shortest_path(G, outlet_nodes[0], venule_boundary_nodes[0])
    art_to_ven_path = nx.shortest_path(
        G, arteriole_boundary_nodes[0], venule_boundary_nodes[0]
    )

    art_path_edge_ids = _edge_ids_from_node_path(G, input_to_art_path)
    ven_path_edge_ids = _edge_ids_from_node_path(G, output_to_ven_path)
    cap_path_edge_ids = _edge_ids_from_node_path(G, art_to_ven_path)

    art_orders = []
    for edge_id in art_path_edge_ids:
        u, v, key = edge_id
        label = G[u][v][key]["branch_order"]
        assert label.startswith("Art")
        art_orders.append(_extract_order_number(label))
    assert art_orders == sorted(art_orders)

    ven_orders = []
    for edge_id in ven_path_edge_ids:
        u, v, key = edge_id
        label = G[u][v][key]["branch_order"]
        assert label.startswith("Ven")
        ven_orders.append(_extract_order_number(label))
    assert ven_orders == sorted(ven_orders)

    for edge_id in cap_path_edge_ids:
        if edge_id in art_path_edge_ids or edge_id in ven_path_edge_ids:
            continue
        u, v, key = edge_id
        label = G[u][v][key]["branch_order"]
        assert label.startswith("B")

    assert G[1][10][0]["branch_order"].startswith("Art")
    assert G[8][11][0]["branch_order"].startswith("Ven")
    assert G[2][3][0]["branch_order"].startswith("B")
    assert G[2][4][0]["branch_order"].startswith("B")


def _build_linear_capillary_chain() -> nx.MultiGraph:
    """Four nodes in a line: inlet 0 → 1 → 2 → 3, for deterministic B01..B03."""
    G = nx.MultiGraph()
    for node, x in enumerate((0.0, 1.0, 2.0, 3.0)):
        G.add_node(node, pos=np.asarray((x, 0.0, 0.0), dtype=float))
    for u, v in ((0, 1), (1, 2), (2, 3)):
        G.add_edge(u, v, length=1.0)
    return G


def test_strict_without_small_vessels_assigns_capillary_orders_and_warns():
    G = _build_linear_capillary_chain()

    with pytest.warns(
        MissingSmallVesselAssignmentWarning,
        match="small arterioles and venules were not assigned",
    ):
        summary = assign_vessel_branch_orders(
            G,
            inlet_nodes=[0],
            outlet_nodes=[3],
            arteriole_boundary_nodes=[],
            venule_boundary_nodes=[],
            strict_hierarchical=True,
            expects_hierarchical=False,
        )

    assert summary["mode"] == "capillary"
    assert G[0][1][0]["branch_order"] == "B01"
    assert G[1][2][0]["branch_order"] == "B02"
    assert G[2][3][0]["branch_order"] == "B03"


def test_strict_with_small_vessel_terminals_keeps_hierarchical_path_without_warning():
    G, input_nodes, outlet_nodes, arteriole_boundary_nodes, venule_boundary_nodes = (
        _build_demo_graph()
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingSmallVesselAssignmentWarning)
        summary = assign_vessel_branch_orders(
            G,
            inlet_nodes=input_nodes,
            outlet_nodes=outlet_nodes,
            arteriole_boundary_nodes=arteriole_boundary_nodes,
            venule_boundary_nodes=venule_boundary_nodes,
            strict_hierarchical=True,
            expects_hierarchical=True,
        )

    assert summary["mode"] == "hierarchical"
    assert G[0][1][0]["branch_order"] == "Art1"
    assert G[1][2][0]["branch_order"] == "Art2"
    assert G[2][3][0]["branch_order"] == "B01"
    assert G[8][9][0]["branch_order"] == "Ven1"


def test_strict_when_small_vessel_assignment_expected_but_missing_still_raises():
    G = _build_linear_capillary_chain()

    with pytest.raises(ValueError, match="hierarchical assignment prerequisites"):
        assign_vessel_branch_orders(
            G,
            inlet_nodes=[0],
            outlet_nodes=[3],
            arteriole_boundary_nodes=[],
            venule_boundary_nodes=[],
            strict_hierarchical=True,
            expects_hierarchical=True,
        )


if __name__ == "__main__":
    G, input_nodes, outlet_nodes, arteriole_boundary_nodes, venule_boundary_nodes = (
        _build_demo_graph()
    )
    assign_hierarchical_branch_orders(
        G,
        inlet_nodes=input_nodes,
        outlet_nodes=outlet_nodes,
        arteriole_boundary_nodes=arteriole_boundary_nodes,
        venule_boundary_nodes=venule_boundary_nodes,
    )
    html_path = _render_rotatable_3d_html(
        G,
        Path(__file__).resolve().parents[1]
        / "examples"
        / "plots"
        / "branch_order_hierarchy_demo_3d.html",
        input_nodes=input_nodes,
        outlet_nodes=outlet_nodes,
        arteriole_boundary_nodes=arteriole_boundary_nodes,
        venule_boundary_nodes=venule_boundary_nodes,
    )
    print(f"Saved rotatable 3D vessel-type visualization to: {html_path}")
