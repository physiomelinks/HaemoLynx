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
    BRANCH_ORDER_CATEGORY_SEQUENCE,
    LARGE_MASK_VESSEL_TYPE,
    SMALL_MASK_VESSEL_TYPE,
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


def test_hierarchical_branch_order_assigns_large_art_and_large_ven_tiers():
    """A Large_Art/Large_Ven tier sits outside Art/Ven, handing off at the
    existing arteriole/venule inlet-outlet nodes -- reusing those exact
    nodes as the hand-off point means every pre-existing Art/B/Ven
    assertion from test_hierarchical_branch_order_pipeline_flow must still
    hold unchanged; only the two newly prepended edges are new territory."""
    G, input_nodes, outlet_nodes, arteriole_boundary_nodes, venule_boundary_nodes = (
        _build_demo_graph()
    )
    # True image-edge stumps, one hop upstream/downstream of the graph's
    # existing inlet/outlet, which becomes the large-vessel boundary (the
    # point where the large-vessel mask's own coverage ends).
    true_inlet, true_outlet = 100, 101
    G.add_node(true_inlet, pos=np.asarray((-1.0, 0.0, 0.0), dtype=float))
    G.add_node(true_outlet, pos=np.asarray((8.0, 0.0, 0.0), dtype=float))
    G.add_edge(
        true_inlet, input_nodes[0], length=1.0, weight=1.0,
        **{LARGE_MASK_VESSEL_TYPE: "arteriole"},
    )
    G.add_edge(
        outlet_nodes[0], true_outlet, length=1.0, weight=1.0,
        **{LARGE_MASK_VESSEL_TYPE: "venule"},
    )
    large_arteriole_boundary_nodes = list(input_nodes)
    large_venule_boundary_nodes = list(outlet_nodes)

    summary = assign_vessel_branch_orders(
        G,
        inlet_nodes=[true_inlet],
        outlet_nodes=[true_outlet],
        arteriole_boundary_nodes=arteriole_boundary_nodes,
        venule_boundary_nodes=venule_boundary_nodes,
        large_arteriole_boundary_nodes=large_arteriole_boundary_nodes,
        large_venule_boundary_nodes=large_venule_boundary_nodes,
    )

    assert summary["mode"] == "hierarchical"
    assert summary["large_arteriole_edge_count"] == 1
    assert summary["large_venule_edge_count"] == 1
    assert G[true_inlet][input_nodes[0]][0]["branch_order"] == "Large_Art1"
    assert G[outlet_nodes[0]][true_outlet][0]["branch_order"] == "Large_Ven1"

    # Every pre-existing Art/B/Ven assertion holds unchanged: the hand-off
    # nodes are the same nodes the old inlet/outlet already were.
    assert G[0][1][0]["branch_order"] == "Art1"
    assert G[1][2][0]["branch_order"] == "Art2"
    assert G[1][10][0]["branch_order"].startswith("Art")
    assert G[2][3][0]["branch_order"].startswith("B")
    assert G[2][4][0]["branch_order"].startswith("B")
    assert G[8][9][0]["branch_order"] == "Ven1"
    assert G[8][11][0]["branch_order"].startswith("Ven")


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


# --- BRANCH_ORDER_CATEGORY_SEQUENCE is the one place this hierarchy lives --


def test_branch_order_category_sequence_is_the_arterial_to_venous_hierarchy():
    """Pins the exact hierarchy statistics.bifurcation and
    visualization._helpers both derive their own sort-rank tables from --
    see that constant's own docstring for why nothing should go back to
    hand-copying this order."""
    assert BRANCH_ORDER_CATEGORY_SEQUENCE == (
        "large_arteriole",
        "arteriole",
        "capillary",
        "venule",
        "large_venule",
    )


def test_statistics_and_visualization_branch_order_ranks_agree():
    """Regression: statistics.bifurcation and visualization._helpers used to
    each keep an independent, hand-written copy of this ranking, with
    nothing checking they agreed. Both now derive their rank tables from
    BRANCH_ORDER_CATEGORY_SEQUENCE's own order, so this only re-confirms
    the derivation -- a reversion to independent hardcoded dicts would show
    up here as soon as one drifted from the other."""
    from haemolynx.statistics.bifurcation import _BRANCH_ORDER_SORT_GROUPS as stats_groups
    from haemolynx.visualization._helpers import _BRANCH_ORDER_SORT_GROUPS as viz_groups

    shared_prefixes = ("large_art", "art", "bo", "ven", "large_ven")
    assert all(prefix in stats_groups for prefix in shared_prefixes)
    assert all(prefix in viz_groups for prefix in shared_prefixes)
    for prefix in shared_prefixes:
        assert stats_groups[prefix] == viz_groups[prefix], (
            f"{prefix!r} ranks differently in statistics.bifurcation "
            f"({stats_groups[prefix]}) and visualization._helpers "
            f"({viz_groups[prefix]})"
        )
    # visualization._helpers additionally accepts the raw "b" prefix
    # (statistics.bifurcation normalizes "B01" to "BO1" before ever
    # consulting its own table -- see _normalize_branch_order_tag) -- it
    # must rank the same as "bo", not a different category.
    assert viz_groups["b"] == viz_groups["bo"]


def test_a_bridge_is_no_order_and_no_step():
    """A zero-resistance bridge is the radius of the thick vessel a vessel
    opens into: it gets no branch order, and the vessel beyond it is one
    order from the thick vessel, not two -- with or without the hierarchy."""
    from haemolynx.graph import IS_ZERO_RESISTANCE

    def network():
        G = nx.MultiGraph()
        G.add_edge(0, 1)  # thick vessel from the inlet
        G.add_edge(1, 2)  # thick vessel on to the outlet side
        G.add_edge(1, 3, **{IS_ZERO_RESISTANCE: True, "branch_order": "B02"})
        G.add_edge(3, 4)  # the vessel opening into it
        G.add_edge(4, 5)
        G.add_edge(5, 2)
        return G

    G = network()
    summary = assign_vessel_branch_orders(G, [0])
    assert summary["mode"] == "capillary"
    assert "branch_order" not in G.edges[1, 3, 0]
    assert G.edges[0, 1, 0]["branch_order"] == "B01"
    assert G.edges[3, 4, 0]["branch_order"] == "B02"
    assert G.edges[4, 5, 0]["branch_order"] == "B03"

    G = network()
    assign_hierarchical_branch_orders(
        G, inlet_nodes=[0], outlet_nodes=[2], arteriole_boundary_nodes=[1],
        venule_boundary_nodes=[5],
    )
    assert "branch_order" not in G.edges[1, 3, 0]
    assert all(
        d.get("branch_order") for u, v, d in G.edges(data=True) if not d.get(IS_ZERO_RESISTANCE)
    )


# --- The Large_Art/Large_Ven tier on a real large vessel ---------------------
#
# Found on a solved E14.5 run (outputs/vessel_classes_check/REPORT.md, findings
# 2 and 3): the Large tier stopped at the first side branch, and the Art tier
# then crossed the capillary bed into the large vein.


def _labels(G: nx.MultiGraph) -> dict[tuple, str]:
    return {
        _edge_id(u, v, k): d.get("branch_order") for u, v, k, d in G.edges(keys=True, data=True)
    }


def _tiered_graph(large_arteriole, large_venule, other) -> nx.MultiGraph:
    """Edges given as node pairs; the first two groups tagged as lying in the
    large arteriole and the large venule mask."""
    G = nx.MultiGraph()
    for edges, tag in ((large_arteriole, "arteriole"), (large_venule, "venule"), (other, None)):
        for u, v in edges:
            G.add_edge(u, v, length=1.0, **({LARGE_MASK_VESSEL_TYPE: tag} if tag else {}))
    return G


def _large_arteriole_with_side_branches():
    """A large arteriole 0-1-2-3-4 (inlet 0) whose mask ends at 4. Small
    arterioles leave it at 1 and 2 and carry on from 4, each meeting the
    capillary bed at its arteriole boundary node (10, 20, 40). The bed drains
    through a small venule (boundary 60) into a large vein 61-62-63 (outlet 63).

    The large boundary nodes are what the large masks' labelling reports:
    every labelled node with an edge outside its mask, here every junction on
    the trunk where a side branch leaves (1, 2) and the mask's end (4).
    """
    G = _tiered_graph(
        large_arteriole=[(0, 1), (1, 2), (2, 3), (3, 4)],
        large_venule=[(61, 62), (62, 63)],
        other=[
            (1, 10), (2, 20), (4, 40),  # small arterioles
            (10, 50), (20, 50), (40, 51), (50, 52), (51, 52), (52, 60),  # capillaries
            (60, 61),  # small venule
        ],
    )
    return G, dict(
        inlet_nodes=[0],
        outlet_nodes=[63],
        arteriole_boundary_nodes=[10, 20, 40],
        venule_boundary_nodes=[60],
        large_arteriole_boundary_nodes=[1, 2, 4],
        large_venule_boundary_nodes=[61],
    )


def _capillaries_from_large_arteriole_to_large_vein():
    """A large arteriole 0-1-2 (inlet 0) and a large vein 8-9-10 (outlet 10).
    A capillary leaves the arteriole at 1 and joins the vein at 8 with no small
    vessel on either side; a small arteriole leaves at 2 and meets the bed at
    3, which drains through a small venule (boundary 7) into the vein at 8.
    """
    G = _tiered_graph(
        large_arteriole=[(0, 1), (1, 2)],
        large_venule=[(8, 9), (9, 10)],
        other=[(1, 5), (5, 8), (2, 3), (3, 4), (4, 7), (7, 8)],
    )
    return G, dict(
        inlet_nodes=[0],
        outlet_nodes=[10],
        arteriole_boundary_nodes=[3],
        venule_boundary_nodes=[7],
        large_arteriole_boundary_nodes=[1, 2],
        large_venule_boundary_nodes=[8],
    )


def test_the_large_tier_runs_along_the_large_vessel_past_its_side_branches():
    """Regression: the Large_Art search from the inlet stopped at every large
    boundary node, and every junction where a side branch leaves the large
    arteriole is one -- so only the inlet's own vessel was Large_Art and the
    rest of the trunk became Art (E14.5: 1 of 48 vessels)."""
    G, nodes = _large_arteriole_with_side_branches()

    summary = assign_vessel_branch_orders(G, **nodes)

    labels = _labels(G)
    assert summary["mode"] == "hierarchical"
    assert [labels[e] for e in [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0)]] == [
        "Large_Art1", "Large_Art2", "Large_Art3", "Large_Art4",
    ]
    assert summary["large_arteriole_edge_count"] == 4
    # Each side branch is an arteriole of the first order from the large one.
    assert labels[(1, 10, 0)] == labels[(2, 20, 0)] == labels[(4, 40, 0)] == "Art1"
    assert [labels[e] for e in [(10, 50, 0), (20, 50, 0), (40, 51, 0)]] == ["B01"] * 3
    assert labels[(50, 52, 0)] == labels[(51, 52, 0)] == "B02"
    assert labels[(52, 60, 0)] == "B03"
    assert labels[(60, 61, 0)] == "Ven1"
    assert labels[(62, 63, 0)] == "Large_Ven1"
    assert labels[(61, 62, 0)] == "Large_Ven2"


def test_the_arteriole_tier_does_not_cross_the_capillary_bed_into_the_large_vein():
    """Regression: the Art search started at every large boundary node and was
    stopped only by arteriole boundary nodes, so a capillary leaving the large
    arteriole directly took it across the bed into the large vein before the
    venous tiers ran (E14.5: the outlet's own vessel was Art11, Large_Ven had
    no vessel). A capillary leaving a large vessel is a capillary."""
    G, nodes = _capillaries_from_large_arteriole_to_large_vein()

    summary = assign_vessel_branch_orders(G, **nodes)

    labels = _labels(G)
    assert labels[(0, 1, 0)] == "Large_Art1"
    assert labels[(1, 2, 0)] == "Large_Art2"
    assert labels[(2, 3, 0)] == "Art1"
    assert labels[(9, 10, 0)] == "Large_Ven1"
    assert labels[(8, 9, 0)] == "Large_Ven2"
    assert labels[(7, 8, 0)] == "Ven1"
    # The capillary from the large arteriole to the large vein, counted from
    # the arteriole it leaves; and the bed beyond the small arteriole.
    assert labels[(1, 5, 0)] == "B01"
    assert labels[(5, 8, 0)] == "B02"
    assert labels[(3, 4, 0)] == "B01"
    assert labels[(4, 7, 0)] == "B02"
    assert summary["arteriole_edge_count"] == 1
    assert summary["large_venule_edge_count"] == 2
    venous_side = {7, 8, 9, 10}
    assert not [
        e for e, label in labels.items()
        if label.startswith("Art") and set(e[:2]) & venous_side
    ]


def test_the_tiers_are_the_same_whichever_node_ids_the_vessels_carry():
    """The tiers come from the graph's shape and tags, not from the order its
    nodes were numbered or its edges added in."""
    G, nodes = _capillaries_from_large_arteriole_to_large_vein()
    assign_vessel_branch_orders(G, **nodes)
    relabel = {n: 100 - n for n in G.nodes}
    H = nx.MultiGraph()
    for u, v, d in reversed(list(G.edges(data=True))):
        H.add_edge(relabel[v], relabel[u], **{k: x for k, x in d.items() if k != "branch_order"})
    renamed = {name: [relabel[n] for n in ids] for name, ids in nodes.items()}

    assign_vessel_branch_orders(H, **renamed)

    assert {
        frozenset((u, v)): d["branch_order"] for u, v, d in G.edges(data=True)
    } == {
        frozenset((100 - u, 100 - v)): d["branch_order"] for u, v, d in H.edges(data=True)
    }


def test_a_large_vessel_mask_tag_alone_does_not_add_a_large_tier():
    """With no large boundary nodes (no Large tier asked for) the orders are
    what they were before the large-vessel tags existed, tags or not."""
    untagged, nodes = _capillaries_from_large_arteriole_to_large_vein()
    for data in untagged.edges.values():
        data.pop(LARGE_MASK_VESSEL_TYPE, None)
    tagged, _ = _capillaries_from_large_arteriole_to_large_vein()
    plain = {k: v for k, v in nodes.items() if not k.startswith("large_")}

    assign_vessel_branch_orders(untagged, **plain)
    assign_vessel_branch_orders(tagged, **plain)

    assert _labels(tagged) == _labels(untagged)
    # Today's rule: Art from the inlet up to the arteriole boundary nodes, Ven
    # from the outlet up to the venule boundary nodes. Pinned so the fix to
    # the Large tier cannot move it.
    assert _labels(untagged) == {
        (0, 1, 0): "Art1", (1, 2, 0): "Art2", (1, 5, 0): "Art2", (2, 3, 0): "Art3",
        (5, 8, 0): "Art3", (8, 9, 0): "Art4", (9, 10, 0): "Art5", (7, 8, 0): "Art4",
        (4, 7, 0): "Art5", (3, 4, 0): "Art4",
    }


def test_a_large_tier_with_no_tagged_vessel_at_the_inlet_falls_back_to_the_plain_tiers():
    """Large boundary nodes but no vessel tagged in a large mask reaching the
    inlet or outlet (a graph made before the tags, or an inlet off the large
    vessel): there is no Large tier, and Art and Ven run from the inlets and
    outlets as they do without one, with a warning."""
    G, nodes = _large_arteriole_with_side_branches()
    for data in G.edges.values():
        data.pop(LARGE_MASK_VESSEL_TYPE, None)
    expected, _ = _large_arteriole_with_side_branches()
    assign_vessel_branch_orders(
        expected, **{k: v for k, v in nodes.items() if not k.startswith("large_")}
    )

    with pytest.warns(UserWarning, match="no vessel tagged in the large"):
        summary = assign_vessel_branch_orders(G, **nodes)

    assert summary["large_arteriole_edge_count"] == summary["large_venule_edge_count"] == 0
    assert _labels(G) == _labels(expected)


def _segment(start, end) -> list[tuple[float, float, float]]:
    """Points every micron from *start* to *end*, both included."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    steps = max(int(np.ceil(np.linalg.norm(end - start))), 1)
    return [tuple(start + (end - start) * t) for t in np.linspace(0.0, 1.0, steps + 1)]


def _branching_large_vessel_network():
    """1 um voxels, every vessel in the plane y = 4. A large arteriole runs
    along z at x = 5 from the z = 0 face (a0-a1-a2-a3) and a large vein
    along z at x = 25 (v0-v1-v2-v3), each in its own 5 x 5 um mask. A small
    arteriole leaves the arteriole at a1 (z = 10) and its bed drains through
    a small venule into the vein at v1; a capillary leaves the arteriole at
    a2 (z = 20) and joins the vein at v2 directly."""
    from haemolynx.pipeline.stages import SkeletonisedVolume, VesselNetwork

    shape = (40, 9, 30)
    pos = {
        "a0": (0, 4, 5), "a1": (10, 4, 5), "a2": (20, 4, 5), "a3": (30, 4, 5),
        "v0": (0, 4, 25), "v1": (10, 4, 25), "v2": (20, 4, 25), "v3": (30, 4, 25),
        "s1": (10, 4, 12), "w1": (10, 4, 18), "c": (20, 4, 15),
    }
    ids = {name: index for index, name in enumerate(pos)}
    G = nx.MultiGraph()
    for name, p in pos.items():
        G.add_node(ids[name], pos=tuple(float(c) for c in p))
    for a, b in [
        ("a0", "a1"), ("a1", "a2"), ("a2", "a3"), ("v0", "v1"), ("v1", "v2"), ("v2", "v3"),
        ("a1", "s1"), ("s1", "w1"), ("w1", "v1"), ("a2", "c"), ("c", "v2"),
    ]:
        points = _segment(pos[a], pos[b])
        G.add_edge(ids[a], ids[b], voxels=points, length=float(np.linalg.norm(
            np.subtract(pos[b], pos[a]))))
    large_art = np.zeros(shape, dtype=bool); large_art[0:32, 2:7, 3:8] = True
    large_ven = np.zeros(shape, dtype=bool); large_ven[0:32, 2:7, 23:28] = True
    small_art = np.zeros(shape, dtype=bool); small_art[9:12, 3:6, 8:13] = True
    small_ven = np.zeros(shape, dtype=bool); small_ven[9:12, 3:6, 18:23] = True
    image = np.zeros(shape, dtype=np.uint8)
    volume = SkeletonisedVolume(
        image=image, skeleton=image.copy(), voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0), output_dir=Path("."),
    )
    network = VesselNetwork(
        graph=G, volume=volume,
        large_arteriole_mask=large_art, large_venule_mask=large_ven,
        small_arteriole_mask=small_art, small_venule_mask=small_ven,
    )
    return network, ids


def _large_tier_settings(tmp_path):
    from haemolynx.pipeline import default_schema

    settings = default_schema().defaults()
    settings.update({
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": True,
        "cut_network_at_large_vessel_volumes": False,
        "assign_large_vessel_branch_orders": True,
        "automated_vessel_assignment_fast_mode": False,
        "automated_vessel_assignment_enable_overlap_cleanup": False,
        "use_small_vessel_masks_for_boundary_assignment": True,
        # Arteriole/venule pairs: with no venule mask the masks are global.
        "large_venule_mask_path": "large_venule_mask.tif",
        "small_venule_mask_path": "small_venule_mask.tif",
        "write_small_vessel_boundary_labelling_3d_html": False,
        "visualize_results": False,
        "plot_dir": tmp_path,
    })
    for name in (
        "inlet_nodes", "outlet_nodes", "arteriole_boundary_nodes", "venule_boundary_nodes",
        "large_arteriole_boundary_nodes", "large_venule_boundary_nodes",
        "inlet_node_coordinates", "outlet_node_coordinates",
        "arteriole_boundary_node_coordinates", "venule_boundary_node_coordinates",
        "arteriole_boundary_node_volumes", "venule_boundary_node_volumes",
    ):
        settings[name] = []
    return settings


def test_the_large_tier_follows_the_large_masks_through_a_run(tmp_path):
    """From the masks to the orders, as a run assigns them: the Boundaries
    stage tags the vessels lying in the large masks, and the orders read
    those tags. The arteriole's whole trunk is Large_Art and the vein's whole
    trunk Large_Ven, though vessels leave both mid-way; the capillary from
    the arteriole straight to the vein is a capillary."""
    from haemolynx.pipeline.stages import _assign_branch_orders, assign_boundaries

    network, ids = _branching_large_vessel_network()
    settings = _large_tier_settings(tmp_path)

    assign_boundaries(settings, network)
    _assign_branch_orders(settings, network.graph)

    G = network.graph
    assert settings["inlet_nodes"] == [ids["a0"]]
    assert settings["outlet_nodes"] == [ids["v0"]]

    def label(a, b):
        return G.edges[ids[a], ids[b], 0]["branch_order"]

    assert [label("a0", "a1"), label("a1", "a2"), label("a2", "a3")] == [
        "Large_Art1", "Large_Art2", "Large_Art3",
    ]
    assert [label("v0", "v1"), label("v1", "v2"), label("v2", "v3")] == [
        "Large_Ven1", "Large_Ven2", "Large_Ven3",
    ]
    assert label("a1", "s1") == "Art1"
    assert label("w1", "v1") == "Ven1"
    assert label("s1", "w1") == "B01"
    assert [label("a2", "c"), label("c", "v2")] == ["B01", "B02"]


# --- The Art/Ven tier along a small vessel's own mask ------------------------
#
# The small masks' labelling makes every small-arteriole node with an edge
# outside the mask a boundary node: every junction where a capillary leaves the
# vessel, not only where its mask ends. The Art search stopped at the first of
# them, and the rest of the small arteriole was counted as capillaries; Ven
# likewise. The fix for the Large tier (above) did not cover it.


def _mask_tagged_graph(
    small_arteriole, small_venule, other, large_arteriole=(), large_venule=()
) -> nx.MultiGraph:
    """Edges given as node pairs, each group tagged as lying in that mask."""
    G = nx.MultiGraph()
    for edges, tags in (
        (large_arteriole, {LARGE_MASK_VESSEL_TYPE: "arteriole"}),
        (large_venule, {LARGE_MASK_VESSEL_TYPE: "venule"}),
        (small_arteriole, {SMALL_MASK_VESSEL_TYPE: "arteriole"}),
        (small_venule, {SMALL_MASK_VESSEL_TYPE: "venule"}),
        (other, {}),
    ):
        for u, v in edges:
            G.add_edge(u, v, length=1.0, **tags)
    return G


def _small_vessels_with_side_capillaries():
    """A small arteriole 0-1-2-3-4 (inlet 0) and a small venule 13-14-15-16
    (outlet 16), each wholly in its small mask. A capillary leaves the
    arteriole at 2, mid-way, and its mask ends at 4; the bed drains into the
    venule at 13, where its mask starts, and at 15, mid-way.

    The boundary nodes are what the small masks' labelling reports: every
    labelled node with an edge outside its mask -- the side-capillary
    junctions (2, 15) as well as the masks' ends (4, 13).
    """
    G = _mask_tagged_graph(
        small_arteriole=[(0, 1), (1, 2), (2, 3), (3, 4)],
        small_venule=[(13, 14), (14, 15), (15, 16)],
        other=[(2, 10), (4, 11), (10, 12), (11, 12), (12, 13), (12, 22), (22, 15)],
    )
    return G, dict(
        inlet_nodes=[0],
        outlet_nodes=[16],
        arteriole_boundary_nodes=[2, 4],
        venule_boundary_nodes=[13, 15],
    )


def test_the_art_and_ven_tiers_run_along_the_small_masks_past_side_capillaries():
    """Regression: Art ended at 2, the first junction where a capillary leaves
    the small arteriole, and 2-3-4 were counted as capillaries (B01); Ven ended
    at 15 and 13-14-15 were B04-B05. Each tier now runs on along its small
    mask; only the vessels leaving it are capillaries, counted from it."""
    G, nodes = _small_vessels_with_side_capillaries()

    summary = assign_vessel_branch_orders(G, **nodes)

    assert summary["mode"] == "hierarchical"
    assert summary["arteriole_edge_count"] == 4
    assert summary["venule_edge_count"] == 3
    assert _labels(G) == {
        (0, 1, 0): "Art1", (1, 2, 0): "Art2", (2, 3, 0): "Art3", (3, 4, 0): "Art4",
        (15, 16, 0): "Ven1", (14, 15, 0): "Ven2", (13, 14, 0): "Ven3",
        (2, 10, 0): "B01", (4, 11, 0): "B01", (10, 12, 0): "B02", (11, 12, 0): "B02",
        (12, 13, 0): "B03", (12, 22, 0): "B03", (15, 22, 0): "B04",
    }


def test_boundary_nodes_chosen_by_hand_still_end_the_tiers():
    """With no small-mask tags -- boundary nodes picked by hand (coordinates,
    volumes, node IDs) -- Art and Ven end at the boundary nodes, as before:
    the user chose them as the tiers' ends."""
    G, nodes = _small_vessels_with_side_capillaries()
    for data in G.edges.values():
        data.pop(SMALL_MASK_VESSEL_TYPE, None)

    assign_vessel_branch_orders(G, **nodes)

    assert _labels(G) == {
        (0, 1, 0): "Art1", (1, 2, 0): "Art2", (15, 16, 0): "Ven1",
        (2, 3, 0): "B01", (3, 4, 0): "B01", (2, 10, 0): "B01", (4, 11, 0): "B01",
        (10, 12, 0): "B02", (11, 12, 0): "B02", (12, 13, 0): "B03", (12, 22, 0): "B03",
        (13, 14, 0): "B04", (15, 22, 0): "B04", (14, 15, 0): "B05",
    }


def test_a_small_mask_piece_reached_only_through_capillaries_is_not_taken():
    """The tier runs on along its small mask, never across a capillary: a
    piece of the arteriole mask in the bed (12-30-31), joined to the small
    arteriole only through capillaries, stays capillary."""
    G, nodes = _small_vessels_with_side_capillaries()
    G.add_edge(12, 30, length=1.0, **{SMALL_MASK_VESSEL_TYPE: "arteriole"})
    G.add_edge(30, 31, length=1.0, **{SMALL_MASK_VESSEL_TYPE: "arteriole"})
    nodes["arteriole_boundary_nodes"] = [2, 4, 12]

    assign_vessel_branch_orders(G, **nodes)

    labels = _labels(G)
    assert labels[(12, 30, 0)] == "B01"
    assert labels[(30, 31, 0)] == "B02"
    assert sorted(e for e, label in labels.items() if label.startswith("Art")) == [
        (0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0),
    ]


def _small_vessels_beside_large_vessels():
    """A large arteriole 0-1-2 (inlet 0) and a large vein 13-14-15 (outlet
    15). A small arteriole leaves the large one at 1 through a vessel too
    little in its mask to be tagged (1-3, as where it opens out of the large
    mask), runs on in its mask 3-4-5, and has a capillary leave it at 4. A
    small venule 9-10-11 takes a capillary at 10 and reaches the vein at 13
    through 11-13, also untagged."""
    G = _mask_tagged_graph(
        large_arteriole=[(0, 1), (1, 2)],
        large_venule=[(13, 14), (14, 15)],
        small_arteriole=[(3, 4), (4, 5)],
        small_venule=[(9, 10), (10, 11)],
        other=[(1, 3), (4, 6), (5, 7), (6, 8), (7, 8), (8, 9), (8, 12), (12, 10), (11, 13)],
    )
    return G, dict(
        inlet_nodes=[0],
        outlet_nodes=[15],
        arteriole_boundary_nodes=[3, 4, 5],
        venule_boundary_nodes=[9, 10, 11],
        large_arteriole_boundary_nodes=[1],
        large_venule_boundary_nodes=[13],
    )


def test_beside_a_large_tier_the_small_tiers_run_on_along_their_masks():
    """Regression: from the hand-off at 1, Art took the side branch only up
    to the first boundary node (1-3) and 3-4-5 were capillaries (B01); Ven
    likewise took 11-13 and left 10-11, 9-10 as B05, B04. The side branch up
    to the small mask stays Art1/Ven1, and the tier runs on along it."""
    G, nodes = _small_vessels_beside_large_vessels()

    summary = assign_vessel_branch_orders(G, **nodes)

    assert summary["arteriole_edge_count"] == 3
    assert summary["venule_edge_count"] == 3
    assert _labels(G) == {
        (0, 1, 0): "Large_Art1", (1, 2, 0): "Large_Art2",
        (14, 15, 0): "Large_Ven1", (13, 14, 0): "Large_Ven2",
        (1, 3, 0): "Art1", (3, 4, 0): "Art2", (4, 5, 0): "Art3",
        (11, 13, 0): "Ven1", (10, 11, 0): "Ven2", (9, 10, 0): "Ven3",
        (4, 6, 0): "B01", (5, 7, 0): "B01", (6, 8, 0): "B02", (7, 8, 0): "B02",
        (8, 9, 0): "B03", (8, 12, 0): "B03", (10, 12, 0): "B04",
    }


def _small_vessel_network():
    """1 um voxels, every vessel in the plane y = 4, no large masks. A small
    arteriole runs along x at z = 10 from the inlet i (x = 0) through p to q
    (x = 20), where its 5 x 5 um mask ends; a small venule runs in its own
    mask from r (x = 30) through t to the outlet o (x = 49). A capillary
    leaves the arteriole at p and joins the venule at t, round through c1 and
    c3 at z = 20; another runs straight from q to r."""
    from haemolynx.pipeline.stages import SkeletonisedVolume, VesselNetwork

    shape = (30, 9, 50)
    pos = {
        "i": (10, 4, 0), "p": (10, 4, 10), "q": (10, 4, 20), "r": (10, 4, 30),
        "t": (10, 4, 40), "o": (10, 4, 49), "c1": (20, 4, 10), "c3": (20, 4, 40),
    }
    ids = {name: index for index, name in enumerate(pos)}
    G = nx.MultiGraph()
    for name, p in pos.items():
        G.add_node(ids[name], pos=tuple(float(c) for c in p))
    for a, b in [
        ("i", "p"), ("p", "q"), ("q", "r"), ("r", "t"), ("t", "o"),
        ("p", "c1"), ("c1", "c3"), ("c3", "t"),
    ]:
        G.add_edge(ids[a], ids[b], voxels=_segment(pos[a], pos[b]), length=float(
            np.linalg.norm(np.subtract(pos[b], pos[a]))))
    small_art = np.zeros(shape, dtype=bool); small_art[8:13, 2:7, 0:22] = True
    small_ven = np.zeros(shape, dtype=bool); small_ven[8:13, 2:7, 29:50] = True
    image = np.zeros(shape, dtype=np.uint8)
    volume = SkeletonisedVolume(
        image=image, skeleton=image.copy(), voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0), output_dir=Path("."),
    )
    network = VesselNetwork(
        graph=G, volume=volume, small_arteriole_mask=small_art, small_venule_mask=small_ven,
    )
    return network, ids


def _small_mask_settings(tmp_path, ids):
    from haemolynx.pipeline import default_schema

    settings = default_schema().defaults()
    settings.update({
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": False,
        "use_small_vessel_masks_for_boundary_assignment": True,
        "small_venule_mask_path": "small_venule_mask.tif",
        "inlet_node_selection_method": "node_ids",
        "inlet_node_ids": [ids["i"]],
        "outlet_node_selection_method": "node_ids",
        "outlet_node_ids": [ids["o"]],
        "write_small_vessel_boundary_labelling_3d_html": False,
        "visualize_results": False,
        "plot_dir": tmp_path,
    })
    for name in (
        "inlet_nodes", "outlet_nodes", "arteriole_boundary_nodes", "venule_boundary_nodes",
        "large_arteriole_boundary_nodes", "large_venule_boundary_nodes",
        "inlet_node_coordinates", "outlet_node_coordinates",
        "arteriole_boundary_node_coordinates", "venule_boundary_node_coordinates",
        "arteriole_boundary_node_volumes", "venule_boundary_node_volumes",
    ):
        settings[name] = []
    return settings


def test_the_small_tiers_follow_the_small_masks_through_a_run(tmp_path):
    """From the masks to the orders, as a run assigns them: the Boundaries
    stage tags the vessels lying in the small masks, and Art and Ven run on
    along them past the capillaries leaving at p and t. A later run whose
    boundary nodes are picked by hand clears the tags, and its tiers end at
    the nodes picked."""
    from haemolynx.pipeline.stages import _assign_branch_orders, assign_boundaries

    network, ids = _small_vessel_network()
    settings = _small_mask_settings(tmp_path, ids)
    G = network.graph

    def label(a, b):
        return G.edges[ids[a], ids[b], 0]["branch_order"]

    def tag(a, b):
        return G.edges[ids[a], ids[b], 0].get(SMALL_MASK_VESSEL_TYPE)

    assign_boundaries(settings, network)
    _assign_branch_orders(settings, G)

    assert sorted(settings["arteriole_boundary_nodes"]) == sorted([ids["p"], ids["q"]])
    assert sorted(settings["venule_boundary_nodes"]) == sorted([ids["r"], ids["t"]])
    assert [tag("i", "p"), tag("p", "q"), tag("r", "t"), tag("t", "o")] == [
        "arteriole", "arteriole", "venule", "venule",
    ]
    assert tag("q", "r") is tag("p", "c1") is tag("c3", "t") is None
    assert [label("i", "p"), label("p", "q")] == ["Art1", "Art2"]
    assert [label("t", "o"), label("r", "t")] == ["Ven1", "Ven2"]
    assert [label("p", "c1"), label("q", "r"), label("c1", "c3"), label("c3", "t")] == [
        "B01", "B01", "B02", "B03",
    ]

    settings["use_small_vessel_masks_for_boundary_assignment"] = False
    settings.update({
        "arteriole_boundary_selection_method": "node_ids",
        "arteriole_boundary_node_ids": [ids["p"], ids["q"]],
        "venule_boundary_selection_method": "node_ids",
        "venule_boundary_node_ids": [ids["r"], ids["t"]],
    })
    assign_boundaries(settings, network)
    _assign_branch_orders(settings, G)

    assert not [d for d in G.edges.values() if SMALL_MASK_VESSEL_TYPE in d]
    assert [label("i", "p"), label("p", "q")] == ["Art1", "B01"]
    assert [label("t", "o"), label("r", "t")] == ["Ven1", "B02"]


# --- A Large tier from the large masks alone -----------------------------------
#
# Hierarchical mode needs arteriole and venule boundary nodes, which only the
# small masks or a hand pick give. A run with the large masks alone fell back
# to B orders from the inlets, so its large arteriole was B01 and the
# Large_Art/Large_Ven tags were computed and never read (E14.5, 2026-10-21).


def _large_vessels_without_small_ones():
    G, nodes = _capillaries_from_large_arteriole_to_large_vein()
    return G, dict(nodes, arteriole_boundary_nodes=[], venule_boundary_nodes=[])


def test_the_large_masks_alone_give_the_large_tiers_and_capillaries_beside_them():
    G, nodes = _large_vessels_without_small_ones()

    summary = assign_vessel_branch_orders(G, **nodes)

    labels = _labels(G)
    assert summary["mode"] == "large_tier"
    assert summary["large_arteriole_edge_count"] == 2
    assert summary["large_venule_edge_count"] == 2
    assert [labels[(0, 1, 0)], labels[(1, 2, 0)]] == ["Large_Art1", "Large_Art2"]
    assert [labels[(9, 10, 0)], labels[(8, 9, 0)]] == ["Large_Ven1", "Large_Ven2"]
    # No Art/Ven tier: every other vessel is a capillary counted from where it
    # leaves the large arteriole.
    assert [labels[(1, 5, 0)], labels[(5, 8, 0)]] == ["B01", "B02"]
    assert [labels[(2, 3, 0)], labels[(3, 4, 0)], labels[(4, 7, 0)], labels[(7, 8, 0)]] == [
        "B01", "B02", "B03", "B03",  # 8 is 2 hops from 1 through 5 as well
    ]


def test_strict_orders_do_not_complain_of_small_vessels_a_run_never_asked_for():
    G, nodes = _large_vessels_without_small_ones()

    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingSmallVesselAssignmentWarning)
        summary = assign_vessel_branch_orders(G, **nodes, strict_hierarchical=True)

    assert summary["mode"] == "large_tier"


def test_large_masks_alone_with_no_tagged_vessel_fall_back_to_capillary_orders():
    from haemolynx.graph import MissingLargeVesselTierWarning

    G, nodes = _large_vessels_without_small_ones()
    for _u, _v, data in G.edges(data=True):
        data.pop(LARGE_MASK_VESSEL_TYPE, None)

    with pytest.warns(MissingLargeVesselTierWarning):
        summary = assign_vessel_branch_orders(G, **nodes)

    assert summary["mode"] == "capillary"
    assert _labels(G)[(0, 1, 0)] == "B01"


def test_the_capillary_beds_feed_from_the_large_arteriole_without_small_vessels():
    from haemolynx.graph.branch_order import capillary_start_nodes

    G, nodes = _large_vessels_without_small_ones()
    assign_vessel_branch_orders(G, **nodes)

    feeds, drains = capillary_start_nodes([0], [10], [], [], graph=G)

    assert sorted(feeds) == [0, 1, 2], "the inlet and both places a capillary leaves the tier"
    assert sorted(drains) == [8, 10]


def test_the_large_tier_follows_the_large_masks_alone_through_a_run(tmp_path):
    from haemolynx.pipeline.stages import _assign_branch_orders, assign_boundaries

    network, ids = _branching_large_vessel_network()
    network.small_arteriole_mask = network.small_venule_mask = None
    settings = _large_tier_settings(tmp_path)
    settings["use_small_vessel_masks_for_boundary_assignment"] = False

    assign_boundaries(settings, network)
    summary = _assign_branch_orders(settings, network.graph)

    G = network.graph

    def label(a, b):
        return G.edges[ids[a], ids[b], 0]["branch_order"]

    assert summary["mode"] == "large_tier"
    assert settings["arteriole_boundary_nodes"] == settings["venule_boundary_nodes"] == []
    assert [label("a0", "a1"), label("a1", "a2"), label("a2", "a3")] == [
        "Large_Art1", "Large_Art2", "Large_Art3",
    ]
    assert [label("v0", "v1"), label("v1", "v2"), label("v2", "v3")] == [
        "Large_Ven1", "Large_Ven2", "Large_Ven3",
    ]
    assert [label("a1", "s1"), label("s1", "w1"), label("w1", "v1")] == ["B01", "B02", "B03"]
    assert [label("a2", "c"), label("c", "v2")] == ["B01", "B02"]


def test_small_masks_finding_no_venule_boundary_node_warn_what_is_left():
    """Without strict assignment the run fell back in silence, and lost the
    Art tier the arteriole side had found too."""
    G, nodes = _capillaries_from_large_arteriole_to_large_vein()
    nodes = dict(nodes, venule_boundary_nodes=[])

    with pytest.warns(MissingSmallVesselAssignmentWarning, match=r"venule_boundary_nodes came out empty.*Large_Art/Large_Ven"):
        summary = assign_vessel_branch_orders(G, **nodes, expects_hierarchical=True)

    assert summary["mode"] == "large_tier"


def test_a_run_expecting_no_small_vessels_gets_no_such_warning():
    G, nodes = _large_vessels_without_small_ones()

    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingSmallVesselAssignmentWarning)
        assign_vessel_branch_orders(G, **nodes)
