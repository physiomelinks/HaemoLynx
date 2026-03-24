"""Integrated synthetic statistics test for branch-aware vessel graphs.

This test builds a synthetic network with:
- defined input node
- defined arteriole->capillary transition node
- branched capillary tree
- defined capillary->venule transition node
- defined output node

It then runs the in-package branch-order/statistics pipeline, exports:
- <image_stem>_statistics.csv
- <image_stem>_branch_statistics.csv

and also writes two precomputed ground-truth CSV files for validation:
- <image_stem>_statistics_ground_truth.csv
- <image_stem>_branch_statistics_ground_truth.csv

Run as script:  python tests/test_synthetic_network_statistics.py
Run with pytest: pytest tests/test_synthetic_network_statistics.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
DEMO_OUTPUT_DIR = REPO_ROOT / "examples" / "outputs" / "synthetic_network_statistics"

from ImageLynx.graph import assign_hierarchical_branch_orders, calculate_edge_length
from ImageLynx.statistics import (
    compute_branch_order_statistics,
    compute_comprehensive_vessel_statistics,
    export_branch_order_statistics_to_csv,
    export_statistics_to_csv,
)


def _build_synthetic_network() -> tuple[nx.MultiGraph, dict[str, int]]:
    """Create a small branched network with explicit transition nodes."""
    G = nx.MultiGraph()

    # Coordinates are (z, y, x) to match ImageLynx conventions.
    positions = {
        0: (0.0, 0.0, 0.0),    # input terminal
        1: (0.0, 0.0, 10.0),
        2: (0.0, 0.0, 20.0),   # arteriole -> capillary transition
        3: (5.0, 0.0, 30.0),   # capillary branch A
        4: (-5.0, 0.0, 30.0),  # capillary branch B
        5: (0.0, 0.0, 40.0),   # capillary -> venule transition
        6: (0.0, 0.0, 50.0),   # output terminal
    }
    for node_id, pos in positions.items():
        G.add_node(node_id, pos=np.asarray(pos, dtype=float))

    # Store edge weights only; length is intentionally derived via
    # calculate_edge_length to exercise the pipeline length function.
    weighted_edges = [
        (0, 1, 12.0),  # Art1
        (1, 2, 15.0),  # Art2
        (2, 3, 14.0),  # BO1
        (3, 5, 13.0),  # BO2
        (2, 4, 16.0),  # BO1
        (4, 5, 12.0),  # BO2
        (5, 6, 11.0),  # Ven1
    ]
    for u, v, weight in weighted_edges:
        G.add_edge(u, v, weight=float(weight))

    node_roles = {
        "input_node": 0,
        "arteriole_boundary_node": 2,
        "venule_boundary_node": 5,
        "output_node": 6,
    }
    return G, node_roles


def _apply_pipeline_length_measurements(G: nx.MultiGraph) -> None:
    """Populate edge length using ImageLynx pipeline helper."""
    for u, v, key, data in G.edges(keys=True, data=True):
        length = float(calculate_edge_length(u, v, data))
        data["length"] = length
        # Keep shortest-path weighting aligned to physical path length.
        data["weight"] = length


def _write_ground_truth_statistics_csv(
    expected_stats: dict[str, float | int | str], output_csv_path: Path
) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Expected Value", "Notes"])
        for metric, value in expected_stats.items():
            writer.writerow([metric, value, "Precomputed synthetic ground truth"])


def _write_ground_truth_branch_csv(
    expected_branch_stats: dict[str, dict[str, float | int | str]],
    output_csv_path: Path,
) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Branch Order",
                "Expected Edge Count",
                "Expected Mean Length (microns)",
                "Expected Mean Tortuosity Index",
                "Notes",
            ]
        )
        for branch in ["Art1", "Art2", "BO1", "BO2", "Ven1"]:
            vals = expected_branch_stats[branch]
            writer.writerow(
                [
                    branch,
                    vals["Edge Count"],
                    vals["Mean Length (microns)"],
                    vals["Mean Tortuosity Index"],
                    "Precomputed synthetic ground truth",
                ]
            )


def _write_branch_labelled_3d_html(G: nx.MultiGraph, output_html_path: Path) -> bool:
    """Write a rotatable 3D HTML with branch labels on each edge."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return False

    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        return False

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()

    # Draw edges and annotate each edge midpoint with branch label.
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    line_z: list[float | None] = []
    text_x: list[float] = []
    text_y: list[float] = []
    text_z: list[float] = []
    text_labels: list[str] = []

    for u, v, _, data in G.edges(keys=True, data=True):
        pu = np.asarray(pos[u], dtype=float)
        pv = np.asarray(pos[v], dtype=float)
        line_x += [float(pu[2]), float(pv[2]), None]
        line_y += [float(pu[1]), float(pv[1]), None]
        line_z += [float(pu[0]), float(pv[0]), None]
        mid = 0.5 * (pu + pv)
        text_x.append(float(mid[2]))
        text_y.append(float(mid[1]))
        text_z.append(float(mid[0]))
        text_labels.append(str(data.get("branch_order", "Unassigned")))

    fig.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line=dict(color="rgba(60,60,60,0.9)", width=6),
            name="Vessel edges",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=text_x,
            y=text_y,
            z=text_z,
            mode="text",
            text=text_labels,
            textposition="top center",
            textfont=dict(size=11, color="black"),
            name="Branch labels",
        )
    )

    node_x = [float(np.asarray(pos[n], dtype=float)[2]) for n in G.nodes()]
    node_y = [float(np.asarray(pos[n], dtype=float)[1]) for n in G.nodes()]
    node_z = [float(np.asarray(pos[n], dtype=float)[0]) for n in G.nodes()]
    fig.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            marker=dict(size=5, color="#2ca02c"),
            name="Nodes",
        )
    )

    fig.update_layout(
        title="Synthetic network statistics test (branch-labelled)",
        showlegend=True,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
    )
    fig.write_html(str(output_html_path), include_plotlyjs="cdn")
    return True


@pytest.mark.plotting
def test_synthetic_network_statistics(tmp_path: Path) -> None:
    pytest.importorskip("plotly.graph_objects")

    image_stem = "synthetic_network"
    out_dir = tmp_path / "synthetic_statistics_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    G, roles = _build_synthetic_network()
    _apply_pipeline_length_measurements(G)

    assign_hierarchical_branch_orders(
        G,
        starting_nodes=[roles["input_node"]],
        output_nodes=[roles["output_node"]],
        arteriole_boundary_nodes=[roles["arteriole_boundary_node"]],
        venule_boundary_nodes=[roles["venule_boundary_node"]],
    )

    node_positions = nx.get_node_attributes(G, "pos")
    stats = compute_comprehensive_vessel_statistics(
        G,
        node_positions=node_positions,
        image_dimensions=(60, 20, 60),
        statistics_mode="fast",
    )
    branch_stats = compute_branch_order_statistics(G, node_positions=node_positions)

    # Pipeline-generated CSVs.
    stats_csv = out_dir / f"{image_stem}_statistics.csv"
    branch_csv = out_dir / f"{image_stem}_branch_statistics.csv"
    export_statistics_to_csv(stats, stats_csv)
    export_branch_order_statistics_to_csv(branch_stats, branch_csv)

    # Precomputed ground-truth values.
    straight_diag = math.sqrt(125.0)  # sqrt(5^2 + 10^2)
    expected_stats = {
        "Total Nodes": 7,
        "Total Edges": 7,
        "Total Edge Length (microns)": 93.0,
        "Average Edge Length (microns)": 93.0 / 7.0,
        "Average Degree": 2.0,
        "Number of Branching Points": 2,
        "Statistics Mode": "fast",
    }
    expected_branch_stats = {
        "Art1": {
            "Edge Count": 1,
            "Mean Length (microns)": 12.0,
            "Mean Tortuosity Index": 1.2,
        },
        "Art2": {
            "Edge Count": 1,
            "Mean Length (microns)": 15.0,
            "Mean Tortuosity Index": 1.5,
        },
        "BO1": {
            "Edge Count": 2,
            "Mean Length (microns)": 15.0,
            "Mean Tortuosity Index": 15.0 / straight_diag,
        },
        "BO2": {
            "Edge Count": 2,
            "Mean Length (microns)": 12.5,
            "Mean Tortuosity Index": 12.5 / straight_diag,
        },
        "Ven1": {
            "Edge Count": 1,
            "Mean Length (microns)": 11.0,
            "Mean Tortuosity Index": 1.1,
        },
    }

    gt_stats_csv = out_dir / f"{image_stem}_statistics_ground_truth.csv"
    gt_branch_csv = out_dir / f"{image_stem}_branch_statistics_ground_truth.csv"
    _write_ground_truth_statistics_csv(expected_stats, gt_stats_csv)
    _write_ground_truth_branch_csv(expected_branch_stats, gt_branch_csv)

    # Numeric validation against precomputed values.
    assert stats["Total Nodes"] == expected_stats["Total Nodes"]
    assert stats["Total Edges"] == expected_stats["Total Edges"]
    assert stats["Number of Branching Points"] == expected_stats["Number of Branching Points"]
    assert stats["Statistics Mode"] == expected_stats["Statistics Mode"]
    assert float(stats["Total Edge Length (microns)"]) == pytest.approx(
        float(expected_stats["Total Edge Length (microns)"]), rel=1e-12
    )
    assert float(stats["Average Edge Length (microns)"]) == pytest.approx(
        float(expected_stats["Average Edge Length (microns)"]), rel=1e-12
    )
    assert float(stats["Average Degree"]) == pytest.approx(
        float(expected_stats["Average Degree"]), rel=1e-12
    )

    assert list(branch_stats.keys()) == ["Art1", "Art2", "BO1", "BO2", "Ven1"]
    for branch, expected in expected_branch_stats.items():
        actual = branch_stats[branch]
        assert int(actual["Edge Count"]) == int(expected["Edge Count"])
        assert float(actual["Mean Length (microns)"]) == pytest.approx(
            float(expected["Mean Length (microns)"]), rel=1e-12
        )
        assert float(actual["Mean Tortuosity Index"]) == pytest.approx(
            float(expected["Mean Tortuosity Index"]), rel=1e-12
        )

    html_path = out_dir / f"{image_stem}_branch_labelled_3d.html"
    assert _write_branch_labelled_3d_html(G, html_path)

    # Also emit a stable demo copy under examples/outputs for easy inspection
    # when running this test file directly.
    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo_stats_csv = DEMO_OUTPUT_DIR / f"{image_stem}_statistics.csv"
    demo_branch_csv = DEMO_OUTPUT_DIR / f"{image_stem}_branch_statistics.csv"
    demo_gt_stats_csv = DEMO_OUTPUT_DIR / f"{image_stem}_statistics_ground_truth.csv"
    demo_gt_branch_csv = DEMO_OUTPUT_DIR / f"{image_stem}_branch_statistics_ground_truth.csv"
    demo_html_path = DEMO_OUTPUT_DIR / f"{image_stem}_branch_labelled_3d.html"
    export_statistics_to_csv(stats, demo_stats_csv)
    export_branch_order_statistics_to_csv(branch_stats, demo_branch_csv)
    _write_ground_truth_statistics_csv(expected_stats, demo_gt_stats_csv)
    _write_ground_truth_branch_csv(expected_branch_stats, demo_gt_branch_csv)
    assert _write_branch_labelled_3d_html(G, demo_html_path)

    # Ensure all expected output artifacts were generated.
    for p in [stats_csv, branch_csv, gt_stats_csv, gt_branch_csv, html_path]:
        assert p.exists()
        assert p.stat().st_size > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
