"""Integrated synthetic statistics test for branch-aware vessel graphs.

This test builds a synthetic network with:
- defined input node
- defined arteriole->capillary transition node
- branched capillary tree
- defined capillary->venule transition node
- defined output node

It then runs the in-package branch-order/statistics pipeline, checks the results against
precomputed values, exports:
- <image_stem>_statistics.csv
- <image_stem>_branch_statistics.csv

and reads both CSVs back to check they carry the same values.

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


def _read_statistics_csv(path: Path) -> dict[str, str]:
    """Metric -> printed value, with the unit put back on the name as the stats dict has it."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["Section"] != "Metadata"]
    return {
        (f"{row['Metric']} ({row['Unit']})" if row["Unit"] else row["Metric"]): row["Value"]
        for row in rows
    }


def _read_branch_csv(path: Path) -> dict[str, dict[str, str]]:
    """Branch order -> its printed row, skipping the '#' ordering note."""
    with path.open(newline="", encoding="utf-8") as f:
        return {
            row["Branch Order"]: row
            for row in csv.DictReader(f)
            if not row["Branch Order"].startswith("#")
        }


def test_synthetic_network_statistics(tmp_path: Path) -> None:
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
        voxel_size=(1.0, 1.0, 1.0),
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

    # The exported CSVs print six significant figures; read back, they carry the same values.
    exported = _read_statistics_csv(stats_csv)
    for metric, value in expected_stats.items():
        if isinstance(value, str):
            assert exported[metric] == value, metric
        else:
            assert float(exported[metric]) == pytest.approx(float(value), rel=1e-5), metric

    exported_branches = _read_branch_csv(branch_csv)
    assert list(exported_branches) == ["Art1", "Art2", "BO1", "BO2", "Ven1"]
    for branch, expected in expected_branch_stats.items():
        row = exported_branches[branch]
        assert int(row["Edge Count"]) == int(expected["Edge Count"])
        assert float(row["Mean Length (microns)"]) == pytest.approx(
            float(expected["Mean Length (microns)"]), rel=1e-5
        )
        assert float(row["Mean Tortuosity Index"]) == pytest.approx(
            float(expected["Mean Tortuosity Index"]), rel=1e-5
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
